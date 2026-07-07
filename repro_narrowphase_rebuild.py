"""Minimal standalone repro for the Newton CUDA-700 collision crash.

WHAT THE BUG IS (confirmed 2026-07-06)
--------------------------------------
CollisionPipeline.collide(state, contacts) launches its FIRST kernel,
``compute_shape_aabbs``, with ``dim = self.model.shape_count`` and reads
``self.model.shape_transform / shape_body / shape_type / ...`` -- i.e. the
device arrays of the Model the pipeline was BUILT on (``self.model``), NOT the
Model backing the ``state`` argument.

In IsaacLab, ``env.sim.reset()`` re-finalizes the Model (reassigning
``NewtonManager._model``), so the OLD Model's device arrays are freed. But the
collision pipeline object is NOT rebuilt (``_initialize_contacts`` only builds a
new one ``if _collision_pipeline is None``, and reset historically never nulled
it). The stale pipeline still points at the freed arrays via ``self.model``, so
the next ``collide()`` dereferences freed device memory -> CUDA 700, surfacing
inside the GJK/MPR narrow-phase kernel:

    Warp Error: Error launching kernel:
        create_narrow_phase_kernel_gjk_mpr__locals__narrow_phase_kernel_gjk_mpr
        on device cuda:0: CUDA error detected: 700

This repro reproduces that dangling-pointer condition with pure Newton + Warp:
build a pipeline on model #1, then FREE the model arrays the pipeline reads and
churn the allocator (so the blocks are reclaimed/overwritten), then call
collide() again -> CUDA 700.

CONTROLS (isolate the trigger)
------------------------------
    uv run repro_narrowphase_rebuild.py                       # crash (frees arrays)
    uv run repro_narrowphase_rebuild.py --no-free-model-arrays # keep arrays -> clean
    uv run repro_narrowphase_rebuild.py --no-rebuild           # baseline #1 only -> clean
    uv run repro_narrowphase_rebuild.py --verbose-cuda         # name the faulting kernel

If the default faults inside narrow_phase_kernel_gjk_mpr / compute_shape_aabbs
but --no-free-model-arrays is clean, that confirms the stale-pipeline /
freed-model-arrays root cause. The IsaacLab-side fix is to rebuild the
CollisionPipeline (null _collision_pipeline + _contacts) on hard reset.
"""

import argparse
import math

import warp as wp

import newton


def build_box_pile(num_boxes: int) -> "newton.Model":
    """Finalize a fresh Model: a pile of overlapping boxes + a ground plane.

    Boxes overlap so box-box pairs survive broad phase and get routed to the
    GJK/MPR narrow-phase kernel (box-box is NOT a primitive fast-path combo).
    """
    builder = newton.ModelBuilder()
    side = max(1, int(math.ceil(math.sqrt(num_boxes))))
    spacing = 0.15  # < box full-width (0.2) so neighbours overlap
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= num_boxes:
                break
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(i * spacing, j * spacing, 0.5 + 0.01 * idx),
                    wp.quat_identity(),
                ),
                mass=1.0,
            )
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            idx += 1
    builder.add_ground_plane()
    return builder.finalize()


def run_collide(model, tag, broad_phase="explicit"):
    from newton._src.sim.collide import CollisionPipeline

    pipeline = CollisionPipeline(model, broad_phase=broad_phase)
    state = model.state()
    contacts = pipeline.contacts() if hasattr(pipeline, "contacts") else model.contacts()
    print(f"[repro] {tag}: running collide() (eager, no graph)...")
    pipeline.collide(state, contacts)
    wp.synchronize_device()
    print(f"[repro] {tag}: collide() OK")
    return pipeline, state, contacts


# arrays that CollisionPipeline.collide()'s first kernel (compute_shape_aabbs)
# reads from self.model:
_MODEL_ARRAYS_READ_BY_COLLIDE = (
    "shape_transform", "shape_body", "shape_type", "shape_scale",
    "shape_collision_radius", "shape_source_ptr", "shape_margin", "shape_gap",
    "shape_collision_aabb_lower", "shape_collision_aabb_upper",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-boxes", type=int, default=128)
    parser.add_argument("--num-boxes-2", type=int, default=8)
    parser.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--broad-phase", default="explicit", choices=["explicit", "sap", "nxn"])
    parser.add_argument(
        "--free-model-arrays", action=argparse.BooleanOptionalAction, default=True,
        help="Free the device arrays the stale pipeline reads + churn the allocator, "
             "reproducing IsaacLab's dangling-pointer state after env.sim.reset().")
    parser.add_argument("--verbose-cuda", action="store_true",
                        help="verify_cuda + verbose: name the faulting kernel, sync each launch.")
    args = parser.parse_args()

    wp.init()
    if args.verbose_cuda:
        wp.config.verbose = True
        wp.config.verify_cuda = True
        print("[repro] verify_cuda=True (sync + error-check + kernel name on fault)")

    device = wp.get_device()
    print(f"[repro] device = {device}  cuda={device.is_cuda}")
    print(f"[repro] warp={wp.config.version}  newton={newton.__version__}")

    print(f"[repro] building model #1 ({args.num_boxes} boxes)...")
    model1 = build_box_pile(args.num_boxes)
    pipe1, state1, contacts1 = run_collide(model1, "pipeline #1", broad_phase=args.broad_phase)

    if not args.rebuild:
        print("[repro] --no-rebuild: DONE (baseline, no crash).")
        return

    print(f"[repro] RE-FINALIZING model #2 ({args.num_boxes_2} boxes) -- mimics env.sim.reset()...")
    model2 = build_box_pile(args.num_boxes_2)
    state2 = model2.state()

    if args.free_model_arrays:
        import gc

        m = pipe1.model  # the pipeline's cached (soon-to-be-stale) model
        freed = []
        for attr in _MODEL_ARRAYS_READ_BY_COLLIDE:
            arr = getattr(m, attr, None)
            if isinstance(arr, wp.array) and arr.device.is_cuda:
                freed.append(attr)
                setattr(m, attr, None)  # drop ref so the block can be reclaimed
        arr = None
        gc.collect()
        wp.synchronize_device()
        # Churn the allocator so the freed blocks are reused/overwritten by
        # unrelated allocations -> the pipeline's baked dim=shape_count now reads
        # reclaimed memory (dangling pointers), exactly like the IsaacLab case.
        churn = [wp.zeros(1 << 16, dtype=wp.float32, device=device) for _ in range(128)]  # noqa: F841
        wp.synchronize_device()
        print(f"[repro]   freed + churned pipeline model arrays: {freed}")

    print("[repro] REUSING stale pipeline against a fresh state (expect CUDA 700)...")
    print("[repro] stale collide(): running (eager, no graph)...")
    pipe1.collide(state2, contacts1)
    wp.synchronize_device()
    print("[repro] stale collide(): OK (no crash)")
    print("[repro] DONE without crash.")


if __name__ == "__main__":
    main()
