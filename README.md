# Newton CUDA-graph stale-buffer crash on solver reinit (CUDA 700 / graph-create error 1)

Minimal, **self-contained, asset-free** reproducer for a Newton crash that happens when a
CUDA graph is replayed after the solver's device buffers have been reallocated by a
simulation reset.

## Symptom

In an IsaacLab + Newton workload (H2/G1 + deformable tablecloth) we hit:

```
Warp CUDA error 700: an illegal memory access was encountered
    (in function wp_cuda_graph_launch, warp.cu:4316)
... cascade of 700 in wp_free_device_async (warp.cu:869)
```

This standalone reproducer triggers the same underlying fault. It has two modes:

* `--mode launch_700` (default): launch the captured graph once successfully, then rebuild
  the solver/model under the live graph (freeing + reallocating the buffers it captured),
  then relaunch. This faults **exactly like the real workload**:

  ```
  Warp CUDA error 700: an illegal memory access was encountered
      (in function wp_cuda_graph_launch, warp.cu:4316)
  ... cascade of 700 in wp_free_device_async (warp.cu:869),
      wp_cuda_unload_module, wp_cuda_stream_destroy during teardown
  ```

* `--mode create_err`: reinit first, then replay the stale graph. Warp instantiates the
  graph exec lazily at first launch against already-freed buffers, so it faults earlier at
  graph *instantiation*:

  ```
  Warp CUDA error 1: invalid argument (in function wp_cuda_graph_create_exec, warp.cu:3620)
  ```

  Same root cause, different manifestation depending on whether the graph exec was
  instantiated before or after the buffers were freed.


## Root cause

A CUDA graph captures the **device pointers** of the buffers touched during capture
(solver internals, contacts / soft-contact buffers from `create_soft_contacts` in the
narrow phase, states, etc.).

If a simulation reset then **reallocates** those buffers — e.g. by constructing a fresh
solver + `State` + `Contacts` (or a fresh `Model`) — the previously-captured graph still
references the **old, now-freed** pointers. Replaying (or even re-instantiating) that
graph dereferences freed device memory → CUDA illegal access.

In IsaacLab terms this is `env.sim.reset()` → `PhysicsManager.reset(soft=False)` →
`start_simulation()` + `initialize_solver()`, which rebuild the solver/state/contacts
while a CUDA graph captured before the reset is still held and later launched.

## Reproduce

Requires an NVIDIA GPU. Uses [uv](https://docs.astral.sh/uv/) to pull Newton from GitHub
`main` together with the `sim` extras (mujoco-warp).

```bash
# Create the environment and install Newton (git main) + warp + mujoco-warp.
uv sync

# Default: launch graph OK, rebuild solver/model under the live graph, relaunch -> CUDA 700
uv run repro_cuda700_reset.py --verbose-cuda

# Alternate: reinit before first launch -> faults at graph instantiation (CUDA err 1)
uv run repro_cuda700_reset.py --mode create_err

# Baseline: no reinit -> runs clean
uv run repro_cuda700_reset.py --mode create_err --no-reinit

# Reinit AND re-capture the graph against the fresh buffers -> runs clean
uv run repro_cuda700_reset.py --mode create_err --recapture
```

To pin the exact warp build this was last verified crashing on:

```bash
uv sync --extra pinned
```

## Environment

- GPU: NVIDIA L40 (also seen on Blackwell RTX PRO 5000), driver 570.158.01, CUDA 12.8
- Newton: `main` (verified crashing at HEAD `9bff8911`), also on an earlier snapshot
- warp-lang: `1.16.0.dev20260706` (also `1.14.0`)
- mujoco-warp: `3.10.0.1`
- Python 3.12, Linux x86_64

## Verdict matrix (reproduced on two Newton/Warp combos)

| Config              | warp 1.14.0 / newton snapshot | warp 1.16.0.dev20260706 / newton main `9bff8911` |
|---------------------|-------------------------------|--------------------------------------------------|
| `--mode launch_700` (default) | **CUDA 700 @ graph_launch** | **CUDA 700 @ graph_launch** |
| `--mode create_err` | **CUDA err1 @ graph_create_exec** | **CUDA err1 @ graph_create_exec** |
| `--mode create_err --no-reinit` | clean | clean |
| `--mode create_err --recapture` | clean | clean |

**Updating to the latest Newton (`main`) does not fix it.** Re-capturing the graph after
the reset is the only currently-known safe path.

## What we'd like from Newton

Either:

1. Make a solver/state reset **invalidate** any CUDA graph that captured the old buffers
   (so a stale graph can't be replayed), and/or provide a documented hook to force
   re-capture; **or**
2. Keep buffer identity stable across an in-place solver reset so previously-captured
   graphs remain valid; **or**
3. At minimum, raise a **clear, actionable error** ("CUDA graph was captured before a
   solver reset; re-capture it before the next launch") instead of a raw CUDA-700 /
   `wp_cuda_graph_create_exec` failure.

## Files

- `repro_cuda700_reset.py` — the reproducer (cloth grid + ground plane + VBD solver + soft contacts).
- `repro_narrowphase_rebuild.py` — **sharper, graph-free reproducer** (see below).

---

# Sharper repro: stale CollisionPipeline reused across a model re-finalize (no CUDA graph)

`repro_narrowphase_rebuild.py` isolates the **true root cause** with NO CUDA graph at all —
every launch is eager. It is the most direct minimal repro for the Newton team.

## Mechanism

`CollisionPipeline.collide(state, contacts)` launches its first kernel, `compute_shape_aabbs`,
with `dim = self.model.shape_count`, reading `self.model.shape_transform / shape_body /
shape_type / ...` — i.e. the device arrays of the `Model` the **pipeline** was built on
(`self.model`), *not* the model backing the `state` argument.

In IsaacLab, `env.sim.reset()` re-finalizes the `Model` (reassigning `NewtonManager._model`),
freeing the OLD model's device arrays. The `CollisionPipeline` object is **not** rebuilt
(`_initialize_contacts` only builds a new one `if _collision_pipeline is None`, and reset
historically never nulled it), so it still points at the freed arrays via `self.model`. The
next `collide()` dereferences freed device memory → CUDA 700, surfacing as:

```
Warp Error: Error launching kernel: compute_shape_aabbs on device cuda:0: CUDA error detected: 700
RuntimeError: CUDA error detected: 700
```

(without `CUDA_LAUNCH_BLOCKING=1` the async fault is deferred and instead surfaces at the
later `narrow_phase_kernel_gjk_mpr` launch / the next `wp_cuda_context_synchronize`, then a
cascade of `wp_free_device_async` / `wp_cuda_unload_module` / `wp_cuda_stream_destroy` 700s
at teardown — matching the real workload).

## Reproduce

```bash
# Crash: build pipeline on model #1, FREE the model arrays it reads + churn the
# allocator (reproducing IsaacLab's dangling-pointer state), then collide() again.
CUDA_LAUNCH_BLOCKING=1 uv run repro_narrowphase_rebuild.py --verbose-cuda

# Control: identical, but DON'T free the arrays -> clean, exit 0.
uv run repro_narrowphase_rebuild.py --no-free-model-arrays

# Baseline: pipeline #1 only, no rebuild -> clean.
uv run repro_narrowphase_rebuild.py --no-rebuild
```

The A/B (`--free-model-arrays` faults, `--no-free-model-arrays` is clean, everything else
identical) isolates the freed-model-array read as the sole trigger.

## What we'd like from Newton (narrow-phase / pipeline)

1. **Bounds-safety / clear error:** `compute_shape_aabbs` and `narrow_phase_kernel_gjk_mpr`
   index `shape_*[shape_a]` with only a `>= 0` guard (no upper bound vs the live array
   sizes). A model/state or freed-buffer mismatch should raise a clear error rather than
   perform an out-of-bounds device read → raw CUDA 700.
2. **(Ideally)** a documented invariant that a `CollisionPipeline` is bound to the `Model`
   it was constructed on, and/or a cheap `pipeline.rebind(model)` / validity check so callers
   can detect a stale pipeline after a model re-finalize.

The **caller-side fix** (validated in IsaacLab) is simply to rebuild the pipeline on hard
reset — null `_collision_pipeline` + `_contacts` so a fresh pipeline is built on the
re-finalized model.
