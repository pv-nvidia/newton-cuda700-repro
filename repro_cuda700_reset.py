"""Minimal standalone repro: replaying a CUDA graph after a solver/model reinit -> CUDA error 700.

No IsaacLab, no USD, no assets. Pure Newton + Warp.

Background
----------
In an IsaacLab + Newton workload we observed a hard crash:

    Warp CUDA error 700: an illegal memory access was encountered
        (in function wp_cuda_graph_launch, warp.cu:4316)
    ... cascade of 700 in wp_free_device_async (warp.cu:869)

It is triggered when a full simulation reset (which reinitializes the Newton
solver and reallocates its device buffers, including the soft-contact /
collision buffers) happens *after* a CUDA graph has already been captured, and
then that previously-captured graph is replayed. The graph still references the
OLD (now-freed) device pointers, so the launch dereferences freed memory.

In IsaacLab terms: ``env.sim.reset()`` -> ``PhysicsManager.reset(soft=False)``
-> ``start_simulation()`` + ``initialize_solver()`` re-create the Model/State/
solver and their soft-contact buffers (``soft_contact_max = shape_count *
particle_count``), but a graph captured before the reset is now stale.

This script reproduces the same class of failure with the smallest possible
Newton setup: a hanging cloth grid with soft contacts, a VBD solver, a captured
CUDA graph, then a solver/model/state re-initialization, then a graph replay.

Expected result
---------------
* WITHOUT ``--reinit``: N steps run cleanly (baseline).
* WITH ``--reinit`` (default): after reinitializing the solver/state/contacts,
  replaying the previously-captured graph faults with CUDA error 700.

Run
---
    python repro_cuda700_reset.py            # reproduces the crash
    python repro_cuda700_reset.py --no-reinit  # baseline, should be clean
    python repro_cuda700_reset.py --recapture  # reinit AND recapture -> should be clean

If ``--recapture`` is clean but the default crashes, that confirms the bug is
"stale graph replayed across a solver reinit" and that re-capturing the graph
after the reset is the (only currently known) safe path.
"""

import argparse

import warp as wp

import newton


class ClothScene:
    """Minimal cloth-on-ground scene with soft contacts + VBD solver + CUDA graph."""

    def __init__(self, sim_substeps: int = 10):
        self.sim_substeps = sim_substeps
        self.frame_dt = 1.0 / 60.0
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.graph = None

        self._build()

    def _build(self, dim: int = 40):
        """(Re)build model, solver, states and contacts from scratch.

        This mirrors what IsaacLab's PhysicsManager.reset(soft=False) does:
        finalize a fresh Model (new device buffers) and construct a new solver
        + state + contacts bound to those buffers. ``dim`` controls the cloth
        grid resolution; changing it between builds forces differently-sized
        particle / contact buffers (and different device addresses), which is
        what makes a stale captured graph dereference invalid memory.
        """
        self._dim = dim
        builder = newton.ModelBuilder()

        # A cloth grid that will fall and contact the ground plane -> exercises
        # the soft-contact / collision buffers (create_soft_contacts).
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e1,
            edge_ke=10.0,
        )
        builder.add_ground_plane()

        # VBD requires vertex coloring for its parallel Gauss-Seidel sweeps.
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(model=self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def free_old_buffers(self):
        """Drop all Python references to old solver/state/contacts so their device
        buffers are freed by Warp's allocator, WITHOUT rebuilding new ones yet.

        This isolates the 'buffers freed under a live graph' condition: the graph
        was already successfully instantiated + launched once, then the memory it
        references is returned to the pool. The next launch should then fault at
        wp_cuda_graph_launch (CUDA 700) rather than at graph instantiation.
        """
        self.solver = None
        self.state_0 = None
        self.state_1 = None
        self.control = None
        self.contacts = None
        import gc

        gc.collect()
        # Force the Warp allocator to actually release freed blocks back to CUDA.
        try:
            wp.get_device().synchronize()
            wp.get_device().memory_pool.release_all() if hasattr(wp.get_device(), "memory_pool") else None
        except Exception as e:  # noqa: BLE001
            print(f"[repro]   (mempool release note: {e})")

    def reinit_solver(self, full: bool = False, dim: int | None = None):
        """Simulate a sim reset, leaving the captured graph pointing at old buffers.

        * ``full=True``  : rebuild everything (Model + solver + state + contacts).
        * ``full=False`` : keep the Model, re-create solver + state + contacts.
          This more closely matches IsaacLab's ``initialize_solver()``, which
          builds a fresh solver + State/contacts bound to the existing model,
          reallocating the soft-contact / collision device buffers touched by
          ``create_soft_contacts`` in the narrow phase.
        * ``dim`` (with ``full=True``): rebuild the Model at a DIFFERENT cloth
          resolution, forcing differently-sized buffers at new device addresses.
        """
        old_graph = self.graph
        if full:
            self.solver = None
            self.state_0 = self.state_1 = self.control = self.contacts = None
            self.model = None
            import gc

            gc.collect()
            self._build(dim=dim if dim is not None else self._dim)  # new model + everything
        else:
            self.solver = newton.solvers.SolverVBD(model=self.model, iterations=10)
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            self.contacts = self.model.contacts()
        self.graph = old_graph  # keep the stale graph handle (the footgun)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument(
        "--reinit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reinitialize the solver/model/state after capture, then replay the STALE graph.",
    )
    parser.add_argument(
        "--recapture",
        action="store_true",
        help="After --reinit, re-capture the graph against the fresh buffers (should be safe).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="On reinit, rebuild the whole Model (not just solver/state/contacts).",
    )
    parser.add_argument(
        "--mode",
        choices=["create_err", "launch_700"],
        default="launch_700",
        help="create_err: reinit then replay stale graph (faults at graph instantiation, err1). "
        "launch_700: launch graph successfully ONCE, then free/realloc buffers, then relaunch "
        "(targets CUDA 700 at wp_cuda_graph_launch, matching the real workload).",
    )
    parser.add_argument(
        "--verbose-cuda",
        action="store_true",
        help="Enable Warp verbose + verify_cuda (sync+error-check after each launch) to pinpoint "
        "WHICH launch first errored, i.e. whether 700 is a re-surfaced earlier async error.",
    )
    parser.add_argument(
        "--resize-dim",
        type=int,
        default=80,
        help="In launch_700 mode, rebuild the cloth at this grid dim (default 80 vs initial 40) "
        "to force differently-sized buffers at new addresses under the live graph.",
    )
    args = parser.parse_args()

    if args.verbose_cuda:
        wp.config.verbose = True
        print("[repro] wp.config.verbose=True (verify_cuda enabled after capture)")

    wp.init()
    device = wp.get_device()
    print(f"[repro] device = {device}  cuda={device.is_cuda}")
    print(f"[repro] warp={wp.config.version}  newton={newton.__version__}")

    scene = ClothScene(sim_substeps=args.substeps)

    # Warm up once eagerly so any lazy buffers are allocated before capture.
    scene.simulate()
    wp.synchronize_device()

    print("[repro] capturing CUDA graph...")
    scene.capture()
    wp.synchronize_device()

    if args.mode == "launch_700":
        # 1) Launch the captured graph ONCE successfully -> forces Warp to
        #    instantiate the graph exec against the CURRENT (valid) buffers.
        print("[repro] launch_700: launching graph once (should instantiate + run OK)...")
        scene.step()
        wp.synchronize_device()
        print("[repro]   first launch ok (graph exec instantiated)")

        # 2) Tear down the old solver/state/contacts and rebuild at a DIFFERENT
        #    cloth resolution. This frees the buffers the graph captured and
        #    allocates differently-sized ones (new addresses / layout) -- exactly
        #    the hazard env.sim.reset() creates when it rebuilds the Newton solver
        #    under a live graph. The stale graph's kernels still use the OLD
        #    pointers/sizes.
        print(f"[repro] launch_700: rebuilding model at dim={args.resize_dim} under live graph...")
        scene.reinit_solver(full=True, dim=args.resize_dim)
        wp.synchronize_device()

        # Turn on per-launch CUDA error verification now (illegal during capture).
        if args.verbose_cuda:
            wp.config.verify_cuda = True
            print("[repro] verify_cuda=True (sync+error-check after each launch)")

        # 3) Re-launch the SAME already-instantiated graph -> should now fault at
        #    wp_cuda_graph_launch with CUDA 700 (illegal access on freed mem).
        print(f"[repro] launch_700: re-launching stale graph {args.steps}x (expect CUDA 700)...")
        for i in range(args.steps):
            wp.capture_launch(scene.graph)
            wp.synchronize_device()
            print(f"[repro]   relaunch {i} ok")
        print("[repro] DONE without crash.")
        return

    if args.reinit:
        print("[repro] reinitializing solver/model/state (simulates env.sim.reset())...")
        scene.reinit_solver(full=args.full)
        wp.synchronize_device()
        if args.recapture:
            print("[repro] re-capturing graph against fresh buffers...")
            scene.capture()
            wp.synchronize_device()

    print(f"[repro] running {args.steps} steps (replaying graph)...")
    for i in range(args.steps):
        scene.step()
        wp.synchronize_device()
        print(f"[repro]   step {i} ok")

    print("[repro] DONE without crash.")


if __name__ == "__main__":
    main()
