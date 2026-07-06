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

## Is the 700 a re-surfaced earlier async error?

No. With `--verbose-cuda` (which sets `wp.config.verify_cuda=True`, forcing a
`cudaDeviceSynchronize` + explicit error check after **every** launch), the first launch
and the rebuild both complete cleanly; the error originates at the **first relaunch after
the rebuild**, at `wp_cuda_graph_launch`. The subsequent 700s in `wp_free_device_async` /
`wp_cuda_unload_module` / `wp_cuda_stream_destroy` are the *cascade* from CUDA's sticky
error state during teardown, not independent faults.

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

Requires an NVIDIA GPU + a Newton install with the `sim` extras (mujoco-warp).

```bash
# Default: launch graph OK, rebuild solver/model under the live graph, relaunch -> CUDA 700
python repro_cuda700_reset.py --verbose-cuda

# Alternate: reinit before first launch -> faults at graph instantiation (CUDA err 1)
python repro_cuda700_reset.py --mode create_err

# Baseline: no reinit -> runs clean
python repro_cuda700_reset.py --mode create_err --no-reinit

# Reinit AND re-capture the graph against the fresh buffers -> runs clean
python repro_cuda700_reset.py --mode create_err --recapture
```

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
