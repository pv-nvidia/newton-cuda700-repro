# Bug: CUDA graph launched after a solver/model reset faults with CUDA 700 (illegal memory access)

## Description

When a CUDA graph is captured over a Newton simulation step and the solver/model is then
reset (rebuilding `State`/`Contacts` and, in general, reallocating device buffers), the
previously-captured graph still references the old, now-freed device memory. The next
launch of that graph faults with an illegal memory access:

```
Warp CUDA error 700: an illegal memory access was encountered
    (in function wp_cuda_graph_launch, warp.cu:4515)
RuntimeError: Graph launch error
```

This surfaces in an IsaacLab + Newton deformable workload where `env.sim.reset()` rebuilds
the solver while a CUDA graph is live, but it reproduces with pure Newton + Warp (no
IsaacLab, no USD, no assets).

## Reproducer

Self-contained, asset-free: https://github.com/pv-nvidia/newton-cuda700-repro

```bash
uv sync
uv run repro_cuda700_reset.py --verbose-cuda
```

Sequence: build cloth grid + ground + VBD solver + soft contacts → capture a CUDA graph →
launch it once successfully → rebuild the model under the live graph (frees + reallocates
the captured buffers) → relaunch the stale graph → CUDA 700 at `wp_cuda_graph_launch`.

`--mode create_err --no-reinit` (no reset) and `--mode create_err --recapture` (re-capture
after reset) both run clean, confirming the reset-under-live-graph is the trigger.

## Environment

- Newton `main` (reproduced at HEAD `e425867`; also `9bff8911` and an earlier snapshot)
- warp-lang `1.16.0.dev20260706` (also `1.14.0`), mujoco-warp `3.10.0.1`
- NVIDIA L40, driver 570.158.01, CUDA 12.8, Python 3.12, Linux x86_64

## Suggested fix (any of)

1. Invalidate CUDA graphs captured against buffers freed by a solver/state reset (or expose
   a hook to force re-capture), so a stale graph cannot be replayed.
2. Keep buffer identity stable across an in-place solver reset so captured graphs stay valid.
3. At minimum, raise a clear, actionable error instead of a raw CUDA-700.
