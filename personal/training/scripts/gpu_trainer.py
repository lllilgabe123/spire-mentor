"""Run ppo.trainer with the CUDA torch in the main process only.

Every spawned collection worker re-imports torch, and the CUDA build makes each
one commit ~1.5 GB (its DLLs), which overflows the paging file at 16 workers.
So the venv holds the small CPU torch, and the CUDA build sits in a separate
folder (torch-cuda) that only this process puts on sys.path: torch is imported
from there, then the folder is taken off sys.path again *before* the trainer
starts its pool. Workers inherit the parent's sys.path, so they import the
venv's CPU torch, which is all they need (single-threaded inference). The
learner update, the slow part on CPU, runs on the GPU (--device cuda).

The __main__ guard matters: a spawned worker runs this file as __mp_main__,
and must not import the CUDA build.
"""
import os
import runpy
import sys

if __name__ == "__main__":
    cuda_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torch-cuda")
    sys.path.insert(0, cuda_dir)
    import torch  # noqa: E402

    sys.path.remove(cuda_dir)
    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA torch from {cuda_dir} sees no GPU (torch {torch.__version__})")
    print(f"learner torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)
    sys.argv[0] = "ppo.trainer"
    runpy.run_module("ppo.trainer", run_name="__main__", alter_sys=True)
