Requirements and installation notes

- The `requirements.txt` file lists the common Python packages used across the repo. Use a virtual environment and install with:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements/requirements.txt

- GPU / FlashAttention / CUDA: if you plan to train large models or use FlashAttention, install those packages separately following the project's docs (FlashAttention often requires building from source or wheels compatible with your CUDA version).

- If you need exact pinned versions for reproducibility, update the `requirements.txt` with exact versions and consider adding a `requirements/constraints.txt` or a lockfile.

- Some notebooks may rely on system packages (OpenBLAS, libsndfile, etc.). If a pip install fails, check the error output and install the missing OS packages via Homebrew or your OS package manager.

