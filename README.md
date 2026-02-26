# Quick Start

```bash
git clone git@github.com:Astera-org/strange-loop

# additional dependency
git clone git@github.com:tlh24/att3ntion

cd strange-loop

# Because att3ntion compiles against CUDA, you must make sure
# to install torch which bundles your version of CUDA libraries.
# To do that, use the --extra switch.

# check your local CUDA version
nvcc --version | grep release

# if 13.0
uv sync --extra cu130

# if 12.4
uv sync --extra cu124

# This will build the att3ntion CUDA kernels.
# The --no-build-isolation here is necessary to build using your
# CUDA-matching torch
uv pip install -e ../att3ntion --no-build-isolation

# enable your environment
source .venv/bin/activate

# confirm your Torch cuda version matches your installed version
python -c 'import torch; print(torch.version.cuda)' # should match nvcc --version

```



