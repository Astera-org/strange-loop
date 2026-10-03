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

# if 12.6
uv sync --extra cu126

# If something goes wrong, this may help
# This is necessary when a previous install may have used cu126, because
# nvidia-*-cu12 and nvidia-*-cu13 extract to the same folder structure, and can
# overwrite each other
uv sync --extra cu130 --reinstall


# This will build the att3ntion CUDA kernels.
# The --no-build-isolation here is necessary to build using your
# CUDA-matching torch
uv pip install -e ../att3ntion --no-build-isolation

# Also, note that Ninja deletes *.o files when building this way.
# For incremental builds, you can also use (must be run from within att3ntion)
cd ../att3ntion 
python setup.py build_ext --inplace


# enable your environment
source .venv/bin/activate

# confirm your Torch cuda version matches your installed version
python -c 'import torch; print(torch.version.cuda)' # should match nvcc --version

```



