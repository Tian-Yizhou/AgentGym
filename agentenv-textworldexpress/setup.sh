#!/bin/bash
# Create (or reuse) the conda env for the TextWorldExpress env server and install this package.
# Java comes from conda-forge (openjdk), so no system-wide Java install is needed.
#
#   bash setup.sh            # env name: agentenv-textworldexpress
#   ENV_NAME=myenv bash setup.sh
set -eo pipefail

ENV_NAME="${ENV_NAME:-agentenv-textworldexpress}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Reusing existing conda env ${ENV_NAME}"
  conda install -n "${ENV_NAME}" -c conda-forge pip openjdk=17 -y
else
  conda create -n "${ENV_NAME}" -c conda-forge python=3.10 pip openjdk=17 -y
fi
conda activate "${ENV_NAME}"
pip install -e "${HERE}[test]"

java -version
python -c "import textworld_express, agentenv_textworldexpress; print('textworld_express', textworld_express.__version__)"
echo "Done. Launch with: conda activate ${ENV_NAME} && textworldexpress --host 127.0.0.1 --port 36005"
