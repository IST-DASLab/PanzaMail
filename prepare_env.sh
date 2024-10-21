# crash in case of error
trap 'trap - ERR RETURN; kill -INT $$ ; return' ERR RETURN

conda create --name panza_refactor python=3.10 -y
conda activate panza_refactor

# install dependencies based on pyproject.toml
pip install -e .[training]