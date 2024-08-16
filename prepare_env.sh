# crash in case of error
trap 'trap - ERR RETURN; kill -INT $$ ; return' ERR RETURN

conda create --name panza python=3.10 -y
conda activate panza

# install dependencies based on pyproject.toml
pip install -e .