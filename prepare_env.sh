# crash in case of error
trap 'trap - ERR RETURN; kill -INT $$ ; return' ERR RETURN

conda create --name panza python=3.9 -y
conda activate panza

conda install nvidia/label/cuda-12.1.0::cuda -y
export CUDA_HOME=$CONDA_PREFIX

conda install gcc_linux-64 -y
conda install gxx_linux-64 -y

conda install pytorch==2.1.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install langdetect langchain langchain-community sentence-transformers faiss-cpu fire nltk gradio
pip install llm-foundry==0.5.0 composer==0.19.1

pip install git+https://github.com/IST-DASLab/spops.git
pip install git+https://github.com/IST-DASLab/peft-rosa.git

pip install -U transformers==4.35.2
