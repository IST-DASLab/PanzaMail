# crash in case of error
trap 'trap - ERR RETURN; kill -INT $$ ; return' ERR RETURN

conda create --name panza python=3.10 -y
conda activate panza

conda install nvidia/label/cuda-12.1.0::cuda -y
export CUDA_HOME=$CONDA_PREFIX

conda install gcc_linux-64 -y
conda install gxx_linux-64 -y

conda install pytorch==2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install langdetect langchain langchain-community sentence-transformers faiss-cpu fire nltk gradio cmake packaging

pip install git+https://github.com/IST-DASLab/llm-foundry
pip install git+https://github.com/IST-DASLab/spops.git
pip install git+https://github.com/IST-DASLab/peft-rosa.git@grad_quant
