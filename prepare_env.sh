# crash in case of error
trap 'trap - ERR RETURN; kill -INT $$ ; return' ERR RETURN

conda create --name panza python=3.10 -y
conda activate panza

conda install pytorch==2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install langdetect langchain langchain-community sentence-transformers faiss-cpu fire bert-score mauve-text evaluate torchmetrics gradio cmake packaging nltk

pip install git+https://github.com/IST-DASLab/llm-foundry
pip install git+https://github.com/IST-DASLab/peft-rosa.git@grad_quant
pip install spops_sm_80
