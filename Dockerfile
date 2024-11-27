#### BEGIN: Load micromamba ####################################################
# We copy micromamba from the micromamba stage because historically the download
# link has been unreliable.
FROM mambaorg/micromamba:1.5.8-jammy as micromamba
#### END: Load micromamba ######################################################


#### BEGIN: Build spops ###################################################
# We use the devel image to build spops.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS spops-build
ARG DEBIAN_FRONTEND=noninteractive

# Install utilities
RUN apt-get update -q \
  && apt-get install -y --no-install-recommends \
  # Required to build spops
  build-essential \
  curl \
  git \
  sudo \
  # Clean up
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Add sancho user with sudo privileges
RUN adduser --disabled-password --gecos "" sancho
RUN usermod -aG sudo sancho
RUN echo "sancho ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sancho

# Add .local/bin to sancho PATH. Needed for pip installs.
RUN echo "export PATH=\$PATH:/home/sancho/.local/bin" >> /home/sancho/.bashrc

# Switch to sancho user
USER sancho
WORKDIR /home/sancho

# Install micromamba
# https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#manual-installation
COPY --from=micromamba --chown=sancho /bin/micromamba .local/bin/micromamba
RUN .local/bin/micromamba shell init -s bash -p /home/sancho/micromamba
RUN .local/bin/micromamba config append channels conda-forge \
  && .local/bin/micromamba config append channels nodefaults \
  && .local/bin/micromamba config set channel_priority strict

# Set up micromamba environment
ARG MAMBA_ROOT_PREFIX=/home/sancho/.micromamba
RUN /home/sancho/.local/bin/micromamba env create -y -n panza python=3.10 \
  && /home/sancho/.local/bin/micromamba clean --all -y -f
RUN /home/sancho/.local/bin/micromamba install -n panza \
  pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install spops dependencies and build flash-attn
# See https://github.com/IST-DASLab/spops/blob/main/setup.py for details.
RUN /home/sancho/.local/bin/micromamba run -n panza \
  pip install --no-cache-dir numpy scipy ninja pybind11
ARG TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
RUN /home/sancho/.local/bin/micromamba run -n panza \
  pip install --user --no-cache-dir \
  git+https://github.com/IST-DASLab/spops.git
#### END: Build spops #####################################################


#### BEGIN: Final image ########################################################
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 as final
ARG DEBIAN_FRONTEND=noninteractive

# Install utilities
RUN apt-get update -q \
  && apt-get install -y --no-install-recommends \
  # General utilities
  build-essential \
  curl \
  git \
  git-lfs \
  htop \
  openssh-server \
  sudo \
  tmux \
  unzip \
  uuid-runtime \
  vim \
  wget \
  # Latex
  texlive-latex-extra \
  texlive-fonts-recommended \
  dvipng \
  cm-super \
  # Clean up
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Add sancho user with sudo privileges
RUN adduser --disabled-password --gecos "" sancho
RUN usermod -aG sudo sancho
RUN echo "sancho ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sancho

# Add .local/bin to sancho PATH. Needed for pip installs.
RUN echo "export PATH=\$PATH:/home/sancho/.local/bin" >> /home/sancho/.bashrc

# Switch to sancho user
USER sancho
WORKDIR /home/sancho

# Set up git
RUN git config --global credential.helper store
RUN git config --global core.editor "vim"

# Install micromamba
# https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#manual-installation
COPY --from=micromamba --chown=sancho /bin/micromamba .local/bin/micromamba
RUN .local/bin/micromamba shell init -s bash -p /home/sancho/micromamba
RUN .local/bin/micromamba config append channels conda-forge \
  && .local/bin/micromamba config append channels nodefaults \
  && .local/bin/micromamba config set channel_priority strict

# Create panza env
RUN /home/sancho/.local/bin/micromamba env create -y -n panza python=3.10
RUN /home/sancho/.local/bin/micromamba install -n panza \
  pytorch==2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
RUN /home/sancho/.local/bin/micromamba run -n panza \
  pip install --no-cache-dir \
  langdetect langchain langchain-community sentence-transformers \
  faiss-cpu fire nltk gradio cmake packaging
RUN /home/sancho/.local/bin/micromamba run -n panza \
  pip install --no-cache-dir \
  git+https://github.com/IST-DASLab/llm-foundry

RUN /home/sancho/.local/bin/micromamba run -n panza \
  pip install --no-cache-dir \
  git+https://github.com/IST-DASLab/peft-rosa.git@grad_quant

# Clean up
RUN /home/sancho/.local/bin/micromamba clean --all -y -f

# Copy spops from spops-build
COPY --from=spops-build --chown=sancho \
  /home/sancho/.local/lib/python3.10/site-packages \
  /home/sancho/.local/lib/python3.10/site-packages
#### END: Final image ########################################################
