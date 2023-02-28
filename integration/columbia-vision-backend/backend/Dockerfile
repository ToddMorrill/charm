FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04
WORKDIR /app
##################################
# Setup base conda envs
##################################
## Install Miniconda in /opt/conda
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV DGLBACKEND=pytorch

ENV CCU_CONTAINER="columbia-vision-backend"

RUN apt-get update && apt-get install -y build-essential libsqlite3-dev sqlite3 bzip2 libbz2-dev zlib1g-dev libssl-dev\
    openssl libgdbm-dev libgdbm-compat-dev liblzma-dev libreadline-dev libncursesw5-dev libffi-dev uuid-dev\
    wget git
    
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

## CHARM environment
ENV CHARM_ENV=/opt/conda/envs/charm \
    CHARM_PYTHON=/opt/conda/envs/charm/bin/python
RUN /opt/conda/bin/conda create -n charm python=3.9

RUN /opt/conda/bin/conda install -n=charm pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

COPY requirements.txt /tmp/requirements.txt

RUN ${CHARM_ENV}/bin/pip install --upgrade pip && \
    ${CHARM_ENV}/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# COPY requirements.txt /tmp/requirements.txt
# RUN ${CHARM_ENV}/bin/pip install --upgrade pip && \
#     ${CHARM_ENV}/bin/pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html && \
#     ${CHARM_ENV}/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
#     rm /tmp/requirements.txt

COPY ccu-1.1-py3-none-any.whl /app/ccu-1.1-py3-none-any.whl
RUN ${CHARM_ENV}/bin/pip install ccu-1.1-py3-none-any.whl

# installing CLIP and other stuff
RUN ${CHARM_ENV}/bin/pip install install ftfy regex tqdm
RUN ${CHARM_ENV}/bin/pip install git+https://github.com/openai/CLIP.git

# the web app will server on port 8000
EXPOSE 8000

COPY . /app
WORKDIR /app

CMD ${CHARM_ENV}/bin/uvicorn main:app --host 0.0.0.0  --port 8000