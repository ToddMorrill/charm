FROM python:3.10.3-slim

RUN mkdir /app
WORKDIR /app

# update this with your container name (e.g. columbia-emotions, columbia-norms
# columbia-vision)
ENV CCU_CONTAINER="columbia-communication-change"

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=secret,id=netrc,dst=/root/.netrc \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt && \
    pip install https://artifactory.sri.com/artifactory/cirano-pypi-local/ccu-1.1-py3-none-any.whl

# if you are installing the ccu library from a local file, use this instead:
# COPY ccu-1.1-py3-none-any.whl /app/ccu-1.1-py3-none-any.whl
# RUN pip install ccu-1.1-py3-none-any.whl


# these are the ports used by the zmq messsage queues
EXPOSE 11880-11890
EXPOSE 12880-12890

# copy python code
COPY . /app

CMD python -u main.py