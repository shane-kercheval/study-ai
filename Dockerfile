FROM python:3.11

WORKDIR /code
ENV PYTHONPATH "${PYTHONPATH}:/code"

ENV PYDEVD_WARN_EVALUATION_TIMEOUT 1000

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    man-db \
    zsh \
    lsof \
    poppler-utils \
    nano
RUN PATH="$PATH:/usr/bin/zsh"

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

