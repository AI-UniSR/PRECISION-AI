FROM rocker/r-ver:4.4.3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    gfortran \
    cmake \
    make \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    pkg-config \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff5-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libgit2-dev \
    libglpk-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a dedicated Python virtual environment
RUN python3 -m venv /opt/pyenv

# Install Python packages inside the venv
RUN /opt/pyenv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/pyenv/bin/pip install --no-cache-dir \
    mlflow==2.22.2 \
    azureml-mlflow==1.60.0.post1 \
    numpy \
    pandas

# Make the venv Python the default one on PATH
ENV PATH="/opt/pyenv/bin:${PATH}"
ENV RETICULATE_PYTHON="/opt/pyenv/bin/python"

# R packages for publication tables
RUN R -q -e "install.packages(c( \
    'argparse', \
    'readr', \
    'jsonlite', \
    'dplyr', \
    'flextable', \
    'officer', \
    'reticulate' \
), repos='https://cloud.r-project.org')"

# Sanity check R
RUN R -q -e "library(argparse); \
library(readr); \
library(jsonlite); \
library(dplyr); \
library(flextable); \
library(officer); \
library(reticulate); \
cat('R packages OK\n')"

# Sanity check Python
RUN /opt/pyenv/bin/python -c "import mlflow, azureml.mlflow, numpy, pandas; print('Python packages OK')"

WORKDIR /app
