# Use the official Jupyter base image
FROM jupyter/base-notebook:latest

# Switch to root to install packages
USER root

# Install system dependencies (optional, add what you need)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user
USER $NB_UID

# Install Python packages with pinned versions for reproducibility
RUN pip install --no-cache-dir \
    numpy==2.1.3 \
    pandas==2.2.3 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    scikit-learn==1.5.2 \
    nbdime==4.0.2 \
    redis==5.2.0 \
    lxml==5.3.0 \
    openpyxl==3.1.5 \
    tqdm==4.67.1

# Configure git and nbdime for Jupyter
RUN nbdime config-git --enable --global

# Set the working directory
WORKDIR /home/jovyan/work

# Expose Jupyter port
EXPOSE 8888

