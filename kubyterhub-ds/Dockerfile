# Use the official Jupyter base image
FROM jupyter/base-notebook:latest

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterhub-ds"
LABEL org.opencontainers.image.description="JupyterHub environment for data science with Redis, Qdrant, ChromaDB, and LanceDB"
LABEL org.opencontainers.image.version="25.12"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterhub-ds"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterhub-ds"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.base.name="jupyter/base-notebook:latest"

# Switch to root to install packages
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user
USER $NB_UID

# Install JupyterHub and related packages
RUN pip install --no-cache-dir \
    jupyterhub==5.4.3 \
    jupyterlab==4.5.0 \
    dockerspawner==12.1.0 \
    oauthenticator==17.3.0 \
    jupyterhub-nativeauthenticator==1.3.0 \
    jupyterhub-ldapauthenticator==2.0.2

# Install JupyterLab real-time collaboration (separate to handle version conflicts)
RUN pip install --no-cache-dir \
    jupyter-collaboration==4.2.0

# Install configurable-http-proxy (use older stable version to avoid Node.js module issues)
RUN npm install -g configurable-http-proxy@4.6.1

# Install mature, stable packages (slower update cycles)
RUN pip install --no-cache-dir \
    scikit-learn==1.7.2 \
    lxml==6.0.2 \
    openpyxl==3.1.5 \
    beautifulsoup4==4.14.3 \
    requests==2.32.5 \
    urllib3==2.6.1 \
    charset-normalizer==3.4.4 \
    PyYAML==6.0.3 \
    fqdn==1.5.1 \
    pandocfilters==1.5.1 \
    defusedxml==0.7.1 \
    pillow==12.0.0 \
    pillow-heif==1.1.1 \
    nbdime==4.0.2 \
    tqdm==4.67.1

# Install frequently updated packages (active development)
RUN pip install --no-cache-dir \
    numpy==2.3.5 \
    pandas==2.3.3 \
    matplotlib==3.10.7 \
    seaborn==0.13.2 \
    redis==7.1.0 \
    qdrant-client==1.16.1 \
    chromadb==1.3.5 \
    lancedb==0.25.3 \
    jsonschema==4.25.1 \
    jsonschema-specifications==2025.9.1 \
    json5==0.12.1 \
    pytz==2025.2 \
    webcolors==25.10.0 \
    redis-memory==0.3.2 \
    litellm==1.80.9

# Configure git and nbdime for Jupyter
RUN nbdime config-git --enable --global

# Switch to root to create JupyterHub configuration directory and copy config
USER root

# Install Docker CLI for DockerSpawner
RUN apt-get update && apt-get install -y \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/jupyterhub

# Copy JupyterHub configuration file
COPY kubyterhub-ds/jupyterhub_config.py /etc/jupyterhub/jupyterhub_config.py

# Copy user creation script, startup script, and default userlist
COPY kubyterhub-ds/create-users.sh /usr/local/bin/create-users.sh
COPY kubyterhub-ds/start-jupyterhub.sh /usr/local/bin/start-jupyterhub.sh
COPY kubyterhub-ds/userlist.txt /etc/jupyterhub/userlist.txt
RUN chmod +x /usr/local/bin/create-users.sh /usr/local/bin/start-jupyterhub.sh

# Create JupyterLab configuration for real-time collaboration
RUN mkdir -p /etc/jupyter && \
    echo 'c.LabApp.collaborative = True' > /etc/jupyter/jupyter_lab_config.py && \
    echo 'c.YDocExtension.disable_rtc = False' >> /etc/jupyter/jupyter_lab_config.py

# Create shared data directory and set permissions
RUN mkdir -p /home/jovyan/work/shared && \
    mkdir -p /home/jovyan/work/private && \
    chown -R jovyan:users /home/jovyan/work

# Set the working directory
WORKDIR /home/jovyan/work

# Expose JupyterHub port (8000) and JupyterLab port (8888)
EXPOSE 8000 8888

# Run JupyterHub as root (required to create users and run LocalProcessSpawner)
CMD ["/usr/local/bin/start-jupyterhub.sh"]