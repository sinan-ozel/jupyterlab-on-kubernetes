FROM python:3.12-slim

ENV K8S_VERSION="1.30.0"

RUN apt update --fix-missing
RUN apt install -y curl
RUN apt install -y wget
RUN apt install -y nodejs
RUN apt install -y npm
RUN apt install -y unzip
RUN apt install -y ca-certificates
RUN apt install -y gnupg
RUN apt install -y tree

RUN curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com bookworm main" | tee /etc/apt/sources.list.d/hashicorp.list

# Update after adding repos
RUN apt update

# Terraform
RUN apt install -y terraform

# Clean up apt
RUN rm -rf /var/cache/apk/*
RUN apt-get autoremove -y
RUN apt-get clean

RUN pip install jupyterlab==4.2.5
RUN pip install ipywidgets==8.1.5
RUN pip install jupyter_contrib_nbextensions==0.7.0
RUN jupyter labextension enable @jupyterlab/toc

# CloudSigma
RUN pip install cloudsigma==2.0.0

# AWS
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN rm awscliv2.zip
RUN ./aws/install

RUN pip install boto3==1.35.24

# Kubernetes
RUN curl -LO https://dl.k8s.io/release/v1.30.0/bin/linux/amd64/kubectl
RUN install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
RUN pip install kubernetes==31.0.0
RUN pip install kustomize==0.6.0

# Helm
RUN curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
RUN chmod 700 get_helm.sh
RUN ./get_helm.sh
RUN rm get_helm.sh

# AWS ekcctl
RUN curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_Linux_amd64.tar.gz"
RUN tar -xzf eksctl_Linux_amd64.tar.gz -C /tmp
RUN rm eksctl_Linux_amd64.tar.gz
RUN mv /tmp/eksctl /usr/local/bin


RUN pip install pyhelm3==0.4.0

# Various Python Libraries
RUN pip install python-dotenv==1.0.1
RUN pip install tqdm==4.66.5
RUN pip install fabric==3.2.2
RUN pip install bcrypt==4.1.1
RUN pip install paramiko==3.4.0
RUN pip install psycopg2-binary==2.9.9

RUN touch /.env
RUN printenv > /.env

EXPOSE 8888
WORKDIR /jupyterlab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir='/jupyterlab/notebooks'", "--allow-root"]
