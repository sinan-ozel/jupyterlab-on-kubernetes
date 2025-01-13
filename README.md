# IaC For Generative AI: LLM Jupyterlab on Kubernetes on AWS

This repo contains the Infrastructure-as-Code to create a Jupyterlab on Kubernetes on AWS.
The Jupyterlab image is completely custom, it contains libraries geared towards LLM work.
The code is on another, local Jupyterlab instance, deployed through docker compose.
The local notebook contains the IaC code required to create a complete Kubernetes environment that support GPU usage.

So this project achieves three goals:

1. Running a custom Jupyter container with GPU support
1. Running this on Kubernetes
1. Creating the Infrastructure-as-Code to create and tear down the infrastructure on AWS to run this system

This is what your environment looks like in the end.
![1_SfmiSe5NHwsgVJgbDgg_Kw](https://github.com/user-attachments/assets/3566e9a5-30e6-4871-80b3-e527cd72a1c4)


## Requirements

* Docker: You need to have this locally. If you have a Windows machine, you will also need WSL 2 running to be able to run the Linux containers.
* An AWS account and AWS CLI installed: You will need the account id and the account secret to login, push an image, and finally to deploy on Kubernetes on AWS.
* (Optional) VS Code - the steps are automated, making it easy to build and push the required image.

## Usage

Locally, use the VS Code tasks `build-kubyterlab-llm` and `push-kubyterlab-llm-to-aws` to build and upload the image. 
You will need a repository called `kubyterlab-llm` on ECR on AWS. Set the region in the settings.
Once the image is up, enter the `iac` folder and run `docker compose up --build`.
This will deploy a local Jupyterlab server which contains the script to create and destroy the environment on AWS.
The notebook called Deploy, creates the system from scratch and the notebook call Teardown deletes everything and backs up the persistent volume into a snapshot.
[https://github.com/sinan-ozel/jupyterlab-on-kubernetes/blob/main/iac/jupyterlab-iac/notebooks/kubyterlab-llm-on-aws/Deploy.ipynb]

On the notebook Deploy, you want to run all cells in order until you reach the point marked "== End of Procedure ==".
There is a URL under the section "Get URL", looking like the following:
![image](https://github.com/user-attachments/assets/e23c2743-2747-42b6-8975-e9f9ee040de0)
This is how you access your environment. It is not over a secure channel, so do not use it to transmit sensitive data or information.

As you go through the notebook, check against the main copy on github and see if everything is going as in the example.
If not, you may need to troubleshoot and debug.
Please note that this has been used only once by me, and there may be errors and incompatibilities. 
It is intended more as a learning tool than a solution.
A working knowledge of AWS infrastructure and Kubernetes is required to get this to work.
