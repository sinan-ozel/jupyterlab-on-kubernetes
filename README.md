# IaC For Generative AI: LLM Jupyterlab on Kubernetes on AWS

This repo contains the Infrastructure-as-Code to create a Jupyterlab on Kubernetes on AWS.
The Jupyterlab image is completely custom, it contains libraries geared towards LLM work.
The code is on another, local Jupyterlab instance, deployed through docker compose.
The local notebook contains the IaC code required to create a complete Kubernetes environment that support GPU usage.

So this project achieves three goals:

1. Running a custom Jupyter container with GPU support
1. Running this on Kubernetes
1. Creating the Infrastructure-as-Code to create and tear down the infrastructure on AWS to run this system

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

Please note that this has been used only once by me, and there may be errors and incompatibilities. 
It is intended more as a learning tool than a solution.
A working knowledge of AWS infrastructure and Kubernetes is required to get this to work.
