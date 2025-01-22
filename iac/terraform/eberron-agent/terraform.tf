terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }

    random = {
      source  = "hashicorp/random"
#       version = "~> 3.5"
    }

    tls = {
      source  = "hashicorp/tls"
#       version = "~> 4.0.4"
    }

    cloudinit = {
      source  = "hashicorp/cloudinit"
#       version = "~> 2.3"
    }

    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.35"
    }
  }

#   required_version = ">= 1.5.0"
}