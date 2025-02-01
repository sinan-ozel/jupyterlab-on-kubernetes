variable "cluster_name" {
  description = "The name of the EKS cluster"
  type        = string
  default     = "eberron-agent"
}

variable "region" {
  description = "The AWS region"
  type        = string
  default     = "ca-central-1"
}

variable "purpose" {
  description = "This is a tag sahred by multiple environments, for example, an EBS is shared for the purpose of holding LLM models."
  type        = string
  default     = "llm"
}

# variable "gpu_instance_type" {
#   description = "EC2 instance type for the GPU node"
#   type        = string
#   default     = "g4dn.2xlarge"
# }

# variable "regular_instance_type" {
#   description = "EC2 instance type for regular nodes"
#   type        = string
#   default     = "t3.medium"
# }

# variable "desired_regular_nodes" {
#   description = "The desired number of regular nodes"
#   type        = number
#   default     = 3
# }

