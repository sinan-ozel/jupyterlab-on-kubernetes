data "aws_availability_zones" "available" {}

module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = var.cluster_name
  cluster_version = "1.31"
  subnet_ids      = module.vpc.private_subnets
  vpc_id          = module.vpc.vpc_id

  eks_managed_node_groups = {
    gpu_nodes = {
      name             = "${var.cluster_name}-gpu"
      desired_capacity = 1
      ami_type         = "AL2_x86_64_GPU"
      max_capacity     = 1
      min_capacity     = 1
      instance_type    = "g4dn.2xlarge"
#       key_name         = var.key_name # Replace with your SSH key name
      labels = {
        "node-role.kubernetes.io/gpu" = "true"
      }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "Exists"
        effect = "NO_SCHEDULE"
      }]
    }

    regular_nodes = {
      name             = "${var.cluster_name}-regular"
      desired_capacity = 3
      ami_type         = "AL2_x86_64"
      max_capacity     = 5
      min_capacity     = 1
      instance_type    = "t3.medium"
      labels = {
        "node-role.kubernetes.io/regular" = "true"
      }
    }
  }
}

