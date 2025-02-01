data "aws_availability_zones" "available" {}

module "eks" {
  source                                   = "terraform-aws-modules/eks/aws"
  cluster_name                             = var.cluster_name
  cluster_version                          = "1.31"
  subnet_ids                               = module.vpc.private_subnets
  vpc_id                                   = module.vpc.vpc_id
  cluster_endpoint_public_access           = true
  enable_cluster_creator_admin_permissions = true

  eks_managed_node_groups = {
    gpu_nodes = {
      name             = "${var.cluster_name}-gpu"
      ami_type         = "AL2_x86_64_GPU"
      instance_types   = ["g4dn.2xlarge"]
      iam_role_arn     = aws_iam_role.eks_nodes.arn
      scaling_config = {
        desired_size = 1
        max_size     = 1
        min_size     = 1
      }
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
      ami_type         = "AL2_x86_64"
      instance_types   = ["t3.medium"]
      iam_role_arn     = aws_iam_role.eks_nodes.arn
      scaling_config = {
        desired_size = 3
        max_size     = 5
        min_size     = 1
      }
      labels = {
        "node-role.kubernetes.io/regular" = "true"
      }
    }
  }

  tags = {
    "Name"                                      = var.cluster_name
    "cluster"                                   = var.cluster_name
    "kubernetes.io/cluster/${var.cluster_name}" = "owned"
  }
}

