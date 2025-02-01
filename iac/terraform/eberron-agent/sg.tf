resource "aws_security_group" "eks_cluster" {
  name_prefix = "${var.cluster_name}-sg"
  description = "Security group for EKS cluster"
  vpc_id      = module.vpc.vpc_id

  # Allow all traffic within the cluster
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [module.vpc.vpc_cidr_block]
    security_groups = [module.eks.cluster_security_group_id]
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    "Name"    = "${var.cluster_name}-sg"
    "cluster" = "${var.cluster_name}"
  }

}
