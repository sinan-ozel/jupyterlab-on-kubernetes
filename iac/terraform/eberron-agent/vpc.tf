module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  name   = "${var.cluster_name}-vpc"
  cidr   = "10.0.0.0/16"
  azs    = slice(data.aws_availability_zones.available.names, 0, 3)

  enable_dns_hostnames = true
  enable_dns_support   = true

  # Public and Private Subnets
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.3.0/24", "10.0.4.0/24"]

  # Ensure public subnets can route through IGW
  enable_nat_gateway           = true
  single_nat_gateway           = true

  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = 1
    "cluster"                                   = var.cluster_name
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"           = 1
    "cluster"                                   = var.cluster_name
  }

  tags = {
    "Name"                                      = "${var.cluster_name}-vpc"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"           = ""
    "kubernetes.io/role/elb"                    = 1
    "cluster"                                   = var.cluster_name
  }

}

# Explicitly defining a route table for public subnets
resource "aws_route_table" "public" {
  vpc_id = module.vpc.vpc_id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = module.vpc.igw_id
  }

  tags = {
    "Name"    = "${var.cluster_name}-public-route-table",
    "cluster" = var.cluster_name
  }

}
