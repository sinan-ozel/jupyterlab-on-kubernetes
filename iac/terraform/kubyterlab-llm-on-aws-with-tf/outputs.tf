output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "gpu_node_group_role_arn" {
  description = "IAM role ARN for GPU node group"
  value       = module.eks.eks_managed_node_groups["gpu_nodes"].iam_role_arn
}

output "regular_node_group_role_arn" {
  description = "IAM role ARN for regular node group"
  value       = module.eks.eks_managed_node_groups["regular_nodes"].iam_role_arn
}

output "security_group_id" {
  description = "Security Group ID used by EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "vpc_id" {
  description = "VPC ID associated with the EKS cluster"
  value       = module.vpc.vpc_id
}

output "gpu_node_group_id" {
  description = "ID of the GPU node group"
  value       = module.eks.eks_managed_node_groups["gpu_nodes"].node_group_id
}

output "regular_node_group_id" {
  description = "ID of the regular node group"
  value       = module.eks.eks_managed_node_groups["regular_nodes"].node_group_id
}
