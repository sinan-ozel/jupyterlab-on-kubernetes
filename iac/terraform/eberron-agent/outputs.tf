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

