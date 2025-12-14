#!/bin/bash

check_docker_tags() {
  local repo=$1
  local name=$2
  echo "$name:"
  response=$(curl -s "https://registry.hub.docker.com/v2/repositories/sinanozel/$repo/tags/")
  if echo "$response" | jq -e '.results' > /dev/null 2>&1; then
    echo "$response" | jq -r '.results[].name' | head -10
  else
    echo "  Repository not found or no tags available"
  fi
  echo ""
}

echo "=== Docker Hub Version Check ==="
echo ""

check_docker_tags "kubyterlab-ds" "📦 kubyterlab-ds"
check_docker_tags "kubyterlab-llm" "🧠 kubyterlab-llm"
check_docker_tags "kubyterlab-img" "🎨 kubyterlab-img"
check_docker_tags "kubyterlab-img-12g" "🖼️ kubyterlab-img-12g"
check_docker_tags "kubyterhub-ds" "👥 kubyterhub-ds"

echo "=== Local Images ==="
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}' | grep -E '(kubyterlab|kubyterhub)' || echo 'No local images found'