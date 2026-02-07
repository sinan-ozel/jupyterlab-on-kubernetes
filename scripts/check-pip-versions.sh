#!/bin/bash
# Check pip package versions in Dockerfile against latest versions on PyPI
# Usage: ./check-pip-versions.sh <dockerfile_path>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dockerfile_path>"
    exit 1
fi

DOCKERFILE="$1"

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: File not found: $DOCKERFILE"
    exit 1
fi

# Extract packages and versions from Dockerfile
echo "Extracting packages from $DOCKERFILE..."
packages=$(grep -oP 'pip3?\s+install\s+\K[a-zA-Z0-9_\-\[\]]+==[\d\.]+' "$DOCKERFILE" | sort -u)

if [ -z "$packages" ]; then
    echo "No pip packages found in Dockerfile"
    exit 0
fi

count=$(echo "$packages" | wc -l)
echo "Found $count packages in $DOCKERFILE"
echo ""

# Print header
printf "%-40s %-15s %-15s %s\n" "Package" "Current" "Latest" "Status"
printf "%s\n" "================================================================================="

# Function to get latest version from PyPI
get_latest_version() {
    local package=$1
    local response=$(curl -s "https://pypi.org/pypi/$package/json" 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$response" ]; then
        echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['info']['version'])" 2>/dev/null
    fi
}

# Process each package
while IFS= read -r line; do
    # Split package name and version
    package=$(echo "$line" | cut -d'=' -f1)
    current_version=$(echo "$line" | cut -d'=' -f3)

    # Get latest version from PyPI
    latest_version=$(get_latest_version "$package")

    # Determine status
    if [ -n "$latest_version" ]; then
        if [ "$latest_version" = "$current_version" ]; then
            status="✓ Up to date"
        else
            status="⚠ Update available"
        fi
        latest_display="$latest_version"
    else
        status="? Unknown"
        latest_display="N/A"
    fi

    printf "%-40s %-15s %-15s %s\n" "$package" "$current_version" "$latest_display" "$status"
done <<< "$packages"
