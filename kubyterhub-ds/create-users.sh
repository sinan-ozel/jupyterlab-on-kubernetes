#!/bin/bash
set -e

USERLIST_FILE="${USERLIST_FILE:-/etc/jupyterhub/userlist.txt}"

echo "Creating users from $USERLIST_FILE"

if [ ! -f "$USERLIST_FILE" ]; then
    echo "WARNING: $USERLIST_FILE not found. No users will be created."
    echo "Create a userlist.txt file with one username per line."
    exit 0
fi

# Read userlist.txt and create users
while IFS= read -r username || [ -n "$username" ]; do
    # Skip empty lines and comments
    username=$(echo "$username" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ -z "$username" ] || [[ "$username" =~ ^# ]]; then
        continue
    fi

    # Check if user already exists
    if id "$username" &>/dev/null; then
        echo "User $username already exists, skipping"
        continue
    fi

    echo "Creating user: $username"

    # Create user with home directory
    useradd -m -s /bin/bash "$username"

    # Set up user's work directory with proper permissions
    mkdir -p "/home/$username/work"
    chown "$username:$username" "/home/$username/work"

    # Create shared directory symlink in user's home
    ln -sf /home/jovyan/work/shared "/home/$username/work/shared" 2>/dev/null || true

    echo "User $username created successfully"
done < "$USERLIST_FILE"

echo "User creation completed"