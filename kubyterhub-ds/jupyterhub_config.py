# JupyterHub Configuration
import os
c = get_config()

# Basic settings
c.JupyterHub.bind_url = 'http://:8000'
c.JupyterHub.hub_bind_url = 'http://:8081'

# Spawner settings - use LocalProcessSpawner with dynamically created users
c.JupyterHub.spawner_class = 'jupyterhub.spawner.LocalProcessSpawner'
c.Spawner.default_url = '/lab'
c.Spawner.cmd = ['jupyter-labhub']

# Each user starts in their own home directory but has access to shared space
c.Spawner.notebook_dir = '~'

# Environment for real-time collaboration
c.Spawner.environment = {
    'JUPYTER_ENABLE_LAB': 'yes',
    'JUPYTERLAB_COLLABORATIVE': 'true',
}# Authentication - use GitHub OAuth
c.JupyterHub.authenticator_class = 'oauthenticator.GitHubOAuthenticator'

# GitHub OAuth settings (set these environment variables)
c.GitHubOAuthenticator.client_id = os.environ.get('GITHUB_CLIENT_ID', '')
c.GitHubOAuthenticator.client_secret = os.environ.get('GITHUB_CLIENT_SECRET', '')

# Build callback URL from environment variables with sensible defaults
hub_host = os.environ.get('HUB_HOST', 'localhost')
hub_port = os.environ.get('HUB_PORT', '8000')
hub_protocol = os.environ.get('HUB_PROTOCOL', 'http')
c.GitHubOAuthenticator.oauth_callback_url = f'{hub_protocol}://{hub_host}:{hub_port}/hub/oauth_callback'

# Alternative: Use full URL override if provided
if os.environ.get('OAUTH_CALLBACK_URL'):
    c.GitHubOAuthenticator.oauth_callback_url = os.environ.get('OAUTH_CALLBACK_URL')

# Control access - uncomment and configure as needed
# Option 1: Allow specific GitHub usernames
# c.GitHubOAuthenticator.allowed_users = {'github-username1', 'github-username2'}

# Option 2: Allow members of specific GitHub organizations
# c.GitHubOAuthenticator.allowed_organizations = {'your-github-org'}

# Option 3: Allow all authenticated GitHub users (less secure)
c.GitHubOAuthenticator.allow_all = True

# Admin users (use GitHub usernames)
c.Authenticator.admin_users = {'admin', 'sinan-ozel'}  # Replace with actual GitHub usernames

# Allow all users to start their own servers
c.JupyterHub.allow_named_servers = True

# Timeout settings
c.Spawner.http_timeout = 120
c.Spawner.start_timeout = 120

# Network settings
c.Spawner.ip = '0.0.0.0'
c.Spawner.port = 8888

# Database settings (use SQLite for simplicity)
c.JupyterHub.db_url = 'sqlite:///jupyterhub.sqlite'

# Logging
c.JupyterHub.log_level = 'INFO'
c.Application.log_level = 'INFO'