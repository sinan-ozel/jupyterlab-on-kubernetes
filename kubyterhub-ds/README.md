# KubiterHub DS - Multi-User Data Science Environment

![Docker Pulls](https://img.shields.io/docker/pulls/sinanozel/kubyterhub-ds)
![Docker Image Size](https://img.shields.io/docker/image-size/sinanozel/kubyterhub-ds/25.12)
![Docker Image Version](https://img.shields.io/docker/v/sinanozel/kubyterhub-ds/25.12)

A production-ready JupyterHub environment for multi-user data science workflows with integrated Redis, Qdrant vector database, ChromaDB, and LanceDB support.

## 👥 User Management

**KubiterHub DS** uses **dynamic user creation** - Linux users are automatically created from a `userlist.txt` file at container startup.

### How It Works:
1. **Create `userlist.txt`** with GitHub usernames (one per line)
2. **Mount the file** into the container at startup
3. **Users are created automatically** as Linux system users
4. **GitHub OAuth maps** to these users for authentication

### Example `userlist.txt`:
```
# Add your GitHub usernames here
sinan-ozel
your-github-username
teammate-username
```

**Important**: Only users listed in `userlist.txt` can log in successfully!

## 🔐 GitHub OAuth Authentication

**Important**: This uses GitHub OAuth (not username/password login). Here's how it works:

### **What You'll Experience:**
1. Visit `http://localhost:8000`
2. Click **"Sign in with GitHub"**
3. Redirected to **GitHub's official login page**
4. GitHub asks: *"Allow KubiterHub DS to access your account?"*
5. Click **"Authorize"**
6. Back to JupyterHub - you're logged in! 🎉

### **Why OAuth (Not Username/Password)?**
- **🔒 More Secure**: Your GitHub password never leaves GitHub
- **🚀 Industry Standard**: Same method used by VS Code, Discord, etc.
- **⚡ One-Click Login**: No need to remember another password
- **🛡️ Revokable**: You can revoke access anytime in GitHub settings

### **One-Time Setup Required:**
You need to create a GitHub OAuth App (takes 2 minutes) - see setup instructions below.

## Features

### JupyterHub Multi-User Environment
- **JupyterHub 5.4.3** - Multi-user server management
- **GitHub OAuth Pre-configured** - Ready for GitHub authentication with environment variables
- **User Management** - Admin interface, named servers, resource control
- **JupyterLab 4.5.0** - Modern notebook interface for each user
- **Kubernetes Ready** - Flexible FQDN configuration for production deployments### Data Science Stack
- **NumPy 2.3.5** & **Pandas 2.3.3** - Core data manipulation
- **Matplotlib 3.10.7** & **Seaborn 0.13.2** - Advanced visualization
- **Scikit-learn 1.7.2** - Machine learning algorithms

### Database & Storage
- **Redis 7.1.0** - In-memory caching and session storage
- **Qdrant 1.16.1** - High-performance vector database
- **ChromaDB 1.3.5** - AI-native open-source embedding database
- **LanceDB 0.25.3** - Serverless vector database

### Development Tools
- **Git integration** - Built-in version control
- **nbdime** - Jupyter notebook diffing and merging
- **Multiple spawner support** - Local, Docker, Kubernetes

## Quick Start

### Prerequisites
- Docker with 4GB+ RAM
- Docker Compose (optional)

### 1. Using Docker Run

```bash
# Pull the image
docker pull sinanozel/kubyterhub-ds:25.12

# Create userlist.txt with your GitHub usernames
echo "your-github-username" > userlist.txt

# Run JupyterHub with basic setup
docker run -p 8000:8000 \
  -v $(pwd)/userlist.txt:/etc/jupyterhub/userlist.txt \
  sinanozel/kubyterhub-ds:25.12

# With GitHub OAuth, user management, and persistent data
docker run -p 8000:8000 \
  -e GITHUB_CLIENT_ID=your_github_client_id \
  -e GITHUB_CLIENT_SECRET=your_github_client_secret \
  -e HUB_PROTOCOL=http \
  -e HUB_HOST=localhost \
  -e HUB_PORT=8000 \
  -v $(pwd)/userlist.txt:/etc/jupyterhub/userlist.txt \
  -v $(pwd)/data/notebooks:/home/jovyan/work \
  -v $(pwd)/data/jupyterhub:/srv/jupyterhub \
  sinanozel/kubyterhub-ds:25.12
```

### 2. Using Docker Compose

Create a `docker-compose.yaml` file:

```yaml
version: '3.8'

services:
  jupyterhub:
    image: sinanozel/kubyterhub-ds:25.12
    container_name: jupyterhub-server
    ports:
      - "8000:8000"
    volumes:
      # User notebooks and data
      - ./data/notebooks:/home/jovyan/work
      # JupyterHub database and user state
      - ./data/jupyterhub:/srv/jupyterhub
    environment:
      - JUPYTERHUB_CRYPT_KEY=your-secret-key-here
      # GitHub OAuth settings
      - GITHUB_CLIENT_ID=your_github_client_id
      - GITHUB_CLIENT_SECRET=your_github_client_secret
      - HUB_PROTOCOL=http
      - HUB_HOST=localhost
      - HUB_PORT=8000

  redis:
    image: redis:7.4.1-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant:v1.13.7
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  redis_data:
  qdrant_data:
```

```bash
docker compose up -d
```

### 3. Set Up GitHub OAuth (Required)

**Before accessing JupyterHub, you must configure GitHub OAuth:**

1. **Create GitHub OAuth App:**
   - Go to GitHub → Settings → Developer settings → OAuth Apps → New OAuth App
   - **Application name**: `KubiterHub DS`
   - **Homepage URL**: `http://localhost:8000` (or your domain)
   - **Authorization callback URL**: `http://localhost:8000/hub/oauth_callback`

2. **Set environment variables:**
   ```bash
   # Copy example and edit with your GitHub credentials
   cp .env.example .env
   # Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET
   ```

3. **Access JupyterHub:**
   - Open `http://localhost:8000`
   - Click "Sign in with GitHub"
   - Authorize the application
   - Start your JupyterLab server!

## Configuration

### GitHub OAuth (Pre-configured)

**This container comes pre-configured for GitHub OAuth authentication.** Simply set these environment variables:

```bash
# Required: GitHub OAuth App credentials
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret

# Development (default)
HUB_PROTOCOL=http
HUB_HOST=localhost
HUB_PORT=8000

# Production/Kubernetes
HUB_PROTOCOL=https
HUB_HOST=your-domain.com
HUB_PORT=443
```

### Kubernetes Deployment

For Kubernetes deployments, configure the FQDN through environment variables:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jupyterhub
spec:
  template:
    spec:
      containers:
      - name: jupyterhub
        image: sinanozel/kubyterhub-ds:25.12
        env:
        - name: GITHUB_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: github-oauth
              key: client-id
        - name: GITHUB_CLIENT_SECRET
          valueFrom:
            secretKeyRef:
              name: github-oauth
              key: client-secret
        - name: HUB_PROTOCOL
          value: "https"
        - name: HUB_HOST
          value: "jupyterhub.your-domain.com"
        - name: HUB_PORT
          value: "443"
```

### Alternative Authentication Options

To use different authenticators, edit `jupyterhub_config.py`:

- **LDAP**: Enterprise directory integration
- **Native**: Built-in user management with web interface

### Resource Management

Control user resources in `jupyterhub_config.py`:

```python
# Memory limits
c.Spawner.mem_limit = '2G'

# CPU limits
c.Spawner.cpu_limit = 1.0

# Timeout settings
c.Spawner.http_timeout = 120
c.Spawner.start_timeout = 300
```

## Database Connections

### Redis Connection
```python
import redis
r = redis.Redis(host='redis', port=6379, decode_responses=True)
r.set('key', 'value')
print(r.get('key'))
```

### Qdrant Vector Database
```python
from qdrant_client import QdrantClient

client = QdrantClient(host="qdrant", port=6333)
collections = client.get_collections()
print(f"Available collections: {collections}")
```

### ChromaDB
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_collection")
print(f"ChromaDB version: {chromadb.__version__}")
```

### LanceDB
```python
import lancedb

db = lancedb.connect("/tmp/my_db")
print(f"LanceDB version: {lancedb.__version__}")
```

## Advanced Usage

### Custom Spawners

For Kubernetes deployment:
```python
c.JupyterHub.spawner_class = 'kubespawner.KubeSpawner'
c.KubeSpawner.image = 'sinanozel/kubyterlab-ds:25.12'
```

For Docker spawning:
```python
c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.DockerSpawner.image = 'sinanozel/kubyterlab-ds:25.12'
```

### SSL/HTTPS Setup

```python
c.JupyterHub.ssl_cert = '/path/to/cert.pem'
c.JupyterHub.ssl_key = '/path/to/key.pem'
c.JupyterHub.port = 443
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JUPYTERHUB_CRYPT_KEY` | Generated | Secret key for secure cookies |
| `JUPYTERHUB_CONFIG` | `/etc/jupyterhub/jupyterhub_config.py` | Config file path |

## Ports

| Port | Service | Description |
|------|---------|-------------|
| 8000 | JupyterHub | Main hub interface |
| 8081 | Hub API | Internal hub communication |
| 8888 | JupyterLab | Individual user sessions |
| 6379 | Redis | Database port |
| 6333 | Qdrant | Vector database HTTP API |

## Troubleshooting

### Users Can't Start Servers
- Check spawner logs: `docker logs jupyterhub-server`
- Verify resource limits in config
- Ensure sufficient system resources

### Database Connection Issues
- Verify Redis/Qdrant containers are running
- Check network connectivity between containers
- Review container logs for errors

### Authentication Problems
- Verify authentication configuration
- Check OAuth app settings (if using OAuth)
- Review JupyterHub logs for auth errors

## Security Considerations

1. **Change default authentication** from DummyAuth in production
2. **Use HTTPS** with proper SSL certificates
3. **Set resource limits** to prevent resource exhaustion
4. **Regular updates** of base images and dependencies
5. **Network isolation** using Docker networks or Kubernetes namespaces

## Building from Source

```bash
git clone https://github.com/sinan-ozel/jupyterlab-on-kubernetes
cd jupyterlab-on-kubernetes/kubyterhub-ds
docker build -t kubyterhub-ds:25.12 .
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

## Support

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and examples
- **Community**: Join our discussions and share experiences