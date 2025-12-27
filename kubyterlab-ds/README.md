# 🧪 KubyterLab-DS - Data Science JupyterLab Environment

> 🚀 A production-ready data science environment with JupyterLab, Redis, PostgreSQL, Qdrant vector database, and embedded LLM services.

## 🎯 Purpose

**KubyterLab-DS** bridges the gap between local development and production deployment:

1. **💻 Local Development** - Run the complete stack locally using `docker-compose` for rapid prototyping and experimentation
2. **☸️ Kubernetes Deployment** - Deploy seamlessly to Kubernetes clusters with minimal configuration changes
3. **🔒 Production Ready** - Use frozen dependency versions for reproducible, stable production environments

This approach follows the "develop locally, deploy globally" philosophy, ensuring your notebooks work consistently from laptop to cloud.

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-sinanozel%2Fkubyterlab--ds-blue?logo=docker)](https://hub.docker.com/r/sinanozel/kubyterlab-ds)
[![Docker Pulls](https://img.shields.io/docker/pulls/sinanozel/kubyterlab-ds)](https://hub.docker.com/r/sinanozel/kubyterlab-ds)
[![Docker Image Size](https://img.shields.io/docker/image-size/sinanozel/kubyterlab-ds/25.11)](https://hub.docker.com/r/sinanozel/kubyterlab-ds)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Services](#-services)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [Data Persistence](#-data-persistence)

## ✨ Features

- 📊 **JupyterLab** - Modern web-based interactive development environment
- 🔴 **Redis** - In-memory data structure store for caching and queues
- 🎯 **Qdrant** - High-performance vector database for semantic search
- 🧠 **ChromaDB** - Lightweight embedded vector database
- 🗄️ **LanceDB** - Apache Arrow-based vector database
- 🐘 **PostgreSQL** - Production-grade relational database
- 🤖 **Ollama Embedding Service** - Text embedding generation (all-minilm-33m)
- 💬 **Ollama LLM Service** - Language model inference (gemma3-270m)

## 🚀 Quick Start

```bash
mkdir -p ~/jupyterlab/{notebooks,data}
docker run -p 8888:8888 -v ~/jupyterlab/notebooks:/home/jovyan/work -v ~/jupyterlab/data:/home/jovyan/.jupyter sinanozel/kubyterlab-ds:25.11
```
This starts without any of the services. To start with the services, see below.

### If having issues with permissions

```bash
# Create directories
mkdir -p ~/jupyterlab/{notebooks,data}

# Set proper permissions (Linux/macOS/WSL only)
sudo chown -R 1000:100 ~/jupyterlab/
sudo chmod -R 755 ~/jupyterlab/

# Run container
docker run -p 8888:8888 \
  -v ~/jupyterlab/notebooks:/home/jovyan/work \
  -v ~/jupyterlab/data:/home/jovyan/.jupyter \
  sinanozel/kubyterlab-ds:25.11
```

### Prerequisites

- Docker

## 🖥️ Setup on Any Computer

### ✅ Pre-Setup Checklist

**Before you start, verify:**

1. **Docker is installed and running**
   ```bash
   docker --version
   docker ps
   ```

2. **Docker Compose is available**
   ```bash
   # Try Docker Compose V2 first (recommended)
   docker compose version

   # Or Docker Compose V1 (legacy)
   docker-compose --version
   ```

3. **Sufficient system resources**
   - **RAM**: At least 8GB available (containers use ~6GB total)
   - **Disk**: At least 10GB free space for images and data
   - **CPU**: 2+ cores recommended

4. **Port availability**
   ```bash
   # Check if required ports are free
   netstat -tulpn | grep -E ':(8888|6379|6333|11434|11435)'

   # Or on Windows
   netstat -an | findstr -E ":(8888|6379|6333|11434|11435)"
   ```

### 🚀 Quick Setup (Any Platform)

1. **Create project directory**
   ```bash
   mkdir kubyterlab-ds-env
   cd kubyterlab-ds-env
   ```

2. **Download the docker-compose.yaml**
   ```bash
   curl -o docker-compose.yaml https://raw.githubusercontent.com/sinan-ozel/jupyterlab-on-kubernetes/main/kubyterlab-ds/docker-compose.yaml
   ```

3. **Create data directories**
   ```bash
   # Linux/macOS/WSL
   mkdir -p notebooks data/{jupyter,redis,qdrant,postgres}

   # Windows PowerShell
   mkdir -Force notebooks, data\jupyter, data\redis, data\qdrant
   ```

4. **Set proper permissions (Linux/macOS/WSL only)**
   ```bash
   # Make sure Jupyter can write to notebooks directory
   sudo chown -R 1000:100 notebooks/
   sudo chmod -R 755 notebooks/

   # Or if you prefer less restrictive permissions
   chmod -R 777 data/ notebooks/
   ```

5. **Start the services**
   ```bash
   # For Docker Compose V2 (recommended)
   KUBYTERLAB_DS_VERSION=25.12 docker compose up -d

   # For Docker Compose V1 (legacy)
   KUBYTERLAB_DS_VERSION=25.12 docker-compose up -d
   ```

6. **Get the access token**
   ```bash
   docker exec jupyter-notebook jupyter server list
   ```

7. **Access JupyterLab**
   - **Local/WSL**: http://localhost:8888/?token=YOUR_TOKEN
   - **Remote**: http://YOUR_SERVER_IP:8888/?token=YOUR_TOKEN

### 🛠️ Platform-Specific Instructions

#### 🐧 **Linux**
```bash
# Standard setup
git clone https://github.com/sinan-ozel/jupyterlab-on-kubernetes.git
cd jupyterlab-on-kubernetes/kubyterlab-ds
sudo chown -R 1000:100 notebooks/
KUBYTERLAB_DS_VERSION=25.12 docker compose up -d
```

#### 🪟 **Windows (WSL2)**
```bash
# In WSL2 terminal
git clone https://github.com/sinan-ozel/jupyterlab-on-kubernetes.git
cd jupyterlab-on-kubernetes/kubyterlab-ds
sudo chown -R 1000:100 notebooks/
KUBYTERLAB_DS_VERSION=25.12 docker compose up -d

# Access from Windows browser: http://localhost:8888/?token=YOUR_TOKEN
```

#### 🪟 **Windows (Native Docker Desktop)**
```powershell
# In PowerShell
git clone https://github.com/sinan-ozel/jupyterlab-on-kubernetes.git
cd jupyterlab-on-kubernetes\kubyterlab-ds
$env:KUBYTERLAB_DS_VERSION="25.12"
docker compose up -d
```

#### 🍎 **macOS**
```bash
# Standard setup
git clone https://github.com/sinan-ozel/jupyterlab-on-kubernetes.git
cd jupyterlab-on-kubernetes/kubyterlab-ds
sudo chown -R 1000:100 notebooks/
KUBYTERLAB_DS_VERSION=25.12 docker compose up -d
```

### 🚨 Troubleshooting Common Issues

#### **❌ Permission Denied: Untitled.ipynb**
```bash
# Fix notebook creation permissions
sudo chown -R 1000:100 notebooks/
sudo chmod -R 755 notebooks/

# Or more permissive (less secure)
sudo chmod -R 777 notebooks/ data/

# Verify inside container
docker exec jupyter-notebook touch /home/jovyan/work/test.txt
docker exec jupyter-notebook ls -la /home/jovyan/work/
```

#### **❌ Port Already in Use (8888, 6379, etc.)**
```bash
# Find what's using the port
sudo netstat -tulpn | grep :8888
# or
sudo lsof -i :8888

# Stop conflicting containers
docker ps | grep 8888
docker stop CONTAINER_NAME

# Or use different ports in docker-compose.yaml
ports:
  - "8889:8888"  # Use port 8889 instead
```

#### **❌ Docker Compose Command Not Found**
```bash
# Install Docker Compose V2
sudo apt update && sudo apt install docker-compose-plugin

# Or install V1
sudo apt install docker-compose

# Or use Docker Desktop (includes both)
```

#### **❌ Container Exits Immediately**
```bash
# Check container logs
docker compose logs jupyter
docker compose logs redis
docker compose logs qdrant

# Common fixes
docker system prune  # Clean up old containers
docker compose down && docker compose up -d
```

#### **❌ Out of Memory/Disk Space**
```bash
# Check Docker resource usage
docker system df

# Clean up unused resources
docker system prune -a

# Adjust container memory limits in docker-compose.yaml
mem_limit: 4g  # Reduce from 8g
```

#### **❌ Cannot Connect to Services (Redis/PostgreSQL/Qdrant)**
```bash
# Verify all services are running
docker compose ps

# Check network connectivity
docker exec jupyter-notebook ping redis
docker exec jupyter-notebook ping postgres
docker exec jupyter-notebook ping qdrant

# Test Redis connection
docker exec jupyter-notebook redis-cli -h redis ping

# Test PostgreSQL connection
docker exec jupyter-notebook psql -h postgres -U jupyter -d kubyterlab -c "SELECT 1;"

# Test Qdrant API
curl http://localhost:6333/collections
```

#### **🪟 Windows-Specific Issues**

**WSL2 Port Access from Windows:**
```bash
# Get WSL IP address
hostname -I

# Access via: http://WSL_IP:8888/?token=YOUR_TOKEN
# Or enable port forwarding (automatic in newer Windows)
```

**Windows Defender/Firewall:**
- Allow Docker Desktop through Windows Defender
- Allow ports 8888, 6379, 6333 through firewall

**Path Issues:**
```powershell
# Use full Windows paths in docker-compose.yaml
volumes:
  - C:\Users\YourName\kubyterlab-ds\notebooks:/home/jovyan/work
```

### ✅ Verification Steps

1. **All containers running**
   ```bash
   docker compose ps
   # Should show all services as "Up"
   ```

2. **JupyterLab accessible**
   ```bash
   curl -I http://localhost:8888
   # Should return HTTP 200 or 405
   ```

3. **Services responding**
   ```bash
   # Redis
   docker exec jupyter-notebook redis-cli -h redis ping

   # PostgreSQL
   docker exec jupyter-notebook psql -h postgres -U jupyter -d kubyterlab -c "SELECT version();"

   # Qdrant
   curl http://localhost:6333/collections

   # Embedding service
   curl http://localhost:11434/api/tags

   # LLM service
   curl http://localhost:11435/api/tags
   ```

4. **File creation works**
   ```bash
   docker exec jupyter-notebook touch /home/jovyan/work/test.ipynb
   docker exec jupyter-notebook ls -la /home/jovyan/work/
   ```

### 1. Create Docker Compose File

Create a `docker-compose.yaml` file:

```yaml
version: '3.8'

services:
  jupyter:
    image: sinanozel/kubyterlab-ds:25.12
    container_name: jupyter-notebook
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data/jupyter:/home/jovyan/.jupyter
      - ~/.ssh:/home/jovyan/.ssh:ro
    environment:
      - GRANT_SUDO=yes
    networks:
      - jupyter-network
    depends_on:
      - redis
      - qdrant
    mem_limit: 8g
    cpus: 2

  redis:
    image: redis:7.4.2-alpine
    container_name: redis
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --appendonly yes
    networks:
      - jupyter-network

  qdrant:
    image: qdrant/qdrant:v1.12.1
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant:/qdrant/storage
    networks:
      - jupyter-network

  embedding:
    image: sinanozel/ollama.0.12.11:all-minilm-33m
    container_name: embedding
    ports:
      - "11434:11434"
    networks:
      - jupyter-network

  llm:
    image: sinanozel/ollama.0.12.11:gemma3-270m
    container_name: llm
    ports:
      - "11435:11434"
    networks:
      - jupyter-network

networks:
  jupyter-network:
    driver: bridge
```

### 2. Start the Environment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### 3. Access JupyterLab

1. Open your browser to `http://localhost:8888`
2. Get the access token:
   ```bash
   docker exec jupyter-notebook jupyter server list
   ```
3. Copy the token from the URL and paste it into the login page

## 🛠️ Services

### 📓 JupyterLab (Port 8888)

Full-featured notebook environment with pre-installed packages:

- **Data Science**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Web Scraping**: BeautifulSoup4, Requests, LXML
- **Data Formats**: OpenPyXL, PyYAML, JSON5
- **Databases**: Redis client, PostgreSQL client (psycopg2), Qdrant client, ChromaDB, LanceDB
- **Utilities**: tqdm, pytz, defusedxml

### 🔴 Redis (Port 6379)

Fast in-memory key-value store with persistence enabled.

### 🐘 PostgreSQL (Port 5432)

Production-grade relational database with persistent storage.

**Connection Details:**
- Host: `postgres`
- Port: `5432`
- Database: `kubyterlab`
- Username: `jupyter`
- Password: `jupyter_password`

### 🎯 Qdrant (Ports 6333, 6334)

Vector database optimized for similarity search and recommendations.

### 🤖 Embedding Service (Port 11434)

Ollama service running all-minilm-33m model for text embeddings.

### 💬 LLM Service (Port 11435)

Ollama service running gemma3-270m for language model inference.

## 💡 Usage Examples

### 🔴 Redis Example

```python
import redis

# Connect to Redis
r = redis.Redis(host='redis', port=6379, decode_responses=True)

# Basic operations
r.set('user:1', 'John Doe')
name = r.get('user:1')
print(f"User: {name}")

# Caching with expiration
r.setex('session:abc123', 3600, 'user_data')

# Lists
r.lpush('tasks', 'task1', 'task2', 'task3')
tasks = r.lrange('tasks', 0, -1)
```

### 🐘 PostgreSQL Example

```python
import psycopg2
import pandas as pd

# Connect to PostgreSQL
conn = psycopg2.connect(
    host='postgres',
    port=5432,
    database='kubyterlab',
    user='jupyter',
    password='jupyter_password'
)

# Create a table
cur = conn.cursor()
cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        email VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# Insert data
cur.execute("INSERT INTO users (name, email) VALUES (%s, %s)",
           ("John Doe", "john@example.com"))
conn.commit()

# Query with pandas
df = pd.read_sql_query("SELECT * FROM users", conn)
print(df)

# Close connection
cur.close()
conn.close()
```

### 🎯 Qdrant Vector Search

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect to Qdrant
client = QdrantClient(host="qdrant", port=6333)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Insert vectors
points = [
    PointStruct(id=1, vector=[0.1] * 384, payload={"text": "Hello world"}),
    PointStruct(id=2, vector=[0.2] * 384, payload={"text": "Data science"}),
]
client.upsert(collection_name="documents", points=points)

# Search
results = client.search(
    collection_name="documents",
    query_vector=[0.15] * 384,
    limit=3
)
```

### 🧠 ChromaDB Example

```python
import chromadb

# Initialize ChromaDB
client = chromadb.Client()

# Create collection
collection = client.create_collection(name="my_docs")

# Add documents
collection.add(
    documents=["This is document 1", "This is document 2"],
    metadatas=[{"source": "web"}, {"source": "api"}],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["document about web"],
    n_results=2
)
print(results)
```

### 🗄️ LanceDB Example

```python
import lancedb

# Connect to LanceDB
db = lancedb.connect("./data/lancedb")

# Create table
data = [
    {"vector": [1.0, 2.0], "text": "First item"},
    {"vector": [3.0, 4.0], "text": "Second item"},
]
table = db.create_table("my_table", data=data)

# Search
results = table.search([1.5, 2.5]).limit(5).to_pandas()
print(results)
```

### 🤖 Ollama Embedding Service

```python
import requests

# Generate embeddings
response = requests.post(
    "http://embedding:11434/api/embeddings",
    json={"model": "all-minilm", "prompt": "Hello, world!"}
)
embeddings = response.json()['embedding']
print(f"Embedding dimension: {len(embeddings)}")
```

### 💬 Ollama LLM Service

```python
import requests

# Generate text
response = requests.post(
    "http://llm:11434/api/generate",
    json={
        "model": "gemma3:270m",
        "prompt": "Explain machine learning in simple terms:",
        "stream": False
    }
)
result = response.json()
print(result['response'])
```

### 🔗 Complete RAG Pipeline

```python
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. Generate embedding
def get_embedding(text):
    response = requests.post(
        "http://embedding:11434/api/embeddings",
        json={"model": "all-minilm", "prompt": text}
    )
    return response.json()['embedding']

# 2. Store in Qdrant
client = QdrantClient(host="qdrant", port=6333)
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

docs = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
]

for i, doc in enumerate(docs):
    vector = get_embedding(doc)
    client.upsert(
        collection_name="knowledge_base",
        points=[PointStruct(id=i, vector=vector, payload={"text": doc})]
    )

# 3. Search and generate answer
query = "What is machine learning?"
query_vector = get_embedding(query)
search_results = client.search(
    collection_name="knowledge_base",
    query_vector=query_vector,
    limit=2
)

context = "\n".join([hit.payload["text"] for hit in search_results])
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

response = requests.post(
    "http://llm:11434/api/generate",
    json={"model": "gemma3:270m", "prompt": prompt, "stream": False}
)
print(response.json()['response'])
```

## ⚙️ Configuration

### Environment Variables

Modify the `docker-compose.yaml` to customize:

```yaml
environment:
  - GRANT_SUDO=yes           # Allow sudo in container
  - JUPYTER_ENABLE_LAB=yes   # Use JupyterLab interface
```

### Resource Limits

Adjust memory and CPU limits:

```yaml
mem_limit: 8g    # Maximum memory
cpus: 2          # CPU cores
```

## 💾 Data Persistence

Data is persisted in the following volumes:

- `./notebooks` - Your Jupyter notebooks
- `./data/redis` - Redis data files
- `./data/qdrant` - Qdrant vector storage
- `~/.jupyter` - JupyterLab configuration
- `~/.ssh` - SSH keys (read-only)

## 🧹 Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (⚠️ deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## 📚 Installed Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.1.3 | Numerical computing |
| pandas | 2.2.3 | Data manipulation |
| matplotlib | 3.9.2 | Plotting |
| seaborn | 0.13.2 | Statistical visualization |
| scikit-learn | 1.5.2 | Machine learning |
| redis | 5.2.0 | Redis client |
| qdrant-client | 1.12.1 | Qdrant vector DB |
| chromadb | 0.5.23 | ChromaDB vector DB |
| lancedb | 0.17.0 | LanceDB vector DB |
| beautifulsoup4 | 4.12.3 | Web scraping |
| requests | 2.32.3 | HTTP library |
| lxml | 5.3.0 | XML/HTML parser |
| openpyxl | 3.1.5 | Excel files |
| PyYAML | 6.0.2 | YAML parser |
| tqdm | 4.67.1 | Progress bars |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Built on top of:
- [Jupyter Docker Stacks](https://github.com/jupyter/docker-stacks)
- [Redis](https://redis.io/)
- [Qdrant](https://qdrant.tech/)
- [Ollama](https://ollama.ai/)

---

Made with ❤️ for data scientists and ML engineers
