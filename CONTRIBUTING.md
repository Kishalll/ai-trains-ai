# AI-Institute Technical Architecture and Contributor Guide

This document covers the technical architecture, directory layout, RAG embedding mechanics, technology stack, development and server deployment environments, reindexing procedures, and external database integrations.

---

## 1. System Architecture

The AI-Institute architecture decouples client interaction from model inference and data retrieval.

```text
User / HTTP Client
        │
        ▼
FastAPI API Server (`api/`) or CLI (`cli.py`)
        │
        ▼
ChatOrchestrator (`inference/chat.py`)
        │
        ├──► Layer 1: Guardrails (`inference/guardrails.py`)
        │        ├── Length check (max 1000 chars)
        │        ├── Non-English script regex (Tamil, Telugu, Malayalam, Kannada, Devanagari)
        │        ├── Transliterated slang token matching (Hinglish, Tanglish, Manglish)
        │        ├── Adversarial jailbreak regex
        │        └── Role off-topic patterns
        │
        ├──► Layer 3: Context Retrieval (`inference/rag.py`)
        │        ├── Local query embedding (`data_processing/embedder.py`)
        │        └── Cosine similarity search in ChromaDB vector store
        │
        ├──► Conversation State (`inference/history.py`)
        │        └── Recent turns loaded from SQLite (`conversations.db`)
        │
        ├──► Layer 2: Ollama Inference (`ai-institute-<role>:<tier>`)
        │        ├── Pass 1: Stream inspection & buffer check
        │        │      ├── Normal response: streams live to caller
        │        │      └── Tool call detected: suppresses streaming
        │        │
        │        └── Silent Tool Execution (`tools/executor.py`)
        │               ├── Execute DbTool / ApiTool / FileTool / StaticTool
        │               └── Feed `<tool_result>` back into model for Pass 2 final stream
        │
        └──► Layer 4: Output Filter (`inference/output_filter.py`)
                 ├── System prompt leakage detection
                 ├── Fabricated phone number redaction
                 └── Blocked terms / hallucinated entities check
```

---

## 2. Directory Structure and File Reference

```text
ai-institute/
├── .venv/                      # Python virtual environment (ignored in git)
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Core runtime dependencies (typer, pyyaml, rich, jinja2)
├── config.yaml                 # Global framework configuration
├── pyproject.toml              # Packaging specification for pip and standalone wheel builds
├── cli.py                      # Typer CLI implementation containing all 26 commands
├── .deployments.json           # Active server deployments registry
├── PUBLIC_URL.txt              # Live public reverse tunnel URL (synced from tunnel daemon)
├── commands.md                 # Complete CLI command reference and options guide
├── README.md                   # User-facing overview and CLI reference
├── CONTRIBUTING.md             # Technical architecture and guide (this file)
│
├── scripts/                    # Deployment templates and service scripts
│   ├── tunnel.py               # Outbound port 443 reverse SSH tunnel supervisor
│   └── ai-institute-tunnel.service.template # Systemd service unit template for port 443 reverse tunnel
│
├── api/                        # REST API implementation
│   ├── __init__.py
│   ├── models.py               # Pydantic request and response schemas
│   ├── routes.py               # FastAPI route handlers for chat, SSE streaming, health, roles
│   └── server.py               # FastAPI application factory and ASGI entrypoint
│
├── data_processing/            # Ingestion, chunking, and embedding
│   ├── __init__.py
│   ├── ingest.py               # Document loaders for TXT, MD, CSV, JSON, and PDF
│   ├── chunker.py              # Sentence/paragraph chunker with configurable overlap
│   └── embedder.py             # SentenceTransformers embedding model wrapper
│
├── inference/                  # Core inference runtime and defense layers
│   ├── __init__.py
│   ├── chat.py                 # ChatOrchestrator managing multi-pass flow and streaming
│   ├── guardrails.py           # Layer 1 input validation and pattern blocking
│   ├── history.py              # SQLite multi-turn conversation storage
│   ├── output_filter.py        # Layer 4 response sanitization and leak redaction
│   └── rag.py                  # ChromaDB vector store interface per role
│
├── tools/                      # Tool execution framework
│   ├── __init__.py
│   ├── base.py                 # Base Tool abstract class with Jinja2 rendering
│   ├── db_tool.py              # Parameterized SQL query execution for SQLite and MySQL
│   ├── api_tool.py             # HTTP GET/POST external API integrations
│   ├── file_tool.py            # Local CSV and JSON tabular search
│   ├── static_tool.py          # Fixed policy and FAQ responses
│   ├── executor.py             # Tool invocation parser, executor, and rate limiter
│   ├── registry.py             # Parser for role tools.yaml configuration
│   └── templates/              # Production templates for Koha, REST, and CSV tools
│
├── training/                   # Fine-tuning and evaluation pipelines
│   ├── __init__.py
│   ├── dataset.py              # Training example validation, deduplication, and statistics
│   ├── trainer.py              # LoRA/QLoRA trainer using Hugging Face PEFT and TRL
│   ├── export.py               # Ollama Modelfile generation and registration
│   └── pentest.py              # Automated multi-layer adversarial attack suite
│
├── teachers/                   # Synthetic dataset generation
│   ├── __init__.py
│   ├── base.py                 # Abstract teacher class and JSON parsing logic
│   ├── ollama_teacher.py       # Local Ollama structured generator
│   ├── nim_teacher.py          # NVIDIA NIM API generator
│   ├── gemini_teacher.py       # Google Gemini API generator
│   ├── openai_teacher.py       # OpenAI API generator
│   ├── factory.py              # Provider registry factory
│   └── prompts/                # Guided-mode generation templates
│
└── roles/                      # Active assistant roles directory
    └── <role_name>/
        ├── role.yaml           # Role scope, model target, and filter rules
        ├── tools.yaml          # Tool definitions for this role
        ├── data/               # Ingested knowledge documents
        ├── training_data/      # Validated training JSON files
        ├── prompts/            # Generated guided-mode prompts
        ├── playbook.md         # Generated teacher orchestration playbook
        ├── conversations.db    # SQLite conversation memory
        ├── model/              # LoRA adapter weights and exported Modelfiles
        └── vectorstore/        # Persistent ChromaDB vector database
```

---

## 3. Embedding and RAG Pipeline

### Embedding Model
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Device: CPU (fp32)
- Caching: Downloaded once and cached locally in `~/.cache/huggingface/hub/`.
- Execution: Managed via `data_processing/embedder.py` with singleton instance caching to avoid redundant model loads during a process lifecycle.

### Document Chunking
- Module: `data_processing/chunker.py`
- Target chunk size: 500 characters
- Chunk overlap: 50 characters
- Strategy: Paragraph splitting (`\n\n`) followed by sentence boundary fallback (`. `, `? `, `! `).
- Chunk IDs: Deterministic sequential IDs formatted as `<filename>_c0`, `<filename>_c1`.
- Metadata: Each chunk records `source` (filename) and `chunk_id`.

### Vector Store Implementation
- Database: ChromaDB persistent client (`chromadb.PersistentClient`).
- Storage location: `roles/<role_name>/vectorstore/`
- Similarity metric: Cosine distance.
- Retrieval query: Returns the top-k (default: 3) highest scoring text chunks.
- Deduplication: Chunks with identical hashes or IDs are not duplicated during incremental ingestion.

### KV Cache Prefix Pre-Warming & Memory Lifecycle
When an LLM evaluates a query, evaluating a long system prompt with RAG context on CPU can take 6 to 8 seconds, while cold-loading model weights from cloud storage into GPU VRAM on first boot can take 30 to 55 seconds. AI-Institute eliminates cold-start latency using document-aware pre-warming and memory pinning in `inference/chat.py`:
1. `ChatOrchestrator.warmup()` retrieves the default rule chunks from the vector store.
2. It compiles the full system prompt matching the exact prefix that live queries will use.
3. It sends a short probe query (`"hi"`) with `keep_alive: -1`, `num_predict: 1`, and a 120-second timeout to Ollama.
4. Ollama transfers the model weights into GPU VRAM (or system RAM on CPU), processes the system prompt, and caches the key-value (KV) attention matrix.
5. Subsequent user queries matching that system prefix reuse the cached KV state and pinned weights, dropping query latency to 0.4s to 1.1s on GPU (and ~1.8s on CPU).
6. Upon session exit or server stop, `orchestrator.unload()` sends `keep_alive: 0` to Ollama to evict model weights from VRAM/RAM, and triggers `torch.cuda.empty_cache()` to release PyTorch embedding tensor allocations on the GPU.

---

## 4. Technology Stack and Rationale

| Technology | Purpose | Implementation Rationale |
|---|---|---|
| `typer` & `rich` | CLI and terminal output | Strong type validation, intuitive command hierarchies, and formatted tables/panels. |
| `fastapi` & `uvicorn` | REST API runtime | High-performance asynchronous ASGI server supporting Server-Sent Events (SSE) streaming. |
| `chromadb` | Vector database | Embedded SQLite and Parquet storage requiring no separate daemon or external service. |
| `sentence-transformers` | Text embeddings | Fast, local 384-dimensional dense vectors with no cloud API dependency. |
| `torch` & `peft` | Fine-tuning engine | Parameter-efficient fine-tuning via LoRA on attention projection layers (`q_proj`, `v_proj`). |
| `trl` | SFT training | Streamlines Supervised Fine-Tuning using `SFTTrainer` with token masking. |
| `ollama` | Model serving runtime | Manages GGUF quantized models with KV prompt caching, memory keep-alive, and token streaming. |
| `jinja2` | Templating engine | Safe template rendering for dynamic database queries and structured text responses. |
| `sqlite3` | Conversation memory | Zero-configuration serverless ACID storage for multi-turn session persistence. |
| `mysqlclient` | Live database connectivity | Robust C-based client for querying live external databases like Koha ILS. |

### CPU vs GPU Training Optimization
In `training/trainer.py`, hardware detection (`detect_hardware()`) configures precision based on available hardware:
- CPU training: Enforces `torch.float32`. PyTorch CPU backward passes in `bfloat16` lack vectorized oneDNN kernels and run in slow scalar emulation (~9.38s per step). `float32` fully utilizes multi-threaded AVX2 and AVX_VNNI instructions (~0.057s per step), delivering a 160x speedup on multi-core CPUs.
- GPU training: Uses 4-bit NormalFloat quantization (QLoRA) with `bitsandbytes` when CUDA is detected, minimizing VRAM consumption.

---

## 5. Development Machine Setup

Follow these steps to set up a full development environment including fine-tuning and evaluation tools:

### Prerequisites
- Linux OS (tested on Linux 7.1.8-arch1-3)
- Python 3.10, 3.11, or 3.12
- Git
- Ollama installed (`curl -fsSL https://ollama.com/install.sh | sh`)

### Setup Instructions

1. Clone the repository and enter the directory:
   ```bash
   git clone git@github.com/Kishalll/ai-institute.git
   cd ai-institute
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install optional groups as needed:
   AI-Institute manages optional dependencies via `deps.py`. You can install them with pip:
   ```bash
   # RAG vector store dependencies:
   pip install chromadb sentence-transformers

   # REST API server dependencies:
   pip install fastapi uvicorn ollama

   # Training and fine-tuning dependencies:
   pip install torch transformers peft trl accelerate datasets

   # External MySQL client (optional):
   pip install mysqlclient
   ```

5. Pull the base student model in Ollama:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull qwen2.5:1.5b
   ```

6. Verify the installation:
   ```bash
   .venv/bin/ai-institute list-roles
   ```
   (Alternatively: `ai-institute list-roles` with virtual environment activated).

7. Train the model:
   Because model weights are excluded from Git to prevent repository bloat, fine-tune the student model on the included training dataset:
   ```bash
   .venv/bin/ai-institute train librarian --epochs 3
   ```
   This trains on the role dataset in `roles/librarian/training_data/` and saves the LoRA adapter weights to `roles/librarian/model/`.
   Alternatively, if you already have trained weights from another machine or an AWS instance, you can copy the `roles/librarian/model/` directory into your cloned repository instead of retraining.

8. Export and register the model in Ollama:
   ```bash
   .venv/bin/ai-institute export librarian --model 3b
   ```
   This creates the Modelfile and registers `ai-institute-librarian:3b` in your local Ollama instance.

9. Test the assistant:
   Start an interactive terminal chat session with live token streaming:
   ```bash
   .venv/bin/ai-institute test librarian
   ```
   You can also test a single query:
   ```bash
   .venv/bin/ai-institute query librarian "What are the library timings?"
   ```

---

## 6. Production Server Deployment (Package Bundle Workflow)

Deploying to a production server requires no repository cloning and avoids installing 4+ GB of PyTorch and fine-tuning libraries.

### Pre-Moving Steps (Development Machine)
1. Generate the standalone deployment bundle:
   ```bash
   .venv/bin/ai-institute package librarian
   ```
   (Alternatively: `ai-institute package librarian` with virtual environment activated).

2. Verify the output bundle:
   This creates `dist/librarian-deploy.zip` (~8.6 MB) containing:
   - `ai_institute-0.1.0-py3-none-any.whl` (lightweight runtime wheel)
   - `librarian.tar.gz` (archive copy of role config, tools, and pre-computed ChromaDB vector store)
   - `Modelfile.3b` (Ollama model definition)
   - `install.sh` (automated setup script)
   All original files, model weights, and role folders on the development machine remain completely untouched.

3. Transfer the single zip file to your server:
   ```bash
   scp dist/librarian-deploy.zip user@your-server-ip:~/
   ```

### Server-Side Installation Steps
1. Prerequisites on the server:
   - Python 3.10+ installed
   - Ollama installed (`curl -fsSL https://ollama.com/install.sh | sh`) and running (`ollama serve`)

2. Create a clean virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Unpack and run the automated installer:
   ```bash
   unzip librarian-deploy.zip
   bash install.sh
   ```
   This automated script:
   - Installs the runtime wheel into the virtual environment (~250 MB footprint, no training dependencies).
   - Runs `ollama create ai-institute-librarian:3b -f Modelfile.3b` (automatically pulling the base model if needed and applying system prompt configuration).
   - Restores the role data and ChromaDB vector store via `ai-institute restore -i librarian.tar.gz`.

### Server-Side Post-Installation Operations
1. Test interactively in terminal:
   ```bash
   ai-institute test librarian
   ```

2. Verify RAG reindexing works on the server:
   ```bash
   ai-institute reindex librarian
   ```

3. Start the background API service:
   ```bash
   # Standard local background deployment:
   ai-institute deploy librarian --port 8080 --host 0.0.0.0

   # Or deploy with integrated public reverse tunnel for your team:
   ai-institute deploy librarian --port 8080 --host 0.0.0.0 --tunnel

   # Or deploy with 24/7 OS systemd autostart on system boot:
   ai-institute deploy librarian --port 8080 --host 0.0.0.0 --tunnel --autostart
   ```
   Note on deployment options:
   - `--background` / `-d` (default true): Runs in the background as a detached process.
   - `--tunnel`: Establishes an outbound port 443 reverse SSH tunnel and outputs the public HTTPS URL.
   - `--autostart`: Creates and enables systemd service units for 24/7 automatic reboot persistence.
   - Remote host (`0.0.0.0`): Listens on all available network interfaces. Always pass `--host 0.0.0.0` when hosting on a remote server, virtual machine, or container so external clients and web browsers can connect over the network.

4. Check active deployments:
   ```bash
   ai-institute status
   ```

5. Stop the deployment and release model memory:
   ```bash
   ai-institute stop librarian

   # If deployed with --autostart, also stops and disables the systemd services:
   ai-institute stop librarian --autostart
   ```

6. Expose the deployment publicly (Firewall-Resilient Reverse Tunnel):
   In institutional and enterprise hosting environments, inbound ports (such as 8080) are frequently blocked by network firewalls, and outbound tunneling tools (like Cloudflare Tunnel or standard SSH) are restricted or blocked on ports 22 or UDP/7844.

   AI-Institute includes an integrated reverse tunnel operating over outbound HTTPS port 443 via `--tunnel`:
   ```bash
   # Deploy with integrated reverse tunnel
   ai-institute deploy librarian --port 8080 --host 0.0.0.0 --tunnel

   # Or with 24/7 systemd reboot autostart + tunnel
   ai-institute deploy librarian --port 8080 --host 0.0.0.0 --tunnel --autostart

   # To manually supervise standalone tunnels without the API server, use the helper script:
   python scripts/tunnel.py --port 8080
   ```

   **Key Technical Features:**
   - **Port 443 Egress**: Establishes a secure reverse SSH tunnel through `a.pinggy.io:443`, bypassing campus hardware firewalls (e.g., Fortinet FortiGate) without requiring root network privileges or open inbound ports.
   - **Background Daemon (`-d` / `--background`)**: `deploy` spawns the tunnel detached in the background, prints the assigned public URLs and PID, and returns control to your terminal.
   - **Clean Stop (`stop <role>`)**: Terminates running background tunnel processes and cleans up `.tunnel.pid`.
   - **Live Ingress URLs**: Outputs the assigned public HTTPS address, dark-mode web playground URL (`/`), and interactive Swagger documentation URL (`/docs`).
   - **URL Synchronization**: Automatically syncs the active tunnel URL into `PUBLIC_URL.txt`.
   - **Systemd Daemonization**: For 24/7 background operation managed by OS init, a systemd template is provided at `scripts/ai-institute-tunnel.service.template`.

---

## 7. Data Ingestion, Maintenance, and Reindexing

### Adding New Documents
When new policy files, book catalogs, or rule documents are available, add them directly:
```bash
.venv/bin/ai-institute add-data librarian --file /path/to/updated_rules.txt
```
The ingestion process:
1. Reads the document format (`data_processing/ingest.py`).
2. Generates text chunks with overlap (`data_processing/chunker.py`).
3. Computes dense embeddings using `sentence-transformers`.
4. Writes vectors and metadata into `roles/<role>/vectorstore/`.
5. Copies the raw document into `roles/<role>/data/`.

### Reindexing Existing Data
If embedding configurations change or vector data needs refreshing from disk:
```bash
.venv/bin/ai-institute reindex librarian
```
The reindex workflow:
1. Opens the role's ChromaDB instance.
2. Clears all existing indexed chunks.
3. Reads all documents currently present in `roles/<role>/data/`.
4. Re-chunks and re-embeds all content.
5. Re-populates the vector store with updated embeddings.
This command requires only the lightweight runtime dependencies and can be executed on production servers without training tools.

### Updating Datasets, Retraining, and Exporting

When new dialogue scenarios, guardrail refusal examples, or tool-calling data are created, add them directly to the role dataset before retraining.

1. Adding new training examples:
   Add examples into the appropriate JSON file in `roles/<role>/training_data/` or import an external batch:
   ```bash
   .venv/bin/ai-institute import-training librarian --file /path/to/new_examples.json
   ```
   Verify the total counts and category balance:
   ```bash
   .venv/bin/ai-institute review-training librarian
   ```

2. Fine-tuning the updated model:
   ```bash
   .venv/bin/ai-institute train librarian --epochs 3 --batch-size 2 --grad-accum 4
   ```

3. Choosing training flag values:
   - `--epochs` / `-e`:
     - `1`: Fastest option. Use this for quick sanity testing to verify that tokenization, data formatting, and GPU kernel compilation succeed.
     - `3` (Recommended): Best overall training quality. Ensures strong role-locking, tool-calling syntax fidelity, and loss convergence without memorization.
     - `5` or more: Slower. Only use when training on very small datasets (under 100 examples) that need extra repetitions. Avoid on larger datasets to prevent overfitting.
   - `--batch-size` / `-b`:
     - `2` (Default): Recommended for standard 16 GB GPUs (like NVIDIA T4) and CPU systems to avoid out-of-memory errors.
     - `4` or higher: Increases throughput and reduces total training time if running on larger GPUs (24 GB+ VRAM like A10G or A100).
   - `--grad-accum` / `-ga`:
     - `4` (Default): Smooths out noisy gradient updates by accumulating gradients across 4 steps, producing an effective batch size of 8 without extra VRAM usage.
     - `1` or `2`: Speeds up step transitions when native `--batch-size` is already 4 or higher.
   - `--learning-rate` / `-lr`:
     - `2e-4` (Default): Standard learning rate for AdamW optimizer with cosine decay on LoRA adapters.
     - `1e-4`: Use for gentle, incremental fine-tuning when introducing small batches of new edge-case data without shifting existing behavior.
   - `--max-steps`:
     - Useful for quick pipeline dry-runs (for example, `--max-steps 10`) to confirm model setup before committing to a full multi-epoch run.

4. Re-exporting and testing:
   Once fine-tuning completes, register the new weights in Ollama:
   ```bash
   .venv/bin/ai-institute export librarian --model 3b
   ```
   Then run interactive tests to verify the updated behavior:
   ```bash
   .venv/bin/ai-institute test librarian
   ```

---

## 8. Live Database Integration (Koha ILS MySQL)

AI-Institute supports parameterized SQL queries against external relational databases, such as the Koha Integrated Library System (ILS).

### Configuration (`roles/<role>/tools.yaml`)
Define database tools in the role `tools.yaml` file:

```yaml
tools:
  - name: search_catalog
    type: database
    description: "Search library catalog by title or author to check real-time availability and shelf location."
    parameters:
      query:
        type: string
        description: "Book title or author name to search"
        required: true
    connection:
      type: mysql
      host: 192.168.1.50
      port: 3306
      database: koha
      user: koha_readonly
      password_env: KOHA_DB_PASSWORD
    query: |
      SELECT b.title, b.author, b.biblionumber,
             COUNT(i.itemnumber) as total_copies,
             SUM(CASE WHEN i.onloan IS NULL AND i.itemlost = 0 THEN 1 ELSE 0 END) as available_copies,
             MAX(i.itemcallnumber) as call_number,
             MAX(i.location) as section
      FROM biblio b
      JOIN items i ON b.biblionumber = i.biblionumber
      WHERE b.title LIKE %s OR b.author LIKE %s
      GROUP BY b.biblionumber, b.title, b.author
      LIMIT 5
    response_template: |
      Found {{ results | length }} matching record(s):
      {% for book in results %}
      - Title: {{ book.title }}
        Author: {{ book.author }}
        Availability: {{ book.available_copies }} of {{ book.total_copies }} copies available
        Call Number: {{ book.call_number }}
        Section: {{ book.section }}
      {% endfor %}
```

### Security and Implementation Contracts
1. Read-Only Database User:
   Always configure a dedicated MySQL user with read-only permissions on the target database:
   ```sql
   CREATE USER 'koha_readonly'@'%' IDENTIFIED BY 'secure_password';
   GRANT SELECT ON koha.biblio TO 'koha_readonly'@'%';
   GRANT SELECT ON koha.biblioitems TO 'koha_readonly'@'%';
   GRANT SELECT ON koha.items TO 'koha_readonly'@'%';
   FLUSH PRIVILEGES;
   ```
2. Password Handling via Environment Variables:
   Passwords are never stored in `tools.yaml`. The `password_env` field specifies the environment variable to read (for example, `export KOHA_DB_PASSWORD="secure_password"`).
3. Query Validation:
   The `DbTool` class (`tools/db_tool.py`) verifies that incoming SQL statements start with `SELECT` or `WITH`. Multi-statement queries containing semicolons are rejected to prevent SQL injection.
4. Parameter Expansion:
   When a SQL query contains multiple parameter placeholders for the same input argument (such as checking both title and author with `%s`), `DbTool` automatically duplicates the arguments to match the parameter count.
5. Network Configuration:
   Ensure inbound TCP port 3306 is allowed through the firewall between the assistant application server and the database server.
