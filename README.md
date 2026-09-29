# AI-Institute

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![Runtime: Ollama](https://img.shields.io/badge/Runtime-Ollama-black.svg)](https://ollama.com/)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Vector DB: ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Training: PyTorch / PEFT](https://img.shields.io/badge/Training-PyTorch_%2F_PEFT-EE4C2C.svg)](https://pytorch.org/)
[![Models: Qwen2.5](https://img.shields.io/badge/Models-Qwen2.5-ffd21e.svg)](https://huggingface.co/Qwen)

AI-Institute is a framework for building, training, evaluating, and deploying role-locked AI assistants. A place where AI teaches AI. 

Generic large language models hallucinate facts, give conflicting answers, leak instructions, and wander off topic. AI-Institute solves this by constraining small open models (such as Qwen2.5 1.5B and 3B) into strict campus personas (such as a library assistant, registrar, or departmental help desk) through fine-tuning, retrieval-augmented generation, tool calling, and multi-layer defenses.

---

## Core Architecture: 4-Layer Defense Stack

Every user message passes through a 4-layer defense pipeline managed by the `ChatOrchestrator` before reaching the user:

1. Layer 1: Input Guardrails (`inference/guardrails.py`)
   - Length verification: Rejects inputs exceeding 1000 characters to prevent context-stuffing attacks.
   - Regional script detection: Catches non-English scripts (Tamil, Telugu, Malayalam, Kannada, Devanagari) and returns a polite English redirection.
   - Transliterated text detection: Identifies romanized regional slang (Hinglish, Tanglish, Manglish) using token matching and redirects to English.
   - Jailbreak filtering: Intercepts persona breaks, instruction override attempts, DAN patterns, and developer mode bypasses.
   - Role-specific off-topic regex: Blocks out-of-scope queries configured in the role definition (for example, rejecting hostel fee questions in a library assistant).
   - Early exit: Blocked queries return an instant refusal without querying the LLM or vector store.

2. Layer 2: Fine-Tuned Student Model (`inference/chat.py` via Ollama)
   - Small open models (Qwen2.5 1.5B or 3B) fine-tuned using LoRA on campus QA, polite in-character refusals, and structured tool invocation patterns.
   - Refuses out-of-scope topics politely even when adversarial prompts bypass regex filters.
   - Emits structured tool calls (`<tool_call>name(param="value")</tool_call>`) when real-time external data is needed.

3. Layer 3: RAG Context Grounding (`inference/rag.py`)
   - Persistent ChromaDB vector store maintained per role.
   - Encodes user queries using local sentence embeddings (`all-MiniLM-L6-v2`).
   - Retrieves top matching document chunks and injects them into the grounded system prompt.
   - Instructs the model to rely strictly on provided facts and refuse ungrounded claims.

4. Layer 4: Output Filter (`inference/output_filter.py`)
   - System prompt leak prevention: Redacts responses disclosing role instructions, system tags, or internal guidelines.
   - Telephone number sanitization: Replaces fabricated phone numbers with the role redirect contact unless a trusted source document was retrieved in context.
   - Blocked term filtering: Enforces role-specific geographic and hallucination blacklists.

5. Background Tool Calling and Live Streaming
   - Pass 1: The model generates initial tokens. The orchestrator buffers the first tokens to inspect for tool calls.
   - If a tool call is detected: Output streaming is suppressed. The tool executes silently against the database or API. The tool result is wrapped in `<tool_result>` tags and fed back to the model.
   - Pass 2: The model generates the final conversational answer, which streams live to the client token by token.
   - Normal responses stream immediately with zero buffering delay.

6. Conversation Memory (`inference/history.py`)
   - Multi-turn conversation state persisted in a per-role SQLite database (`conversations.db`).
   - Maintains contextual continuity across turns while isolating distinct user sessions.

---

## Key Features

- Synthetic Data Generation: Generates structured training datasets using teacher models (Ollama, NVIDIA NIM, Google Gemini, OpenAI) guided by role questionnaires.
- LoRA Fine-Tuning: Automated training with hardware optimization. Runs float32 oneDNN training on CPU to avoid bfloat16 emulation slowdowns, or 4-bit QLoRA on CUDA GPUs.
- Ollama Export: Automatically packages trained weights into versioned Modelfiles and registers models directly into local Ollama instances.
- Extensible Tool Framework: Connects assistants to real-time external data sources including SQLite databases, MySQL (such as Koha Integrated Library System), REST APIs, and local tabular files.
- REST API Server: Production-ready FastAPI service with health checks, role discovery, and Server-Sent Events (SSE) live token streaming.
- Document-Aware Pre-Warming: Preloads document context and system prompts into Ollama's KV cache on startup, dropping first-turn CPU latency from ~8s to under 2s and GPU response times to ~0.4s.
- Automatic Memory Lifecycle: Uses `keep_alive: -1` during runtime to lock models in GPU VRAM or system RAM, with a 120s warmup ceiling to accommodate cloud storage cold starts. Evicts models (`keep_alive: 0`) and flushes PyTorch GPU cache (`torch.cuda.empty_cache()`) on shutdown to free memory cleanly.
- Adversarial Pentest Suite: Built-in automated security evaluation testing 50+ college-specific attack vectors across all defense layers.
- Complete Portability: Backup and restore entire role bundles (configs, tools, and pre-computed vector stores) as compressed archives.

---

## CLI Command Reference

All commands can be run directly via `.venv/bin/ai-institute <command>`, or `ai-institute <command>` (with the virtual environment activated).

### Role Management

#### `create-role`
Initializes a new assistant role directory with basic configuration.
```bash
.venv/bin/ai-institute create-role <name> --description "<desc>" [--student-model <model>] [--teacher-mode <mode>]
```
- `name` (required): Alphanumeric role identifier.
- `--description`, `-d` (required): Scope and purpose description.
- `--student-model`, `-m`: Base model identifier (default: `qwen2.5:3b`).
- `--teacher-mode`, `-t`: Teacher guidance style, `smart` or `guided` (default: `smart`).

#### `list-roles`
Displays all configured roles in a formatted table.
```bash
.venv/bin/ai-institute list-roles
```

#### `info`
Shows detailed configuration, ingested files, tools, and status for a role.
```bash
.venv/bin/ai-institute info <role>
```

#### `delete-role`
Deletes a role and removes its directory.
```bash
.venv/bin/ai-institute delete-role <role> [--force / -f]
```

---

### Data and Vector Store Management

#### `add-data`
Copies documents to the role data directory and indexes them into ChromaDB.
```bash
.venv/bin/ai-institute add-data <role> [--file <path>] [--dir <path>]
```
- Supported formats: TXT, MD, CSV, JSON, PDF.

#### `list-data`
Lists all ingested documents and their indexed chunk counts.
```bash
.venv/bin/ai-institute list-data <role>
```

#### `remove-data`
Deletes a data file and purges its embeddings from ChromaDB.
```bash
.venv/bin/ai-institute remove-data <role> --file <filename>
```

#### `reindex`
Clears and rebuilds the vector store from all files currently in the role data folder.
```bash
.venv/bin/ai-institute reindex <role>
```

#### `search`
Runs a similarity search query directly against the role vector store.
```bash
.venv/bin/ai-institute search <role> "<query>" [--top-k <int>]
```

---

### Dataset Preparation and Prompts

#### `import-training`
Validates, deduplicates, and stores training examples in the role dataset folder.
```bash
.venv/bin/ai-institute import-training <role> [--file <path>] [--dir <path>]
```

#### `review-training`
Displays category distribution, total counts, and sample previews for a training dataset.
```bash
.venv/bin/ai-institute review-training <role>
```

#### `generate-prompts`
Runs an interactive questionnaire and creates 7 guided-mode prompt templates.
```bash
.venv/bin/ai-institute generate-prompts <role> [--force / -f]
```

#### `generate-playbook`
Runs an interactive questionnaire and creates a `playbook.md` specification for AI coding assistants.
```bash
.venv/bin/ai-institute generate-playbook <role> [--force / -f]
```

#### `auto-generate`
Uses a teacher LLM to generate synthetic training examples.
```bash
.venv/bin/ai-institute auto-generate <role> [--provider <name>] [--model <name>] [--count <int>] [--category <cat>]
```
- `--provider`, `-p`: `ollama`, `nim`, `gemini`, or `openai` (default: `ollama`).
- `--count`, `-c`: Number of examples to generate (default: 20).
- `--category`: Target specific category (`qa`, `refusal`, `jailbreak`, etc.).

---

### Fine-Tuning and Model Export

#### `train`
Executes LoRA fine-tuning on the role training dataset.
```bash
.venv/bin/ai-institute train <role> [--epochs <int>] [--batch-size <int>] [--learning-rate <float>] [--max-steps <int>]
```
- `--epochs`, `-e`: Training epochs (default: 1).
- `--batch-size`, `-b`: Batch size per device (default: 2).
- `--learning-rate`, `-lr`: Learning rate (default: 2e-4).
- `--max-steps`: Early stopping step count for quick testing.

#### `train-status`
Shows the latest training run metrics, loss, duration, and hardware target.
```bash
.venv/bin/ai-institute train-status <role>
```

#### `export`
Builds versioned Modelfiles and registers the model in Ollama.
```bash
.venv/bin/ai-institute export <role> [--model <tier>]
```
- `--model`, `-m`: Model tier override (`1.5b`, `3b`).

---

### Chat, Verification, and Pentesting

#### `query`
Sends a single query through the defense stack and streams the output to stdout.
```bash
.venv/bin/ai-institute query <role> "<message>" [--session <id>]
```

#### `test`
Launches an interactive multi-turn terminal chat session with live streaming.
```bash
.venv/bin/ai-institute test <role> [--session <id>]
```
- Interactive commands: `:exit` (quits and unloads model), `:clear` (wipes session memory), `:history` (prints conversation turns).

#### `pentest`
Runs automated adversarial test attacks against the role defense stack.
```bash
.venv/bin/ai-institute pentest <role> [--attempts <int>] [--patterns <path>]
```
- Evaluates attack blocks across Layer 1 (Guardrails), Layer 2 (Model Refusal), and Layer 4 (Output Filter).
- Automatically cleans up test session records from SQLite memory.

---

### Deployment and Lifecycle

#### `deploy`
Starts the Uvicorn REST API server in the background.
```bash
.venv/bin/ai-institute deploy <role> [--port <int>] [--host <str>] [--foreground]
```
- Automatically checks Ollama liveness and executes pipeline pre-warming.
- Records PID and host information in `.deployments.json`.

#### `status`
Displays active server deployments and process status.
```bash
.venv/bin/ai-institute status
```

#### `stop`
Gracefully stops a running deployment and unloads the model from Ollama RAM.
```bash
.venv/bin/ai-institute stop <role>
```

#### `backup`
Packages a role directory (config, tools, and pre-computed vector store) into a tarball.
```bash
.venv/bin/ai-institute backup <role> --output <path.tar.gz>
```

#### `restore`
Extracts a role tarball into the roles directory.
```bash
.venv/bin/ai-institute restore --input <path.tar.gz> [--force]
```

#### `package`
Bundles the runtime wheel, role data archive copy, Modelfile, and automated `install.sh` script into a single deployable zip file without modifying original files.
```bash
.venv/bin/ai-institute package librarian [--output <path.zip>] [--model <tier>]
```
(Alternatively: `ai-institute package librarian` with virtual environment activated)

---

## REST API Reference

The deployed API exposes the following endpoints under `/api/v1`:

### 1. Chat Completion (`POST /api/v1/chat`)
Sends a message and returns the complete response after generation finishes.
- Request Body:
  ```json
  {
    "role": "librarian",
    "message": "When is the library open?",
    "session_id": "optional-session-id"
  }
  ```
- Response:
  ```json
  {
    "response": "The library is open from 8:00 AM to 10:00 PM on weekdays.",
    "session_id": "optional-session-id",
    "sources": ["rules.txt"],
    "blocked": false,
    "elapsed": 1.84
  }
  ```

### 2. Live Token Streaming (`POST /api/v1/chat/stream`)
Streams tokens in real time using Server-Sent Events (`text/event-stream`). Background tool calls are suppressed during execution.
- Stream events:
  ```text
  data: {"type": "token", "content": "The"}
  data: {"type": "token", "content": " library"}
  data: {"type": "token", "content": " is"}
  ...
  data: {"type": "meta", "session_id": "...", "blocked": false, "layer": "model", "sources": ["rules.txt"], "elapsed": 1.84}
  ```

### 3. Health Check (`GET /api/v1/health`)
Returns server operational status and list of loaded roles.

### 4. Role Discovery (`GET /api/v1/roles` and `GET /api/v1/roles/{name}`)
Returns role summaries, questionnaire definitions, registered tools, and ingested files.

---

## Quickstart Workflow

Follow this sequence to build and run an assistant from scratch:

```bash
# 1. Create the role
.venv/bin/ai-institute create-role librarian --description "VIT Library Assistant" --student-model "qwen2.5:3b"

# 2. Ingest domain documents
.venv/bin/ai-institute add-data librarian --file /path/to/library_rules.txt
.venv/bin/ai-institute add-data librarian --file /path/to/catalog.csv

# 3. Generate teacher specification and training data
.venv/bin/ai-institute generate-playbook librarian
.venv/bin/ai-institute auto-generate librarian --count 50

# 4. Train the model
.venv/bin/ai-institute train librarian --epochs 3

# 5. Export to Ollama
.venv/bin/ai-institute export librarian --model 3b

# 6. Verify defenses
.venv/bin/ai-institute pentest librarian --attempts 20

# 7. Test in terminal
.venv/bin/ai-institute test librarian

# 8. Deploy as a background REST API
.venv/bin/ai-institute deploy librarian --port 8080
```

---

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Free for students, educators, researchers, and noncommercial use.

For commercial licensing or enterprise use, contact: kishal2007@gmail.com

