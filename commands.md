# AI-Institute CLI Command Reference

This document provides a comprehensive reference for all 26 commands implemented in the `ai-institute` CLI ([`cli.py`](file:///root/ai-trains-ai/cli.py)).

---

## Command Execution

All commands can be executed using either the active virtual environment or direct executable invocation:

```bash
# With virtual environment activated:
ai-institute <command> [OPTIONS] [ARGUMENTS]

# Or directly through virtual environment binary:
.venv/bin/ai-institute <command> [OPTIONS] [ARGUMENTS]
```

---

## Table of Contents

- [1. Role Management](#1-role-management)
  - [`create-role`](#create-role)
  - [`list-roles`](#list-roles)
  - [`info`](#info)
  - [`delete-role`](#delete-role)
- [2. Knowledge Ingestion & Vector Store (RAG)](#2-knowledge-ingestion--vector-store-rag)
  - [`add-data`](#add-data)
  - [`list-data`](#list-data)
  - [`remove-data`](#remove-data)
  - [`reindex`](#reindex)
  - [`search`](#search)
- [3. Dataset Preparation & Synthetic Data Generation](#3-dataset-preparation--synthetic-data-generation)
  - [`import-training`](#import-training)
  - [`review-training`](#review-training)
  - [`generate-prompts`](#generate-prompts)
  - [`generate-playbook`](#generate-playbook)
  - [`auto-generate`](#auto-generate)
- [4. Fine-Tuning & Model Registration](#4-fine-tuning--model-registration)
  - [`train`](#train)
  - [`train-status`](#train-status)
  - [`export`](#export)
- [5. Inference, Verification & Adversarial Pentesting](#5-inference-verification--adversarial-pentesting)
  - [`query`](#query)
  - [`test`](#test)
  - [`pentest`](#pentest)
- [6. Deployment, Public Tunneling & OS Autostart](#6-deployment-public-tunneling--os-autostart)
  - [`deploy`](#deploy)
  - [`status`](#status)
  - [`stop`](#stop)
- [7. Packaging, Backup & Portability](#7-packaging-backup--portability)
  - [`backup`](#backup)
  - [`restore`](#restore)
  - [`package`](#package)

---

## 1. Role Management

### `create-role`
Initializes a new assistant role directory structure under `roles/<name>/` with a base configuration file (`role.yaml`).

**Usage:**
```bash
ai-institute create-role <name> --description "<desc>" [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | `string` | **Yes** | Unique identifier for the role (e.g. `librarian`, `registrar`, `admissions`). |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--description` | `-d` | `string` | *Required* | Clear description of the assistant's scope, responsibilities, and operational boundaries. |
| `--student-model` | `-m` | `string` | `qwen2.5:3b` | Target small student LLM to fine-tune (e.g. `qwen2.5:0.5b`, `qwen2.5:1.5b`, `qwen2.5:3b`, `qwen2.5:7b`). |
| `--teacher-mode` | `-t` | `string` | `smart` | Teacher AI orchestration style: `smart` (playbook-driven) or `guided` (prompt-template driven). |

**Examples:**
```bash
# Basic role creation
ai-institute create-role librarian -d "VIT Campus Library Assistant"

# Specify a lighter model and guided teacher mode
ai-institute create-role proctor -d "Hostel Discipline Assistant" -m qwen2.5:1.5b -t guided
```

---

### `list-roles`
Renders a formatted table listing all configured roles, showing their target models, teacher modes, and descriptions.

**Usage:**
```bash
ai-institute list-roles
```

**Options:**
None.

---

### `info`
Displays detailed role configuration, including saved questionnaire parameters, guardrail rules, registered tool names, and ingested data files.

**Usage:**
```bash
ai-institute info <name>
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | `string` | **Yes** | Target role identifier to inspect. |

---

### `delete-role`
Permanently deletes an entire role directory (`roles/<name>/`), including its vector store, training data, conversation history, and fine-tuned weights.

**Usage:**
```bash
ai-institute delete-role <name> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | `string` | **Yes** | Role name to remove. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--force` | `-f` | `bool` | `False` | Bypasses the confirmation prompt and deletes immediately. |

---

## 2. Knowledge Ingestion & Vector Store (RAG)

### `add-data`
Ingests plain text, markdown, CSV, JSON, or PDF documents into the role's `data/` folder, chunks them, computes sentence embeddings (`all-MiniLM-L6-v2`), and indexes them into ChromaDB.

**Usage:**
```bash
ai-institute add-data <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--file` | `-f` | `path` | `None` | Path to a single document (`.txt`, `.md`, `.csv`, `.json`, `.pdf`). |
| `--dir` | `-d` | `path` | `None` | Path to a directory of documents to ingest in bulk. |

**Examples:**
```bash
# Add a single file
ai-institute add-data librarian -f docs/library_rules.txt

# Bulk ingest a directory
ai-institute add-data librarian -d /data/campus_policies/
```

---

### `list-data`
Displays a table of all files currently ingested for the role along with file sizes and indexed vector chunk counts.

**Usage:**
```bash
ai-institute list-data <role>
```

---

### `remove-data`
Deletes a specific data file from `roles/<role>/data/` and purges all of its associated vector embeddings from the ChromaDB collection.

**Usage:**
```bash
ai-institute remove-data <role> --file <filename>
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--file` | `-f` | `string` | *Required* | Base name of the file to remove (e.g. `old_rules.txt`). |

---

### `reindex`
Wipes and recalculates the role's entire ChromaDB vector store by re-reading all files currently inside `roles/<role>/data/`.

**Usage:**
```bash
ai-institute reindex <role>
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |

---

### `search`
Performs a direct similarity search against the role's ChromaDB collection, displaying top-matching chunks with distance scores without invoking an LLM.

**Usage:**
```bash
ai-institute search <role> "<query>" [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |
| `query` | `string` | **Yes** | Natural language text query to embed and search. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--top-k` | `-k` | `int` | `3` | Number of most relevant matching chunks to retrieve. |

---

## 3. Dataset Preparation & Synthetic Data Generation

### `import-training`
Validates, deduplicates, and saves external training dataset JSON files into `roles/<role>/training_data/`.

**Usage:**
```bash
ai-institute import-training <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--file` | `-f` | `path` | `None` | Path to a single training data JSON file. |
| `--dir` | `-d` | `path` | `None` | Path to a directory containing training data JSON files. |

---

### `review-training`
Analyzes all validated training JSON files in `roles/<role>/training_data/`, reporting category distributions, token statistics, and sample instruction previews.

**Usage:**
```bash
ai-institute review-training <role>
```

---

### `generate-prompts`
Runs an interactive requirements questionnaire and generates 7 guided-mode prompt templates in `roles/<role>/prompts/` (used for teacher LLM prompting).

**Usage:**
```bash
ai-institute generate-prompts <role> [OPTIONS]
```

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--force` | `-f` | `bool` | `False` | Forces re-prompting of the questionnaire even if previous answers exist in `role.yaml`. |

---

### `generate-playbook`
Runs the questionnaire and compiles a comprehensive `playbook.md` in `roles/<role>/` containing strict role specifications, category targets, and the explicit "Zero Assumptions" rule.

**Usage:**
```bash
ai-institute generate-playbook <role> [OPTIONS]
```

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--force` | `-f` | `bool` | `False` | Re-runs the questionnaire rather than reusing existing `role.yaml` answers. |

---

### `auto-generate`
Leverages a Teacher LLM to automatically generate synthetic training examples adhering strictly to the role scope.

**Usage:**
```bash
ai-institute auto-generate <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--provider` | `-p` | `string` | `ollama` | Teacher provider: `ollama`, `nim`, `gemini`, or `openai`. |
| `--model` | `-m` | `string` | `None` | Model name override for the teacher provider. |
| `--count` | `-c` | `int` | `20` | Total number of training examples to synthesize. |
| `--category` | — | `string` | `None` | Specific category to generate (e.g. `refusal`, `jailbreak`, `tool_calling_catalog`). Defaults to even distribution. |

---

## 4. Fine-Tuning & Model Registration

### `train`
Executes parameter-efficient fine-tuning (LoRA / QLoRA) on the base student model using validated datasets in `roles/<role>/training_data/`.

**Usage:**
```bash
ai-institute train <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--epochs` | `-e` | `int` | `1` | Number of training epochs over the dataset. |
| `--batch-size` | `-b` | `int` | `2` | Batch size per device. |
| `--grad-accum` | `-ga` | `int` | `4` | Number of gradient accumulation steps before updating weights. |
| `--learning-rate` | `-lr` | `float` | `0.0002` | Optimizer learning rate. |
| `--max-steps` | — | `int` | `None` | Maximum training steps (useful for smoke testing pipeline). |

**Examples:**
```bash
# Production training on GPU
ai-institute train librarian --epochs 3 --batch-size 4 --grad-accum 4

# Quick pipeline test on CPU
ai-institute train librarian --max-steps 10
```

---

### `train-status`
Reads `roles/<role>/model/training_status.json` and renders a summary table of the latest training metrics (loss, duration, device type, steps).

**Usage:**
```bash
ai-institute train-status <role>
```

---

### `export`
Merges trained LoRA adapter weights with base model weights into `roles/<role>/model/merged/`, automatically converts them to 16-bit GGUF format (`<role>-<tier>.gguf`) using `llama.cpp`, and registers the model in Ollama via a custom `Modelfile`.

**Usage:**
```bash
ai-institute export <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | `-m` | `string` | `None` | Model tier override tag (e.g. `1.5b` or `3b`). |

---

## 5. Inference, Verification & Adversarial Pentesting

### `query`
Sends a single prompt to the assistant role through the full 4-layer defense orchestrator and prints the response with source attribution.

**Usage:**
```bash
ai-institute query <role> "<message>" [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |
| `message` | `string` | **Yes** | Natural language user message. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--session` | `-s` | `string` | `None` | Session ID to track multi-turn conversation memory in SQLite. |

---

### `test`
Launches an interactive real-time streaming chat session with the assistant directly in your terminal. Pre-warms model memory on launch and cleans it up upon exit.

**Usage:**
```bash
ai-institute test <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--session` | `-s` | `string` | `None` | Session identifier to resume prior conversational context. |

---

### `pentest`
Executes an automated adversarial security benchmark using 50+ college-specific attack vectors across prompt leaks, jailbreaks, regional slang, and math solvers.

**Usage:**
```bash
ai-institute pentest <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target assistant role name. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--attempts` | `-a` | `int` | `50` | Number of adversarial attack attempts to execute. |
| `--report` | — | `bool` | `True` | Displays detailed breakdown of bypasses and layer block rates. |
| `--patterns` | `-p` | `path` | `None` | Custom JSON attack dataset path (default: `tests/jailbreak_patterns.json`). |

---

## 6. Deployment, Public Tunneling & OS Autostart

### `deploy`
Starts the Uvicorn REST API server for a role with optional public HTTPS tunneling and systemd autostart on system boot.

**Usage:**
```bash
ai-institute deploy <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Role to deploy (e.g. `librarian`). |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--port` | `-p` | `int` | `8080` | Port for the FastAPI server to bind to. |
| `--host` | `-h` | `string` | `127.0.0.1` | Network interface to listen on. Use `0.0.0.0` for remote access. |
| `--background` /<br>`--foreground` | `-d` /<br>`-f` | `bool` | `True` | Runs in background as a daemon by default. Pass `-f` to run in foreground. |
| `--tunnel` | — | `bool` | `False` | Automatically establishes a secure reverse SSH tunnel over outbound HTTPS port 443, giving you a live public HTTPS address for your team. |
| `--autostart` | — | `bool` | `False` | Generates, enables, and starts a systemd service (`ai-institute-<role>.service`) for 24/7 reboot persistence. (Requires root/sudo). |

**Examples:**
```bash
# 1. Simple local background server
ai-institute deploy librarian --port 8080

# 2. Local server + live public link for your team
ai-institute deploy librarian --port 8080 --host 0.0.0.0 --tunnel

# 3. Full production: 24/7 autostart on system reboot + public tunnel
sudo ai-institute deploy librarian --port 8080 --host 0.0.0.0 --tunnel --autostart
```

---

### `status`
Displays a status table of all active server deployments, their local bind addresses, live public tunnel URLs, PIDs/systemd services, and operational health.

**Usage:**
```bash
ai-institute status
```

---

### `stop`
Gracefully shuts down a running role deployment, cleans up associated background tunnel processes, unloads model weights from Ollama memory (`keep_alive: 0`), and flushes GPU VRAM.

**Usage:**
```bash
ai-institute stop <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Name of the deployed role to stop. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--autostart` | — | `bool` | `False` | Stops and disables the systemd service unit if previously installed with `--autostart`. |

---

## 7. Packaging, Backup & Portability

### `backup`
Packages the role's configuration, tool definitions, knowledge documents, training data, and pre-computed vector store into a portable `.tar.gz` archive.

**Usage:**
```bash
ai-institute backup <role> --output <archive.tar.gz>
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Target role name to archive. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--output` | `-o` | `path` | *Required* | Destination file path for the compressed archive. |

---

### `restore`
Restores a complete role directory from a `.tar.gz` backup archive into the `roles/` directory.

**Usage:**
```bash
ai-institute restore --input <archive.tar.gz> [OPTIONS]
```

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `-i` | `path` | *Required* | Path to the source `.tar.gz` backup file. |
| `--force` | `-f` | `bool` | `False` | Overwrites existing role directory if it already exists. |

---

### `package`
Bundles the runtime wheel, role data archive copy, Modelfile, and an automated `install.sh` script into a standalone deployment zip file (`dist/<role>-deploy.zip`).

**Usage:**
```bash
ai-institute package <role> [OPTIONS]
```

**Arguments:**
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `role` | `string` | **Yes** | Role name to package for production deployment. |

**Options:**
| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--output` | `-o` | `path` | `None` | Custom path for the output `.zip` file. Defaults to `dist/<role>-deploy.zip`. |
| `--model` | `-m` | `string` | `None` | Target model tier tag override (`1.5b` or `3b`). |

---

*Document generated automatically for [`ai-trains-ai`](file:///root/ai-trains-ai) CLI architecture.*
