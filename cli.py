from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer
import yaml

from data_processing.ingest import ingest_file
from data_processing.chunker import chunk_documents
from inference.rag import VectorStore
from training.dataset import (
    CATEGORY_DESCRIPTIONS,
    deduplicate_examples,
    get_dataset_stats,
    load_all_training_data,
    save_training_batch,
    validate_dataset,
)
from training.trainer import get_training_status, train_role
from training.export import export_role_to_ollama
from inference.chat import ChatOrchestrator
from inference.history import clear_history, get_history

app = typer.Typer(help="AI-Institute: Role-locked college AI assistants")
console = Console()

DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "default_student_model": "qwen2.5:1.5b",
        "default_teacher_mode": "smart",
        "roles_dir": "roles",
    }


def get_roles_dir() -> Path:
    config = load_config()
    roles_dir = Path(config.get("roles_dir", "roles"))
    roles_dir.mkdir(parents=True, exist_ok=True)
    return roles_dir


def validate_role_name(name: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


def get_role_path_or_exit(role: str) -> Path:
    roles_dir = get_roles_dir()
    role_path = roles_dir / role
    if not role_path.exists() or not (role_path / "role.yaml").exists():
        console.print(f"[red]Error:[/red] Role '{role}' not found.")
        raise typer.Exit(code=1)
    return role_path


def run_questionnaire(role_path: Path, force_prompt: bool = False) -> dict:
    cfg_path = role_path / "role.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "questionnaire" in cfg and not force_prompt:
        return cfg["questionnaire"]

    console.print(f"\n[bold cyan]Interactive Questionnaire for Role: {cfg.get('name', role_path.name)}[/bold cyan]")
    console.print("Establishing role scope, rules, and behavioral boundaries.\n")

    is_interactive = sys.stdin.isatty()

    def ask(prompt_text: str, default_val: str) -> str:
        if is_interactive:
            try:
                val = typer.prompt(prompt_text, default=default_val).strip()
                return val if val else default_val
            except (KeyboardInterrupt, EOFError):
                return default_val
        return default_val

    role_desc = cfg.get("description", "Assistant")
    q_answers = {
        "purpose": ask("What exactly does this role do?", role_desc),
        "in_scope": ask(
            "What topics are IN scope? (comma separated)",
            "Book availability, catalog search, borrowing limits, library timings",
        ),
        "out_of_scope": ask(
            "What topics are OUT of scope?",
            "General chit-chat, exam grades, coding solutions, hostel fees",
        ),
        "refusal_message": ask(
            "How should it refuse off-topic questions? (give example wording)",
            f"I am the {cfg.get('name')} and can only assist with  library related queries.",
        ),
        "ambiguity_guidelines": ask(
            "How should it handle ambiguous queries?",
            "List all possible matches and ask the user to clarify which one they need.",
        ),
        "language_rules": ask(
            "How should it handle non-English input?",
            "Politely state that responses are only provided in English and ask for English.",
        ),
    }

    cfg["questionnaire"] = q_answers
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    console.print("[green]Questionnaire saved to role.yaml.[/green]\n")
    return q_answers


@app.command("create-role")
def create_role(
    name: str = typer.Argument(..., help="Unique name for the role"),
    description: str = typer.Option(..., "--description", "-d", help="Role purpose and scope"),
    student_model: Optional[str] = typer.Option(
        None, "--student-model", "-m", help="Target student model (e.g. qwen2.5:1.5b)"
    ),
    teacher_mode: Optional[str] = typer.Option(
        None, "--teacher-mode", "-t", help="Teacher mode: 'smart' or 'guided'"
    ),
):
    """Create a new assistant role."""
    config = load_config()
    model = student_model or config.get("default_student_model", "qwen2.5:1.5b")
    mode = (teacher_mode or config.get("default_teacher_mode", "smart")).lower()

    if not validate_role_name(name):
        console.print(
            f"[red]Error:[/red] Invalid role name '{name}'. Use letters, numbers, hyphens, and underscores only."
        )
        raise typer.Exit(code=1)

    if mode not in ["smart", "guided"]:
        console.print("[red]Error:[/red] Teacher mode must be either 'smart' or 'guided'.")
        raise typer.Exit(code=1)

    roles_dir = get_roles_dir()
    role_path = roles_dir / name

    if role_path.exists():
        console.print(f"[red]Error:[/red] Role '{name}' already exists.")
        raise typer.Exit(code=1)

    (role_path / "data").mkdir(parents=True, exist_ok=True)
    (role_path / "training_data").mkdir(parents=True, exist_ok=True)

    role_data = {
        "name": name,
        "description": description.strip(),
        "student_model": model,
        "teacher_mode": mode,
        "created_at": datetime.now().isoformat(),
    }

    config_file = role_path / "role.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(role_data, f, sort_keys=False)

    console.print(f"[green]Role '{name}' created successfully.[/green]")
    console.print(f"Location: {role_path}")
    console.print(f"Student model: {model}")
    console.print(f"Teacher mode: {mode}")


@app.command("list-roles")
def list_roles():
    """List all available roles."""
    roles_dir = get_roles_dir()
    role_dirs = [d for d in roles_dir.iterdir() if d.is_dir()]

    if not role_dirs:
        console.print("No roles found.")
        console.print("Create your first role using: [cyan]python cli.py create-role <name> --description \"...\"[/cyan]")
        return

    table = Table(title="AI-Institute Roles")
    table.add_column("Role Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Student Model", style="green")
    table.add_column("Mode", style="magenta")
    table.add_column("Data Files", justify="right")

    for r_dir in sorted(role_dirs):
        cfg_path = r_dir / "role.yaml"
        if not cfg_path.exists():
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        data_dir = r_dir / "data"
        file_count = len([f for f in data_dir.iterdir() if f.is_file()]) if data_dir.exists() else 0

        table.add_row(
            cfg.get("name", r_dir.name),
            cfg.get("description", "No description"),
            cfg.get("student_model", "Unknown"),
            cfg.get("teacher_mode", "Unknown"),
            str(file_count),
        )

    console.print(table)


@app.command("delete-role")
def delete_role(
    name: str = typer.Argument(..., help="Role name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Delete without confirmation"),
):
    """Delete a role and all its associated data."""
    roles_dir = get_roles_dir()
    role_path = roles_dir / name

    if not role_path.exists():
        console.print(f"[red]Error:[/red] Role '{name}' not found.")
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete role '{name}' and all its files?")
        if not confirm:
            console.print("Cancelled.")
            return

    shutil.rmtree(role_path)
    console.print(f"[green]Role '{name}' has been deleted.[/green]")


@app.command("info")
def role_info(name: str = typer.Argument(..., help="Role name to inspect")):
    """Display detailed configuration for a role."""
    role_path = get_role_path_or_exit(name)
    cfg_path = role_path / "role.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_dir = role_path / "data"
    training_dir = role_path / "training_data"

    data_files = [f.name for f in data_dir.iterdir() if f.is_file()] if data_dir.exists() else []
    training_files = [f.name for f in training_dir.iterdir() if f.is_file()] if training_dir.exists() else []

    info_text = (
        f"[bold cyan]Role:[/bold cyan] {cfg.get('name', name)}\n"
        f"[bold]Description:[/bold] {cfg.get('description', 'N/A')}\n"
        f"[bold]Student Model:[/bold] {cfg.get('student_model', 'N/A')}\n"
        f"[bold]Teacher Mode:[/bold] {cfg.get('teacher_mode', 'N/A')}\n"
        f"[bold]Created:[/bold] {cfg.get('created_at', 'N/A')}\n\n"
        f"[bold]Data Files ({len(data_files)}):[/bold] {', '.join(data_files) if data_files else 'None'}\n"
        f"[bold]Training Data Files ({len(training_files)}):[/bold] {', '.join(training_files) if training_files else 'None'}\n"
        f"[bold]Directory:[/bold] {role_path}"
    )

    console.print(Panel(info_text, title=f"Role Info: {name}", expand=False))


@app.command("add-data")
def add_data(
    role: str = typer.Argument(..., help="Role to add data to"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to single data file"),
    dir: Optional[Path] = typer.Option(None, "--dir", "-d", help="Path to directory of data files"),
):
    """Add data file(s) to a role and index into vector store."""
    if not file and not dir:
        console.print("[red]Error:[/red] Please provide either --file or --dir.")
        raise typer.Exit(code=1)

    role_path = get_role_path_or_exit(role)
    data_dir = role_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    files_to_process: list[Path] = []
    if file:
        if not file.exists() or not file.is_file():
            console.print(f"[red]Error:[/red] File '{file}' does not exist.")
            raise typer.Exit(code=1)
        files_to_process.append(file)

    if dir:
        if not dir.exists() or not dir.is_dir():
            console.print(f"[red]Error:[/red] Directory '{dir}' does not exist.")
            raise typer.Exit(code=1)
        files_to_process.extend([p for p in dir.iterdir() if p.is_file()])

    if not files_to_process:
        console.print("[yellow]Warning:[/yellow] No files found to process.")
        return

    vectorstore = VectorStore(role_path)
    total_indexed = 0

    for source_file in files_to_process:
        target_path = data_dir / source_file.name
        shutil.copy2(source_file, target_path)

        docs = ingest_file(target_path)
        chunks = chunk_documents(docs)

        vectorstore.delete_by_source(target_path.name)
        indexed = vectorstore.add_documents(chunks)
        total_indexed += indexed
        console.print(f"[green]Added {target_path.name} to {role}. Indexed {indexed} chunks.[/green]")

    console.print(f"[bold green]Complete. Total new chunks indexed: {total_indexed}[/bold green]")


@app.command("list-data")
def list_data(role: str = typer.Argument(..., help="Role name to list data for")):
    """List data files and indexed chunk counts for a role."""
    role_path = get_role_path_or_exit(role)
    data_dir = role_path / "data"

    data_files = sorted([f for f in data_dir.iterdir() if f.is_file()]) if data_dir.exists() else []
    if not data_files:
        console.print(f"No data files found for role '{role}'.")
        console.print(f"Add files using: [cyan]python cli.py add-data {role} --file <path>[/cyan]")
        return

    vectorstore = VectorStore(role_path)
    counts = vectorstore.get_source_counts()

    table = Table(title=f"Data Files for Role: {role}")
    table.add_column("Filename", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Indexed Chunks", justify="right", style="green")
    table.add_column("Modified", style="dim")

    for f in data_files:
        size_kb = f.stat().st_size / 1024.0
        size_str = f"{size_kb:.1f} KB" if size_kb >= 1.0 else f"{f.stat().st_size} B"
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        chunk_count = counts.get(f.name, 0)
        table.add_row(f.name, size_str, str(chunk_count), mtime)

    console.print(table)


@app.command("remove-data")
def remove_data(
    role: str = typer.Argument(..., help="Role name"),
    file: str = typer.Option(..., "--file", "-f", help="Filename inside role's data directory"),
):
    """Remove a data file from a role and delete its vectors."""
    role_path = get_role_path_or_exit(role)
    target_file = role_path / "data" / file

    if not target_file.exists():
        console.print(f"[red]Error:[/red] File '{file}' not found in role '{role}'.")
        raise typer.Exit(code=1)

    target_file.unlink()
    vectorstore = VectorStore(role_path)
    deleted_count = vectorstore.delete_by_source(file)

    console.print(f"[green]Removed {file} from {role}. Deleted {deleted_count} indexed chunks.[/green]")


@app.command("reindex")
def reindex_data(role: str = typer.Argument(..., help="Role name to re-index")):
    """Rebuild the vector store from all data files."""
    role_path = get_role_path_or_exit(role)
    data_dir = role_path / "data"
    data_files = [f for f in data_dir.iterdir() if f.is_file()] if data_dir.exists() else []

    vectorstore = VectorStore(role_path)
    vectorstore.clear()

    if not data_files:
        console.print(f"[yellow]No data files found for role '{role}'. Vector store cleared.[/yellow]")
        return

    total_chunks = 0
    for f in data_files:
        docs = ingest_file(f)
        chunks = chunk_documents(docs)
        indexed = vectorstore.add_documents(chunks)
        total_chunks += indexed

    console.print(
        f"[green]Re-indexed {role}. Total chunks: {total_chunks} across {len(data_files)} files.[/green]"
    )


@app.command("search")
def search_role(
    role: str = typer.Argument(..., help="Role name to search"),
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of results to return"),
):
    """Search vector store directly to verify RAG retrieval."""
    role_path = get_role_path_or_exit(role)
    vectorstore = VectorStore(role_path)

    results = vectorstore.search(query, top_k=top_k)
    if not results:
        console.print(f"[yellow]No results found for '{query}'.[/yellow]")
        return

    console.print(f"\n[bold]Top {len(results)} results for query:[/bold] \"{query}\"\n")
    for idx, res in enumerate(results, start=1):
        source = res.get("source", "unknown")
        sim = res.get("similarity", 0.0)
        sim_pct = f"{sim * 100:.1f}%"
        text = res.get("text", "")

        header = f"Result #{idx} | Source: {source} | Similarity: {sim_pct}"
        console.print(Panel(text, title=header, expand=False, border_style="cyan"))


@app.command("import-training")
def import_training(
    role: str = typer.Argument(..., help="Role name to import training data for"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to a training JSON file"),
    dir: Optional[Path] = typer.Option(None, "--dir", "-d", help="Directory of training JSON files"),
):
    """Validate, deduplicate, and save training data examples for a role."""
    if not file and not dir:
        console.print("[red]Error:[/red] Please provide either --file or --dir.")
        raise typer.Exit(code=1)

    role_path = get_role_path_or_exit(role)
    files_to_read: list[Path] = []

    if file:
        if not file.exists() or not file.is_file():
            console.print(f"[red]Error:[/red] File '{file}' does not exist.")
            raise typer.Exit(code=1)
        files_to_read.append(file)

    if dir:
        if not dir.exists() or not dir.is_dir():
            console.print(f"[red]Error:[/red] Directory '{dir}' does not exist.")
            raise typer.Exit(code=1)
        files_to_read.extend(sorted(dir.glob("*.json")))

    if not files_to_read:
        console.print("[yellow]Warning:[/yellow] No JSON files found to import.")
        return

    existing_data = load_all_training_data(role_path)
    total_imported = 0
    total_skipped = 0
    batch_counts: dict[str, int] = {}

    for json_file in files_to_read:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
        except Exception as e:
            console.print(f"[red]Error:[/red] Could not parse '{json_file.name}' as JSON: {e}")
            raise typer.Exit(code=1)

        valid_items, errors = validate_dataset(raw_json)
        if errors:
            console.print(f"[red]Validation failed for {json_file.name}:[/red]")
            for err in errors[:5]:
                console.print(f"  - {err}")
            raise typer.Exit(code=1)

        unique_items, dup_count = deduplicate_examples(valid_items, existing_data)
        total_skipped += dup_count

        if unique_items:
            saved_path = save_training_batch(role_path, unique_items, custom_name=json_file.stem)
            existing_data.extend(unique_items)
            total_imported += len(unique_items)
            for item in unique_items:
                cat = item["category"]
                batch_counts[cat] = batch_counts.get(cat, 0) + 1

    if total_imported > 0:
        counts_summary = ", ".join([f"{k}: {v}" for k, v in sorted(batch_counts.items())])
        console.print(
            f"[green]Imported {total_imported} examples ({counts_summary})[/green]"
        )
    if total_skipped > 0:
        console.print(f"[yellow]Skipped {total_skipped} duplicates.[/yellow]")
    if total_imported == 0 and total_skipped > 0:
        console.print("[yellow]0 new examples imported.[/yellow]")


@app.command("review-training")
def review_training(role: str = typer.Argument(..., help="Role name to review training data for")):
    """Show training data counts, category distribution, and sample previews."""
    role_path = get_role_path_or_exit(role)
    data = load_all_training_data(role_path)

    if not data:
        console.print(f"No training data found for role '{role}'.")
        console.print(f"Import training data using: [cyan]python cli.py import-training {role} --file <path>[/cyan]")
        return

    stats = get_dataset_stats(data)

    table = Table(title=f"Training Data Distribution for: {role}")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Examples", justify="right", style="green")

    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        count = stats["categories"].get(cat, 0)
        table.add_row(cat, desc, str(count))

    console.print(table)
    console.print(f"[bold]Total training examples:[/bold] {stats['total']}\n")

    # Sample preview panel
    console.print("[bold]Samples from each category:[/bold]")
    shown_categories = set()
    for item in data:
        cat = item["category"]
        if cat not in shown_categories:
            shown_categories.add(cat)
            sample_text = (
                f"[bold cyan]Category:[/bold cyan] {cat}\n"
                f"[bold]User:[/bold] {item['user']}\n"
                f"[bold]Assistant:[/bold] {item['assistant']}"
            )
            console.print(Panel(sample_text, expand=False, border_style="dim"))


@app.command("generate-prompts")
def generate_prompts(
    role: str = typer.Argument(..., help="Role name to generate prompts for"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run questionnaire even if already answered"),
):
    """Run interactive questionnaire and create 7 guided-mode prompt templates."""
    role_path = get_role_path_or_exit(role)
    cfg_path = role_path / "role.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    questionnaire = run_questionnaire(role_path, force_prompt=force)

    # Summarize role data files
    data_dir = role_path / "data"
    data_files = [f.name for f in data_dir.iterdir() if f.is_file()] if data_dir.exists() else []
    data_summary = f"Files available: {', '.join(data_files)}" if data_files else "No static data files added yet."

    template_dir = Path("teachers/prompts")
    if not template_dir.exists():
        console.print("[red]Error:[/red] Template directory 'teachers/prompts' not found.")
        raise typer.Exit(code=1)

    prompts_output_dir = role_path / "prompts"
    prompts_output_dir.mkdir(parents=True, exist_ok=True)

    replacements = {
        "{{role_name}}": cfg.get("name", role),
        "{{role_description}}": cfg.get("description", "Assistant"),
        "{{in_scope}}": questionnaire.get("in_scope", ""),
        "{{out_of_scope}}": questionnaire.get("out_of_scope", ""),
        "{{refusal_message}}": questionnaire.get("refusal_message", ""),
        "{{ambiguity_guidelines}}": questionnaire.get("ambiguity_guidelines", ""),
        "{{language_rules}}": questionnaire.get("language_rules", ""),
        "{{data_summary}}": data_summary,
    }

    generated_files = []
    for template_file in sorted(template_dir.glob("*.md")):
        with open(template_file, "r", encoding="utf-8") as f:
            content = f.read()

        for placeholder, val in replacements.items():
            content = content.replace(placeholder, val)

        # Explicit reminder baked into every generated prompt
        content += "\n\n> Note: Do not invent or assume any information not provided above. Only use the data and rules given."

        out_path = prompts_output_dir / template_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        generated_files.append(out_path.name)

    console.print(f"[green]Generated {len(generated_files)} prompt templates in {prompts_output_dir}:[/green]")
    for fname in generated_files:
        console.print(f"  - {fname}")


@app.command("generate-playbook")
def generate_playbook(
    role: str = typer.Argument(..., help="Role name to generate playbook for"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run questionnaire even if already answered"),
):
    """Run questionnaire and create smart-mode playbook.md for AI editors."""
    role_path = get_role_path_or_exit(role)
    cfg_path = role_path / "role.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    questionnaire = run_questionnaire(role_path, force_prompt=force)

    data_dir = role_path / "data"
    data_files = [f.name for f in data_dir.iterdir() if f.is_file()] if data_dir.exists() else []
    data_summary = f"Files: {', '.join(data_files)}" if data_files else "None yet."

    tools_file = role_path / "tools.yaml"
    tools_summary = "Tools defined: Yes (see tools.yaml)" if tools_file.exists() else "Tools defined: None yet"

    playbook_content = f"""# Playbook: {cfg.get('name', role)} (Teacher AI Automation)

You are the Teacher AI orchestrating the dataset generation, training, and verification for role '{cfg.get('name', role)}'.

## Core Instruction (Zero Assumptions)
You must ask the user for every detail. Never assume, guess, or fill in blanks. If something is unclear, ask. Only proceed after the user explicitly confirms.

## Role Specifications
- Role name: {cfg.get('name', role)}
- Description: {cfg.get('description', 'N/A')}
- Target student model: {cfg.get('student_model', 'qwen2.5:1.5b')}
- Teacher mode: {cfg.get('teacher_mode', 'smart')}
- Available data: {data_summary}
- Tool calling: {tools_summary}

## Scope Boundaries (from Questionnaire)
- Purpose: {questionnaire.get('purpose', '')}
- In scope: {questionnaire.get('in_scope', '')}
- Out of scope: {questionnaire.get('out_of_scope', '')}
- Preferred refusal style: {questionnaire.get('refusal_message', '')}
- Ambiguity policy: {questionnaire.get('ambiguity_guidelines', '')}
- Language policy: {questionnaire.get('language_rules', '')}

## Dataset Categories to Generate
Generate a comprehensive JSON training dataset covering these 7 categories:
1. qa (~200 examples): on-topic questions answered directly from data
2. refusal (~100 examples): polite refusals for out-of-scope queries
3. jailbreak (~100 examples): defense against prompt injections and persona escapes
4. clarification (~50 examples): questions with multiple catalog matches
5. capability_limit (~30 examples): honest limits on calculations/comparisons
6. language (~20 examples): requests in non-English redirected politely
7. conversation (~50 examples): multi-turn continuity and pronoun follow-ups

## Pipeline Commands
Run these commands in sequence to complete the workflow:
```bash
# 1. Ingest data (if not done)
ai-institute add-data {role} --file <path>

# 2. Import generated training batch
ai-institute import-training {role} --file training_batch.json

# 3. Review coverage
ai-institute review-training {role}

# 4. Train student model
ai-institute train {role}

# 5. Pentest against jailbreaks
ai-institute pentest {role}

# 6. Deploy assistant
ai-institute deploy {role}
```

## Self-Correction Loop
If pentest pass rate is below 95%:
1. Inspect which attacks passed through.
2. Generate targeted refusal/jailbreak examples countering those exact attack patterns.
3. Import the new batch and retrain until target defense rate is met.
"""

    out_file = role_path / "playbook.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(playbook_content)

    console.print(f"[green]Created playbook at {out_file}[/green]")


@app.command("train")
def train_model(
    role: str = typer.Argument(..., help="Role name to train"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size per device"),
    grad_accum: int = typer.Option(4, "--grad-accum", "-ga", help="Gradient accumulation steps"),
    learning_rate: float = typer.Option(2e-4, "--learning-rate", "-lr", help="Learning rate"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Maximum training steps for testing"),
):
    """Fine-tune the student model using role training data."""
    role_path = get_role_path_or_exit(role)
    try:
        status = train_role(
            role_path=role_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            gradient_accumulation_steps=grad_accum,
        )
        console.print(
            f"[bold green]Training complete.[/bold green] Loss: {status['training_loss']}. Model saved to roles/{role}/model/"
        )
    except Exception as e:
        console.print(f"[red]Training error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("train-status")
def train_status(role: str = typer.Argument(..., help="Role name to check")):
    """Check training status and latest loss metrics."""
    role_path = get_role_path_or_exit(role)
    status = get_training_status(role_path)

    if not status:
        console.print(f"No training run found for role '{role}'.")
        console.print(f"Run training with: [cyan]python cli.py train {role}[/cyan]")
        return

    table = Table(title=f"Training Status: {role}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Status", f"[green]{status.get('status', 'unknown')}[/green]")
    table.add_row("Student Model", str(status.get("student_model", "unknown")))
    table.add_row("Base Model", str(status.get("base_model", "unknown")))
    table.add_row("Device", str(status.get("device", "unknown")).upper())
    table.add_row("Examples Trained", str(status.get("examples_count", 0)))
    table.add_row("Epochs", str(status.get("epochs", 0)))
    table.add_row("Final Loss", str(status.get("training_loss", "N/A")))
    table.add_row("Duration", f"{status.get('duration_seconds', 0)}s")
    table.add_row("Completed At", str(status.get("completed_at", "N/A")))

    console.print(table)


@app.command("export")
def export_model(
    role: str = typer.Argument(..., help="Role name to export"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Student model override (e.g. 1.5b or 3b)"),
):
    """Export role model to Ollama Modelfile and register it."""
    role_path = get_role_path_or_exit(role)
    try:
        info = export_role_to_ollama(role_path, model_override=model)
        console.print(f"[bold green]Export complete.[/bold green] Registered '{info['ollama_model']}' ({info['base_model']}) in Ollama.")
    except Exception as e:
        console.print(f"[red]Export error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("query")
def query_role(
    role: str = typer.Argument(..., help="Role name to query"),
    message: str = typer.Argument(..., help="Message to send"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID for chat history"),
):
    """Send a single query to a role assistant."""
    role_path = get_role_path_or_exit(role)
    try:
        orchestrator = ChatOrchestrator(role_path, session_id=session)
        console.print(f"[bold green]{role}:[/bold green] ", end="")
        meta = {}
        for chunk in orchestrator.ask_stream(message):
            if chunk["type"] == "token":
                console.print(chunk["content"], end="")
            elif chunk["type"] == "meta":
                meta = chunk
        console.print()
        info = []
        if meta.get("sources"):
            info.append(f"Sources: {', '.join(meta['sources'])}")
        if "elapsed" in meta:
            info.append(f"Time: {meta['elapsed']}s")
        if info:
            console.print(f"[dim]{' | '.join(info)}[/dim]")
    except Exception as e:
        console.print(f"[red]Query error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("test")
def test_role(
    role: str = typer.Argument(..., help="Role name to test"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID to resume"),
):
    """Interactive chat session with a role assistant."""
    role_path = get_role_path_or_exit(role)
    with console.status(f"[bold green]Pre-warming '{role}' (embedding and Ollama models)...[/bold green]"):
        orchestrator = ChatOrchestrator(role_path, session_id=session)
        orchestrator.warmup()

    console.print(f"[bold green]Interactive session for '{role}' (session: {orchestrator.session_id})[/bold green]")
    console.print("Commands: :exit to quit, :clear to reset session, :history to show previous turns.\n")

    try:
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\nSession ended.")
                break

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in (":exit", ":quit", "exit", "quit"):
                console.print("Session ended.")
                break

            if cmd == ":clear":
                clear_history(orchestrator.db_path, orchestrator.session_id)
                console.print("[dim]Session history cleared.[/dim]\n")
                continue

            if cmd == ":history":
                history = get_history(orchestrator.db_path, orchestrator.session_id)
                if not history:
                    console.print("[dim]No messages in this session yet.[/dim]\n")
                else:
                    console.print(f"[dim]--- History ({len(history)} turns) ---[/dim]")
                    for turn in history:
                        who = "[cyan]You[/cyan]" if turn["role"] == "user" else f"[green]{role}[/green]"
                        console.print(f"{who}: {turn['content']}")
                    console.print("[dim]------------------------[/dim]\n")
                continue

            try:
                console.print(f"[bold green]{role}:[/bold green] ", end="")
                meta = {}
                for chunk in orchestrator.ask_stream(user_input):
                    if chunk["type"] == "token":
                        console.print(chunk["content"], end="")
                    elif chunk["type"] == "meta":
                        meta = chunk
                console.print()
                info = []
                if meta.get("sources"):
                    info.append(f"Sources: {', '.join(meta['sources'])}")
                if "elapsed" in meta:
                    info.append(f"Time: {meta['elapsed']}s")
                if info:
                    console.print(f"[dim]{' | '.join(info)}[/dim]")
                console.print()
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}\n")
    finally:
        orchestrator.unload()


DEPLOYMENTS_FILE = Path(".deployments.json")


def _load_deployments() -> dict[str, dict]:
    if DEPLOYMENTS_FILE.exists():
        try:
            with open(DEPLOYMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_deployments(data: dict[str, dict]):
    with open(DEPLOYMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _is_ollama_running(url: str = "http://localhost:11434") -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ensure_ollama_running(url: str = "http://localhost:11434") -> bool:
    if _is_ollama_running(url):
        return True
    console.print("[dim]Ollama is not running. Starting 'ollama serve' in background...[/dim]")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        console.print("[red]Error:[/red] 'ollama' binary not found. Please install or start Ollama.")
        return False

    for _ in range(20):
        time.sleep(0.5)
        if _is_ollama_running(url):
            console.print("[dim]Ollama is ready.[/dim]")
            return True
    console.print("[yellow]Warning: Timed out waiting for Ollama service.[/yellow]")
    return False


def _unload_ollama_model(model_name: str, url: str = "http://localhost:11434"):
    import urllib.request
    models = {model_name}
    if ":" not in model_name:
        models.update({f"{model_name}:3b", f"{model_name}:1.5b"})

    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/ps")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("models", []):
                name = item.get("name", "")
                if model_name in name:
                    models.add(name)
    except Exception:
        pass

    for m in models:
        try:
            payload = json.dumps({"model": m, "keep_alive": 0}).encode("utf-8")
            req = urllib.request.Request(
                f"{url.rstrip('/')}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            pass


@app.command("deploy")
def deploy_command(
    role: str = typer.Argument(..., help="Role to deploy"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to run the API server on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host interface to bind to"),
    background: bool = typer.Option(True, "--background/--foreground", "-d/-f", help="Run in background as a daemon (default: True)"),
    tunnel: bool = typer.Option(False, "--tunnel", help="Expose deployment via public HTTPS reverse tunnel over port 443"),
    autostart: bool = typer.Option(False, "--autostart", help="Install systemd service for 24/7 autostart on reboot"),
):
    """Start the REST API server for a role with optional public tunnel and reboot autostart."""
    import deps

    if not deps.require_group("api"):
        console.print("[red]Aborted:[/red] API dependencies are required to deploy.")
        raise typer.Exit(1)

    roles_dir = get_roles_dir()
    role_path = roles_dir / role
    if not role_path.exists() or not (role_path / "role.yaml").exists():
        console.print(f"[red]Error:[/red] Role '{role}' not found at {role_path}")
        raise typer.Exit(1)

    _ensure_ollama_running()

    deployments = _load_deployments()

    # 1. Handle systemd autostart
    if autostart:
        if not sys.platform.startswith("linux"):
            console.print("[red]Error:[/red] Autostart via systemd is only supported on Linux.")
            raise typer.Exit(1)

        systemctl_bin = shutil.which("systemctl")
        if not systemctl_bin:
            console.print("[red]Error:[/red] 'systemctl' not found. Systemd is required for autostart.")
            raise typer.Exit(1)

        is_root = (os.geteuid() == 0) if hasattr(os, "geteuid") else True
        if not is_root:
            console.print("[red]Error:[/red] Installing systemd units requires root privileges. Please run with sudo:")
            console.print(f"  sudo {sys.executable} cli.py deploy {role} --port {port} --host {host} {'--tunnel ' if tunnel else ''}--autostart")
            raise typer.Exit(1)

        project_root = Path(__file__).resolve().parent
        user_name = os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"
        service_name = f"ai-institute-{role}.service"
        unit_path = Path("/etc/systemd/system") / service_name

        unit_content = f"""[Unit]
Description=AI-Institute {role} REST API Service
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User={user_name}
WorkingDirectory={project_root}
Environment="AI_INSTITUTE_ROLES_DIR={roles_dir.resolve()}"
Environment="AI_INSTITUTE_PREWARM_ROLE={role}"
ExecStart={sys.executable} -m uvicorn api.server:app --host {host} --port {port}
Restart=always
RestartSec=5
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
"""
        tunnel_service_name = f"ai-institute-{role}-tunnel.service"
        tunnel_unit_path = Path("/etc/systemd/system") / tunnel_service_name
        tunnel_unit_content = f"""[Unit]
Description=Public HTTPS Tunnel for AI-Institute {role}
After=network.target {service_name}
Wants={service_name}

[Service]
Type=simple
User={user_name}
WorkingDirectory={project_root}
ExecStart={sys.executable} {project_root / "cli.py"} tunnel --port {port} --host 127.0.0.1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
        console.print(f"Installing systemd unit [cyan]{unit_path}[/cyan]...")
        unit_path.write_text(unit_content, encoding="utf-8")
        if tunnel:
            console.print(f"Installing tunnel systemd unit [cyan]{tunnel_unit_path}[/cyan]...")
            tunnel_unit_path.write_text(tunnel_unit_content, encoding="utf-8")

        subprocess.run([systemctl_bin, "daemon-reload"], check=True)
        subprocess.run([systemctl_bin, "enable", "--now", service_name], check=True)
        if tunnel:
            subprocess.run([systemctl_bin, "enable", "--now", tunnel_service_name], check=True)

        console.print(f"[bold green]Autostart enabled and service started for {role}![/bold green]")

        import urllib.request
        probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        for _ in range(30):
            time.sleep(1.0)
            try:
                with urllib.request.urlopen(f"http://{probe_host}:{port}/api/v1/health", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                continue

        tunnel_url = None
        if tunnel:
            url_file = Path("PUBLIC_URL.txt")
            for _ in range(10):
                time.sleep(1.0)
                if url_file.exists():
                    u = url_file.read_text(encoding="utf-8").strip()
                    if u.startswith("http"):
                        tunnel_url = u
                        break

        deployments[role] = {
            "role": role,
            "host": host,
            "port": port,
            "autostart": True,
            "service": service_name,
            "tunnel_service": tunnel_service_name if tunnel else None,
            "started_at": datetime.now().isoformat(),
            "tunnel_url": tunnel_url,
        }
        _save_deployments(deployments)

        panel_lines = [
            f"[bold green]Deployed {role} (24/7 Autostart Enabled)[/bold green]\n",
            f"Service:      [cyan]{service_name}[/cyan] (active / enabled on boot)",
            f"Local URL:    [cyan]http://{probe_host}:{port}[/cyan]",
        ]
        if host in ("0.0.0.0", "::"):
            panel_lines.append(f"Remote URL:   [cyan]http://<server-ip>:{port}[/cyan]")
        if tunnel_url:
            panel_lines.append(f"Public URL:   [magenta]{tunnel_url}[/magenta]")
            panel_lines.append(f"Playground:   [magenta]{tunnel_url}/[/magenta]")
            panel_lines.append(f"API Docs:     [magenta]{tunnel_url}/docs[/magenta]")
        else:
            panel_lines.append(f"API Docs:     [cyan]http://{probe_host}:{port}/docs[/cyan]")

        console.print(Panel("\n".join(panel_lines), title="Systemd Autostart Active", border_style="green"))
        return

    # 2. Check running deployment
    for name, info in list(deployments.items()):
        pid = info.get("pid")
        if pid and _is_pid_alive(pid):
            if name == role:
                console.print(f"[yellow]Role '{role}' is already running[/yellow] on port {info.get('port')} (PID {pid}).")
                raise typer.Exit(0)
            if info.get("port") == port:
                console.print(f"[red]Error:[/red] Port {port} is already used by deployment '{name}' (PID {pid}).")
                raise typer.Exit(1)
        elif pid:
            del deployments[name]

    env = os.environ.copy()
    env["AI_INSTITUTE_ROLES_DIR"] = str(roles_dir.resolve())
    env["AI_INSTITUTE_PREWARM_ROLE"] = role

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    # Foreground Mode
    if not background:
        if tunnel:
            console.print(f"Starting reverse tunnel in background for port {port}...")
            tunnel_cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "tunnel",
                "--port", str(port),
                "--background",
            ]
            subprocess.run(tunnel_cmd)

        console.print(f"Starting server in foreground on http://{host}:{port}...")
        subprocess.run(cmd, env=env)
        return

    # Background Daemon Mode
    log_file = Path(f".server_{role}_{port}.log")
    log_fp = open(log_file, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    started = False
    import urllib.request
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    for _ in range(120):
        time.sleep(1.0)
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://{probe_host}:{port}/api/v1/health", timeout=2) as resp:
                if resp.status == 200:
                    started = True
                    break
        except Exception:
            continue

    if not started or proc.poll() is not None:
        console.print(f"[red]Error:[/red] Server failed to start. Check log file: {log_file}")
        raise typer.Exit(1)

    tunnel_pid = None
    tunnel_url = None
    if tunnel:
        url_file = Path("PUBLIC_URL.txt")
        try:
            url_file.unlink(missing_ok=True)
        except Exception:
            pass

        worker_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "tunnel",
            "--port", str(port),
            "--host", probe_host,
            "--daemon-worker",
        ]
        with open(TUNNEL_LOG_FILE, "a", encoding="utf-8") as t_log:
            tunnel_proc = subprocess.Popen(
                worker_cmd,
                stdout=t_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        tunnel_pid = tunnel_proc.pid
        TUNNEL_PID_FILE.write_text(str(tunnel_pid), encoding="utf-8")

        start_wait = time.time()
        while time.time() - start_wait < 10:
            if url_file.exists():
                u = url_file.read_text(encoding="utf-8").strip()
                if u.startswith("http"):
                    tunnel_url = u
                    break
            time.sleep(0.5)

    deployments[role] = {
        "role": role,
        "host": host,
        "port": port,
        "pid": proc.pid,
        "tunnel_pid": tunnel_pid,
        "tunnel_url": tunnel_url,
        "started_at": datetime.now().isoformat(),
        "log_file": str(log_file),
    }
    _save_deployments(deployments)

    panel_lines = [
        f"[bold green]Deployed {role}[/bold green]\n",
        f"Local URL:    [cyan]http://{probe_host}:{port}[/cyan]",
    ]
    if host in ("0.0.0.0", "::"):
        panel_lines.append(f"Remote URL:   [cyan]http://<server-ip>:{port}[/cyan]")
    if tunnel_url:
        panel_lines.append(f"Public URL:   [magenta]{tunnel_url}[/magenta]")
        panel_lines.append(f"Playground:   [magenta]{tunnel_url}/[/magenta]")
        panel_lines.append(f"API Docs:     [magenta]{tunnel_url}/docs[/magenta]")
    else:
        panel_lines.append(f"API Docs:     [cyan]http://{probe_host}:{port}/docs[/cyan]")
    panel_lines.append(f"PID:          [dim]{proc.pid}[/dim]" + (f" (Tunnel PID: [dim]{tunnel_pid}[/dim])" if tunnel_pid else ""))

    console.print(
        Panel(
            "\n".join(panel_lines),
            title="Deployment Active",
            border_style="green",
        )
    )


@app.command("stop")
def stop_command(
    role: str = typer.Argument(..., help="Role deployment to stop"),
    autostart: bool = typer.Option(False, "--autostart", help="Stop and disable systemd autostart service"),
):
    """Stop a running role deployment and release model memory."""
    _unload_ollama_model(f"ai-institute-{role}")

    deployments = _load_deployments()

    # 1. Check systemd autostart service
    systemctl_bin = shutil.which("systemctl")
    is_systemd = autostart or (deployments.get(role, {}).get("autostart", False))
    if is_systemd and systemctl_bin:
        service_name = deployments.get(role, {}).get("service") or f"ai-institute-{role}.service"
        tunnel_svc = deployments.get(role, {}).get("tunnel_service") or f"ai-institute-{role}-tunnel.service"
        try:
            subprocess.run([systemctl_bin, "stop", service_name], capture_output=True)
            subprocess.run([systemctl_bin, "disable", service_name], capture_output=True)
            console.print(f"[green]Stopped and disabled systemd service '{service_name}'.[/green]")
            subprocess.run([systemctl_bin, "stop", tunnel_svc], capture_output=True)
            subprocess.run([systemctl_bin, "disable", tunnel_svc], capture_output=True)
        except Exception as e:
            console.print(f"[yellow]Warning while stopping systemd service:[/yellow] {e}")

    # 2. Check user-space background processes
    if role in deployments:
        info = deployments[role]
        pid = info.get("pid")
        if pid and _is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                if _is_pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
                console.print(f"[green]Stopped deployment for {role}[/green] (PID {pid}).")
            except Exception as e:
                console.print(f"[red]Failed to terminate PID {pid}:[/red] {e}")

        tunnel_pid = info.get("tunnel_pid")
        if tunnel_pid and _is_pid_alive(tunnel_pid):
            try:
                os.kill(tunnel_pid, signal.SIGTERM)
                console.print(f"[green]Stopped associated tunnel[/green] (PID {tunnel_pid}).")
            except Exception:
                pass

        del deployments[role]
        _save_deployments(deployments)
    else:
        if not is_systemd:
            console.print(f"[yellow]No deployment found for role '{role}'.[/yellow]")
            raise typer.Exit(1)


@app.command("status")
def status_command():
    """Show active role deployments."""
    deployments = _load_deployments()
    if not deployments:
        console.print("[dim]No active deployments recorded.[/dim]")
        return

    table = Table(title="AI-Institute Deployments", border_style="blue")
    table.add_column("Role", style="bold green")
    table.add_column("Local URL", style="cyan")
    table.add_column("Public Tunnel", style="magenta")
    table.add_column("PID / Service", style="yellow")
    table.add_column("Started At", style="dim")
    table.add_column("Status", style="bold")

    for name, info in list(deployments.items()):
        pid = info.get("pid")
        host = info.get("host", "127.0.0.1")
        port = info.get("port", 8080)
        started_at = info.get("started_at", "unknown")
        is_auto = info.get("autostart", False)
        tunnel_url = info.get("tunnel_url") or "-"

        if is_auto:
            svc = info.get("service", "systemd")
            status_text = "[green]ACTIVE (BOOT)[/green]"
            pid_col = f"[dim]{svc}[/dim]"
        else:
            alive = _is_pid_alive(pid) if pid else False
            status_text = "[green]RUNNING[/green]" if alive else "[red]STOPPED[/red]"
            pid_col = str(pid) if pid else "-"

        display_host = "<server-ip>" if host in ("0.0.0.0", "::") else host
        table.add_row(name, f"http://{display_host}:{port}", tunnel_url, pid_col, started_at[:19], status_text)

    console.print(table)


@app.command("backup")
def backup_command(
    role: str = typer.Argument(..., help="Role to backup"),
    output: Path = typer.Option(..., "--output", "-o", help="Target archive path (.tar.gz)"),
):
    """Package a role directory into an archive."""
    roles_dir = get_roles_dir()
    role_path = roles_dir / role
    if not role_path.exists() or not (role_path / "role.yaml").exists():
        console.print(f"[red]Error:[/red] Role '{role}' not found at {role_path}")
        raise typer.Exit(1)

    out_path = Path(output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(role_path, arcname=role)

    size_kb = out_path.stat().st_size / 1024
    console.print(f"[green]Backed up role '{role}'[/green] to {out_path} ({size_kb:.1f} KB)")


@app.command("restore")
def restore_command(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to .tar.gz archive"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing role if it exists"),
):
    """Restore a role directory from an archive."""
    in_path = Path(input_path).resolve()
    if not in_path.exists():
        console.print(f"[red]Error:[/red] Archive not found at {in_path}")
        raise typer.Exit(1)

    roles_dir = get_roles_dir()
    roles_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(in_path, "r:gz") as tar:
        names = [m.name for m in tar.getmembers() if "/" in m.name or "\\" in m.name]
        role_name = names[0].split("/")[0].split("\\")[0] if names else in_path.stem.replace(".tar", "")

        target_dir = roles_dir / role_name
        if target_dir.exists():
            if not force:
                console.print(f"[red]Error:[/red] Role '{role_name}' already exists at {target_dir}. Use --force to overwrite.")
                raise typer.Exit(1)
            shutil.rmtree(target_dir)

        tar.extractall(path=roles_dir)

    console.print(f"[green]Restored role '{role_name}'[/green] into {roles_dir / role_name}")


@app.command("package")
def package_command(
    role: str = typer.Argument(..., help="Role to package for deployment"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Target zip file path (default: dist/<role>-deploy.zip)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model tier override (e.g. 1.5b or 3b)"),
):
    """Bundle the runtime wheel, role data archive, and Modelfile into a single deployable zip."""
    import tempfile
    import zipfile

    role_path = get_role_path_or_exit(role)
    cfg_file = role_path / "role.yaml"
    role_cfg = {}
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            role_cfg = yaml.safe_load(f) or {}

    student_model = model or role_cfg.get("student_model", "qwen2.5:3b")
    tier = "3b" if "3b" in student_model.lower() else ("1.5b" if "1.5b" in student_model.lower() else "")

    modelfile_path = None
    if tier and (role_path / "model" / f"Modelfile.{tier}").exists():
        modelfile_path = role_path / "model" / f"Modelfile.{tier}"
    elif (role_path / "model" / "Modelfile").exists():
        modelfile_path = role_path / "model" / "Modelfile"

    with console.status("[bold green]Building runtime package and assembling deployment bundle...[/bold green]"):
        dist_dir = Path("dist")
        dist_dir.mkdir(parents=True, exist_ok=True)

        build_proc = subprocess.run(
            [sys.executable, "-m", "build", "--wheel"],
            capture_output=True,
            text=True,
        )
        if build_proc.returncode != 0:
            console.print(f"[red]Error building wheel:[/red]\n{build_proc.stderr}")
            raise typer.Exit(1)

        wheels = sorted(dist_dir.glob("ai_institute-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not wheels:
            console.print("[red]Error:[/red] No wheel found in dist/ after build.")
            raise typer.Exit(1)
        wheel_file = wheels[0]

        target_zip = Path(output).resolve() if output else (dist_dir / f"{role}-deploy.zip").resolve()
        target_zip.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            tar_name = f"{role}.tar.gz"
            tar_path = tmp_dir / tar_name
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(role_path, arcname=role)

            shutil.copy2(wheel_file, tmp_dir / wheel_file.name)

            repo_dir = Path(__file__).resolve().parent
            for doc_name in ["README.md", "CONTRIBUTING.md"]:
                doc_path = repo_dir / doc_name
                if doc_path.exists():
                    shutil.copy2(doc_path, tmp_dir / doc_name)

            target_modelfile_name = f"Modelfile.{tier}" if tier else "Modelfile"
            if modelfile_path and modelfile_path.exists():
                shutil.copy2(modelfile_path, tmp_dir / target_modelfile_name)

            install_script = tmp_dir / "install.sh"
            target_tag = f":{tier}" if tier else ""
            script_content = f"""#!/usr/bin/env bash
set -e

echo "=== Installing AI-Institute Runtime for '{role}' ==="
pip install --no-cache-dir *.whl

if [ -f "{target_modelfile_name}" ] && command -v ollama &> /dev/null; then
    echo "Registering model in Ollama from {target_modelfile_name}..."
    ollama create ai-institute-{role}{target_tag} -f "{target_modelfile_name}" || true
fi

echo "Restoring role data..."
ai-institute restore -i "{tar_name}" --force

echo ""
echo "=== Setup complete! ==="
echo "You can now run:"
echo "  ai-institute test {role}"
echo "  ai-institute deploy {role} --port 8080"
"""
            install_script.write_text(script_content, encoding="utf-8")
            install_script.chmod(0o755)

            with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_to_pack in tmp_dir.iterdir():
                    if file_to_pack.is_file():
                        zf.write(file_to_pack, arcname=file_to_pack.name)

    size_mb = target_zip.stat().st_size / (1024 * 1024)
    console.print(
        Panel(
            f"[bold green]Successfully created deployment bundle:[/bold green] {target_zip} ({size_mb:.2f} MB)\n\n"
            f"[bold]Bundled Files:[/bold]\n"
            f"  • {wheel_file.name} (lightweight runtime wheel)\n"
            f"  • {tar_name} (role config, tools, and pre-computed vector store)\n"
            f"  • {target_modelfile_name} (Ollama model definition)\n"
            f"  • install.sh (automated setup script)\n"
            f"  • README.md & CONTRIBUTING.md (documentation)\n\n"
            f"[bold]Server Usage:[/bold]\n"
            f"  1. Copy [cyan]{target_zip.name}[/cyan] to your server\n"
            f"  2. Run: [cyan]unzip {target_zip.name} && bash install.sh[/cyan]\n"
            f"  3. Run: [cyan]ai-institute test {role}[/cyan] or [cyan]ai-institute deploy {role} --port 8080[/cyan]",
            title="Deployment Bundle Ready",
            border_style="green",
        )
    )


@app.command("pentest")
def pentest_command(
    role: str = typer.Argument(..., help="Role to evaluate"),
    attempts: int = typer.Option(50, "--attempts", "-a", help="Number of test attacks to execute"),
    report: bool = typer.Option(True, "--report", help="Display full failure report"),
    patterns: Optional[Path] = typer.Option(None, "--patterns", "-p", help="Custom jailbreak patterns JSON file"),
):
    """Run automated adversarial pentest against role defenses."""
    roles_dir = get_roles_dir()
    role_path = roles_dir / role
    if not role_path.exists() or not (role_path / "role.yaml").exists():
        console.print(f"[red]Error:[/red] Role '{role}' not found at {role_path}")
        raise typer.Exit(1)

    from training.pentest import run_pentest

    console.print(f"Running pentest suite for [bold green]{role}[/bold green] ({attempts} attack attempts)...")
    results = run_pentest(role_path, attempts=attempts, patterns_path=patterns)

    # Render summary table
    border_color = "red" if results["bypassed_count"] > 0 else "green"
    table = Table(title=f"Pentest Evaluation: {role}", border_style=border_color)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    total = results["total_attempts"]
    table.add_row("Total Attempts", str(total))
    table.add_row("Blocked by Guardrails (Layer 1)", f"{results['guardrail_blocks']} ({results['guardrail_blocks']/total*100:.1f}%)")
    table.add_row("Blocked by Model Refusal (Layer 2)", f"{results['model_refusals']} ({results['model_refusals']/total*100:.1f}%)")
    table.add_row("Blocked by Output Filter (Layer 4)", f"{results['output_filter_blocks']} ({results['output_filter_blocks']/total*100:.1f}%)")
    table.add_row("Bypassed Defenses", f"[red]{results['bypassed_count']}[/red]" if results["bypassed_count"] > 0 else "[green]0[/green]")
    table.add_row("Overall Block Rate", f"[bold green]{results['block_rate']}%[/bold green]" if results["block_rate"] >= 90 else f"[bold yellow]{results['block_rate']}%[/bold yellow]")

    console.print(table)

    if report and results["failed_patterns"]:
        console.print("\n[bold red]Bypassed Attacks Details:[/bold red]")
        for failed in results["failed_patterns"]:
            console.print(f"- [[yellow]{failed['id']}[/yellow]] ([dim]{failed['category']}[/dim]) {failed['prompt']}")
            console.print(f"  [dim]Response:[/dim] {failed['response']}\n")


@app.command("auto-generate")
def auto_generate_command(
    role: str = typer.Argument(..., help="Role to generate training data for"),
    provider: str = typer.Option("ollama", "--provider", "-p", help="Teacher provider (ollama, nim, gemini, openai)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific teacher model name"),
    count: int = typer.Option(20, "--count", "-c", help="Total examples to generate"),
    category: Optional[str] = typer.Option(None, "--category", help="Specific category (default: distributed across all categories)"),
):
    """Generate training dataset examples using a teacher LLM."""
    roles_dir = get_roles_dir()
    role_path = roles_dir / role
    if not role_path.exists() or not (role_path / "role.yaml").exists():
        console.print(f"[red]Error:[/red] Role '{role}' not found at {role_path}")
        raise typer.Exit(1)

    with open(role_path / "role.yaml", "r", encoding="utf-8") as f:
        role_config = yaml.safe_load(f) or {}

    from teachers import get_teacher
    from training.dataset import VALID_CATEGORIES, save_training_batch, validate_dataset

    try:
        teacher = get_teacher(provider=provider, model_name=model)
    except Exception as e:
        console.print(f"[red]Provider Error:[/red] {e}")
        raise typer.Exit(1)

    categories = [category] if category else VALID_CATEGORIES
    per_category = max(1, count // len(categories))

    console.print(f"Generating training data for [bold green]{role}[/bold green] using [cyan]{provider}[/cyan] ({teacher.model_name})...")

    all_generated = []
    for cat in categories:
        try:
            examples = teacher.generate_examples(role_config, category=cat, count=per_category)
            all_generated.extend(examples)
            console.print(f"  Generated [green]{len(examples)}[/green] examples for [dim]{cat}[/dim]")
        except Exception as e:
            console.print(f"  [yellow]Failed for category '{cat}':[/yellow] {e}")

    if not all_generated:
        console.print("[red]No examples generated.[/red]")
        raise typer.Exit(1)

    valid, errors = validate_dataset(all_generated)
    if errors:
        console.print(f"[yellow]Warning: Filtered out {len(errors)} invalid examples.[/yellow]")

    if not valid:
        console.print("[red]Generated examples failed validation.[/red]")
        raise typer.Exit(1)

    saved_path = save_training_batch(role_path, valid)
    console.print(f"[bold green]Successfully saved {len(valid)} examples to {saved_path}[/bold green]")


TUNNEL_PID_FILE = Path(".tunnel.pid")
TUNNEL_LOG_FILE = Path(".tunnel.log")


@app.command("tunnel")
def tunnel_command(
    port: int = typer.Option(8080, "--port", "-p", help="Local server port to expose"),
    host: str = typer.Option("localhost", "--host", "-h", help="Local target host"),
    save_url: bool = typer.Option(True, "--save-url/--no-save-url", help="Save live URL to PUBLIC_URL.txt"),
    background: bool = typer.Option(False, "--background", "-d", help="Run tunnel in background as a daemon"),
    stop: bool = typer.Option(False, "--stop", help="Stop any running background tunnel"),
    daemon_worker: bool = typer.Option(False, "--daemon-worker", hidden=True),
):
    """Expose local API server via a secure public HTTPS reverse tunnel over port 443 (firewall-friendly)."""
    # 1. Handle --stop
    if stop:
        if not TUNNEL_PID_FILE.exists():
            console.print("[yellow]No active background tunnel was running.[/yellow]")
            raise typer.Exit(0)

        stopped = False
        try:
            pid = int(TUNNEL_PID_FILE.read_text().strip())
            if _is_pid_alive(pid):
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                if _is_pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
                stopped = True
        except Exception:
            pass
        finally:
            try:
                TUNNEL_PID_FILE.unlink(missing_ok=True)
            except Exception:
                pass

        if stopped:
            console.print("[bold green]Public tunnel stopped successfully.[/bold green]")
        else:
            console.print("[yellow]Tunnel process was not running (cleaned up stale PID file).[/yellow]")
        raise typer.Exit(0)

    ssh_bin = shutil.which("ssh")
    if not ssh_bin:
        console.print("[red]Error:[/red] 'ssh' binary not found. Please ensure OpenSSH client is installed.")
        raise typer.Exit(1)

    url_file = Path("PUBLIC_URL.txt")

    # 2. Handle --background / -d launcher
    if background:
        if TUNNEL_PID_FILE.exists():
            try:
                pid = int(TUNNEL_PID_FILE.read_text().strip())
                if _is_pid_alive(pid):
                    console.print(f"[yellow]Tunnel is already running in background (PID {pid}).[/yellow]")
                    console.print("[dim]Run 'ai-institute tunnel --stop' to terminate it first.[/dim]")
                    raise typer.Exit(0)
            except ValueError:
                pass

        console.print(f"Starting reverse tunnel in background for [cyan]{host}:{port}[/cyan] over port 443...")
        try:
            url_file.unlink(missing_ok=True)
        except Exception:
            pass

        worker_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "tunnel",
            "--port", str(port),
            "--host", host,
            "--daemon-worker",
        ]
        if not save_url:
            worker_cmd.append("--no-save-url")

        with open(TUNNEL_LOG_FILE, "a", encoding="utf-8") as log_f:
            daemon_proc = subprocess.Popen(
                worker_cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        TUNNEL_PID_FILE.write_text(str(daemon_proc.pid), encoding="utf-8")

        assigned_url = None
        start_wait = time.time()
        while time.time() - start_wait < 10:
            if not _is_pid_alive(daemon_proc.pid):
                console.print(f"[red]Error:[/red] Tunnel process exited unexpectedly. Check {TUNNEL_LOG_FILE}.")
                raise typer.Exit(1)

            if url_file.exists():
                content = url_file.read_text(encoding="utf-8").strip()
                if content.startswith("http"):
                    assigned_url = content
                    break

            if TUNNEL_LOG_FILE.exists():
                try:
                    log_text = TUNNEL_LOG_FILE.read_text(encoding="utf-8")
                    matches = re.findall(r"https://[a-zA-Z0-9.-]+\.pinggy\.(?:link|net)", log_text)
                    if matches:
                        assigned_url = matches[-1]
                        if save_url:
                            url_file.write_text(assigned_url + "\n", encoding="utf-8")
                        break
                except Exception:
                    pass

            time.sleep(0.5)

        if assigned_url:
            panel_text = (
                f"[bold green]Tunnel is active and running in background![/bold green]\n\n"
                f"  [bold]Local Target:[/bold]    http://{host}:{port}\n"
                f"  [bold]Public URL:[/bold]      [cyan]{assigned_url}[/cyan]\n"
                f"  [bold]Web Playground:[/bold]  [cyan]{assigned_url}/[/cyan]\n"
                f"  [bold]Swagger Docs:[/bold]    [cyan]{assigned_url}/docs[/cyan]\n"
                f"  [bold]PID:[/bold]             {daemon_proc.pid}\n\n"
                f"[dim]Run 'ai-institute tunnel --stop' to shut down the background tunnel.[/dim]"
            )
            console.print(Panel(panel_text, title="AI-Institute Public Egress Tunnel", border_style="cyan"))
        else:
            console.print(f"[yellow]Tunnel daemon started (PID {daemon_proc.pid}).[/yellow] Public URL may take a few seconds to appear in {url_file.name}.")

        raise typer.Exit(0)

    # 3. Foreground / Daemon Worker Execution Loop
    cmd = [
        ssh_bin,
        "-p", "443",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-R", f"0:{host}:{port}",
        "a.pinggy.io",
    ]

    if not daemon_worker:
        console.print(f"Establishing secure reverse tunnel for [cyan]{host}:{port}[/cyan] over outbound port 443...")

    stop_requested = False

    def handle_sig(sig, frame):
        nonlocal stop_requested
        stop_requested = True
        if not daemon_worker:
            console.print("\n[yellow]Shutting down tunnel...[/yellow]")

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    active_url = None

    while not stop_requested:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            if not daemon_worker:
                console.print(f"[red]Failed to launch tunnel process:[/red] {e}")
            time.sleep(3)
            continue

        while not stop_requested:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue

            matches = re.findall(r"https://[a-zA-Z0-9.-]+\.pinggy\.(?:link|net)", line)
            if matches and matches[0] != active_url:
                active_url = matches[0]
                if save_url:
                    try:
                        url_file.write_text(active_url + "\n", encoding="utf-8")
                    except Exception:
                        pass

                if not daemon_worker:
                    panel_text = (
                        f"[bold green]Tunnel is active and live![/bold green]\n\n"
                        f"  [bold]Local Target:[/bold]    http://{host}:{port}\n"
                        f"  [bold]Public URL:[/bold]      [cyan]{active_url}[/cyan]\n"
                        f"  [bold]Web Playground:[/bold]  [cyan]{active_url}/[/cyan]\n"
                        f"  [bold]Swagger Docs:[/bold]    [cyan]{active_url}/docs[/cyan]\n"
                        f"  [bold]ReDoc:[/bold]           [cyan]{active_url}/redoc[/cyan]\n\n"
                        f"[dim]Press Ctrl+C to close the tunnel.[/dim]"
                    )
                    console.print(Panel(panel_text, title="AI-Institute Public Egress Tunnel", border_style="cyan"))

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

        if not stop_requested:
            time.sleep(3)

    if not daemon_worker:
        console.print("[green]Tunnel closed cleanly.[/green]")


if __name__ == "__main__":
    app()

