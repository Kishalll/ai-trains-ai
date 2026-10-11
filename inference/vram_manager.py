"""
GPU VRAM Budgeting & Multi-Role Resource Manager for AI-Trains-AI.

Inspects physical GPU VRAM (via torch.cuda or nvidia-smi), tracks Ollama loaded models,
estimates role footprints (Model weights + SentenceTransformer Laya Decision Engine),
and interactively prompts for model unloading/dropping when VRAM capacity is exceeded.
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def get_gpu_vram_info() -> Dict[str, Any]:
    """
    Query physical GPU VRAM in real-time.
    Uses torch.cuda if available, falling back to nvidia-smi.
    """
    # 1. Try torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            used_bytes = total_bytes - free_bytes
            return {
                "available": True,
                "device_name": device_name,
                "total_mb": round(total_bytes / (1024 * 1024), 2),
                "used_mb": round(used_bytes / (1024 * 1024), 2),
                "free_mb": round(free_bytes / (1024 * 1024), 2),
                "total_gb": round(total_bytes / (1024 ** 3), 2),
                "used_gb": round(used_bytes / (1024 ** 3), 2),
                "free_gb": round(free_bytes / (1024 ** 3), 2),
                "source": "torch.cuda",
            }
    except Exception:
        pass

    # 2. Try nvidia-smi fallback
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = res.stdout.strip().split("\n")
            if lines and lines[0]:
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 4:
                    name = parts[0]
                    total_mb = float(parts[1])
                    free_mb = float(parts[2])
                    used_mb = float(parts[3])
                    return {
                        "available": True,
                        "device_name": name,
                        "total_mb": round(total_mb, 2),
                        "used_mb": round(used_mb, 2),
                        "free_mb": round(free_mb, 2),
                        "total_gb": round(total_mb / 1024, 2),
                        "used_gb": round(used_mb / 1024, 2),
                        "free_gb": round(free_mb / 1024, 2),
                        "source": "nvidia-smi",
                    }
        except Exception:
            pass

    return {
        "available": False,
        "device_name": "CPU",
        "total_mb": 0.0,
        "used_mb": 0.0,
        "free_mb": 0.0,
        "total_gb": 0.0,
        "used_gb": 0.0,
        "free_gb": 0.0,
        "source": "none",
    }


def get_ollama_models_info(base_url: str = OLLAMA_BASE_URL) -> Dict[str, Any]:
    """
    Retrieve loaded models (/api/ps) and available local models (/api/tags) from Ollama.
    """
    loaded = {}
    available = {}

    # Query /api/ps (loaded in VRAM/RAM)
    try:
        req = urllib.request.Request(f"{base_url}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                name = m.get("name") or m.get("model")
                size_vram = m.get("size_vram", m.get("size", 0))
                loaded[name] = {
                    "name": name,
                    "size_vram_bytes": size_vram,
                    "size_vram_mb": round(size_vram / (1024 * 1024), 2),
                    "size_vram_gb": round(size_vram / (1024 ** 3), 2),
                    "details": m.get("details", {}),
                }
    except Exception:
        pass

    # Query /api/tags (local models on disk)
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                name = m.get("name") or m.get("model")
                size_bytes = m.get("size", 0)
                available[name] = {
                    "name": name,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "size_gb": round(size_bytes / (1024 ** 3), 2),
                }
    except Exception:
        pass

    return {"loaded": loaded, "available": available}


def unload_ollama_model(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Unload a model from Ollama VRAM by sending keep_alive: 0.
    """
    try:
        payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def estimate_role_vram(role_name: str, roles_dir: Path | str = "roles") -> Dict[str, Any]:
    """
    Estimate VRAM requirements for a specific role.
    Checks model weights + SentenceTransformer memory if Decision Engine is enabled.
    """
    resolved_dir = Path(roles_dir) / role_name
    cfg_file = resolved_dir / "role.yaml"
    cfg = {}
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass

    student_model = cfg.get("student_model", "qwen2.5:3b")
    tier = "3b" if "3b" in student_model.lower() else ("1.5b" if "1.5b" in student_model.lower() else "")
    custom_model_tags = [
        f"ai-institute-{role_name}:{tier}" if tier else f"ai-institute-{role_name}",
        f"ai-{role_name}:{tier}" if tier else f"ai-{role_name}",
        f"{role_name}:{tier}" if tier else role_name,
    ]

    # Check Ollama info
    ollama_info = get_ollama_models_info()
    loaded_models = ollama_info["loaded"]
    available_models = ollama_info["available"]

    matched_model_name = student_model
    for tag in custom_model_tags:
        if tag in loaded_models or tag in available_models:
            matched_model_name = tag
            break

    # Determine Model VRAM
    if matched_model_name in loaded_models:
        model_vram_mb = loaded_models[matched_model_name]["size_vram_mb"]
        is_loaded = True
    elif matched_model_name in available_models:
        # Disk size * 1.05 accounts for context KV cache
        model_vram_mb = available_models[matched_model_name]["size_mb"] * 1.05
        is_loaded = False
    else:
        # Heuristic estimation by parameter size
        if "1.5b" in matched_model_name.lower():
            model_vram_mb = 3300.0
        elif "3b" in matched_model_name.lower():
            model_vram_mb = 6400.0
        elif "7b" in matched_model_name.lower():
            model_vram_mb = 9200.0
        elif "14b" in matched_model_name.lower():
            model_vram_mb = 18000.0
        else:
            model_vram_mb = 4000.0
        is_loaded = False

    # Check Decision Engine (SentenceTransformer Laya reflex)
    de_config = cfg.get("decision_engine", {})
    has_de_file = (resolved_dir / "decision_engine.yaml").exists()
    de_enabled = (isinstance(de_config, dict) and de_config.get("enabled", False)) or has_de_file
    de_vram_mb = 350.0 if de_enabled else 0.0

    total_mb = model_vram_mb + de_vram_mb

    return {
        "role": role_name,
        "model_name": matched_model_name,
        "model_vram_mb": round(model_vram_mb, 2),
        "model_vram_gb": round(model_vram_mb / 1024, 2),
        "de_enabled": de_enabled,
        "de_vram_mb": de_vram_mb,
        "total_role_vram_mb": round(total_mb, 2),
        "total_role_vram_gb": round(total_mb / 1024, 2),
        "is_loaded": is_loaded,
    }


def calculate_deployment_budget(roles: List[str], roles_dir: Path | str = "roles") -> Dict[str, Any]:
    """
    Calculate full VRAM budget for a proposed multi-role deployment.
    """
    gpu_info = get_gpu_vram_info()
    ollama_info = get_ollama_models_info()
    loaded_models = ollama_info["loaded"]

    estimates = [estimate_role_vram(r, roles_dir) for r in roles]

    # Models needed by target deployment
    target_model_names = {e["model_name"] for e in estimates}

    # Loaded models currently in Ollama that are NOT in target deployment
    unrelated_loaded = [
        info for name, info in loaded_models.items()
        if name not in target_model_names
    ]
    unrelated_loaded_mb = sum(m["size_vram_mb"] for m in unrelated_loaded)

    # Base system buffer (OS / window manager / display server)
    system_buffer_mb = 1000.0

    total_role_vram_mb = sum(e["total_role_vram_mb"] for e in estimates)
    # Total projected VRAM if unrelated models remain loaded
    projected_total_mb = total_role_vram_mb + system_buffer_mb + unrelated_loaded_mb

    total_gpu_mb = gpu_info["total_mb"]
    exceeds_vram = gpu_info["available"] and (projected_total_mb > total_gpu_mb)
    deficit_mb = max(0.0, projected_total_mb - total_gpu_mb)

    return {
        "gpu_info": gpu_info,
        "role_estimates": estimates,
        "total_role_vram_mb": round(total_role_vram_mb, 2),
        "total_role_vram_gb": round(total_role_vram_mb / 1024, 2),
        "unrelated_loaded_models": unrelated_loaded,
        "unrelated_loaded_mb": round(unrelated_loaded_mb, 2),
        "system_buffer_mb": system_buffer_mb,
        "projected_total_mb": round(projected_total_mb, 2),
        "projected_total_gb": round(projected_total_mb / 1024, 2),
        "exceeds_vram": exceeds_vram,
        "deficit_mb": round(deficit_mb, 2),
        "deficit_gb": round(deficit_mb / 1024, 2),
    }


def manage_vram_budget_interactive(
    roles: List[str],
    roles_dir: Path | str = "roles",
    console: Optional[Console] = None,
    interactive: bool = True,
) -> List[str]:
    """
    Verifies VRAM availability for target roles.
    If VRAM is exceeded, interactively prompts the user which model(s) or role(s) to drop/unload.
    Returns the final approved list of roles to deploy.
    """
    c = console or Console()
    current_roles = list(roles)

    while True:
        budget = calculate_deployment_budget(current_roles, roles_dir)
        gpu = budget["gpu_info"]

        if not gpu["available"]:
            c.print("[dim]No CUDA GPU detected; proceeding with CPU allocation.[/dim]")
            return current_roles

        # Display pre-deployment memory breakdown table
        table = Table(
            title=f"GPU VRAM Budget: {gpu['device_name']} ({gpu['total_gb']} GB Total)",
            border_style="cyan",
        )
        table.add_column("Component / Role", style="bold")
        table.add_column("Model / Engine", style="dim")
        table.add_column("VRAM Required", justify="right")
        table.add_column("System-1 Laya", justify="center")
        table.add_column("Status", justify="center")

        for est in budget["role_estimates"]:
            de_badge = "[green]Enabled[/green]" if est["de_enabled"] else "[dim]Bypassed[/dim]"
            status_badge = "[cyan]Warm/Loaded[/cyan]" if est["is_loaded"] else "[yellow]To Load[/yellow]"
            table.add_row(
                est["role"],
                est["model_name"],
                f"{est['total_role_vram_gb']} GB",
                de_badge,
                status_badge,
            )

        if budget["unrelated_loaded_models"]:
            for m in budget["unrelated_loaded_models"]:
                table.add_row(
                    f"[dim](loaded) {m['name']}[/dim]",
                    "Unrelated in VRAM",
                    f"{m['size_vram_gb']} GB",
                    "[dim]—[/dim]",
                    "[yellow]Consuming VRAM[/yellow]",
                )

        table.add_section()
        table.add_row(
            "[bold]System / OS Buffer[/bold]",
            "Display / OS",
            f"{round(budget['system_buffer_mb'] / 1024, 2)} GB",
            "—",
            "[dim]Reserved[/dim]",
        )
        table.add_row(
            "[bold cyan]Projected Total Usage[/bold cyan]",
            "—",
            f"[bold cyan]{budget['projected_total_gb']} GB / {gpu['total_gb']} GB[/bold cyan]",
            "—",
            "[bold red]OVER CAPACITY[/bold red]" if budget["exceeds_vram"] else "[bold green]SAFE[/bold green]",
        )

        c.print(table)

        if not budget["exceeds_vram"]:
            c.print(f"[bold green]✓ VRAM Budget Check Passed[/bold green] (Headroom: {round((gpu['total_mb'] - budget['projected_total_mb']) / 1024, 2)} GB free)\n")
            return current_roles

        # Exceeds VRAM! Prompt user
        c.print(Panel(
            f"[bold red]⚠️ GPU VRAM Budget Exceeded![/bold red]\n"
            f"Projected allocation: [bold red]{budget['projected_total_gb']} GB[/bold red] "
            f"exceeds GPU capacity: [bold green]{gpu['total_gb']} GB[/bold green] by "
            f"[bold yellow]{budget['deficit_gb']} GB[/bold yellow].\n\n"
            f"To prevent CUDA Out-of-Memory (OOM) crashes, please choose an action below.",
            title="Resource Overflow Alert",
            border_style="red",
        ))

        if not interactive or not sys.stdin.isatty():
            c.print("[red]Non-interactive environment detected. Aborting to avoid GPU crash.[/red]")
            raise RuntimeError(f"VRAM budget exceeded by {budget['deficit_gb']} GB in non-interactive mode.")

        options = []
        # Option 1: Unload unrelated Ollama models if any exist
        if budget["unrelated_loaded_models"]:
            options.append("unload_unrelated")
            c.print("[bold cyan]1)[/bold cyan] Unload unrelated models currently in Ollama VRAM to free space:")
            for m in budget["unrelated_loaded_models"]:
                c.print(f"    - [yellow]{m['name']}[/yellow] ({m['size_vram_gb']} GB)")

        # Option 2: Drop one of the target roles
        if len(current_roles) > 1:
            c.print("\n[bold cyan]2)[/bold cyan] Drop one of the deployment roles:")
            for idx, r in enumerate(current_roles, start=1):
                c.print(f"    [white]{chr(96 + idx)})[/white] Drop role '[cyan]{r}[/cyan]'")

        c.print("\n[bold cyan]q)[/bold cyan] Abort deployment\n")

        choice = Prompt.ask("Select an action to resolve VRAM shortage", default="1" if budget["unrelated_loaded_models"] else "a").strip().lower()

        if choice in ("q", "quit", "exit"):
            c.print("[yellow]Deployment cancelled by user.[/yellow]")
            sys.exit(0)

        # Handle unloading unrelated models
        if choice in ("1", "unload") and budget["unrelated_loaded_models"]:
            for m in budget["unrelated_loaded_models"]:
                c.print(f"Flushing [yellow]{m['name']}[/yellow] from VRAM...")
                unload_ollama_model(m["name"])
            c.print("[green]Unloaded unrelated models. Recalculating VRAM budget...[/green]\n")
            continue

        # Handle dropping a role
        dropped = False
        for idx, r in enumerate(current_roles, start=1):
            if choice == chr(96 + idx) or choice == str(idx + (1 if budget["unrelated_loaded_models"] else 0)):
                c.print(f"[yellow]Dropping role '{r}' from deployment.[/yellow]")
                est = next((e for e in budget["role_estimates"] if e["role"] == r), None)
                if est and est.get("is_loaded"):
                    c.print(f"Flushing dropped model [yellow]{est['model_name']}[/yellow] from GPU VRAM...")
                    unload_ollama_model(est["model_name"])
                current_roles.remove(r)
                dropped = True
                break

        if not dropped and len(current_roles) == 1:
            c.print("[red]Cannot deploy 0 roles. Deployment aborted.[/red]")
            sys.exit(1)
        elif not dropped:
            c.print("[red]Invalid selection. Please choose a valid option.[/red]\n")
