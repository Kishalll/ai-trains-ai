import importlib.util
import os
import shutil
import subprocess
import sys

DEPENDENCY_GROUPS = {
    "rag": {
        "description": "RAG vector search and embeddings",
        "packages": ["chromadb", "sentence-transformers"],
        "download_size": "~230MB",
    },
    "training": {
        "description": "Model fine-tuning and export",
        "packages": ["torch", "transformers", "peft", "trl", "accelerate", "datasets", "gguf", "sentencepiece"],
        "download_size": "~500MB (CPU) / ~2.5GB (GPU)",
    },
    "api": {
        "description": "REST API server and Ollama client",
        "packages": ["fastapi", "uvicorn", "ollama"],
        "download_size": "~30MB",
    },
    "db": {
        "description": "MySQL database connection",
        "packages": ["mysqlclient"],
        "download_size": "~10MB",
    },
    "teacher-gemini": {
        "description": "Gemini API teacher integration",
        "packages": ["google-generativeai"],
        "download_size": "~20MB",
    },
    "teacher-openai": {
        "description": "OpenAI API teacher integration",
        "packages": ["openai"],
        "download_size": "~15MB",
    },
    "teacher-nim": {
        "description": "NVIDIA NIM API teacher integration",
        "packages": ["openai"],
        "download_size": "~15MB",
    },
}

IMPORT_NAME_MAP = {
    "sentence-transformers": "sentence_transformers",
    "google-generativeai": "google.generativeai",
    "pyyaml": "yaml",
    "mysqlclient": "MySQLdb",
}


def has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU and driver are available on the machine."""
    if os.path.exists("/dev/nvidia0"):
        return True
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            return res.returncode == 0
        except (subprocess.SubprocessError, OSError):
            pass
    return False


def get_download_size(group_name: str) -> str:
    """Return estimated download size based on hardware."""
    if group_name == "training":
        return "~2.5GB (GPU build)" if has_nvidia_gpu() else "~500MB (CPU build)"
    group = DEPENDENCY_GROUPS.get(group_name, {})
    return group.get("download_size", "")


def is_installed(package_name: str) -> bool:
    module_name = IMPORT_NAME_MAP.get(package_name, package_name.replace("-", "_"))
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def get_missing_packages(group_name: str) -> list[str]:
    group = DEPENDENCY_GROUPS.get(group_name)
    if not group:
        return []
    return [pkg for pkg in group["packages"] if not is_installed(pkg)]


def install_packages(packages: list[str]) -> bool:
    """Install packages, using CPU-only index for torch if no NVIDIA GPU is present."""
    other_packages = [pkg for pkg in packages if pkg != "torch"]

    if "torch" in packages:
        if has_nvidia_gpu():
            print("NVIDIA GPU detected: installing CUDA-enabled PyTorch...")
            torch_cmd = [sys.executable, "-m", "pip", "install", "torch"]
        else:
            print("No NVIDIA GPU detected: installing lightweight CPU PyTorch...")
            torch_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        res = subprocess.run(torch_cmd)
        if res.returncode != 0:
            return False

    if other_packages:
        print(f"Installing {', '.join(other_packages)} via pip...")
        cmd = [sys.executable, "-m", "pip", "install"] + other_packages
        res = subprocess.run(cmd)
        if res.returncode != 0:
            return False

    return True


def check_torch_hardware() -> None:
    """Warn if a GPU is present but only CPU PyTorch is installed."""
    if not is_installed("torch") or not has_nvidia_gpu():
        return
    try:
        import importlib
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            print("\nNotice: NVIDIA GPU detected, but CPU-only PyTorch is currently installed.")
            choice = input("Upgrade PyTorch to CUDA version? [Y/n]: ").strip().lower()
            if choice not in ["n", "no"]:
                print("Upgrading PyTorch to CUDA version...")
                subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch"])
    except Exception:
        pass


def check_build_tools() -> None:
    """Ensure gcc and Python.h are present for GPU kernel compilation (Triton)."""
    if not has_nvidia_gpu():
        return
    import shutil
    import sysconfig

    has_gcc = shutil.which("gcc") is not None
    inc_dir = sysconfig.get_path("include")
    has_python_h = os.path.exists(os.path.join(inc_dir, "Python.h")) if inc_dir else False

    if not (has_gcc and has_python_h):
        print("\nNotice: NVIDIA GPU detected, but C compiler or Python headers are missing (required by Triton).")
        if shutil.which("apt-get"):
            try:
                print("Attempting automatic install of build-essential and python3-dev...")
                subprocess.run(["sudo", "apt-get", "update", "-qq"])
                subprocess.run(["sudo", "apt-get", "install", "-y", "build-essential", "python3-dev"])
            except Exception:
                print("Auto-install failed. Please run: sudo apt install -y build-essential python3-dev")
        else:
            print("Please ensure gcc and python3-dev are installed on this server.")


def require_group(group_name: str, auto_install: bool = False) -> bool:
    if group_name not in DEPENDENCY_GROUPS:
        return True

    if group_name == "training":
        check_torch_hardware()
        check_build_tools()

    missing = get_missing_packages(group_name)
    if not missing:
        return True

    group = DEPENDENCY_GROUPS[group_name]
    size_str = get_download_size(group_name)
    size_note = f" (download size: {size_str})" if size_str else ""

    print(f"\nMissing dependencies for {group['description']}{size_note}:")
    print(f"  {', '.join(missing)}")

    if not auto_install:
        try:
            choice = input("\nInstall now? [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return False

        if choice in ["n", "no"]:
            print("Installation skipped.")
            return False

    if install_packages(missing):
        print("Installation complete.\n")
        return True

    print("Installation failed. Please check your network connection and try again.")
    return False
