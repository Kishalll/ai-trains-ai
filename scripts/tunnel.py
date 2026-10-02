#!/usr/bin/env python3
"""AI-Institute Public Egress Tunnel Supervisor (Pinggy over port 443).

Establishes and maintains a secure reverse SSH tunnel through a.pinggy.io:443,
extracts the assigned public HTTPS URL, and synchronizes it to PUBLIC_URL.txt.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

running = True


def handle_sig(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, handle_sig)
signal.signal(signal.SIGTERM, handle_sig)


def main():
    parser = argparse.ArgumentParser(description="Public HTTPS reverse tunnel over port 443")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Local port to expose")
    parser.add_argument("--host", "-H", type=str, default="localhost", help="Local host to forward to")
    parser.add_argument("--url-file", type=Path, default=Path("PUBLIC_URL.txt"), help="File to write live URL")
    args = parser.parse_args()

    ssh_bin = shutil.which("ssh")
    if not ssh_bin:
        print("[ERROR] 'ssh' binary not found. Please ensure OpenSSH client is installed.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        ssh_bin,
        "-p", "443",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-R", f"0:{args.host}:{args.port}",
        "a.pinggy.io",
    ]

    active_url = None

    while running:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            time.sleep(3)
            continue

        while running:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue

            matches = re.findall(r"https://[a-zA-Z0-9.-]+\.pinggy\.(?:link|net)", line)
            if matches and matches[0] != active_url:
                active_url = matches[0]
                try:
                    args.url_file.write_text(active_url + "\n", encoding="utf-8")
                except Exception:
                    pass

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

        if running:
            time.sleep(3)


if __name__ == "__main__":
    main()
