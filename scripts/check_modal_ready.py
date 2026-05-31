"""Offline Modal/Qwen readiness check for live LLM experiments.

This script validates configuration only; it does not contact Modal, Qwen, or
OpenAI. Use it after copying `.env.example` to `.env` and filling in the Modal
URL or hosted Qwen settings.

Run:
    python scripts/check_modal_ready.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PACKAGES = {
    "sentence-transformers": "sentence_transformers",
    "faiss-cpu": "faiss",
    "python-dotenv": "dotenv",
}

TRAINING_PACKAGES = {
    **SIMULATION_PACKAGES,
    "tiktoken": "tiktoken",
    "torch": "torch",
}


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get(env_file: dict[str, str], key: str) -> str:
    return os.environ.get(key) or env_file.get(key, "")


def _ok(label: str) -> str:
    return f"[OK] {label}"


def _warn(label: str) -> str:
    return f"[WARN] {label}"


def _fail(label: str) -> str:
    return f"[FAIL] {label}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--profile",
        choices=["simulation", "training"],
        default="training",
        help="simulation checks Exp 1/live-sweep deps; training also checks Exp 3 deps.",
    )
    parser.add_argument(
        "--allow-missing-packages",
        action="store_true",
        help="Return success even if local LLM-track packages are absent.",
    )
    args = parser.parse_args()

    env_file = _load_env_file(args.env)
    lines: list[str] = []
    failures = 0

    if args.env.exists():
        lines.append(_ok(f"Found {args.env}"))
    else:
        lines.append(_fail(f"Missing {args.env}; copy .env.example to .env first"))
        failures += 1

    qwen_key = _get(env_file, "QWEN_API_KEY") or _get(env_file, "DASHSCOPE_API_KEY")
    openai_key = _get(env_file, "OPENAI_API_KEY")
    if qwen_key:
        lines.append(_ok("Qwen/DashScope key is set"))
    elif openai_key:
        lines.append(_ok("OpenAI-compatible key is set"))
    else:
        lines.append(_fail("No QWEN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY found"))
        failures += 1

    model = (
        _get(env_file, "QWEN_MODEL")
        or _get(env_file, "LLM_MODEL")
        or _get(env_file, "OPENAI_MODEL")
    )
    if model:
        lines.append(_ok(f"Model: {model}"))
    else:
        lines.append(_warn("No model set; code will fall back to qwen-plus or gpt-4o-mini"))

    base_url = (
        _get(env_file, "QWEN_BASE_URL")
        or _get(env_file, "LLM_BASE_URL")
        or _get(env_file, "OPENAI_BASE_URL")
    )
    if base_url:
        parsed = urlparse(base_url)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            lines.append(_ok(f"Base URL shape looks valid: {base_url}"))
        else:
            lines.append(_fail(f"Base URL is not a valid HTTP URL: {base_url}"))
            failures += 1
        if parsed.path.rstrip("/").endswith("/v1"):
            lines.append(_ok("Base URL ends with /v1"))
        else:
            lines.append(_warn("Base URL does not end with /v1; append /v1 to Modal URLs"))
    else:
        lines.append(_warn("No base URL set; hosted Qwen/OpenAI defaults will be used"))

    if qwen_key == "EMPTY":
        if base_url and "modal.run" in base_url:
            lines.append(_ok("QWEN_API_KEY=EMPTY is appropriate for the default Modal vLLM server"))
        else:
            lines.append(_warn("QWEN_API_KEY=EMPTY only works for unauthenticated local/Modal vLLM endpoints"))

    required_packages = (
        SIMULATION_PACKAGES if args.profile == "simulation" else TRAINING_PACKAGES
    )
    missing_packages: list[str] = []
    for package, module in required_packages.items():
        if importlib.util.find_spec(module) is None:
            missing_packages.append(package)
    if missing_packages:
        if not args.allow_missing_packages:
            lines.append(_fail("Missing local LLM-track packages: " + ", ".join(missing_packages)))
            failures += 1
        else:
            lines.append(_warn("Missing local LLM-track packages: " + ", ".join(missing_packages)))
    else:
        lines.append(_ok(f"Local {args.profile} packages are installed"))

    if (ROOT / "data" / "benign_memories.seed.jsonl").exists():
        lines.append(_ok("Committed benign seed corpus exists"))
    else:
        lines.append(_fail("Missing data/benign_memories.seed.jsonl"))
        failures += 1

    print("\n".join(lines))
    if failures:
        print("\nModal/live LLM path is not ready yet.")
        return 1
    print("\nModal/live LLM path looks ready for a dry run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
