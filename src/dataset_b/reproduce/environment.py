"""Install and check the per-model environment profiles."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .manifest import ReproduceError


PROFILE_DIRECTORY = "environments/reproducibility"
DEFAULT_ENV_DIRECTORY = ".venv-repro"
EXECUTION_ENV = {"PYTHONDONTWRITEBYTECODE": "1"}


def load_profiles(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / PROFILE_DIRECTORY / "profiles.json"
    if not path.is_file():
        raise ReproduceError(f"Environment profiles do not exist: {PROFILE_DIRECTORY}/profiles.json")
    profiles = json.loads(path.read_text(encoding="utf-8")).get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ReproduceError("Environment profiles must be a non-empty object")
    for name, profile in profiles.items():
        if profile.get("compute") not in {"cpu", "cuda"}:
            raise ReproduceError(f"Profile {name} has an unknown compute route")
        for field in ("python", "requirements", "models", "smoke_imports", "version_evidence"):
            if not profile.get(field):
                raise ReproduceError(f"Profile {name} is missing {field}")
    return profiles


def locked_versions(project_root: Path, profile: Mapping[str, Any]) -> dict[str, str]:
    """Read exact pins; an unpinned requirement is a profile error."""
    path = project_root / PROFILE_DIRECTORY / profile["requirements"]
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith(("#", "--")):
            continue
        name, separator, version = text.partition("==")
        if not separator or not version:
            raise ReproduceError(f"Requirement must pin an exact version: {text}")
        pins[_normalize(name)] = version
    return pins


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def python_path(env_root: Path, profile_name: str) -> Path:
    folder = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return env_root / profile_name / folder


def assess_fingerprint(
    project_root: Path,
    name: str,
    profile: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> list[str]:
    """Return every reason this environment cannot serve its profile."""
    errors = []
    if observed.get("python") != profile["python"]:
        errors.append(f"{name}: Python {observed.get('python')} differs from {profile['python']}")
    installed = {_normalize(key): value for key, value in observed.get("packages", {}).items()}
    for package, version in locked_versions(project_root, profile).items():
        if installed.get(package) != version:
            errors.append(f"{name}: {package} is {installed.get(package)}, locked {version}")
    for module, error in observed.get("import_errors", {}).items():
        errors.append(f"{name}: import of {module} failed: {error}")
    if profile["compute"] == "cuda":
        torch_state = observed.get("torch", {})
        if torch_state.get("cuda_available") is not True:
            errors.append(f"{name}: CUDA is required and not available")
        elif torch_state.get("finite_cuda_operation") is not True:
            errors.append(f"{name}: finite CUDA tensor operation failed")
    return errors


def _run_fingerprint(python: Path, project_root: Path, profile: Mapping[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(python), "-B", "-m", "src.dataset_b.reproduce.fingerprint", "--imports", ",".join(profile["smoke_imports"])],
        cwd=project_root,
        env={**os.environ, **EXECUTION_ENV},
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ReproduceError(f"Fingerprint failed: {completed.stderr.strip()[-500:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def install_profile(env_root: Path, project_root: Path, name: str, profile: Mapping[str, Any]) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise ReproduceError("The uv installer is required to create environment profiles")
    target = env_root / name
    if not python_path(env_root, name).is_file():
        command = [uv, "venv", "--python", profile["python"], str(target)]
        # uv can fail once right after downloading a new Python build, then succeed.
        if subprocess.run(command, cwd=project_root).returncode != 0:
            subprocess.run(command, check=True, cwd=project_root)
    requirements = project_root / PROFILE_DIRECTORY / profile["requirements"]
    subprocess.run(
        [
            uv, "pip", "install", "--python", str(python_path(env_root, name)), "-r", str(requirements),
            "--index-strategy", "unsafe-best-match",
        ],
        check=True,
        cwd=project_root,
    )


def check_environments(
    project_root: Path,
    env_root: Path,
    *,
    install: bool = False,
    output_directory: str = "reports/reproducibility/environment",
) -> dict[str, Any]:
    """Fingerprint every profile and fail on any difference from its lock."""
    profiles = load_profiles(project_root)
    errors: list[str] = []
    results: dict[str, Any] = {}
    directory = project_root / output_directory
    directory.mkdir(parents=True, exist_ok=True)
    for name, profile in profiles.items():
        try:
            if install:
                install_profile(env_root, project_root, name, profile)
            python = python_path(env_root, name)
            if not python.is_file():
                errors.append(f"{name}: environment is not installed")
                results[name] = {"status": "FAIL"}
                continue
            observed = _run_fingerprint(python, project_root, profile)
        except (ReproduceError, OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            results[name] = {"status": "FAIL"}
            continue
        problems = assess_fingerprint(project_root, name, profile, observed)
        errors += problems
        results[name] = {
            "status": "FAIL" if problems else "PASS",
            "python": observed.get("python"),
            "torch": observed.get("torch"),
            "nvidia_driver": observed.get("nvidia_driver"),
        }
        payload = {
            "profile": name,
            "status": results[name]["status"],
            "errors": problems,
            "version_evidence": profile["version_evidence"],
            "fingerprint": observed,
        }
        (directory / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return {"status": "FAIL" if errors else "PASS", "profiles": results, "errors": errors, "host_python": sys.version.split()[0]}
