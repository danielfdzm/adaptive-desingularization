#!/usr/bin/env python3
"""Browse, verify, and rebuild this paper's results without changing its records.

The interface and checksum verifier use only the Python standard library.
Figure/table builders run in a temporary copy of the curated workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "experiments" / "catalog.json"
FIGURE_DIR = "paper_1111/revision-validation/publication-figures"
FIGURES = {
    "pb-solution": ("build_pb_solution_maps.py", ["figure_pb_solution_maps"]),
    "algorithm": ("build_algorithm.py", ["figure_alg41_cohort"]),
    "forcing-solutions": ("build_forcing_solutions.py", ["figure_forcing_solutions"]),
    "local": ("build_local.py", ["figure_local"]),
    "recovery-tradeoff": ("build_recovery_tradeoff.py", ["figure_recovery_tradeoff"]),
    "budget": ("build_budget.py", ["figure_budget"]),
    "supplementary": ("build_supplement.py", [
        "figure_recovery", "figure_ecdf", "figure_ecdf_pb_dimensions",
        "figure_solution_1d", "figure_order", "figure_solution_pb",
        "figure_variance_family", "figure_dimension",
    ]),
}
FIGURE_GROUPS = {
    "main": ["pb-solution", "algorithm", "forcing-solutions"],
    "supplement": ["local", "recovery-tradeoff", "budget", "supplementary"],
    "all": list(FIGURES),
}
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache"}


class ReleaseError(Exception):
    """A missing, changed, or unsafe release input."""


def contained(root: Path, relative: str) -> Path:
    """Resolve a catalog/manifest path without allowing escape from its root."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ReleaseError(f"Invalid relative path: {relative!r}")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts:
        raise ReleaseError(f"Path must remain inside the release: {relative}")
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ReleaseError(f"Path points outside the release: {relative}")
    return path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path, *, release: bool = False) -> dict[str, Path]:
    files = {}
    for directory, dirs, names in os.walk(root):
        here = Path(directory)
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS
                         and not (release and here == root and d == "outputs"))
        for name in dirs:
            path = here / name
            if path.is_symlink():
                raise ReleaseError(f"Unexpected directory symlink: {path.relative_to(root)}")
        for name in sorted(names):
            if name == ".DS_Store" or name.endswith((".pyc", ".pyo")):
                continue
            if release and here == root and name in {"SHA256SUMS", ".git"}:
                continue
            path = here / name
            if path.is_symlink():
                raise ReleaseError(f"Unexpected file symlink: {path.relative_to(root)}")
            files[path.relative_to(root).as_posix()] = path
    return files


def load_catalog() -> tuple[dict, Path]:
    if not CATALOG.is_file():
        raise ReleaseError("Missing experiments/catalog.json")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    if catalog.get("schema_version", catalog.get("version")) != 1:
        raise ReleaseError("Unsupported experiment catalog version")
    workspace = contained(ROOT, catalog["path_base"])
    if not workspace.is_dir():
        raise ReleaseError(f"Missing curated workspace: {catalog['path_base']}")
    seen = set()
    for experiment in catalog["experiments"]:
        identifier = experiment["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier) or identifier in seen:
            raise ReleaseError(f"Invalid or duplicate experiment ID: {identifier}")
        seen.add(identifier)
        for relative in experiment.get("records", []):
            path = contained(workspace, relative)
            if not path.exists():
                raise ReleaseError(f"{identifier}: missing records {relative}")
            if path.is_dir() and not any(p.is_file() for p in path.rglob("*")):
                raise ReleaseError(f"{identifier}: empty records directory {relative}")
        for analysis in experiment.get("analysis", []):
            if analysis["kind"] not in {"saved_records", "numerical_validation", "optimization"}:
                raise ReleaseError(f"{identifier}: invalid analysis kind {analysis['kind']}")
            path = contained(workspace, analysis["script"])
            if not path.is_file():
                raise ReleaseError(f"{identifier}: missing analysis {analysis['script']}")
            if not all(isinstance(arg, str) for arg in analysis.get("args", [])):
                raise ReleaseError(f"{identifier}: analysis arguments must be strings")
        for rerun in experiment.get("rerun", []):
            relative = rerun["script"] if isinstance(rerun, dict) else rerun
            if not contained(workspace, relative).is_file():
                raise ReleaseError(f"{identifier}: missing rerun script {relative}")
    for figure in catalog.get("figures", []):
        if not contained(ROOT, figure["file"]).is_file():
            raise ReleaseError(f"Missing published figure {figure['number']}: {figure['file']}")
        if not contained(workspace, figure["builder"]).is_file():
            raise ReleaseError(f"Missing builder for figure {figure['number']}: {figure['builder']}")
    return catalog, workspace


def verify_public_content(files: dict[str, Path], catalog: dict) -> None:
    """Keep manuscript documents out of the public experiment distribution."""
    figures = {figure["file"] for figure in catalog.get("figures", [])}
    for relative, path in files.items():
        name = path.name.lower()
        if relative.split("/")[0].lower() in {"paper", "manuscript"}:
            raise ReleaseError(f"Manuscript files are excluded from this public repository: {relative}")
        if path.suffix.lower() == ".pdf" and relative not in figures:
            raise ReleaseError(f"Only catalogued figure PDFs may be distributed: {relative}")
        if name in {"paper.tex", "supplement.tex"} or name.startswith("supplement-sm"):
            raise ReleaseError(f"Manuscript sources are excluded from this public repository: {relative}")
        if path.suffix.lower() == ".tex":
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in (r"\documentclass", r"\begin{document}", r"\end{document}")):
                raise ReleaseError(f"Only numerical table fragments may be distributed as TeX: {relative}")
        if path.suffix.lower() in {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}:
            raise ReleaseError(f"Distribute inspectable experiment files, not bundled archives: {relative}")


def verify_release() -> tuple[dict, Path]:
    manifest = ROOT / "SHA256SUMS"
    if not manifest.is_file():
        raise ReleaseError("Missing SHA256SUMS; use a complete repository checkout or release.")
    expected = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ReleaseError(f"Malformed checksum entry at SHA256SUMS:{number}")
        sha, relative = match.groups()
        contained(ROOT, relative)
        if relative in expected:
            raise ReleaseError(f"Duplicate checksum entry: {relative}")
        expected[relative] = sha
    actual = inventory(ROOT, release=True)
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = [relative for relative in sorted(expected.keys() & actual.keys())
               if digest(actual[relative]) != expected[relative]]
    problems = []
    for title, paths in (("Missing", missing), ("Unlisted", extra), ("Changed", changed)):
        if paths:
            problems.append(f"{title} files ({len(paths)}):\n  " + "\n  ".join(paths[:20])
                            + ("\n  ..." if len(paths) > 20 else ""))
    if problems:
        raise ReleaseError("\n".join(problems))
    catalog, workspace = load_catalog()
    verify_public_content(actual, catalog)
    for script, _ in FIGURES.values():
        if not contained(workspace, f"{FIGURE_DIR}/{script}").is_file():
            raise ReleaseError(f"Missing publication figure builder: {script}")
    print(f"Verified {len(expected):,} file hashes and {len(catalog['experiments'])} experiment entries.")
    return catalog, workspace


def select_experiments(catalog: dict, selections: list[str]) -> list[dict]:
    if not selections or selections == ["all"]:
        return catalog["experiments"]
    by_id = {item["id"]: item for item in catalog["experiments"]}
    unknown = set(selections) - by_id.keys()
    if unknown:
        raise ReleaseError(f"Unknown experiment: {', '.join(sorted(unknown))}. Use 'list' for IDs.")
    return [by_id[item] for item in dict.fromkeys(selections)]


def show_experiments(catalog: dict, selections: list[str]) -> None:
    for item in select_experiments(catalog, selections):
        print(f"{item['id']}: {item['title']}")
        paper = item.get("paper", {})
        for field in ("sections", "tables", "figures"):
            if paper.get(field):
                print(f"  {field.capitalize()}: {', '.join(map(str, paper[field]))}")
        if selections and selections != ["all"]:
            for relative in item.get("records", []):
                print(f"  Records: {catalog['path_base']}/{relative}")
            for analysis in item.get("analysis", []):
                command = shlex.join(["python", f"{catalog['path_base']}/{analysis['script']}",
                                      *analysis.get("args", [])])
                print(f"  {analysis['kind']}: {command}")
                if analysis.get("notes"):
                    print(f"    {analysis['notes']}")
            for rerun in item.get("rerun", []):
                relative = rerun["script"] if isinstance(rerun, dict) else rerun
                if isinstance(rerun, dict):
                    command = shlex.join(["python", f"{catalog['path_base']}/{relative}",
                                          *rerun.get("args", [])])
                    print(f"  Fresh run entry point: {command}")
                    if rerun.get("notes"):
                        print(f"    {rerun['notes']}")
                else:
                    print(f"  Fresh run entry point: {catalog['path_base']}/{relative}")
            if item.get("notes"):
                print(f"  Note: {item['notes']}")
        print()
    if not selections:
        print("Use 'python reproduce.py list ID' for records and scripts for one experiment.")


def figure_steps(selections: list[str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    selected = []
    for key in selections or ["all"]:
        if key in FIGURE_GROUPS:
            selected.extend(FIGURE_GROUPS[key])
        elif key in FIGURES:
            selected.append(key)
        else:
            choices = ", ".join([*FIGURE_GROUPS, *FIGURES])
            raise ReleaseError(f"Unknown figure selection {key!r}. Choose: {choices}")
    selected = list(dict.fromkeys(selected))
    steps = [(f"{FIGURE_DIR}/{FIGURES[key][0]}", []) for key in selected]
    stems = [stem for key in selected for stem in FIGURES[key][1]]
    return steps, stems


def table_steps(catalog: dict, selections: list[str]) -> list[tuple[str, list[str]]]:
    steps, seen = [], set()
    for item in select_experiments(catalog, selections):
        for analysis in item.get("analysis", []):
            if analysis["kind"] != "saved_records" or analysis.get("purpose") == "figure":
                continue
            args = analysis.get("args", [])
            identity = (analysis["script"], tuple(args))
            if identity not in seen:
                seen.add(identity)
                steps.append((analysis["script"], args))
    if not steps:
        raise ReleaseError("No saved-record table builder for this selection. Use 'list ID' for its numerical checks.")
    return steps


def rebuild(kind: str, selections: list[str]) -> None:
    catalog, original = verify_release()
    if kind == "figures":
        steps, stems = figure_steps(selections)
    else:
        steps, stems = table_steps(catalog, selections), []
    output = contained(ROOT, f"outputs/{kind}")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    print(f"Running {len(steps)} saved-record builder(s) in a temporary workspace.")
    with tempfile.TemporaryDirectory(prefix=".reproduce-", dir=ROOT / "outputs") as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(original, workspace, ignore=shutil.ignore_patterns(*IGNORED_DIRS, "*.pyc"))
        before = {name: (path.stat().st_mtime_ns, path.stat().st_size)
                  for name, path in inventory(workspace).items()}
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                           MPLCONFIGDIR=str(Path(temporary) / "matplotlib"))
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[variable] = "1"
        for index, (relative, args) in enumerate(steps, 1):
            script = contained(workspace, relative)
            if not script.is_file():
                raise ReleaseError(f"Missing required builder: {relative}")
            logfile = logs / f"{index:02d}-{script.stem}.txt"
            print(f"[{index}/{len(steps)}] {relative}", flush=True)
            with logfile.open("w", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, str(script), *args],
                                        cwd=script.parent, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                tail = "\n".join(logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
                raise ReleaseError(f"Builder failed ({result.returncode}); see {logfile.relative_to(ROOT)}\n{tail}")
        after = inventory(workspace)
        changed = [name for name, path in after.items()
                   if before.get(name) != (path.stat().st_mtime_ns, path.stat().st_size)]
        exported = []
        if kind == "figures":
            for stem in stems:
                for extension in ("pdf", "svg", "png"):
                    relative = f"{FIGURE_DIR}/{stem}.{extension}"
                    path = contained(workspace, relative)
                    if not path.is_file():
                        raise ReleaseError(f"Builder did not produce expected figure: {relative}")
                    shutil.copy2(path, output / path.name)
                    exported.append(path.name)
            # Retain numerical samples and fresh provenance/layout audits as well.
            for name in changed:
                if name.startswith(FIGURE_DIR + "/") and name.endswith((".json", ".npz")):
                    path = after[name]
                    shutil.copy2(path, output / path.name)
                    exported.append(path.name)
        else:
            for name in changed:
                path = after[name]
                destination = contained(output / "workspace", name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                exported.append("workspace/" + name)
        (output / "last-run.json").write_text(json.dumps({
            "command": kind, "selections": selections or ["all"],
            "python": sys.version, "steps": [{"script": s, "args": a} for s, a in steps],
            "outputs": sorted(exported), "stored_records_modified": False,
        }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(exported)} generated files and builder logs in {output.relative_to(ROOT)}/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")
    listing = commands.add_parser("list", help="Browse the paper's experiment inventory")
    listing.add_argument("selection", nargs="*", metavar="ID")
    commands.add_parser("check", help="Verify every supplied file and catalog path (no dependencies)")
    figures = commands.add_parser("figures", help="Rebuild publication figures into outputs/figures/")
    figures.add_argument("selection", nargs="*", metavar="GROUP",
                         help="all (default), main, supplement, or an individual builder")
    tables = commands.add_parser("tables", help="Rebuild summaries/table fragments into outputs/tables/")
    tables.add_argument("selection", nargs="*", metavar="ID", help="Experiment ID(s), or all (default)")
    args = parser.parse_args()
    try:
        if args.command is None:
            parser.print_help()
        elif args.command == "check":
            verify_release()
        elif args.command == "list":
            catalog, _ = load_catalog()
            show_experiments(catalog, args.selection)
        else:
            rebuild(args.command, args.selection)
    except (ReleaseError, OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
