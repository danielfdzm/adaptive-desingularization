"""Verify that the derived solver changes only rejected-restart handling."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
old_text = (HERE / "src/chain_solver.py").read_text()
new_text = (HERE / "restart_safe.py").read_text()


def methods(text, name):
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == name)
    return {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}


old = methods(old_text, "Solver")
new = methods(new_text, "RestartSafeSolver")
run = copy.deepcopy(new["run"])
for node in ast.walk(run):
    if isinstance(node, ast.Try):
        node.handlers = [h for h in node.handlers if not (
            isinstance(h.type, ast.Name) and h.type.id == "GuardedRestartReturn")]
run_equal = ast.dump(old["run"], include_attributes=False) == ast.dump(run, include_attributes=False)

original_body = copy.deepcopy(old["restart_geometry"].body[1:])  # omit docstring
new_body = copy.deepcopy(new["restart_geometry"].body[0].body)  # one successful attempt
linear = []
for node in new_body:
    if isinstance(node, ast.Try):
        linear += node.body
    elif isinstance(node, ast.If):
        continue  # only the rejected-objective branch
    else:
        linear.append(node)
for node in ast.walk(ast.Module(body=linear, type_ignores=[])):
    if isinstance(node, ast.Name) and node.id == "objective":
        node.id = "_"  # the frozen method discarded the objective
restart_equal = ast.dump(ast.Module(body=original_body, type_ignores=[]), include_attributes=False) == ast.dump(
    ast.Module(body=linear, type_ignores=[]), include_attributes=False)
frozen = json.loads((HERE / "frozen-source-sha256.json").read_text())
record = {
    "derived_solver_version": "restart-safe-v1",
    "derived_solver_sha256": {"restart_safe.py": hashlib.sha256((HERE / "restart_safe.py").read_bytes()).hexdigest()},
    "frozen_sources_unchanged": all(hashlib.sha256((HERE / "src" / n).read_bytes()).hexdigest() == h for n, h in frozen.items()),
    "run_ast_equal_after_removing_only_new_exception_handler": run_equal,
    "restart_geometry_finite_path_ast_equal": restart_equal,
    "fit_current_finite_path": "Calls frozen fit_current once and returns its result; retries only after InadmissibleChart.",
    "guarded_restart_policy": "Two rejected restart attempts counted as local returns; explicit guarded_restart stop with unchanged physical state and coefficients.",
    "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(HERE / "restart-safe-static-audit.json").write_text(json.dumps(record, indent=2) + "\n")
parts = []
for name in ("fit_current", "restart_geometry", "run"):
    a = ast.get_source_segment(old_text, old[name]).splitlines(keepends=True)
    b = ast.get_source_segment(new_text, new[name]).splitlines(keepends=True)
    parts.extend(difflib.unified_diff(a, b, fromfile="frozen/" + name, tofile="restart-safe/" + name))
(HERE / "restart-safe.diff").write_text("".join(parts))
print(json.dumps(record, indent=2))
assert record["frozen_sources_unchanged"] and run_equal and restart_equal
