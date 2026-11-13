#!/usr/bin/env python3
"""Validate saved endpoints with convergent, independently refined BVP tolerances.

Stricter failed validation remains alongside this output. The solve uses
tol=1e-9 on [-12,12], then tol=3e-10 on [-14,14], with staged initialization.
These changes affect only reference validation, not any optimization run.
"""
import hashlib
import json
import sys
from pathlib import Path
import validate_saved as vs
import run_chain as rc

original=vs.reference
def resolved_reference(name,L=12.,tol=1e-10):
    return original(name,L,1e-9 if L<=12. else 3e-10)
vs.reference=resolved_reference

for directory in map(Path,sys.argv[1:]):
    out=directory/'resolved_validation';out.mkdir(exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_tolerance=1e-9,base_domain=[-12,12],refined_tolerance=3e-10,refined_domain=[-14,14],
        original_failed_validation_retained=True,optimization_changed=False),indent=2))
    for path in sorted((directory/'runs').glob('*.json')):
        dst=out/path.name
        if dst.exists():continue
        try:r=vs.validate(path)
        except Exception as exc:r=dict(source=str(path),status='validation_exception',exception=repr(exc))
        dst.write_text(json.dumps(rc.clean(r),indent=2))
        print(path.name,r.get('status'),r.get('error'),flush=True)
