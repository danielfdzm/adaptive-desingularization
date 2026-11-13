#!/usr/bin/env python3
"""Check an extracted release with hashes and optional safe CLI/import smoke tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,default=HERE.parents[2])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    root=args.package_root.resolve()
    manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
    for name,record in manifest.items():
        path=root/name
        if not path.is_file():raise RuntimeError(f'Missing packaged file: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError(f'Packaged hash mismatch: {name}')
        if path.stat().st_size!=record['bytes']:
            raise RuntimeError(f'Packaged size mismatch: {name}')
    info=json.loads((root/'PACKAGE_INFO.json').read_text())
    workspace=root/info['workspace']
    campaign=workspace/info['campaign']
    master=json.loads((campaign/'master-summary.json').read_text())
    assert master['complete'] and not master['missing']
    assert (master['found_accuracy_records'],master['found_new_guarded_accuracy_records'],
            master['found_controlled_benchmark_records'],master['fresh_variance_records'])==(258,152,54,90)
    assert master['found_supplementary_records']==116 and master['supplementary']['complete']
    diagnostics=json.loads((campaign/'manuscript-validation-summary.json').read_text())
    assert diagnostics['complete'] and not diagnostics['provisional']
    assert diagnostics['coverage']['first_order_pb']['found']==60
    assert diagnostics['coverage']['directional']['found']==30
    smoke=[]
    if args.smoke:
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
        env['MPLCONFIGDIR']=str(root/'matplotlib-cache')
        entries=['summarize_results.py','summarize_validation.py','package_release.py','compare_independent.py','audit_supplementary.py','first-order/campaign.py',
                 'first-order/run_case.py','directional/src/chain_experiments.py',
                 'directional/run_controlled.py','directional/audit_endpoints.py',
                 'independent/run.py','variance/run_matched.py','supplementary-2d/campaign.py','supplementary-2d/run_case.py']
        for name in entries:
            done=subprocess.run([sys.executable,str(campaign/name),'--help'],cwd=workspace,
                                env=env,text=True,capture_output=True,timeout=90)
            smoke.append(dict(script=name,returncode=done.returncode,
                              help_mentions_usage='usage:' in done.stdout.lower(),stderr=done.stderr))
            if done.returncode or 'usage:' not in done.stdout.lower():
                raise RuntimeError(f'Smoke check failed for {name}: {done.stderr}')
    report=dict(passed=True,manifest_files=len(manifest),record_counts=info['record_counts'],
                complete_master_summary=True,smoke_checks=smoke,
                scope='Hashes, coverage and optional --help imports only; no optimization or PDE reference solve.')
    (root/'PACKAGE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
