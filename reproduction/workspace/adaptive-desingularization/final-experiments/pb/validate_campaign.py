#!/usr/bin/env python3
"""Validate all declared datasets, optionally waiting for ongoing runs to finish.

This serial validation worker can run alongside at most two optimization
workers. A completed aggregate is only a readiness marker: audits additionally
check every expected per-run record and all frozen solver hashes.
"""
from __future__ import annotations
import argparse,subprocess,sys,time
from pathlib import Path
from reproduce import studies
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    for name,*_ in studies():
        directory=args.root/name
        if args.wait:
            announced=False
            while not (directory/'results.json').exists():
                if not announced:
                    print(f'Waiting for complete dataset {name}',flush=True);announced=True
                time.sleep(5)
        subprocess.run([sys.executable,str(HERE/'audit.py'),str(directory)],check=True)
        subprocess.run([sys.executable,str(HERE/'validate.py'),str(directory)],check=True)
    subprocess.run([sys.executable,str(HERE/'summarize_validation.py'),'--root',str(args.root)],check=True)
    subprocess.run([sys.executable,str(HERE/'training_comparison.py'),'--root',str(args.root)],check=True)

if __name__=='__main__':main()
