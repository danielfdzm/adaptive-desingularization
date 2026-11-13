#!/usr/bin/env python3
"""Run the declared PB campaign, audits, independent validation, and summaries.

The full campaign has 135 runs. Existing datasets require --resume, which
checks the solver hashes and every numerical run setting before continuing.
"""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def studies():
    out=[('results',[5,10,20],[0,1],[10,15],['standard','stable','adaptive'],12,8128,1.),
         ('calibrated',[10],[0,1],[10,15],['standard','stable','adaptive'],12,8128,100.)]
    for label,power,qseed in [('scramble1',12,18128),('scramble2',12,28128),('refined',13,8128)]:
        out.append(('retrain_calibrated_'+label,[10],[0],[10,11],['standard','stable','adaptive'],power,qseed,100.))
        out.append(('retrain_fixed_'+label,[5,20],[0],[10,11],['adaptive'],power,qseed,1.))
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output-root',type=Path,default=HERE/'reproduction',
                        help='Fresh output root; distributed measured datasets are not overwritten')
    parser.add_argument('--only',nargs='+',help='Optional study directory names')
    parser.add_argument('--skip-validation',action='store_true')
    args=parser.parse_args()
    output_root=args.output_root
    dirs=[]
    for name,dims,geoms,seeds,methods,power,qseed,reaction in studies():
        if args.only and name not in args.only:continue
        output=output_root/name
        cmd=[sys.executable,str(HERE/'run.py'),'--dimensions',*map(str,dims),
             '--geometries',*map(str,geoms),'--seeds',*map(str,seeds),'--methods',*methods,
             '--power',str(power),'--quadrature-seed',str(qseed),'--reaction-strength',str(reaction),
             '--max-fits','1500','--jobs',str(args.jobs),'--output',str(output)]
        if args.resume:cmd.append('--resume')
        subprocess.run(cmd,check=True)
        subprocess.run([sys.executable,str(HERE/'audit.py'),str(output)],check=True)
        if not args.skip_validation:
            subprocess.run([sys.executable,str(HERE/'validate.py'),str(output)],check=True)
        dirs.append(str(output))
    subprocess.run([sys.executable,str(HERE/'report.py'),*dirs,'--output',str(output_root/'report')],check=True)
    for dataset,figure in [('results','fixed-pb-trajectories.pdf'),('calibrated','calibrated-trajectories.pdf')]:
        if any(Path(d).name==dataset for d in dirs):
            subprocess.run([sys.executable,str(HERE/'plot.py'),str(output_root/dataset),
                            '--output',str(output_root/'report'/figure)],check=True)
    if not args.only:
        subprocess.run([sys.executable,str(HERE/'training_comparison.py'),'--root',str(output_root)],check=True)
        if not args.skip_validation:
            subprocess.run([sys.executable,str(HERE/'summarize_validation.py'),'--root',str(output_root)],check=True)

if __name__=='__main__':main()
