#!/usr/bin/env python3
"""Re-evaluate all new PB endpoints/transfers with independent refined quadrature."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=HERE/'source/final-experiments'
VALIDATOR=ROOT/'adaptive-desingularization/final-experiments/pb/validate.py'

def main():
    records=sorted((HERE/'accuracy').glob('pb_*.json'))
    assert len(records)==60,'Independent quadrature checks require the complete PB NC/C cohort.'
    staging=HERE/'pb-validation-input';staging.mkdir(exist_ok=True)
    hashes={}
    for path in records:
        r=json.loads(path.read_text())
        name=f"d{r['dimension']}_g{r['geometry_id']}_seed{r['seed']}_{r['method']}.json"
        target=staging/name
        if not target.exists():target.symlink_to(path.resolve())
        hashes[str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert all(hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h
                   for p,h in r['reproduction']['source_hashes'].items())
    provenance=dict(dimensions=[5,10,20],geometries=[0,1],seeds=[10,15],methods=['stable','adaptive'],
        source_hashes={'pb/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (SOURCE/'pb').glob('*.py')})
    (staging/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    sys.path.insert(0,str(SOURCE/'pb'))
    spec=importlib.util.spec_from_file_location('independent_pb_validator',VALIDATOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # The original validator reads source hashes relative to its own directory.
    # Redirect that read to the executed frozen source copies; numerical audit code is unchanged.
    module.__file__=str(SOURCE/'pb/validate.py')
    result=module.validate(staging,coarse=14,fine=16)
    result.update(validator_path=str(VALIDATOR.relative_to(ROOT)),
                  validator_sha256=hashlib.sha256(VALIDATOR.read_bytes()).hexdigest(),
                  new_input_sha256=hashes,all_recorded_executed_source_hashes_match=True)
    (HERE/'pb-independent-quadrature-validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','groups','kernel_checks','new_input_sha256')},indent=2))
    assert result['passed']

if __name__=='__main__':main()
