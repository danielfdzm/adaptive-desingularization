#!/usr/bin/env python3
"""Independent analytic and refinement checks for the objective controls."""
import json
from pathlib import Path
import numpy as np
from run_controls import Problem, exact_check

HERE=Path(__file__).resolve().parent


def main():
    checks=[]
    for family in ('zero_gap','positive_gap','ordinary'):
        problem=Problem(family)
        for beta,t in ((.03,0.),(.03,1e-12),(-.07,.04),(.04,1.1)):
            result=problem.evaluate(beta,t)
            reference=exact_check(family,beta,t,result['a'],result['c'])
            error_difference=abs(np.sqrt(2*result['F'])-reference['error'])
            gradient_difference=float(np.linalg.norm(result['gradient']-np.array([
                reference['gradient_beta'],reference['gradient_t']])))
            assert error_difference<2e-14,(family,beta,t,error_difference)
            assert gradient_difference<2e-13,(family,beta,t,gradient_difference)
            checks.append(dict(family=family,beta=beta,t=t,error_difference=error_difference,
                               gradient_difference=gradient_difference))
    path=HERE/'results.json'
    endpoints=[]
    if path.exists():
        records=json.loads(path.read_text())['records']
        for r in records:
            assert r['t']>=r['t_floor']*.999999999999999
            assert abs(r['error_grid']-r['error'])<2e-13,r
            assert r['refined_objective_difference']<2e-14,r
            assert r['refined_gradient_difference']<2e-13,r
            assert r['coefficient_residual']<2e-13,r
        endpoints=dict(count=len(records),
                       max_error_difference=max(abs(r['error_grid']-r['error']) for r in records),
                       max_refined_objective_difference=max(r['refined_objective_difference'] for r in records),
                       max_refined_gradient_difference=max(r['refined_gradient_difference'] for r in records),
                       max_coefficient_residual=max(r['coefficient_residual'] for r in records))
    (HERE/'validation.json').write_text(json.dumps(dict(analytic_checks=checks,endpoints=endpoints),indent=2)+'\n')
    print(f'Passed {len(checks)} analytic gradient/error checks; endpoints: {endpoints}')


if __name__=='__main__':main()
