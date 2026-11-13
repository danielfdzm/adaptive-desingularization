#!/usr/bin/env python3
"""Summarize saved validation evidence using only the Python standard library.

No quadrature, optimization, or PDE/reference calculations are performed.
Until all 60 first-order and 30 directional endpoints are audited, the output
is explicitly manuscript-validation-summary.partial.json.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    hashes = {}
    cache = {}

    def read(relative, expected_hash=None, as_json=True):
        if relative not in cache:
            cache[relative] = (HERE / relative).read_bytes()
            hashes[relative] = hashlib.sha256(cache[relative]).hexdigest()
        if expected_hash is not None and hashes[relative] != expected_hash:
            raise RuntimeError(f'Input/source hash mismatch: {relative}')
        return json.loads(cache[relative]) if as_json else cache[relative]

    def positive(value, label, zero_allowed=True):
        if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
            raise ValueError(f'Invalid {label}: {value}')
        return value

    def comparisons(identifier, method, quadratic, estimates, reference):
        quadratic = positive(quadratic, 'quadratic gap')
        ref_nl = positive(estimates[reference], 'reference nonlinear gap')
        ref_total = positive(quadratic + ref_nl, 'reference total gap', False)
        result = []
        for rule, value in estimates.items():
            positive(value, 'nonlinear estimate')
            change = abs(value - ref_nl)
            if ref_nl == 0 and change:
                raise ValueError(f'Undefined nonlinear relative change for {identifier}')
            result.append(dict(id=identifier, method=method, rule=rule, reference_rule=reference,
                quadratic=quadratic, nonlinear=value, reference_nonlinear=ref_nl,
                reference_total_gap=ref_total, total_gap_relative_change=change/ref_total,
                nonlinear_relative_change=change/ref_nl if ref_nl else 0.))
        return result

    def maxima(items):
        return {key: max(items, key=lambda r: r[key]) if items else None
                for key in ('total_gap_relative_change', 'nonlinear_relative_change')}

    def ordering(groups, required_methods):
        complete = []; missing = []
        for start, records in sorted(groups.items()):
            if set(records) != set(required_methods):
                missing.append(start)
                continue
            rules = list(next(iter(records.values())))
            orders = {rule: sorted(required_methods, key=lambda m: (records[m][rule], m)) for rule in rules}
            complete.append(dict(start=start, orders=orders,
                unchanged=all(order == orders[rules[0]] for order in orders.values())))
        return dict(complete_starts=len(complete), incomplete_starts=missing,
                    unchanged_starts=sum(r['unchanged'] for r in complete), rows=complete)

    pb = read('first-order/pb-independent-quadrature-validation.json')
    pb_expected = {f'd{d}_g{g}_seed{s}_{m}' for d in (5, 10, 20)
                   for g in (0, 1) for s in range(10, 15) for m in ('stable', 'adaptive')}
    pb_seen = []; pb_comparisons = []; pb_groups = {}
    pb_reference = 'p16_s9201_proposal0.318198051534'
    for row in pb['runs']:
        identifier = row['id']; pb_seen.append(identifier)
        path = f"accuracy/pb_d{row['dimension']}_g{row['geometry_id']}_s{row['seed']}_{row['method']}.json"
        raw = read('first-order/' + path, pb['new_input_sha256'][path])
        for name, digest in raw['reproduction']['source_hashes'].items():
            read('first-order/' + name, digest, as_json=False)
        q = .5 * (row['h1_relative_error'] * raw['target_norm']) ** 2
        estimates = {'training': raw['nonlinear_gap']}
        estimates.update({v['rule']: v['value'] for v in row['final_nonlinear_estimates']})
        pb_comparisons += comparisons(identifier, row['method'], q, estimates, pb_reference)
        start = f"d{row['dimension']}_g{row['geometry_id']}_seed{row['seed']}"
        pb_groups.setdefault(start, {})[row['method']] = {k: q+v for k, v in estimates.items()}

    directional = read('directional/endpoint-audit.json')
    read('directional/audit_endpoints.py', directional['audit_source_sha256'], as_json=False)
    for name, digest in directional['solver_source_sha256'].items():
        read('directional/src/' + name, digest, as_json=False)
    baseline_manifest = read('historical-baselines/manifest.json')
    dir_expected = {f'higher_d5_g{g}_seed{s}_{m}.json' for g in (0, 1)
                    for s in range(10, 15) for m in ('standard', 'stable', 'adaptive')}
    dir_comparisons = []; dir_groups = {}; metrics = []
    for name, row in sorted(directional['rows'].items()):
        if row['method'] == 'standard':
            raw = read('historical-baselines/' + name, row['input_sha256'])
            if baseline_manifest[name]['sha256'] != row['input_sha256']:
                raise RuntimeError(f'Historical baseline hash mismatch: {name}')
        else:
            raw = read('directional/accuracy-higher/' + name, row['input_sha256'])
        if (raw['method'], raw['geometry'], raw['seed']) != (row['method'], row['geometry'], row['seed']):
            raise RuntimeError(f'Directional identity mismatch: {name}')
        q = row['quadratic']; estimates = row['nonlinear']
        dir_comparisons += comparisons(name, row['method'], q, estimates, 'refined')
        start = f"g{row['geometry']}_seed{row['seed']}"
        dir_groups.setdefault(start, {})[row['method']] = {k: q+v for k, v in estimates.items()}
        metrics.append(dict(id=name, method=row['method'], sample_count=row['basis_refinement_sample_count'],
            sampled_H1_basis_refinement=positive(row['basis_refinement_sampled_H1_relative'], 'sampled H1 change'),
            relative_max_value_basis_refinement=positive(row['basis_refinement_relative_max_value_change'], 'relative value change'),
            absolute_max_value_basis_refinement=positive(row['basis_refinement_max_value_change'], 'absolute value change'),
            coefficient_stationarity=positive(row['coefficient_gradient_inf_normalized'], 'stationarity residual')))

    def coverage(expected, observed, declared_complete):
        found = set(observed)
        return dict(expected=len(expected), found=len(observed), unique_found=len(found),
                    missing=sorted(expected-found), unexpected=sorted(found-expected),
                    audit_declares_complete=declared_complete,
                    complete=declared_complete and len(observed)==len(found) and found==expected)

    covered = dict(first_order_pb=coverage(pb_expected, pb_seen, pb.get('complete', False)),
                   directional=coverage(dir_expected, list(directional['rows']), directional.get('complete', False)))
    complete = all(c['complete'] for c in covered.values())
    all_items = pb_comparisons + dir_comparisons
    metric_maxima = {k: max(metrics, key=lambda r: r[k]) if metrics else None for k in
                    ('sampled_H1_basis_refinement', 'relative_max_value_basis_refinement',
                     'absolute_max_value_basis_refinement', 'coefficient_stationarity')}
    ncc = [r for r in metrics if r['method'] != 'standard']
    metric_maxima['nc_c_coefficient_stationarity'] = max(ncc, key=lambda r: r['coefficient_stationarity']) if ncc else None
    read(Path(__file__).name, as_json=False)
    summary = dict(schema_version=1, complete=complete, provisional=not complete, coverage=covered,
        arithmetic_definition='For each endpoint fix q as its analytic quadratic H1 gap. Compare training and all saved refinement nonlinear estimates N against the designated finest-rule value Nref: total relative change = abs(N-Nref)/(q+Nref), nonlinear relative change = abs(N-Nref)/Nref. This avoids subtracting nearly equal total gaps. Include the reference itself (zero deviation).',
        references=dict(first_order_pb=dict(rule=pb_reference, power=16, seed=9201, proposal_sigma='sigma/sqrt(2)',
                        compared_rules='training plus all five saved refinement rules'),
                        directional=dict(rule='refined', power=16, seed='91000+geometry', proposal_sigma='sigma',
                        compared_rules=['training','independent','refined','narrow'])),
        cohorts=dict(first_order_pb=dict(comparison_count=len(pb_comparisons), maxima=maxima(pb_comparisons),
                         method_ordering=ordering(pb_groups, ('stable','adaptive'))),
                     directional=dict(comparison_count=len(dir_comparisons), maxima=maxima(dir_comparisons),
                         method_ordering=ordering(dir_groups, ('standard','stable','adaptive')))),
        combined_maxima=maxima(all_items), directional_metric_maxima=metric_maxima,
        directional_metrics=metrics,
        basis_refinement_scope='Directional endpoint diagnostics on 1024 fixed Sobol-mixture nodes with positive weights, comparing values and gradients after doubling internal quadrature orders. The sampled H1 quantity is not a certified whole-space H1 bound.',
        stationarity_scope='Directional endpoints only: infinity norm of the coefficient gradient on training quadrature, divided by the squared target H1 norm. The PB validation does not supply this residual.',
        checks=dict(all_recorded_input_and_source_hashes_verified=True,
                    pb_validation_passed=pb['passed'],
                    all_directional_final_charts_admissible=all(r['all_final_charts_admissible'] for r in directional['rows'].values())),
        caveats=['Refinement differences are empirical quadrature checks, not continuum error certificates.',
                 'Rules with a common scramble seed are nested, not mutually independent replications.',
                 'First-order coverage is60 NC/C endpoints; directional coverage is30 S/NC/C endpoints, including10 historical S records.',
                 'Incomplete directional coverage yields provisional maxima and an explicit .partial output.'],
        input_hashes=hashes, summary_generator_sha256=hashes[Path(__file__).name])
    name = 'manuscript-validation-summary' + ('' if complete else '.partial') + '.json'
    target = HERE/name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)
    print(json.dumps(dict(output=name, complete=complete, coverage=covered,
                         combined_maxima=summary['combined_maxima'],
                         directional_metric_maxima=metric_maxima), indent=2))


if __name__ == '__main__':
    main()
