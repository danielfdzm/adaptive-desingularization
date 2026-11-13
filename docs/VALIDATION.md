# Reorganization validation

Checked on November 13, 2026 against the November 12 paper and
supplement retained privately, using Python 3.14.0 and the versions in `requirements.txt`.

| Check | Result |
| --- | --- |
| Experiment inventory | 22 reported experimental and validation groups mapped to the manuscript |
| Publication figures | All 7 builders succeeded and produced all 14 included figures |
| Main figure previews | All 3 regenerated PNGs are byte-identical to the supplied previews |
| Sampled solution data | PB field arrays and prescribed-forcing curve arrays match the original arrays exactly |
| Saved-record table analyses | All 9 builders succeeded; 30 generated analysis files |
| Main recovery table | All 5 rows reproduced from all 700 matched NC/C endpoints |
| Enlarged-cohort summaries | Per-run CSVs are byte-identical and summary JSON agrees with the original |
| Source and record preservation | Every retained original file matches its source SHA-256 |
| Reproduction interface | All 5 integrity, manuscript-exclusion, path-containment, and output-isolation tests pass |
| Reader-facing documentation | All relative file links resolve |
| Public contents | Exactly 14 figure PDFs; no manuscript PDFs, full TeX documents, or bundled archives |

The main cohort audit checks every expected method, seed, geometry, and
dimension: 800 amended independent-recovery records, 900 PB records
including the explicitly reported evaluator controls, 348 reversible-method
comparison runs, and 360 forcing-budget runs. The earlier protocol outcomes,
reported controls, and failures remain available in their indexed groups.

These checks rebuild outputs and verify coverage; they do not rerun the
full optimization campaigns. The original numerical records remain the
evidence for the paper. PDF creation metadata can change when rebuilding;
the publication PDFs supplied under `figures/` are preserved.

Run `python reproduce.py check` to verify the delivered file inventory,
`python reproduce.py figures` to rebuild figures, and
`python reproduce.py tables` to rebuild the saved-record summaries. The
[reproduction guide](REPRODUCING.md) explains the individual commands.
