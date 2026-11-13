# Amendment 4 to the Phase 3 pre-registration (2026-11-11T14:11Z)

Written before any timing run (timing/results/ did not exist).

## Reason

The host is not idle. Besides the cohort runs it serves interactive applications: at
15:09 local time the 1-, 5- and 15-minute load averages were 77, 96 and 74 on 8 cores, and
the S workers of cohort B received about 40-45% of a core each. No other numerical job of
the revision sessions will run during the timing window (confirmed by the Phase-1 and
Phase-2 sessions), but wall-clock times can still be inflated by contention.

## Changes

* Unchanged: the pre-registered measure (end-to-end seconds reported by each runner; median
  over repetitions and starts; range of per-start medians), five serial repetitions, one
  numerical thread, rotated method order, exact reproduction of endpoint error and fits.
* Added fields per timed run: child_cpu_seconds (user + system CPU time of the whole run
  process, including interpreter start and imports, from getrusage(RUSAGE_CHILDREN)), and
  load1_before / load1_after (1-minute load average). CPU time is reported as a secondary
  measure next to the wall-clock times, with the load range.
* Methods: NC, C and C_d (amendments 1 and 3: C is the t-chart method, NC and C_d use the
  exact pair evaluator in the first-order PB and full-span rows).
* The campaign refuses to resume with changed sources and records whether the sources were
  unchanged at the end.
