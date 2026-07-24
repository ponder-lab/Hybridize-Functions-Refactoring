# Evidence-Population Baselines

Per-subject baselines of the consumer-visible evidence populations from the reference-configuration headless-evaluator run, recorded so release-scale regressions in the analysis surface as diffs instead of going unnoticed ([wala/ML#767](https://github.com/wala/ML/issues/767); the motivating loss sat undetected for thirteen releases, [wala/ML#765](https://github.com/wala/ML/issues/765)).

## Layout

`<subject>/<evidence file>`: the subject's rows of that run output, normalized (header preserved, data rows sorted and deduplicated). The baselined evidence files are `statuses.csv` and `parameters.csv`; volatile outputs (timings, `results.csv`) are deliberately not baselined.

## Protocol

- Diff a run against the baselines with [`../diff-evidence-baselines.py`](../diff-evidence-baselines.py), pointing `--run` at the evaluator's working directory. Only subjects present in the run are compared, so a single-subject smoke run (`PROJECTS=NLPGNN`) diffs that subject alone.
- **Cadence**: the NLPGNN smoke cell once per release day against the day's final release; the full six-subject diff at every measurement session.
- **Additions** fold into the baseline via `--update`, riding an ordinary pull request with a one-line rationale in the body.
- **Losses block**: the script exits nonzero on any lost row, and a baseline update removing rows cannot merge without citing its disposition (an issue link) in the pull request body.
- The reference configuration is the one recorded on [wala/ML#765](https://github.com/wala/ML/issues/765): `PERFORM_ANALYSIS`, `TEST_ENTRYPOINTS`, `FOLLOW_TYPE_HINTS`, and `SPECULATIVE` set to `true`, every `CHECK_*` knob `false`.
