#!/bin/bash
#
# audit-barren.sh — list every hybridization candidate the tool computes as performing no
# tensor computation, for a human accuracy pass. A barren flag on a function that actually
# performs a tensor op is a false positive (issue 709); this surfaces the set to review.
#
# Runs the headless evaluator (run-headless-evaluator.sh, its sibling) with
# alwaysCheckTensorComputation on, so functions.csv's `tensor computation` column is
# populated for every function rather than only the P1/P6 candidates the transformation
# consults. It then lists the functions that have a tensor parameter yet perform no tensor
# computation, grouped by subject, with source locations.
#
# Prerequisites are those of run-headless-evaluator.sh: a built headless evaluator (ECLIPSE)
# and a workspace holding the subjects as PyDev projects (WORKSPACE).
#
# Usage:
#   ECLIPSE=/path/to/hybridize-evaluator WORKSPACE=/path/to/workspace \
#     [PROJECTS=A,B] [OUTDIR=/path/to/results] ./audit-barren.sh
#
# PROJECTS restricts the run to a subset (comma-separated project names); unset audits all
# open Python projects. OUTDIR selects where the evaluator's CSVs land (default ./barren-audit).
#
set -eu

ECLIPSE="${ECLIPSE:?Set ECLIPSE to the headless evaluator launcher, e.g. the product hybridize-evaluator binary.}"
WORKSPACE="${WORKSPACE:?Set WORKSPACE to the workspace holding the subjects as PyDev projects.}"

HERE="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${OUTDIR:-$PWD/barren-audit}"
mkdir -p "$OUTDIR"

# The reference analysis configuration, plus the tensor-computation check forced on every
# function. RUNNER writes its CSVs into the current directory, so run from OUTDIR.
( cd "$OUTDIR" && env \
	ECLIPSE="$ECLIPSE" WORKSPACE="$WORKSPACE" \
	PERFORM_ANALYSIS=true CHECK_TENSOR_COMPUTATION=true \
	FOLLOW_TYPE_HINTS=true SPECULATIVE=true TEST_ENTRYPOINTS=true \
	${PROJECTS+PROJECTS="$PROJECTS"} \
	"$HERE/run-headless-evaluator.sh" )

python3 - "$OUTDIR/functions.csv" <<'PY'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))


def value(row, column):
	return row.get(column, "").strip().lower()


barren = [r for r in rows
          if value(r, "tensor parameter") == "true" and value(r, "tensor computation") == "false"]
barren.sort(key=lambda r: (r["subject"], r.get("relative path", ""), r["function"]))

print("Barren candidates (tensor parameter=true, tensor computation=false): %d\n" % len(barren))

subject = None
for r in barren:
	if r["subject"] != subject:
		subject = r["subject"]
		print("== %s ==" % subject)
	print("  %-42s %s" % (r["function"], r.get("relative path", "")))
PY
