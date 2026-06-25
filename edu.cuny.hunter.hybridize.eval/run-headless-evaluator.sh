#!/bin/bash
#
# Headless command-line runner for the Hybridize Functions evaluator (issue 657).
#
# Runs the `edu.cuny.hunter.hybridize.eval.evaluate` application over the open
# Python projects in a workspace, without the IDE, and writes the result CSVs to
# the workspace's working directory (the same outputs as the in-IDE evaluator).
#
# Prerequisites:
#   - The headless evaluator: either the self-contained product
#     (`edu.cuny.hunter.hybridize.eval.product`; set ECLIPSE to its materialized
#     `hybridize-evaluator` launcher), or an Eclipse install containing the eval
#     bundle and its runtime (the ponder-lab PyDev fork, WALA, Ariadne).
#   - A workspace already populated with the subjects as PyDev projects.
#
# Usage:
#   ECLIPSE=/path/to/hybridize-evaluator WORKSPACE=/path/to/workspace ./run-headless-evaluator.sh
#
# The knobs below are this script's defaults for a typical run; the tool itself
# defaults every flag off (it reads them as system properties), so the script
# sets the ones a normal run wants. Override via the environment. PERFORM_CHANGE
# defaults off -- the evaluation applies the transformation only in special cases
# (e.g. the performance evaluation); analysis-only runs leave the subjects
# untouched. Parallelization defaults to off because it is nondeterministic
# (issue 315).
#
set -eu

ECLIPSE="${ECLIPSE:?Set ECLIPSE to the Eclipse launcher of an install containing the eval bundle.}"
WORKSPACE="${WORKSPACE:?Set WORKSPACE to the workspace holding the subjects as PyDev projects.}"

: "${PERFORM_ANALYSIS:=true}"
: "${PERFORM_CHANGE:=false}"
: "${INFER_INPUT_SIGNATURES:=false}"
: "${PROCESS_IN_PARALLEL:=false}"
: "${FOLLOW_TYPE_HINTS:=true}"
: "${SPECULATIVE:=true}"
: "${TEST_ENTRYPOINTS:=true}"
: "${OUTPUT_CALLS:=false}"
: "${MAX_HEAP:=30480m}"

exec "$ECLIPSE" \
	-application edu.cuny.hunter.hybridize.eval.evaluate \
	-data "$WORKSPACE" \
	-consoleLog -nosplash --launcher.suppressErrors \
	-vmargs \
	-Xms1024m "-Xmx${MAX_HEAP}" \
	-Dedu.cuny.hunter.hybridize.eval.performAnalysis="$PERFORM_ANALYSIS" \
	-Dedu.cuny.hunter.hybridize.eval.performChange="$PERFORM_CHANGE" \
	-Dedu.cuny.hunter.hybridize.eval.inferInputSignatures="$INFER_INPUT_SIGNATURES" \
	-Dedu.cuny.hunter.hybridize.eval.processFunctionsInParallel="$PROCESS_IN_PARALLEL" \
	-Dedu.cuny.hunter.hybridize.eval.alwaysFollowTypeHints="$FOLLOW_TYPE_HINTS" \
	-Dedu.cuny.hunter.hybridize.eval.useSpeculativeAnalysis="$SPECULATIVE" \
	-Dedu.cuny.hunter.hybridize.eval.useTestEntrypoints="$TEST_ENTRYPOINTS" \
	-Dedu.cuny.hunter.hybridize.eval.outputCalls="$OUTPUT_CALLS" \
	-Dedu.cuny.hunter.hybridize.eval.alwaysCheckPythonSideEffects=false \
	-Dedu.cuny.hunter.hybridize.eval.alwaysCheckRecursion=false
