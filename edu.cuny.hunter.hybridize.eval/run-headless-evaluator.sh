#!/bin/bash
#
# Headless command-line runner for the Hybridize Functions evaluator (issue 657).
#
# Runs the `edu.cuny.hunter.hybridize.eval.evaluate` application over the open
# Python projects in a workspace, without the IDE, and writes the result CSVs to
# the workspace's working directory (the same outputs as the in-IDE evaluator).
#
# Prerequisites:
#   - An Eclipse installation that contains the Hybridize eval bundle and its
#     runtime (the ponder-lab PyDev fork, WALA, Ariadne) -- e.g. the development
#     Eclipse, or a product built to include `edu.cuny.hunter.hybridize.eval`.
#     Packaging a self-contained product so no pre-existing Eclipse is required
#     is tracked separately.
#   - A workspace already populated with the subjects as PyDev projects.
#
# Usage:
#   ECLIPSE=/path/to/eclipse WORKSPACE=/path/to/workspace ./run-headless-evaluator.sh
#
# Configuration knobs mirror the IDE launch and can be overridden via the
# environment (defaults shown below). Parallelization defaults to off because it
# is nondeterministic (issue 315).
#
set -eu

ECLIPSE="${ECLIPSE:?Set ECLIPSE to the Eclipse launcher of an install containing the eval bundle.}"
WORKSPACE="${WORKSPACE:?Set WORKSPACE to the workspace holding the subjects as PyDev projects.}"

: "${PERFORM_ANALYSIS:=true}"
: "${PERFORM_CHANGE:=true}"
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
	-consoleLog -nosplash \
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
