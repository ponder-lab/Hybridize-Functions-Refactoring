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
#   ECLIPSE=/path/to/hybridize-evaluator WORKSPACE=/path/to/workspace \
#     PERFORM_ANALYSIS=true ./run-headless-evaluator.sh
#
# The evaluator reads its configuration as JVM system properties, every one
# defaulting to off. This script does not restate those defaults; it forwards
# only the flags you set in the environment and lets the tool default the rest,
# so a useful run sets at least PERFORM_ANALYSIS=true. Recognized knobs:
# PERFORM_ANALYSIS, PERFORM_CHANGE, INFER_INPUT_SIGNATURES, CHECK_SIDE_EFFECTS,
# CHECK_RECURSION, PROCESS_IN_PARALLEL, FOLLOW_TYPE_HINTS, SPECULATIVE,
# TEST_ENTRYPOINTS, OUTPUT_CALLS, PROJECTS.
# PERFORM_CHANGE applies the transformation; leave it off except in special cases
# (e.g. the performance evaluation). PROCESS_IN_PARALLEL is nondeterministic
# (issue 315). PROJECTS is a comma-separated list of project names to evaluate a
# subset; unset evaluates all open Python projects.
#
# JVM arguments (heap, GC, modules) come from the product launcher's own
# configuration; this script appends to them with --launcher.appendVmargs rather
# than restating them. Set MAX_HEAP to override the heap when pointing ECLIPSE at
# a non-product Eclipse.
#
set -eu

ECLIPSE="${ECLIPSE:?Set ECLIPSE to the headless evaluator launcher, e.g. the product hybridize-evaluator binary.}"
WORKSPACE="${WORKSPACE:?Set WORKSPACE to the workspace holding the subjects as PyDev projects.}"

exec "$ECLIPSE" \
	-application edu.cuny.hunter.hybridize.eval.evaluate \
	-data "$WORKSPACE" \
	-consoleLog -nosplash --launcher.suppressErrors --launcher.appendVmargs \
	-vmargs \
	${MAX_HEAP+-Xmx"$MAX_HEAP"} \
	${PERFORM_ANALYSIS+-Dedu.cuny.hunter.hybridize.eval.performAnalysis="$PERFORM_ANALYSIS"} \
	${PERFORM_CHANGE+-Dedu.cuny.hunter.hybridize.eval.performChange="$PERFORM_CHANGE"} \
	${INFER_INPUT_SIGNATURES+-Dedu.cuny.hunter.hybridize.eval.inferInputSignatures="$INFER_INPUT_SIGNATURES"} \
	${CHECK_SIDE_EFFECTS+-Dedu.cuny.hunter.hybridize.eval.alwaysCheckPythonSideEffects="$CHECK_SIDE_EFFECTS"} \
	${CHECK_RECURSION+-Dedu.cuny.hunter.hybridize.eval.alwaysCheckRecursion="$CHECK_RECURSION"} \
	${PROCESS_IN_PARALLEL+-Dedu.cuny.hunter.hybridize.eval.processFunctionsInParallel="$PROCESS_IN_PARALLEL"} \
	${FOLLOW_TYPE_HINTS+-Dedu.cuny.hunter.hybridize.eval.alwaysFollowTypeHints="$FOLLOW_TYPE_HINTS"} \
	${SPECULATIVE+-Dedu.cuny.hunter.hybridize.eval.useSpeculativeAnalysis="$SPECULATIVE"} \
	${TEST_ENTRYPOINTS+-Dedu.cuny.hunter.hybridize.eval.useTestEntrypoints="$TEST_ENTRYPOINTS"} \
	${OUTPUT_CALLS+-Dedu.cuny.hunter.hybridize.eval.outputCalls="$OUTPUT_CALLS"} \
	${PROJECTS+-Dedu.cuny.hunter.hybridize.eval.projects="$PROJECTS"}
