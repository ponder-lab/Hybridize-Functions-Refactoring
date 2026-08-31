#!/bin/bash
#
# Headless command-line runner for the Hybridize Functions evaluator (issue 657).
#
# Runs the `edu.cuny.hunter.hybridize.eval.evaluate` application over the open
# Python projects in a workspace, without the IDE, and writes the result CSVs to
# the directory named by OUTDIR (the same outputs as the in-IDE evaluator).
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
#     OUTDIR=/path/to/results PERFORM_ANALYSIS=true ./run-headless-evaluator.sh
#
# ECLIPSE, WORKSPACE, and OUTDIR are required and have no defaults. OUTDIR must be
# absolute (or `.`); it is created if absent and is the working directory of the
# run, so the evaluator writes its CSVs and its own README.md manifest there, and
# pointing it at a checkout overwrites that checkout's README.md.
#
# The remaining knobs are optional. The evaluator reads its configuration as JVM
# system properties, every one defaulting to off. This script does not restate
# those defaults; it forwards only the flags you set in the environment and lets the tool default the rest,
# so a useful run sets at least PERFORM_ANALYSIS=true. Recognized knobs:
# PERFORM_ANALYSIS, PERFORM_CHANGE, INFER_INPUT_SIGNATURES, CHECK_SIDE_EFFECTS,
# CHECK_RECURSION, CHECK_TENSOR_COMPUTATION, CHECK_EAGER_ONLY_CALLS, CHECK_NUMPY_CALLS, CHECK_STATIC_SHAPE_READS, CHECK_STALE_VARIABLE_READS, CHECK_TENSOR_ITERATION, PROCESS_IN_PARALLEL,
# FOLLOW_TYPE_HINTS, SPECULATIVE,
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

# The output directory is required and has no default. The evaluator writes its CSVs and
# its own README.md manifest into its working directory, so defaulting that to the caller's
# working directory lets a run started inside a checkout overwrite that checkout's tracked
# README.md -- silently, since nothing fails and the result looks like an ordinary edit.
# The check lives here rather than in the callers because a direct invocation is exactly what
# debugging the evaluator produces, and it deliberately tests nothing about what the working
# directory contains: any such rule still clobbers the layouts it does not recognize.
# Pass OUTDIR=. to write to the current directory deliberately.
OUTDIR="${OUTDIR:?Set OUTDIR to the directory for the CSVs and the evaluation README. There is no default: the evaluator overwrites a README.md in its working directory, so a run started inside a checkout destroys the README of that checkout. Pass OUTDIR=. to write to the current directory deliberately.}"
# A relative OUTDIR is refused, because this cd composes with the caller's. A caller that has
# already cd'd into its output directory and then passes a relative path applies it twice and
# nests the results a level deeper than intended, silently: the run succeeds, and whatever reads
# the CSVs afterward finds an empty directory and reports lost rows rather than a misplaced run.
# `.` is exempt because applying it twice is the same as applying it once.
case "$OUTDIR" in
	.) OUTDIR="$PWD" ;;
	/*) ;;
	*)
		echo "run-headless-evaluator.sh: OUTDIR must be an absolute path (or \`.\`), not \`$OUTDIR\`." >&2
		echo "  This script changes into OUTDIR, so a relative path composes with any cd the caller" >&2
		echo "  already made and nests the output a level deeper. Use \`\$PWD/$OUTDIR\` instead." >&2
		exit 1
		;;
esac
mkdir -p "$OUTDIR"
cd "$OUTDIR"
echo "run-headless-evaluator.sh: writing evaluator output to $PWD" >&2

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
	${CHECK_TENSOR_COMPUTATION+-Dedu.cuny.hunter.hybridize.eval.alwaysCheckTensorComputation="$CHECK_TENSOR_COMPUTATION"} \
	${CHECK_EAGER_ONLY_CALLS+-Dedu.cuny.hunter.hybridize.eval.alwaysCheckEagerOnlyCalls="$CHECK_EAGER_ONLY_CALLS"} \
	${CHECK_NUMPY_CALLS+-Dedu.cuny.hunter.hybridize.eval.alwaysCheckNumpyCalls="$CHECK_NUMPY_CALLS"} \
	${CHECK_STATIC_SHAPE_READS+-Dedu.cuny.hunter.hybridize.eval.alwaysCheckStaticShapeReads="$CHECK_STATIC_SHAPE_READS"} \
	${CHECK_STALE_VARIABLE_READS+-Dedu.cuny.hunter.hybridize.eval.alwaysCheckStaleVariableReads="$CHECK_STALE_VARIABLE_READS"} \
	${CHECK_TENSOR_ITERATION+-Dedu.cuny.hunter.hybridize.eval.alwaysCheckTensorIteration="$CHECK_TENSOR_ITERATION"} \
	${PROCESS_IN_PARALLEL+-Dedu.cuny.hunter.hybridize.eval.processFunctionsInParallel="$PROCESS_IN_PARALLEL"} \
	${FOLLOW_TYPE_HINTS+-Dedu.cuny.hunter.hybridize.eval.alwaysFollowTypeHints="$FOLLOW_TYPE_HINTS"} \
	${SPECULATIVE+-Dedu.cuny.hunter.hybridize.eval.useSpeculativeAnalysis="$SPECULATIVE"} \
	${TEST_ENTRYPOINTS+-Dedu.cuny.hunter.hybridize.eval.useTestEntrypoints="$TEST_ENTRYPOINTS"} \
	${OUTPUT_CALLS+-Dedu.cuny.hunter.hybridize.eval.outputCalls="$OUTPUT_CALLS"} \
	${PROJECTS+-Dedu.cuny.hunter.hybridize.eval.projects="$PROJECTS"}
