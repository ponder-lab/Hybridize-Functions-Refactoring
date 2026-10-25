package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Collections;
import java.util.Set;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;

/**
 * The graph-only-call analysis behind the {@link PreconditionFailure#HAS_GRAPH_ONLY_CALLS} safety precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1171): a transitive walk over the call graph deciding whether a
 * function, or anything it calls, calls an API that works only while a graph is being built. It is the hybrid-to-eager counterpart of
 * {@link EagerOnlyCallAnalysis}: removing the {@code tf.function} decorator from such a function runs the call eagerly, where it raises or
 * returns {@code None} gradients.
 * <p>
 * The engine names the APIs a call may invoke ({@link TensorFlowApis}); which of them are graph-only is this analysis's policy,
 * {@link #GRAPH_ONLY_APIS}. A call is never matched on its spelling, so an alias of {@code tf.gradients} is caught and a program method
 * named {@code get_gradients} is not.
 * <p>
 * The walk is the shared one of {@link Util#reachesInstruction(CallGraph, CGNode, java.util.function.BiPredicate)}, as for
 * {@link EagerOnlyCallAnalysis}.
 * <p>
 * Only a function known to be hybrid is checked. A function whose hybridity is undetermined is unchecked by design: no hybrid-to-eager rule
 * converts it, so there is nothing to refuse.
 */
class GraphOnlyCallAnalysis {

	/**
	 * The fully-qualified names, as the engine names them ({@link TensorFlowApis}), of the APIs that work only while a graph is being
	 * built. Measured on TensorFlow 2.9.3, each fails when called eagerly: {@code tf.gradients}, {@code tf.compat.v1.gradients} and
	 * {@code tf.keras.backend.gradients} raise {@code RuntimeError: tf.gradients is not supported when eager execution is enabled}, and an
	 * optimizer's {@code get_gradients} and {@code get_updates}, which compute their gradients through {@code tf.gradients}, raise
	 * {@code ValueError: Variable ... has None for gradient}. The engine does not model {@code tf.compat.v1}, so it names a call of
	 * {@code tf.compat.v1.gradients} by its attribute chain, which is listed here as its own name.
	 */
	private static final Set<String> GRAPH_ONLY_APIS = Set.of("tensorflow.gradients", "tensorflow.compat.v1.gradients",
			"tensorflow.keras.backend.gradients", "tensorflow.keras.optimizers.Optimizer.get_gradients",
			"tensorflow.keras.optimizers.Optimizer.get_updates");

	/** The call graph, whose successors the walk follows. */
	private final CallGraph callGraph;

	/** The call-graph builder, which the engine asks for the APIs a call invokes. */
	private final PythonSSAPropagationCallGraphBuilder builder;

	/**
	 * Creates the analysis.
	 *
	 * @param callGraph The call graph.
	 * @param builder The call-graph builder.
	 */
	GraphOnlyCallAnalysis(CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder) {
		this.callGraph = callGraph;
		this.builder = builder;
	}

	/**
	 * True iff {@code node}, transitively over its call-graph successors, calls a graph-only API ({@link #GRAPH_ONLY_APIS}).
	 *
	 * @param node The call-graph node to check.
	 * @return True iff a call of a graph-only API is reachable from {@code node}.
	 */
	boolean callsGraphOnlyApi(CGNode node) {
		return Util.reachesInstruction(this.callGraph, node, (n, instruction) -> instruction instanceof PythonInvokeInstruction invoke
				&& !Collections.disjoint(TensorFlowApis.of(this.builder, n, invoke), GRAPH_ONLY_APIS));
	}
}
