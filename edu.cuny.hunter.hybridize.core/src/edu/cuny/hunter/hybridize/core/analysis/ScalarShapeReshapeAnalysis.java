package edu.cuny.hunter.hybridize.core.analysis;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.ShapeOperandRank;
import com.ibm.wala.cast.python.ml.client.ShapeOperandRank.Rank;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;

/**
 * The scalar-shape-reshape analysis behind the {@link PreconditionFailure#RESHAPES_WITH_SCALAR_SHAPE} safety precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1170): a transitive walk over the call graph deciding whether a
 * function, or anything it calls, calls {@code tf.reshape} with a shape operand that may be rank 0, such as {@code tf.reshape(t, -1)}.
 * Eagerly, TensorFlow's {@code Reshape} kernel accepts a scalar shape; in a graph, its shape inference requires a rank-1 shape operand, so
 * the converted function raises {@code ValueError: Shape must be rank 1 but is rank 0} on its first trace.
 * <p>
 * Both facts come from the engine. {@link TensorFlowApis} names the TensorFlow APIs a call may invoke, and {@link ShapeOperandRank} gives
 * the rank of a {@code tf.reshape} call's shape operand. Only {@link Rank#MAY_BE_RANK_0} refuses. An operand of {@link Rank#UNKNOWN} rank,
 * such as one behind a starred argument, converts as it did before this check.
 * <p>
 * The walk is the shared one of {@link Util#reachesInstruction(CallGraph, CGNode, java.util.function.BiPredicate)}, as for
 * {@link EagerOnlyCallAnalysis}.
 */
class ScalarShapeReshapeAnalysis {

	/** The fully-qualified name of {@code tf.reshape}, as the engine names it ({@link TensorFlowApis}). */
	private static final String RESHAPE_API = "tensorflow.reshape";

	/** The call graph, whose successors the walk follows. */
	private final CallGraph callGraph;

	/** The call-graph builder, which the engine asks for the APIs a call invokes and for a shape operand's values. */
	private final PythonSSAPropagationCallGraphBuilder builder;

	/** The solved tensor-type analysis, which gives a tensor shape operand's rank. */
	private final TensorTypeAnalysis tensorTypeAnalysis;

	/**
	 * Creates the analysis.
	 *
	 * @param callGraph The call graph.
	 * @param builder The call-graph builder.
	 * @param tensorTypeAnalysis The solved tensor-type analysis.
	 */
	ScalarShapeReshapeAnalysis(CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder, TensorTypeAnalysis tensorTypeAnalysis) {
		this.callGraph = callGraph;
		this.builder = builder;
		this.tensorTypeAnalysis = tensorTypeAnalysis;
	}

	/**
	 * True iff {@code node}, transitively over its call-graph successors, calls {@code tf.reshape} with a shape operand that may be rank 0.
	 *
	 * @param node The call-graph node to check.
	 * @return True iff a reshape by a shape that may be a scalar is reachable from {@code node}.
	 */
	boolean reshapesWithScalarShape(CGNode node) {
		return Util.reachesInstruction(this.callGraph, node,
				(n, instruction) -> instruction instanceof PythonInvokeInstruction invoke && this.reshapesWithScalarShape(n, invoke));
	}

	/**
	 * True iff {@code invoke} may invoke {@code tf.reshape} and the engine reports that its shape operand may be rank 0.
	 *
	 * @param node The node whose IR contains {@code invoke}.
	 * @param invoke The call.
	 * @return True iff {@code invoke} may reshape by a scalar shape.
	 */
	private boolean reshapesWithScalarShape(CGNode node, PythonInvokeInstruction invoke) {
		return TensorFlowApis.of(this.builder, node, invoke).contains(RESHAPE_API)
				&& ShapeOperandRank.ofReshape(this.builder, this.tensorTypeAnalysis, node, invoke) == Rank.MAY_BE_RANK_0;
	}
}
