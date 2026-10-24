package edu.cuny.hunter.hybridize.core.analysis;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.Iterator2Iterable;
import com.ibm.wala.util.collections.Pair;

/**
 * The scalar-reshape-shape analysis behind the {@link PreconditionFailure#RESHAPES_WITH_SCALAR_SHAPE} safety precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1170): a transitive walk over the call graph deciding whether a
 * function, or anything it reaches, calls {@code tf.reshape} with a rank-0 shape operand, such as {@code tf.reshape(t, -1)}. Eagerly,
 * TensorFlow's {@code Reshape} kernel accepts a scalar shape; in a graph, {@code Reshape}'s shape inference requires a rank-1 shape
 * operand, so the converted function raises {@code ValueError: Shape must be rank 1 but is rank 0} on its first trace, with a bare
 * decorator or a signature alike.
 * <p>
 * Both facts come from the engine. The op is identified by {@link TensorFlowApis}, which names the TensorFlow APIs a call may invoke,
 * following the call graph's resolved callees, so an alias of {@code tf.reshape} is identified as the op it is, and a user function named
 * like it is not. The shape operand is a scalar when its points-to set holds a Python number, such as the literal {@code -1}, or when the
 * tensor-type analysis types it as a rank-0 tensor, such as {@code tf.constant(-1)}. Either may hold in only some contexts, and a may-fact
 * refuses: the check is conservative, so a shape operand that is a scalar on some path and a vector on another is refused too.
 * <p>
 * Only program bodies are scanned, and the walk passes through TensorFlow library nodes to reach program callbacks, mirroring
 * {@link EagerOnlyCallAnalysis}.
 */
class ScalarReshapeShapeAnalysis {

	/** The fully-qualified name of {@code tf.reshape}, as the engine names it ({@link TensorFlowApis}). */
	private static final String RESHAPE_OP_FQN = "tensorflow.reshape";

	/** The position of {@code tf.reshape}'s shape operand, from 0 for the first argument after the callee. */
	private static final int SHAPE_POSITION = 1;

	/** The keyword of {@code tf.reshape}'s shape operand. */
	private static final String SHAPE_KEYWORD = "shape";

	/** The call graph, whose successors the walk follows. */
	private final CallGraph callGraph;

	/** The call-graph builder, which the engine asks to name the TensorFlow APIs a call invokes. */
	private final PythonSSAPropagationCallGraphBuilder builder;

	/** The pointer analysis, whose points-to sets tell a Python number apart. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/** The ranks the tensor-type analysis gives each local, by node and value number; a rank is absent where a type's dimensions are ⊤. */
	private final Map<CGNode, Map<Integer, Set<Integer>>> ranks;

	/**
	 * Creates the analysis.
	 *
	 * @param callGraph The call graph.
	 * @param builder The call-graph builder, which the engine asks to name the TensorFlow APIs a call invokes.
	 * @param tensorTypeAnalysis The tensor-type analysis, which gives a tensor's rank.
	 */
	ScalarReshapeShapeAnalysis(CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder, TensorTypeAnalysis tensorTypeAnalysis) {
		this.callGraph = callGraph;
		this.builder = builder;
		this.pointerAnalysis = builder.getPointerAnalysis();
		this.ranks = indexRanks(tensorTypeAnalysis);
	}

	/**
	 * Indexes the ranks {@code tensorTypeAnalysis} gives its local pointer keys by node and value number, once.
	 *
	 * @param tensorTypeAnalysis The tensor-type analysis.
	 * @return Each local's ranks, by node and value number.
	 */
	private static Map<CGNode, Map<Integer, Set<Integer>>> indexRanks(TensorTypeAnalysis tensorTypeAnalysis) {
		Map<CGNode, Map<Integer, Set<Integer>>> index = new HashMap<>();

		for (Pair<PointerKey, TensorVariable> pair : tensorTypeAnalysis)
			if (pair.fst instanceof LocalPointerKey local && pair.snd != null)
				for (TensorType type : pair.snd.getTypes())
					// A None member is no tensor, and ⊤ dimensions give no rank.
					if (!type.isNone() && type.getDims() != null)
						index.computeIfAbsent(local.getNode(), _ -> new HashMap<>())
								.computeIfAbsent(local.getValueNumber(), _ -> new HashSet<>()).add(type.getDims().size());

		return index;
	}

	/**
	 * True iff {@code node}, transitively over its call-graph successors, calls {@code tf.reshape} with a rank-0 shape operand.
	 *
	 * @param node The call-graph node to check.
	 * @return True iff a reshape by a scalar shape is reachable from {@code node}.
	 */
	boolean reshapesWithScalarShape(CGNode node) {
		return this.reshapesWithScalarShape(node, new HashSet<>());
	}

	private boolean reshapesWithScalarShape(CGNode node, Set<CGNode> seen) {
		if (!seen.add(node))
			return false;

		if (!Util.isTensorFlowNode(node)) {
			IR ir = node.getIR();

			if (ir != null)
				for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
					if (instruction instanceof PythonInvokeInstruction invoke && this.isScalarReshape(node, invoke))
						return true;
		}

		for (Iterator<CGNode> succNodes = this.callGraph.getSuccNodes(node); succNodes.hasNext();)
			if (this.reshapesWithScalarShape(succNodes.next(), seen))
				return true;

		return false;
	}

	/**
	 * True iff {@code invoke} may invoke {@code tf.reshape} and its shape operand may be a scalar.
	 *
	 * @param node The node whose IR holds {@code invoke}.
	 * @param invoke The call.
	 * @return True iff {@code invoke} may reshape by a rank-0 shape.
	 */
	private boolean isScalarReshape(CGNode node, PythonInvokeInstruction invoke) {
		if (!TensorFlowApis.of(this.builder, node, invoke).contains(RESHAPE_OP_FQN))
			return false;

		// The callee is the invoke's first use, so the argument at position i is use i + 1.
		if (SHAPE_POSITION + 1 < invoke.getNumberOfPositionalParameters() && this.isScalar(node, invoke.getUse(SHAPE_POSITION + 1)))
			return true;

		return invoke.getKeywords().contains(SHAPE_KEYWORD) && this.isScalar(node, invoke.getUse(SHAPE_KEYWORD));
	}

	/**
	 * True iff the local {@code valueNumber} of {@code node} may be a scalar: its points-to set holds a Python number, or the tensor-type
	 * analysis types it as a rank-0 tensor.
	 *
	 * @param node The node defining the local.
	 * @param valueNumber The local's value number.
	 * @return True iff the local may be a Python number or a rank-0 tensor.
	 */
	private boolean isScalar(CGNode node, int valueNumber) {
		if (valueNumber < 0)
			return false;

		if (this.ranks.getOrDefault(node, Map.of()).getOrDefault(valueNumber, Set.of()).contains(0))
			return true;

		PointerKey key = this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, valueNumber);

		for (InstanceKey instanceKey : this.pointerAnalysis.getPointsToSet(key))
			if (instanceKey instanceof ConstantKey<?> constantKey && constantKey.getValue() instanceof Number)
				return true;

		return false;
	}
}
