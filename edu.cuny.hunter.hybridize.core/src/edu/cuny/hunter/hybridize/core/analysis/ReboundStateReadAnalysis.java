package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.modref.DelegatingExtendedHeapModel;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.ipa.modref.ModRef;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.Iterator2Iterable;

import edu.cuny.hunter.hybridize.core.wala.ml.PythonModRefWithBuiltinFunctions;

/**
 * The analysis behind the precondition of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1076: whether a function
 * reads state that another function rebinds between calls. Measured on TensorFlow 2.9.3, a method that reads a tensor from a field of
 * {@code self}, converted on its own, returns the first call's state once another method has rebound the field: the first trace captures
 * the tensor the field held, and later calls reuse the trace.
 * <p>
 * The reads are those of the function's body and of every node it reaches in the call graph, which a trace of the function runs as well. A
 * read counts when the value it defines is typed as a tensor by the tensor-type analysis, which is the only source of tensor-ness here. The
 * fields a read reads, and the fields a body writes, are the pointer keys {@link ModRef} gives for its instructions, so a read and a write
 * are of the same state exactly when they are the same instance key and field. A write counts when it is made outside the reader's closure,
 * whose own writes are the side-effect precondition's, and outside {@code __init__} and {@code build}, which set state up once: a field set
 * only there is read live. A {@code tf.Variable} updated with {@code assign} is not rebound, since {@code assign} writes the variable, not
 * the field holding it.
 * <p>
 * The elements of a list, tuple, dictionary, or set are not read as state: {@code tf.function} passes a container argument's tensors in
 * afresh on each call, and the code building a container writes its elements as fields. A container held in a field is missed with them,
 * though its element reads are stale too once another function rebuilds it. Objects are told apart by allocation site, so instances
 * allocated at one site are one object here: a getter of an object created afresh on each call is refused, though no instance it reads is
 * ever rebound between two of its reads.
 */
class ReboundStateReadAnalysis {

	private final CallGraph callGraph;

	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	private final Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys;

	private final ModRef<InstanceKey> modRef = new PythonModRefWithBuiltinFunctions();

	private final ExtendedHeapModel heapModel;

	/**
	 * Creates the analysis.
	 *
	 * @param callGraph The call graph, walked from the reader to its callees and over the whole program for the writers.
	 * @param pointerAnalysis The pointer analysis, whose heap model names the fields read and written.
	 * @param tensorTypedKeys The pointer keys the tensor-type analysis types as tensors (see {@link Util#computeTensorTypedOrigins}).
	 */
	ReboundStateReadAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
		this.tensorTypedKeys = tensorTypedKeys;
		// The Python mod visitor requires the pipeline's own AstHeapModel; wrap only a heap model that isn't already extended.
		this.heapModel = pointerAnalysis.getHeapModel() instanceof ExtendedHeapModel ehm ? ehm
				: new DelegatingExtendedHeapModel(pointerAnalysis.getHeapModel());
	}

	/**
	 * Whether the function of {@code nodes} reads a tensor-typed field that another function rebinds.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @return {@code TRUE} if a tensor-typed field the function reads is written by another function outside {@code __init__} and
	 *         {@code build}, {@code FALSE} otherwise.
	 */
	Boolean readsReboundState(Set<CGNode> nodes) {
		Set<CGNode> closure = new HashSet<>();
		Deque<CGNode> worklist = new ArrayDeque<>(nodes);
		closure.addAll(nodes);

		while (!worklist.isEmpty()) {
			CGNode current = worklist.remove();

			for (Iterator<CGNode> succNodes = this.callGraph.getSuccNodes(current); succNodes.hasNext();) {
				CGNode next = succNodes.next();

				if (closure.add(next))
					worklist.add(next);
			}
		}

		Set<PointerKey> tensorFieldsRead = new HashSet<>();

		for (CGNode node : closure)
			tensorFieldsRead.addAll(this.tensorFieldsRead(node));

		if (tensorFieldsRead.isEmpty())
			return Boolean.FALSE;

		for (CGNode node : this.callGraph)
			if (!closure.contains(node) && !isSetUp(node) && !Collections
					.disjoint(Function.getDirectMod(node, this.modRef, this.heapModel, this.pointerAnalysis), tensorFieldsRead))
				return Boolean.TRUE;

		return Boolean.FALSE;
	}

	/**
	 * The fields of objects that {@code node}'s own instructions read where the value read is typed as a tensor.
	 *
	 * @param node A node the function reaches.
	 * @return The instance-field pointer keys read into a tensor-typed value.
	 */
	private Set<PointerKey> tensorFieldsRead(CGNode node) {
		Set<PointerKey> ret = new HashSet<>();
		IR ir = node.getIR();

		if (ir == null)
			return ret;

		for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions())) {
			if (!this.definesTensor(node, instruction))
				continue;

			for (PointerKey pointerKey : this.modRef.getRef(node, this.heapModel, this.pointerAnalysis, instruction, null))
				if (pointerKey instanceof InstanceFieldPointerKey field
						&& !Util.isContainerType(field.getInstanceKey().concreteType().getReference()))
					ret.add(field);
		}

		return ret;
	}

	/**
	 * True iff {@code instruction} defines a value the tensor-type analysis types as a tensor.
	 *
	 * @param node The node of {@code instruction}.
	 * @param instruction An instruction of {@code node}.
	 * @return True iff one of its defs is tensor-typed.
	 */
	private boolean definesTensor(CGNode node, SSAInstruction instruction) {
		for (int i = 0; i < instruction.getNumberOfDefs(); i++)
			if (this.tensorTypedKeys.containsKey(this.heapModel.getPointerKeyForLocal(node, instruction.getDef(i))))
				return true;

		return false;
	}

	/**
	 * True iff {@code node} is an {@code __init__} or a {@code build} method, which sets state up once.
	 *
	 * @param node A node of the call graph.
	 * @return True iff the node's function is named {@code __init__} or {@code build}.
	 */
	private static boolean isSetUp(CGNode node) {
		String name = node.getMethod().getReference().getDeclaringClass().getName().toString();
		return name.endsWith("/__init__") || name.endsWith("/build");
	}
}
