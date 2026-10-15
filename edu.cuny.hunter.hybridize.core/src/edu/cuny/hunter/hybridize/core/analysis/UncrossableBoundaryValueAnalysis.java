package edu.cuny.hunter.hybridize.core.analysis;

import java.util.HashSet;
import java.util.Set;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Iterator2Iterable;

/**
 * The analysis behind the precondition of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1075: whether a value
 * crossing a function's {@code tf.function} boundary does not cross it unchanged. Measured on TensorFlow 2.9.3:
 * <ul>
 * <li>A returned {@code tf.Variable} is converted to a tensor holding its value. Nothing raises at the boundary, but the variable's
 * identity and mutability are lost, so a caller that assigns to it, or compares its {@code ref()}, fails.
 * <li>A returned {@code tf.TensorArray} is converted to a tensor, so a caller that writes to it fails.
 * <li>A {@code tf.TensorSpec} passed as an argument makes the call itself raise a {@code ValueError}.
 * </ul>
 * Each is decided from the points-to sets Ariadne computes, by the type of the allocation reached: its summaries allocate a variable, a
 * tensor array, and a tensor spec under types of their own, distinct from the module objects that construct them. A value inside a returned
 * or passed list, tuple, or dictionary crosses the boundary too, since {@code tf.function} flattens these structures, so the walk follows
 * container elements, transitively, through each container's object catalog. It follows nothing else: an arbitrary object cannot be
 * returned from {@code tf.function} at all. A named tuple is flattened like a tuple, but Ariadne types its instance as its class rather
 * than as a container, so a variable inside one is missed.
 * <p>
 * The return side asks what the function's own body returns, and the argument side what its callers pass, walking the call graph in the
 * caller direction like {@link KerasSymbolicArgumentAnalysis}. Across returns and call sites the quantifier is existential: one return of a
 * variable, or one call passing a spec, is enough to change what the decorated function does. Ignorance is {@code null} rather than
 * {@code FALSE}, the polarity of every sibling safety check: a node or a caller without an IR, or a non-Python invoke, leaves the verdict
 * undetermined.
 */
class UncrossableBoundaryValueAnalysis {

	/**
	 * The allocation types of the returned values the boundary converts to tensors: what Ariadne's {@code tf.Variable} and
	 * {@code tf.TensorArray} summaries allocate.
	 */
	private static final Set<String> UNCROSSABLE_RETURN_TYPE_NAMES = Set.of("Ltensorflow/python/ops/variables/Variable",
			"Ltensorflow/python/ops/tensor_array_ops/TensorArray");

	/** The allocation type of the argument values that make the call raise: what Ariadne's {@code tf.TensorSpec} summary allocates. */
	private static final Set<String> UNCROSSABLE_ARGUMENT_TYPE_NAMES = Set.of("Ltensorflow/framework/TensorSpec");

	private final PythonSSAPropagationCallGraphBuilder builder;

	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	UncrossableBoundaryValueAnalysis(PythonSSAPropagationCallGraphBuilder builder) {
		this.builder = builder;
		this.pointerAnalysis = builder.getPointerAnalysis();
	}

	/**
	 * Whether the body of {@code nodes} returns a value the boundary converts to a tensor.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @return {@code TRUE} if some return reaches a variable or a tensor array, {@code FALSE} if none does, or {@code null} if a node has
	 *         no IR.
	 */
	Boolean returnsUncrossableValue(Set<CGNode> nodes) {
		for (CGNode node : nodes) {
			IR ir = node.getIR();

			if (ir == null)
				return null;

			for (SSAInstruction instruction : ir.getInstructions())
				if (instruction instanceof SSAReturnInstruction ret && ret.getResult() >= 0
						&& this.reaches(node, ret.getResult(), UNCROSSABLE_RETURN_TYPE_NAMES))
					return Boolean.TRUE;
		}

		return Boolean.FALSE;
	}

	/**
	 * Whether some call site of {@code nodes} passes a value that makes the call raise.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @param callGraph The call graph, queried in the caller direction.
	 * @return {@code TRUE} if some call site passes a tensor spec, {@code FALSE} if every examined call site passes none, or {@code null}
	 *         if a caller could not be examined at all.
	 */
	Boolean passesUncrossableArgument(Set<CGNode> nodes, CallGraph callGraph) {
		for (CGNode node : nodes)
			for (CGNode predecessor : Iterator2Iterable.make(callGraph.getPredNodes(node))) {
				IR ir = predecessor.getIR();

				if (ir == null)
					return null;

				for (CallSiteReference site : Iterator2Iterable.make(callGraph.getPossibleSites(predecessor, node)))
					for (SSAAbstractInvokeInstruction instruction : ir.getCalls(site)) {
						if (!(instruction instanceof PythonInvokeInstruction invoke))
							return null;

						// Positional slot 0 is the callee itself; the arguments start at 1.
						for (int slot = 1; slot < invoke.getNumberOfPositionalParameters(); slot++)
							if (this.reaches(predecessor, invoke.getUse(slot), UNCROSSABLE_ARGUMENT_TYPE_NAMES))
								return Boolean.TRUE;

						for (String keyword : invoke.getKeywords())
							if (this.reaches(predecessor, invoke.getUse(keyword), UNCROSSABLE_ARGUMENT_TYPE_NAMES))
								return Boolean.TRUE;
					}
			}

		return Boolean.FALSE;
	}

	/** True iff {@code value} in {@code node}'s frame may hold, directly or as a container element, an allocation of one of the types. */
	private boolean reaches(CGNode node, int value, Set<String> typeNames) {
		PointerKey pointerKey = this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, value);
		Set<InstanceKey> seen = new HashSet<>();

		for (InstanceKey instanceKey : this.pointerAnalysis.getPointsToSet(pointerKey))
			if (this.reaches(instanceKey, typeNames, seen))
				return true;

		return false;
	}

	private boolean reaches(InstanceKey instanceKey, Set<String> typeNames, Set<InstanceKey> seen) {
		if (!seen.add(instanceKey))
			return false;

		TypeReference reference = instanceKey.concreteType().getReference();

		if (typeNames.contains(reference.getName().toString()))
			return true;

		if (!Util.isContainerType(reference))
			return false;

		// A container's elements are the fields its object catalog names, as in `Parameter`'s tensor-container walk.
		PointerKey catalogPointerKey = ((AstPointerKeyFactory) this.builder.getPointerKeyFactory())
				.getPointerKeyForObjectCatalog(instanceKey);

		for (InstanceKey catalogKey : this.pointerAnalysis.getPointsToSet(catalogPointerKey))
			if (catalogKey instanceof ConstantKey<?> constantKey && constantKey.getValue() != null) {
				IField field = instanceKey.concreteType().getField(Atom.findOrCreateAsciiAtom(constantKey.getValue().toString()));

				if (field == null)
					continue;

				for (InstanceKey element : this.pointerAnalysis
						.getPointsToSet(this.builder.getPointerKeyForInstanceField(instanceKey, field)))
					if (this.reaches(element, typeNames, seen))
						return true;
			}

		return false;
	}
}
