package edu.cuny.hunter.hybridize.core.analysis;

import java.util.HashSet;
import java.util.Set;
import java.util.function.BiPredicate;
import java.util.function.Predicate;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.ContainerElements;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
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
 * container elements, transitively, through the element keys the engine lists for each container. It follows nothing else: an arbitrary
 * object cannot be returned from {@code tf.function} at all. A named tuple is flattened like a tuple, but Ariadne types its instance as its
 * class rather than as a container, so a variable inside one is missed.
 * <p>
 * The return side asks what the function's own body returns, and the argument side what its callers pass, examining the arguments at each
 * call site of the function's call-graph predecessors. Across returns and call sites the quantifier is existential: one return of a
 * variable, or one call passing a spec, is enough to change what the decorated function does. Ignorance is {@code null} rather than
 * {@code FALSE}, the polarity of every sibling safety check: a node or a caller without an IR, or a non-Python invoke, leaves the verdict
 * undetermined.
 * <p>
 * The same return-side walk answers the precondition of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1117: whether
 * a function returns a Python primitive, which the boundary converts to a tensor just the same. There the value reached is a primitive
 * constant rather than an allocation of a given type, and booleans count, since a returned {@code bool} becomes a tensor too. The
 * caller-side walk also asks whether any call site uses the result, since a result every caller discards changes nothing it can see.
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
		return this.returns(nodes, hasTypeIn(UNCROSSABLE_RETURN_TYPE_NAMES));
	}

	/**
	 * Whether the body of {@code nodes} may return a Python primitive, directly or as a container element, per
	 * {@link Function#isPrimitiveValue(InstanceKey, boolean)} with booleans counted (issue 1117): a primitive constant, or a Python scalar
	 * of unknown value (wala/ML#1148).
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @return {@code TRUE} if some return reaches a primitive, {@code FALSE} if none does, or {@code null} if a node has no IR.
	 */
	Boolean returnsPrimitive(Set<CGNode> nodes) {
		return this.returns(nodes, instanceKey -> Function.isPrimitiveValue(instanceKey, false));
	}

	/**
	 * Whether the body of {@code nodes} returns, directly or as a container element, an instance satisfying {@code target}.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @param target The predicate a returned instance is tested against.
	 * @return {@code TRUE} if some return reaches such an instance, {@code FALSE} if none does, or {@code null} if a node has no IR.
	 */
	private Boolean returns(Set<CGNode> nodes, Predicate<InstanceKey> target) {
		for (CGNode node : nodes) {
			IR ir = node.getIR();

			if (ir == null)
				return null;

			for (SSAInstruction instruction : ir.getInstructions())
				if (instruction instanceof SSAReturnInstruction ret && ret.getResult() >= 0 && this.reaches(node, ret.getResult(), target))
					return Boolean.TRUE;
		}

		return Boolean.FALSE;
	}

	/**
	 * The predicate an allocation of one of the given types satisfies.
	 *
	 * @param typeNames The type names in question.
	 * @return True of an {@link InstanceKey} whose concrete type is named in {@code typeNames}.
	 */
	private static Predicate<InstanceKey> hasTypeIn(Set<String> typeNames) {
		return instanceKey -> typeNames.contains(instanceKey.concreteType().getReference().getName().toString());
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
		return anyCallSite(nodes, callGraph, (predecessor, invoke) -> {
			// Positional slot 0 is the callee itself; the arguments start at 1.
			for (int slot = 1; slot < invoke.getNumberOfPositionalParameters(); slot++)
				if (this.reaches(predecessor, invoke.getUse(slot), hasTypeIn(UNCROSSABLE_ARGUMENT_TYPE_NAMES)))
					return true;

			for (String keyword : invoke.getKeywords())
				if (this.reaches(predecessor, invoke.getUse(keyword), hasTypeIn(UNCROSSABLE_ARGUMENT_TYPE_NAMES)))
					return true;

			return false;
		});
	}

	/**
	 * Whether some call site of {@code nodes} uses the call's result, per the caller's {@link com.ibm.wala.ssa.DefUse} (issue 1117). A call
	 * made as a statement defines a result nothing uses, so the type the function returns cannot matter to it.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @param callGraph The call graph, queried in the caller direction.
	 * @return {@code TRUE} if some call site's result has a use, {@code FALSE} if no examined call site's does, or {@code null} if a caller
	 *         could not be examined at all.
	 */
	static Boolean usesResult(Set<CGNode> nodes, CallGraph callGraph) {
		return anyCallSite(nodes, callGraph, (predecessor, invoke) -> predecessor.getDU().getNumberOfUses(invoke.getDef()) > 0);
	}

	/**
	 * Whether some call site of {@code nodes} satisfies {@code test}.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @param callGraph The call graph, queried in the caller direction.
	 * @param test The predicate a call site, given by its caller node and invoke instruction, is tested against.
	 * @return {@code TRUE} if some call site satisfies {@code test}, {@code FALSE} if no examined call site does, or {@code null} if a
	 *         caller has no IR or a call site is not a Python invoke.
	 */
	private static Boolean anyCallSite(Set<CGNode> nodes, CallGraph callGraph, BiPredicate<CGNode, PythonInvokeInstruction> test) {
		for (CGNode node : nodes)
			for (CGNode predecessor : Iterator2Iterable.make(callGraph.getPredNodes(node))) {
				IR ir = predecessor.getIR();

				if (ir == null)
					return null;

				for (CallSiteReference site : Iterator2Iterable.make(callGraph.getPossibleSites(predecessor, node)))
					for (SSAAbstractInvokeInstruction instruction : ir.getCalls(site)) {
						if (!(instruction instanceof PythonInvokeInstruction invoke))
							return null;

						if (test.test(predecessor, invoke))
							return Boolean.TRUE;
					}
			}

		return Boolean.FALSE;
	}

	/**
	 * True iff {@code value} in {@code node}'s frame may hold, directly or as a container element, an instance satisfying {@code target}.
	 */
	private boolean reaches(CGNode node, int value, Predicate<InstanceKey> target) {
		PointerKey pointerKey = this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, value);
		Set<InstanceKey> seen = new HashSet<>();

		for (InstanceKey instanceKey : this.pointerAnalysis.getPointsToSet(pointerKey))
			if (this.reaches(instanceKey, target, seen))
				return true;

		return false;
	}

	private boolean reaches(InstanceKey instanceKey, Predicate<InstanceKey> target, Set<InstanceKey> seen) {
		if (!seen.add(instanceKey))
			return false;

		if (target.test(instanceKey))
			return true;

		if (!Util.isContainerType(instanceKey.concreteType().getReference()))
			return false;

		// A container's elements are the element keys the engine lists, as in `Parameter`'s tensor-container walk. These include the
		// channel a concatenation or a repetition stores into, which the container's catalog doesn't name.
		for (PointerKey element : ContainerElements.of(this.builder, instanceKey).allElements())
			for (InstanceKey elementValue : this.pointerAnalysis.getPointsToSet(element))
				if (this.reaches(elementValue, target, seen))
					return true;

		return false;
	}
}
