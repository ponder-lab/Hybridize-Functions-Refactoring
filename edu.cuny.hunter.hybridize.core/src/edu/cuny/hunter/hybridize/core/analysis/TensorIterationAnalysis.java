package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.Pair;

/**
 * The tensor-iteration analysis behind the symbolic-iteration precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/830): whether a function's body iterates a parameter-derived,
 * tensor-typed value with a Python {@code for}. Eagerly such iteration works (the elements are tensors); under {@code tf.function} tracing,
 * the decorated function's inputs are symbolic, and iterating a symbolic tensor raises {@code OperatorNotAllowedInGraphError} even with
 * AutoGraph converting the loop (runtime-verified on the pinned framework). The scope is deliberately anchored on parameter-derived values:
 * an in-body {@code tf.range} loop is AutoGraph-supported and must not fire, and it is the decoration that makes a parameter symbolic at
 * entry.
 * <p>
 * The scan is body-local: parameter value numbers seed a forward def-use taint, and a {@link EachElementGetInstruction} (the IR form of a
 * Python {@code for}-each) whose iterated object is tainted <em>and</em> tensor-typed by the tensor-type analysis fires. The tensor-typed
 * gate keeps ordinary Python-collection parameters (lists, dicts, dataset objects) out: only a value the analysis types as a tensor can
 * become a symbolic tensor at trace time. Iteration through a builtin that iterates its argument ({@code enumerate}, {@code zip}) fires
 * too: the loop iterates the builtin's result, but the builtin's model iterates the tensor
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1098).
 * <p>
 * The same scan backs the item-assignment precondition (issue 1098): a subscript store ({@code y[i] = v}, augmented forms included) into a
 * parameter-derived, tensor-typed value works eagerly on a NumPy array, but a symbolic tensor has no {@code __setitem__}, so tracing raises
 * a {@code TypeError}.
 */
class TensorIterationAnalysis {

	/** The (node, value number) index of the tensor-type analysis, mirroring {@link NumpyParameterFlowAnalysis}'s index. */
	private final Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex;

	/** The pointer analysis, used to resolve the iterated value's producing callee for the {@code tf.range} exemption. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/** The call graph, used to resolve the builtin a tensor is passed to. */
	private final CallGraph callGraph;

	TensorIterationAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis, TensorTypeAnalysis tensorTypeAnalysis) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
		Map<CGNode, Map<Integer, Set<TensorType>>> index = new HashMap<>();

		for (Pair<PointerKey, TensorVariable> pair : tensorTypeAnalysis)
			if (pair.fst instanceof LocalPointerKey local && pair.snd != null)
				// A None member is no tensor here; it matters only to the signature (wala/ML#1040).
				pair.snd.getTypes().stream().filter(type -> !type.isNone())
						.forEach(type -> index.computeIfAbsent(local.getNode(), n -> new HashMap<>())
								.computeIfAbsent(local.getValueNumber(), v -> new HashSet<>()).add(type));

		this.tensorTypeIndex = index;
	}

	/**
	 * True iff {@code node}'s body iterates a parameter-derived, tensor-typed value, directly or through a builtin that iterates it.
	 *
	 * @param node The call-graph node to scan.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a source.
	 * @return True iff a Python {@code for} iterates a parameter-derived tensor.
	 */
	boolean iteratesTensorParameter(CGNode node, boolean method) {
		return anyTaintedUse(node, method, (use, value) -> {
			DefUse defUse = node.getDU();

			// The tf.range exemption applies regardless of the bound's provenance: `for i in tf.range(tf.shape(x)[0])` is
			// AutoGraph-supported (the loop stages to a while loop), so a range result never fires even when its bound is
			// parameter-derived. The container exemption keeps a Python collection of tensors out: its value carries tensor
			// typing through element dataflow, but iterating a fixed-length collection under tracing is legal (the loop
			// unrolls), so only a tensor proper fires (issue 832; NLPGNN's adjacency-list iteration).
			if (use instanceof EachElementGetInstruction each && each.getUse(0) == value)
				return this.isTensorTyped(node, value) && !this.isRangeResult(node, value, defUse) && !this.pointsToContainer(node, value);

			// A builtin whose model iterates the argument (`enumerate(y)`, `zip(x, y)`): the loop iterates the builtin's result, a
			// list the container exemption keeps out, but the builtin iterates the tensor itself, which raises under tracing (issue
			// 1098). No tf.range exemption here: AutoGraph stages only a loop directly over the range, not one through a builtin.
			// The container exemption still applies, so enumerating a Python list of tensors does not fire.
			if (use instanceof PythonInvokeInstruction invoke)
				return this.isTensorTyped(node, value) && !this.pointsToContainer(node, value) && this.builtinIterates(node, invoke, value);

			return false;
		});
	}

	/**
	 * True iff {@code node}'s body stores into a parameter-derived, tensor-typed value by subscript ({@code y[i] = v}, augmented forms
	 * included), which a symbolic tensor does not support (issue 1098).
	 *
	 * @param node The call-graph node to scan.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a source.
	 * @return True iff a subscript store writes into a parameter-derived tensor.
	 */
	boolean assignsIntoTensorParameter(CGNode node, boolean method) {
		// The container exemption keeps a store into a Python collection of tensors out, which tracing supports. Attribute stores share
		// the IR form (https://github.com/wala/ML/issues/1100), but their member is always a string constant, the attribute's name, so
		// a string-keyed write is treated as an attribute store (legal on a symbolic tensor) and does not fire. This under-refuses a
		// string-keyed subscript store into a tensor, which fails eagerly on a NumPy array anyway. A multi-index store (`y[0, 1] = v`)
		// is keyed by a tuple and fires.
		return anyTaintedUse(node, method,
				(use, value) -> use instanceof PythonPropertyWrite write && write.getObjectRef() == value && this.isTensorTyped(node, value)
						&& !this.pointsToContainer(node, value) && !this.isStringConstant(node, write.getMemberRef()));
	}

	/**
	 * True iff {@code sink} holds for some use of a parameter-derived value in {@code node}'s body. Parameter value numbers seed a forward
	 * def-use taint, and every use of a tainted value is tested before its definitions are tainted in turn.
	 *
	 * @param node The call-graph node to scan.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a source.
	 * @param sink The test, given a use and the tainted value it uses.
	 * @return True iff {@code sink} holds for some tainted use.
	 */
	private static boolean anyTaintedUse(CGNode node, boolean method, TaintedUse sink) {
		IR ir = node.getIR();

		if (ir == null)
			return false;

		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();
		Set<Integer> tainted = new HashSet<>();
		Deque<Integer> worklist = new ArrayDeque<>();

		for (int i = method ? 2 : 1; i < parameters.length; i++) {
			tainted.add(parameters[i]);
			worklist.add(parameters[i]);
		}

		if (tainted.isEmpty())
			return false;

		DefUse defUse = node.getDU();

		while (!worklist.isEmpty()) {
			int value = worklist.pop();

			for (Iterator<SSAInstruction> uses = defUse.getUses(value); uses.hasNext();) {
				SSAInstruction use = uses.next();

				if (sink.test(use, value))
					return true;

				for (int d = 0; d < use.getNumberOfDefs(); d++) {
					int def = use.getDef(d);

					if (tainted.add(def))
						worklist.push(def);
				}
			}
		}

		return false;
	}

	/** A test of a use of a parameter-derived value. */
	@FunctionalInterface
	private interface TaintedUse {

		/**
		 * True iff {@code use} is the use sought.
		 *
		 * @param use The instruction using {@code value}.
		 * @param value The parameter-derived value number.
		 * @return True iff the use is the one sought.
		 */
		boolean test(SSAInstruction use, int value);
	}

	/** True iff the tensor-type analysis types {@code value} in {@code node} as a tensor. */
	private boolean isTensorTyped(CGNode node, int value) {
		return !this.tensorTypeIndex.getOrDefault(node, Map.of()).getOrDefault(value, Set.of()).isEmpty();
	}

	/**
	 * True iff {@code invoke} passes {@code value} positionally to a Python builtin whose model iterates the corresponding parameter (an
	 * {@link EachElementGetInstruction} over it in the builtin's summary), as Ariadne models {@code enumerate} and {@code zip}. The
	 * builtin's own model decides, so a builtin that only stores or returns its argument does not fire. Slots at or after a starred
	 * argument are skipped, since their alignment to the builtin's parameters is unknown.
	 */
	private boolean builtinIterates(CGNode node, PythonInvokeInstruction invoke, int value) {
		int starred = invoke.firstStarredPosition();
		int end = starred == -1 ? invoke.getNumberOfPositionalParameters() : starred;

		for (int slot = 1; slot < end; slot++) {
			if (invoke.getUse(slot) != value)
				continue;

			for (CGNode callee : this.callGraph.getPossibleTargets(node, invoke.getCallSite()))
				if (callee.getMethod().getDeclaringClass().getName().toString().startsWith(Util.BUILTIN_PRELUDE_TYPE_NAME_PREFIX)
						&& iteratesParameter(callee, slot))
					return true;
		}

		return false;
	}

	/** True iff {@code callee}'s IR iterates its parameter at {@code index} with an {@link EachElementGetInstruction}. */
	private static boolean iteratesParameter(CGNode callee, int index) {
		IR ir = callee.getIR();

		if (ir == null)
			return false;

		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();

		if (index >= parameters.length)
			return false;

		for (SSAInstruction instruction : ir.getInstructions())
			if (instruction instanceof EachElementGetInstruction each && each.getUse(0) == parameters[index])
				return true;

		return false;
	}

	/**
	 * True iff {@code value} is a string constant in {@code node}, in its symbol table or by points-to. Used to tell an attribute store
	 * from a subscript store, which Ariadne's IR does not distinguish (https://github.com/wala/ML/issues/1100).
	 */
	private boolean isStringConstant(CGNode node, int value) {
		IR ir = node.getIR();
		return ir.getSymbolTable().isStringConstant(value) || Util.resolveStringConstant(node, value, this.pointerAnalysis) != null;
	}

	/**
	 * True iff {@code value}'s points-to set includes a container allocation per {@link Util#isContainerType}, whose set spans the Python
	 * collections (list, tuple, dict, set) and the {@code enumerate} wrapper: such a value is iterated as a Python collection, which
	 * tracing unrolls, so it must not fire even though element dataflow gives it tensor typing. Including {@code enumerate} is harmless
	 * here: an {@code enumerate} result only reaches this exemption after passing the tensor-typed gate, which an enumerate object does
	 * not.
	 */
	private boolean pointsToContainer(CGNode node, int value) {
		PointerKey pointerKey = this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, value);

		for (InstanceKey instanceKey : this.pointerAnalysis.getPointsToSet(pointerKey))
			if (Util.isContainerType(instanceKey.concreteType().getReference()))
				return true;

		return false;
	}

	/** The AutoGraph-supported iterable producer: a {@code tf.range} result stages to a while loop rather than raising. */
	private static final String RANGE_FQN = "tensorflow.range";

	/** True iff {@code value} is defined by a {@code tf.range} invocation. */
	private boolean isRangeResult(CGNode node, int value, DefUse defUse) {
		SSAInstruction def = defUse.getDef(value);

		if (!(def instanceof PythonInvokeInstruction invoke))
			return false;

		return RANGE_FQN.equals(Util.resolveCalleeFullyQualifiedName(node, invoke.getUse(0), defUse, this.pointerAnalysis));
	}
}
