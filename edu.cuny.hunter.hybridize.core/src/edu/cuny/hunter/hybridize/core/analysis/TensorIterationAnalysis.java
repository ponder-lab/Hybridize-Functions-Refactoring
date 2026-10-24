package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
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
 * The analysis behind two preconditions on how a function's body uses a parameter-derived, tensor-typed value: whether it iterates the
 * value (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/830), and whether it assigns into the value by subscript
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1098). Both share one scan, described below.
 * <p>
 * The iteration check fires on a Python {@code for} over such a value. Eagerly such iteration works (the elements are tensors); under
 * {@code tf.function} tracing, the decorated function's inputs are symbolic. AutoGraph converts a {@code for} over a symbolic tensor to a
 * {@code tf.while_loop}, which raises or traces to a different result when the body can't be staged: it mutates Python state (appending
 * each element to a local list traces, but the list has one element), changes a loop-carried value's dtype or shape, returns inside the
 * loop, or converts an element to a Python value. A body that only adds each element to a fixed-shape tensor accumulator traces to the
 * eager result. The check doesn't decide stageability, so it refuses every such loop
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1137). The scope is deliberately anchored on parameter-derived
 * values: an in-body {@code tf.range} loop is AutoGraph-supported and must not fire, and it is the decoration that makes a parameter
 * symbolic at entry.
 * <p>
 * The scan is body-local: parameter value numbers seed a forward def-use taint, and a {@link EachElementGetInstruction} (the IR form of a
 * Python {@code for}-each) whose iterated object is tainted <em>and</em> tensor-typed by the tensor-type analysis fires. The tensor-typed
 * gate keeps ordinary Python-collection parameters (lists, dicts, dataset objects) out: only a value the analysis types as a tensor can
 * become a symbolic tensor at trace time. Passing such a value to a Python builtin that iterates it ({@code enumerate}, {@code zip},
 * {@code list}, and the others of {@link #builtinIterates}) fires too, whether or not a loop iterates the result
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1098).
 * <p>
 * The same scan backs the item-assignment precondition (issue 1098): a subscript store ({@code y[i] = v}, augmented forms included) into a
 * parameter-derived, tensor-typed value works eagerly on a NumPy array, but a symbolic tensor has no {@code __setitem__}, so tracing raises
 * a {@code TypeError}. A slice store ({@code y[1:3] = v}) is not seen: the engine lowers it to a call of the {@code slice} builtin, not to
 * a subscript store (https://github.com/wala/ML/issues/1130).
 * <p>
 * The same scan backs a third precondition (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1169): passing such a
 * value as {@code tf.constant}'s {@code value} works eagerly, where it is a NumPy array or an eager tensor, but {@code tf.constant} rejects
 * a symbolic tensor, so tracing raises a {@code TypeError}. The engine names the API a call invokes ({@link TensorFlowApis}), so an alias
 * of {@code tf.constant} is seen, and a program function named like it is not. The scan is body-local, like the other two: a parameter
 * passed to a helper that gives it to {@code tf.constant} is not seen.
 */
class TensorIterationAnalysis {

	/** The (node, value number) index of the tensor-type analysis, mirroring {@link NumpyParameterFlowAnalysis}'s index. */
	private final Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex;

	/** The pointer analysis, used to resolve the iterated value's producing callee for the {@code tf.range} exemption. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/** The call graph, used to resolve the builtin a tensor is passed to. */
	private final CallGraph callGraph;

	/** The call-graph builder, which the engine asks to name the TensorFlow API a call invokes. */
	private final PythonSSAPropagationCallGraphBuilder builder;

	TensorIterationAnalysis(PythonSSAPropagationCallGraphBuilder builder, TensorTypeAnalysis tensorTypeAnalysis) {
		this.builder = builder;
		this.callGraph = builder.getCallGraph();
		this.pointerAnalysis = builder.getPointerAnalysis();
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

			// A builtin that iterates the argument (`enumerate(y)`, `zip(x, y)`): a loop iterates the builtin's result, a list the
			// container exemption keeps out, but the builtin iterates the tensor itself, which raises under tracing (issue 1098). The
			// container exemption still applies, so enumerating a Python list of tensors does not fire. No tf.range exemption here:
			// AutoGraph stages only a loop directly over the range, not one through a builtin.
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
		// The container exemption keeps a store into a Python collection of tensors out, which tracing supports. An attribute store
		// lowers to the same instruction, but the engine records which syntax a write came from (wala/ML#1100), and an attribute store
		// is legal on a symbolic tensor, so only a subscript store fires.
		return anyTaintedUse(node, method, (use, value) -> use instanceof PythonPropertyWrite write && write.isSubscript()
				&& write.getObjectRef() == value && this.isTensorTyped(node, value) && !this.pointsToContainer(node, value));
	}

	/** The API whose {@code value} argument must not be a symbolic tensor, as the engine names it ({@link TensorFlowApis}). */
	private static final String CONSTANT_FQN = "tensorflow.constant";

	/** The keyword of {@code tf.constant}'s first parameter, the value it converts. */
	private static final String CONSTANT_VALUE_KEYWORD = "value";

	/**
	 * True iff {@code node}'s body passes a parameter-derived, tensor-typed value as {@code tf.constant}'s {@code value}, which a symbolic
	 * tensor cannot be (issue 1169). The container exemption keeps a Python collection out, as for the other two checks. Unlike theirs, the
	 * taint stops at a static read of a tensor's type ({@link ShapeValueBoundaryAnalysis#TYPE_MEMBER_NAMES}), such as {@code x.shape}: it
	 * yields Python values that tracing keeps concrete, so a NumPy array built from them, as in {@code tf.constant(np.arange(x.shape[0]))},
	 * is a trace-time constant rather than a symbolic tensor.
	 *
	 * @param node The call-graph node to scan.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a source.
	 * @return True iff {@code tf.constant} is given a parameter-derived tensor as its value.
	 */
	boolean passesTensorParameterToConstant(CGNode node, boolean method) {
		return anyTaintedUse(node, method,
				(use, value) -> use instanceof PythonInvokeInstruction invoke && passesAsConstantValue(invoke, value)
						&& this.isTensorTyped(node, value) && !this.pointsToContainer(node, value)
						&& TensorFlowApis.of(this.builder, node, invoke).contains(CONSTANT_FQN),
				(use, value) -> this.readsType(node, use, value));
	}

	/**
	 * True iff {@code use} is an attribute read of {@code value}'s static type, a member of
	 * {@link ShapeValueBoundaryAnalysis#TYPE_MEMBER_NAMES}.
	 */
	private boolean readsType(CGNode node, SSAInstruction use, int value) {
		// A subscript under a key such as "shape" reads an element, not the type, so only an attribute read launders.
		if (!(use instanceof PythonPropertyRead read) || read.isSubscript() || read.getObjectRef() != value)
			return false;

		String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);
		return member != null && ShapeValueBoundaryAnalysis.TYPE_MEMBER_NAMES.contains(member);
	}

	/**
	 * True iff {@code invoke} passes {@code value} in {@code tf.constant}'s {@code value} slot: its first positional argument, or its
	 * {@code value} keyword. A call spreading a starred argument first is not aligned, so it is not matched.
	 */
	private static boolean passesAsConstantValue(PythonInvokeInstruction invoke, int value) {
		// The callee is the invoke's first use, so the first argument is use 1.
		if (invoke.getNumberOfPositionalParameters() > 1 && invoke.firstStarredPosition() != 1 && invoke.getUse(1) == value)
			return true;

		return invoke.getKeywords().contains(CONSTANT_VALUE_KEYWORD) && invoke.getUse(CONSTANT_VALUE_KEYWORD) == value;
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
		return anyTaintedUse(node, method, sink, (use, value) -> false);
	}

	/**
	 * True iff {@code sink} holds for some use of a parameter-derived value in {@code node}'s body, as for
	 * {@link #anyTaintedUse(CGNode, boolean, TaintedUse)}, except that a use satisfying {@code launders} does not taint its definitions.
	 *
	 * @param node The call-graph node to scan.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a source.
	 * @param sink The test, given a use and the tainted value it uses.
	 * @param launders The test of a use whose definitions do not carry the taint, given the use and the tainted value it uses.
	 * @return True iff {@code sink} holds for some tainted use.
	 */
	private static boolean anyTaintedUse(CGNode node, boolean method, TaintedUse sink, TaintedUse launders) {
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

				if (launders.test(use, value))
					continue;

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
	 * The Python builtins that iterate their first positional argument whatever else they are given: {@code enumerate(x, start)},
	 * {@code sum(x, start)}, and so on. Whether a builtin iterates an argument is Python's behavior, not the engine's, so it is stated here
	 * rather than read off the engine's builtin summaries. On TensorFlow 2.9.3, each of these called on a tensor parameter runs eagerly and
	 * raises under {@code tf.function}: {@code reversed} with a {@code TypeError}, the rest with {@code OperatorNotAllowedInGraphError}.
	 * {@code sorted} is left out, since AutoGraph converts it, and it runs under {@code tf.function}.
	 */
	private static final Set<String> FIRST_ARGUMENT_ITERATING_BUILTINS = Set.of("all", "any", "enumerate", "list", "reversed", "sum",
			"tuple");

	/**
	 * The Python builtins that iterate their first positional argument only when it is their sole one: {@code min(x)}, {@code max(x)} and
	 * {@code iter(x)} iterate {@code x}, while {@code min(x, y)} compares its arguments and {@code iter(f, sentinel)} calls {@code f}. Each
	 * raises {@code OperatorNotAllowedInGraphError} under {@code tf.function} on TensorFlow 2.9.3 when given only a tensor parameter.
	 */
	private static final Set<String> SOLE_ARGUMENT_ITERATING_BUILTINS = Set.of("iter", "max", "min");

	/**
	 * The Python builtins that iterate every positional argument. {@code zip} raises {@code OperatorNotAllowedInGraphError} under
	 * {@code tf.function} on TensorFlow 2.9.3 when given a tensor parameter.
	 */
	private static final Set<String> ALL_ARGUMENTS_ITERATING_BUILTINS = Set.of("zip");

	/**
	 * True iff {@code invoke} passes {@code value} positionally to a Python builtin that iterates it there, per
	 * {@link #FIRST_ARGUMENT_ITERATING_BUILTINS}, {@link #SOLE_ARGUMENT_ITERATING_BUILTINS}, and {@link #ALL_ARGUMENTS_ITERATING_BUILTINS}.
	 * The call graph decides which builtin is called: a callee is the builtin {@code name} when its declaring class is the engine's prelude
	 * object for it. Slots at or after a starred argument are skipped, since their alignment to the builtin's parameters is unknown. A call
	 * the engine made while lowering a {@code for} loop or a comprehension to the iteration protocol is skipped too, since it is not a call
	 * of the builtin in the source.
	 * <p>
	 * TODO: The engine has no summary for {@code tuple}, {@code min}, {@code max}, {@code any} or {@code all}, so a call to one has no
	 * target, and this doesn't fire for it: https://github.com/wala/ML/issues/1129. Once the engine summarizes them, it fires with no
	 * change here.
	 *
	 * @param node The call-graph node whose body holds {@code invoke}.
	 * @param invoke The call.
	 * @param value The parameter-derived value number.
	 * @return True iff the called builtin iterates {@code value}.
	 */
	private boolean builtinIterates(CGNode node, PythonInvokeInstruction invoke, int value) {
		if (invoke.isIterationProtocol())
			return false;

		int starred = invoke.firstStarredPosition();
		int positional = invoke.getNumberOfPositionalParameters();
		int end = starred == -1 ? positional : starred;

		for (int slot = 1; slot < end; slot++) {
			if (invoke.getUse(slot) != value)
				continue;

			for (CGNode callee : this.callGraph.getPossibleTargets(node, invoke.getCallSite())) {
				String type = callee.getMethod().getDeclaringClass().getName().toString();

				if (!type.startsWith(Util.BUILTIN_PRELUDE_TYPE_NAME_PREFIX))
					continue;

				String name = type.substring(Util.BUILTIN_PRELUDE_TYPE_NAME_PREFIX.length());

				if (ALL_ARGUMENTS_ITERATING_BUILTINS.contains(name) || slot == 1 && FIRST_ARGUMENT_ITERATING_BUILTINS.contains(name)
						|| slot == 1 && starred == -1 && positional == 2 && SOLE_ARGUMENT_ITERATING_BUILTINS.contains(name))
					return true;
			}
		}

		return false;
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
