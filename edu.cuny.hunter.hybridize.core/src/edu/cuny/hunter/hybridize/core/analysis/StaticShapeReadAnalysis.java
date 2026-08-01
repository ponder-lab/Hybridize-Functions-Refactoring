package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.MethodReference;

/**
 * The statically-read-axis analysis behind the unresolved-axis precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/811): an interprocedural, two-color taint walk collecting, for a
 * function's parameters, the axes whose extents the body (transitively) reads <em>statically</em> and consumes where a Python integer is
 * required. Under an emitted {@code input_signature}, a wildcard axis is {@code None} at trace time, so such a consumption raises (or,
 * worse, silently misbehaves through a {@code [:None]} slice); a <em>dynamic</em> read ({@code tf.shape(x)[i]}) returns a tensor and is
 * safe, which is the distinction this walk draws.
 * <p>
 * The walk mirrors {@link NumpyParameterFlowAnalysis}'s structure with three deliberate inversions. First, a {@code .shape} property read
 * is not laundered but is the event of interest: it colors the read value as shape metadata carrying an {@link AxisRead} descriptor
 * (parameter provenance plus covered axes, narrowed by constant subscripts and slices). Second, {@code tf.shape}/{@code size}/{@code rank}
 * results <em>are</em> laundered: they are tensors, valid under a wildcard. Third, descriptors join by union rather than poisoning: where
 * provenance or coverage is lost, the descriptor widens to "any parameter"/"any axis", so the consumer errs toward declining, the
 * conservative default this precondition requires (where the walk cannot establish the property, it must decline).
 * <p>
 * Sinks are the enumerated consumption sites observed in the corpus (extensible by adding a constant and a membership entry; a new sink is
 * not a new {@link PreconditionFailure}): a weight shape passed to {@code add_weight} in a layer's {@code build}, a
 * {@code tf.reshape}/{@code tf.range} target, and integer arithmetic over a dimension. Element-through-container flow into a reshape target
 * list is followed by coloring the written container ({@link PythonPropertyWrite}).
 * <p>
 * Keras lazy-{@code build} reachability: a layer's own {@code build} is a call-graph <em>sibling</em> of {@code call} (both hang off the
 * {@code __call__} trampoline), so it is seeded by name-based sibling lookup with its {@code input_shape} parameter anchored to the
 * function's first non-receiver parameter. A <em>sublayer's</em> {@code build} is forward-reachable (the trampoline interposes) and is
 * seeded with unknown provenance, mirroring {@code Function.subtractBuildProtocolContributions}'s reachability model.
 */
class StaticShapeReadAnalysis {

	/** The call graph, used to follow user-defined callees and to find reachable {@code build} methods. */
	private final CallGraph callGraph;

	/** The pointer analysis, used to resolve attribute names, module roots, and interprocedural integer constants. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/** The scan memo, shared across the checked function's call-graph nodes. */
	private final Map<String, ScanResult> memo = new HashMap<>();

	StaticShapeReadAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
	}

	/** The {@code shape} attribute, whose read is a static shape access. */
	private static final String SHAPE_MEMBER_NAME = "shape";

	/** The {@code dtype} attribute: a trace-time constant, laundering taint. */
	private static final String DTYPE_MEMBER_NAME = "dtype";

	/** The {@code add_weight} member: a shape-consuming sink when a statically-read dimension flows into it (a weight extent). */
	private static final String ADD_WEIGHT_MEMBER_NAME = "add_weight";

	/** Keras's static shape accessor: {@code K.int_shape(x)} is a static read, like {@code x.shape}. */
	private static final String INT_SHAPE_MEMBER_NAME = "int_shape";

	/** The built-in {@code slice} constructor, into which a Python {@code x[a:b:c]} subscript is lowered. */
	private static final String SLICE_BUILTIN_NAME = "slice";

	/** The trailing declaring-class segment naming a Keras lazy-{@code build} method. */
	private static final String BUILD_METHOD_NAME_SUFFIX = "/build";

	/**
	 * Fully-qualified names of the TensorFlow dynamic shape ops, whose results are tensors and therefore valid under a wildcard axis: their
	 * reads are laundered, the central static/dynamic distinction of the precondition.
	 */
	private static final Set<String> DYNAMIC_SHAPE_OP_FQNS = Set.of("tensorflow.shape", "tensorflow.size", "tensorflow.rank");

	/**
	 * Fully-qualified names of the ops whose arguments require Python integers for a shape target: a statically-read dimension flowing into
	 * them is a sink ({@code tf.reshape}'s target list; {@code tf.range}'s bounds, the {@code _qe_masking} consumption).
	 */
	private static final Set<String> SHAPE_TARGET_OP_FQNS = Set.of("tensorflow.reshape", "tensorflow.range");

	/**
	 * Names of the binary operators constituting integer arithmetic over a dimension, matched by {@code toString()} since the operator
	 * enums live in non-exported WALA packages. Comparisons are excluded: {@code x.shape[1] == 4} is {@code False} under a wildcard, not a
	 * raise.
	 */
	private static final Set<String> ARITHMETIC_OPERATOR_NAMES = Set.of("add", "sub", "mul", "div", "rem", "mod", "floordiv", "pow");

	/** The unary-negation operator name (see {@link NumpyParameterFlowAnalysis}'s rationale for name matching). */
	private static final String NEGATION_OPERATOR_NAME = "neg";

	/**
	 * A statically-read axis requirement: the parameters whose axes are read ({@code parameterOrdinals}, 0-based among the checked
	 * function's non-receiver parameters; {@code null} means provenance was lost and <em>any</em> tensor parameter may be the source) and
	 * the covered axes ({@code axes}; {@code null} means coverage was lost and <em>any</em> axis may be read). A negative axis counts from
	 * the end, resolved against the inferred spec's own rank by the consumer.
	 */
	record AxisRead(Set<Integer> parameterOrdinals, Set<Integer> axes) {

		/**
		 * The union join of two axis reads. A {@code null} <em>argument</em> is the identity (no prior read to merge with); a {@code null}
		 * <em>field</em> (unknown provenance or coverage) absorbs, so the union of anything with unknown is unknown.
		 */
		static AxisRead join(AxisRead a, AxisRead b) {
			if (a == null)
				return b;
			if (b == null)
				return a;

			Set<Integer> parameters = a.parameterOrdinals() == null || b.parameterOrdinals() == null ? null
					: union(a.parameterOrdinals(), b.parameterOrdinals());
			Set<Integer> axes = a.axes() == null || b.axes() == null ? null : union(a.axes(), b.axes());

			return new AxisRead(parameters, axes);
		}

		private static Set<Integer> union(Set<Integer> a, Set<Integer> b) {
			Set<Integer> ret = new TreeSet<>(a);
			ret.addAll(b);
			return ret;
		}
	}

	/**
	 * The requirements collected from {@code node} and everything reachable during tracing: for each parameter axis read statically and
	 * consumed at a sink, one {@link AxisRead}. Empty means no statically-read axis is consumed, so any signature is safe on this account.
	 *
	 * @param node The call-graph node of the checked function.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a source.
	 * @return The axis requirements the emitted signature must satisfy.
	 */
	Set<AxisRead> staticallyReadAxes(CGNode node, boolean method) {
		IR ir = node.getIR();

		if (ir == null)
			return Set.of();

		// Parameter value numbers: slot 0 is the function object itself; slot 1 is the receiver for an instance method. The ordinal is
		// the 0-based position among the remaining (non-receiver) parameters, matching the inferred signature's parameter order.
		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();
		Map<Integer, Set<Integer>> valueSeed = new HashMap<>();

		for (int i = method ? 2 : 1; i < parameters.length; i++)
			valueSeed.put(parameters[i], Set.of(i - (method ? 2 : 1)));

		Set<AxisRead> reads = new HashSet<>();

		if (!valueSeed.isEmpty())
			reads.addAll(this.scan(node, valueSeed, Map.of()).reads());

		// The function's own Keras `build` is a sibling under the `__call__` trampoline, not a successor: seed it by name-based lookup,
		// anchoring its `input_shape` parameter to this function's first non-receiver parameter (the Keras protocol passes the first
		// argument's shape).
		if (method && parameters.length > 2)
			reads.addAll(this.scanOwnBuild(node));

		// A sublayer's `build` is forward-reachable through the trampoline, but the Keras library frames between `call` and `build` are
		// summarized, so taint cannot flow through them slot-by-slot: seed every reachable `build`'s `input_shape` with unknown
		// provenance instead. Where the sublayer's input derives from no parameter, this over-approximates and may over-block; accepted
		// safety-first, mirroring the numpy precondition's stance.
		reads.addAll(this.scanReachableBuilds(node));

		return reads;
	}

	/** Scans this function's own class's {@code build} sibling, if any, anchored to the first non-receiver parameter (ordinal 0). */
	private Set<AxisRead> scanOwnBuild(CGNode node) {
		MethodReference reference = node.getMethod().getReference();
		String declaringClassName = reference.getDeclaringClass().getName().toString();
		int lastSegment = declaringClassName.lastIndexOf('/');

		if (lastSegment < 0 || declaringClassName.endsWith(BUILD_METHOD_NAME_SUFFIX))
			return Set.of();

		String buildClassName = declaringClassName.substring(0, lastSegment) + BUILD_METHOD_NAME_SUFFIX;
		Set<AxisRead> reads = new HashSet<>();

		for (CGNode buildNode : this.callGraph)
			if (buildNode.getMethod().getReference().getDeclaringClass().getName().toString().equals(buildClassName))
				reads.addAll(this.scanBuild(buildNode, Set.of(0)));

		return reads;
	}

	/** Scans every {@code build} method forward-reachable from {@code node} (a sublayer's, via the trampoline), unknown provenance. */
	private Set<AxisRead> scanReachableBuilds(CGNode node) {
		Set<AxisRead> reads = new HashSet<>();
		Set<CGNode> seen = new HashSet<>();
		Deque<CGNode> worklist = new ArrayDeque<>();

		seen.add(node);
		worklist.add(node);

		while (!worklist.isEmpty()) {
			CGNode current = worklist.remove();

			if (current != node
					&& current.getMethod().getReference().getDeclaringClass().getName().toString().endsWith(BUILD_METHOD_NAME_SUFFIX))
				reads.addAll(this.scanBuild(current, null));

			for (Iterator<CGNode> successors = this.callGraph.getSuccNodes(current); successors.hasNext();) {
				CGNode next = successors.next();

				if (seen.add(next))
					worklist.add(next);
			}
		}

		return reads;
	}

	/**
	 * Scans a {@code build} method with its {@code input_shape} parameter seeded as shape metadata whose provenance is
	 * {@code parameterOrdinals} ({@code null} = unknown), covering all axes until narrowed by subscripts inside {@code build}.
	 */
	private Set<AxisRead> scanBuild(CGNode buildNode, Set<Integer> parameterOrdinals) {
		IR ir = buildNode.getIR();

		if (ir == null)
			return Set.of();

		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();

		// build(self, input_shape): slot 0 is the function object, slot 1 the receiver, slot 2 the shape.
		if (parameters.length <= 2)
			return Set.of();

		return this.scan(buildNode, Map.of(), Map.of(parameters[2], new AxisRead(parameterOrdinals, null))).reads();
	}

	/**
	 * The result of a scan: the {@link AxisRead}s whose descriptors reached a sink, whether any value taint escaped (reached a non-shape,
	 * non-laundering use), and the descriptor this node's returns carry back to the caller ({@code null} when no shape-tainted value is
	 * returned).
	 */
	private record ScanResult(Set<AxisRead> reads, boolean valueEscapes, AxisRead returnDescriptor, Set<Integer> returnProvenance) {
	}

	/**
	 * Worklist taint propagation over {@code node}'s def-use chains. A <em>value</em> taint marks a tensor value derived from the checked
	 * function's parameters, carrying the set of source-parameter ordinals; a <em>shape</em> taint marks a value derived from a static
	 * shape read, carrying an {@link AxisRead} descriptor. Static reads create shape taint; dynamic reads ({@code tf.shape} etc.) and
	 * {@code dtype} reads launder; constant subscripts and slices narrow a descriptor's covered axes; container writes propagate shape
	 * taint onto the written container (a reshape target list); sinks record the consumed descriptor. Memoized per (node, seeds) with an
	 * optimistic cycle guard, following {@link NumpyParameterFlowAnalysis}.
	 */
	private ScanResult scan(CGNode node, Map<Integer, Set<Integer>> valueSeed, Map<Integer, AxisRead> shapeSeed) {
		String key = this.renderMemoKey(node, valueSeed, shapeSeed);
		ScanResult cached = this.memo.get(key);

		if (cached != null)
			return cached;

		// Optimistic cycle guard: a recursive revisit contributes nothing new.
		this.memo.put(key, new ScanResult(Set.of(), false, null, null));

		IR ir = node.getIR();

		if (ir == null)
			return new ScanResult(Set.of(), false, null, null);

		DefUse defUse = node.getDU();
		Map<Integer, Set<Integer>> valueProvenance = new HashMap<>(valueSeed);
		Map<Integer, AxisRead> shapeDescriptors = new HashMap<>(shapeSeed);
		Deque<Integer> worklist = new ArrayDeque<>();
		worklist.addAll(valueSeed.keySet());
		worklist.addAll(shapeSeed.keySet());
		Set<AxisRead> reads = new HashSet<>();
		boolean valueEscapes = false;
		AxisRead returnDescriptor = null;
		Set<Integer> returnProvenance = null;

		while (!worklist.isEmpty()) {
			int valueNumber = worklist.pop();
			boolean valueColored = valueProvenance.containsKey(valueNumber);

			for (Iterator<SSAInstruction> uses = defUse.getUses(valueNumber); uses.hasNext();) {
				SSAInstruction use = uses.next();

				if (use instanceof PythonPropertyRead read && read.getObjectRef() == valueNumber) {
					String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);

					// A `dtype` read is a trace-time constant: launder.
					if (DTYPE_MEMBER_NAME.equals(member))
						continue;

					// A static `.shape` read of a parameter-derived tensor: the event of interest. The provenance is the tensor's
					// source-parameter set; the descriptor initially covers every axis.
					if (SHAPE_MEMBER_NAME.equals(member) && valueColored) {
						AxisRead descriptor = new AxisRead(valueProvenance.get(valueNumber), null);

						for (int d = 0; d < read.getNumberOfDefs(); d++)
							colorShape(read.getDef(d), descriptor, valueProvenance, shapeDescriptors, worklist);

						continue;
					}

					// A constant subscript of a shape vector extracts one dimension: narrow the descriptor to that axis. A negative
					// index stays negative; the consumer resolves it against the spec's own rank. Subscripting an already-narrowed
					// vector (the `x.shape[-2:]` then `dims[0]` idiom) composes: the index selects within the covered run, whose
					// order is ascending, not within the original shape.
					if (!valueColored && shapeDescriptors.containsKey(valueNumber)) {
						Integer index = this.resolveIntConstant(node, read.getMemberRef(), defUse);
						AxisRead base = shapeDescriptors.get(valueNumber);
						AxisRead narrowed;

						if (index == null)
							narrowed = new AxisRead(base.parameterOrdinals(), null);
						else if (base.axes() == null)
							narrowed = new AxisRead(base.parameterOrdinals(), Set.of(index));
						else
							narrowed = new AxisRead(base.parameterOrdinals(), composeSubscript(base.axes(), index));

						for (int d = 0; d < read.getNumberOfDefs(); d++)
							colorShape(read.getDef(d), narrowed, valueProvenance, shapeDescriptors, worklist);

						continue;
					}
				}

				// A shape-tainted value written into a container (a reshape target list under construction) colors the container.
				if (use instanceof PythonPropertyWrite write && !valueColored && shapeDescriptors.containsKey(valueNumber)
						&& write.getValue() == valueNumber) {
					colorShape(write.getObjectRef(), shapeDescriptors.get(valueNumber), valueProvenance, shapeDescriptors, worklist);
					continue;
				}

				if (use instanceof PythonInvokeInstruction invoke) {
					if (this.handleInvoke(node, invoke, defUse, valueNumber, valueColored, valueProvenance, shapeDescriptors, worklist,
							reads))
						continue;

					// Cross into user-defined callees, carrying each tainted argument's color to the corresponding parameter slot.
					InterproceduralOutcome outcome = this.crossIntoCallees(node, invoke, valueProvenance, shapeDescriptors, reads);

					if (outcome.valueEscaped())
						valueEscapes = true;

					for (int d = 0; d < invoke.getNumberOfDefs(); d++) {
						int def = invoke.getDef(d);

						if (outcome.resultProvenance() != null)
							colorValue(def, outcome.resultProvenance(), valueProvenance, shapeDescriptors, worklist);
						else if (outcome.resultDescriptor() != null)
							colorShape(def, outcome.resultDescriptor(), valueProvenance, shapeDescriptors, worklist);
					}

					continue;
				}

				if (use instanceof SSAReturnInstruction) {
					if (valueColored) {
						valueEscapes = true;
						returnProvenance = returnProvenance == null ? valueProvenance.get(valueNumber)
								: AxisRead.join(new AxisRead(returnProvenance, null), new AxisRead(valueProvenance.get(valueNumber), null))
										.parameterOrdinals();
					} else if (shapeDescriptors.containsKey(valueNumber))
						returnDescriptor = AxisRead.join(returnDescriptor, shapeDescriptors.get(valueNumber));

					continue;
				}

				// Integer arithmetic over a statically-read dimension is a sink: a wildcard axis is None at trace time, and arithmetic
				// on None raises.
				if ((use instanceof SSABinaryOpInstruction binary && isArithmeticOperator(binary)
						|| use instanceof SSAUnaryOpInstruction unary
								&& NEGATION_OPERATOR_NAME.equalsIgnoreCase(String.valueOf(unary.getOpcode())))
						&& !valueColored && shapeDescriptors.containsKey(valueNumber)) {
					reads.add(shapeDescriptors.get(valueNumber));

					// The arithmetic result is itself dimension-derived; keep tracking it toward further sinks.
					for (int d = 0; d < use.getNumberOfDefs(); d++)
						colorShape(use.getDef(d), shapeDescriptors.get(valueNumber), valueProvenance, shapeDescriptors, worklist);

					continue;
				}

				// Any other instruction: a value-tainted operand escapes onward and the definitions inherit its color; a shape-tainted
				// operand propagates its descriptor.
				if (valueColored) {
					valueEscapes = true;

					for (int d = 0; d < use.getNumberOfDefs(); d++)
						colorValue(use.getDef(d), valueProvenance.get(valueNumber), valueProvenance, shapeDescriptors, worklist);
				} else {
					AxisRead descriptor = shapeDescriptors.get(valueNumber);

					for (int d = 0; d < use.getNumberOfDefs(); d++)
						colorShape(use.getDef(d), descriptor, valueProvenance, shapeDescriptors, worklist);
				}
			}
		}

		ScanResult result = new ScanResult(reads, valueEscapes, returnDescriptor, returnProvenance);
		this.memo.put(key, result);
		return result;
	}

	/**
	 * Handles the invoke-instruction cases that need no callee descent: laundering ({@code tf.shape}/{@code size}/{@code rank}), slice
	 * narrowing, static {@code K.int_shape}, and the sink tests ({@code tf.reshape}/{@code tf.range} targets; {@code add_weight}). Returns
	 * true iff the invoke was fully handled here.
	 */
	private boolean handleInvoke(CGNode node, PythonInvokeInstruction invoke, DefUse defUse, int valueNumber, boolean valueColored,
			Map<Integer, Set<Integer>> valueProvenance, Map<Integer, AxisRead> shapeDescriptors, Deque<Integer> worklist,
			Set<AxisRead> reads) {
		String fqn = Util.resolveCalleeFullyQualifiedName(node, invoke.getUse(0), defUse, this.pointerAnalysis);

		// A dynamic shape read returns a tensor, valid under a wildcard: launder entirely.
		if (fqn != null && DYNAMIC_SHAPE_OP_FQNS.contains(fqn))
			return true;

		// A shape-target op consuming a statically-read dimension (directly or through its target list) is a sink.
		if (fqn != null && SHAPE_TARGET_OP_FQNS.contains(fqn)) {
			recordShapeArguments(invoke, shapeDescriptors, reads);
			return true;
		}

		// `K.int_shape(x)` is a static read of `x`'s shape, like `x.shape`.
		if (valueColored && isMemberCall(invoke, defUse, INT_SHAPE_MEMBER_NAME, node, this.pointerAnalysis)) {
			AxisRead descriptor = new AxisRead(valueProvenance.get(valueNumber), null);

			for (int d = 0; d < invoke.getNumberOfDefs(); d++)
				colorShape(invoke.getDef(d), descriptor, valueProvenance, shapeDescriptors, worklist);

			return true;
		}

		// `add_weight(..., shape=...)` requires Python integer extents: a sink for any statically-read dimension among its arguments.
		if (isMemberCall(invoke, defUse, ADD_WEIGHT_MEMBER_NAME, node, this.pointerAnalysis)) {
			recordShapeArguments(invoke, shapeDescriptors, reads);
			return true;
		}

		// A `x[a:b:c]` subscript of a shape vector narrows the covered axes; of a tensor value it yields a sub-tensor (fall through to
		// the generic invoke handling, which keeps the value color).
		if (!valueColored && invokesSliceBuiltin(invoke, defUse) && invoke.getNumberOfUses() > 1 && invoke.getUse(1) == valueNumber
				&& shapeDescriptors.containsKey(valueNumber)) {
			AxisRead base = shapeDescriptors.get(valueNumber);
			// Slicing an already-narrowed vector is not composed (mirroring the numpy walk's policy): coverage drops to unknown,
			// the conservative direction here.
			Set<Integer> dims = base.axes() == null ? this.resolveSliceAxes(node, invoke, defUse) : null;
			AxisRead narrowed = new AxisRead(base.parameterOrdinals(), dims);

			for (int d = 0; d < invoke.getNumberOfDefs(); d++)
				colorShape(invoke.getDef(d), narrowed, valueProvenance, shapeDescriptors, worklist);

			return true;
		}

		return false;
	}

	/** Records every shape-tainted argument of {@code invoke} as a consumed {@link AxisRead}. */
	private static void recordShapeArguments(PythonInvokeInstruction invoke, Map<Integer, AxisRead> shapeDescriptors, Set<AxisRead> reads) {
		for (int j = 1; j < invoke.getNumberOfUses(); j++) {
			AxisRead descriptor = shapeDescriptors.get(invoke.getUse(j));

			if (descriptor != null)
				reads.add(descriptor);
		}
	}

	/** The outcome of descending into an invoke's user-defined callees. */
	private record InterproceduralOutcome(boolean valueEscaped, Set<Integer> resultProvenance, AxisRead resultDescriptor) {
	}

	/**
	 * Crosses into {@code invoke}'s user-defined callees, seeding tainted argument slots, accumulating their sinks into {@code reads}, and
	 * deriving the call-site result's color: value provenance when a value-tainted argument reached a non-shape use (or the callee is a
	 * library summary), else the callees' returned shape descriptor.
	 */
	private InterproceduralOutcome crossIntoCallees(CGNode node, PythonInvokeInstruction invoke, Map<Integer, Set<Integer>> valueProvenance,
			Map<Integer, AxisRead> shapeDescriptors, Set<AxisRead> reads) {
		Map<Integer, Set<Integer>> valueSlots = new TreeMap<>();
		Map<Integer, AxisRead> shapeSlots = new TreeMap<>();

		for (int j = 1; j < invoke.getNumberOfUses(); j++) {
			int argument = invoke.getUse(j);

			if (valueProvenance.containsKey(argument))
				valueSlots.put(j, valueProvenance.get(argument));
			else if (shapeDescriptors.containsKey(argument))
				shapeSlots.put(j, shapeDescriptors.get(argument));
		}

		if (valueSlots.isEmpty() && shapeSlots.isEmpty())
			return new InterproceduralOutcome(false, null, null);

		boolean analyzedCallee = false;
		boolean calleeValueEscapes = false;
		AxisRead calleeReturnDescriptor = null;
		Set<Integer> calleeReturnProvenance = null;

		for (CGNode target : this.callGraph.getPossibleTargets(node, invoke.getCallSite())) {
			if (Util.isTensorFlowNode(target))
				continue;

			IR targetIr = target.getIR();

			if (targetIr == null)
				continue;

			int[] targetParameters = targetIr.getSymbolTable().getParameterValueNumbers();
			Map<Integer, Set<Integer>> targetValueSeed = new HashMap<>();
			Map<Integer, AxisRead> targetShapeSeed = new HashMap<>();

			valueSlots.forEach((slot, provenance) -> {
				if (slot < targetParameters.length)
					targetValueSeed.put(targetParameters[slot], provenance);
			});

			shapeSlots.forEach((slot, descriptor) -> {
				if (slot < targetParameters.length)
					targetShapeSeed.put(targetParameters[slot], descriptor);
			});

			if (targetValueSeed.isEmpty() && targetShapeSeed.isEmpty())
				continue;

			analyzedCallee = true;

			ScanResult result = this.scan(target, targetValueSeed, targetShapeSeed);

			reads.addAll(result.reads());

			if (result.valueEscapes())
				calleeValueEscapes = true;

			if (result.returnDescriptor() != null)
				calleeReturnDescriptor = AxisRead.join(calleeReturnDescriptor, result.returnDescriptor());

			if (result.returnProvenance() != null)
				calleeReturnProvenance = calleeReturnProvenance == null ? result.returnProvenance()
						: AxisRead.join(new AxisRead(calleeReturnProvenance, null), new AxisRead(result.returnProvenance(), null))
								.parameterOrdinals();
		}

		// A library or unanalyzed callee consuming a value-tainted argument yields a parameter-derived value (e.g. `tf.pad(x, ...)`);
		// an analyzed callee's returned taints flow to the result. A returned shape descriptor is preferred over value provenance only
		// when no value provenance returned, keeping value color dominant.
		Set<Integer> mergedValueProvenance = null;

		if (!valueSlots.isEmpty() && (!analyzedCallee || calleeValueEscapes && calleeReturnProvenance == null)) {
			mergedValueProvenance = new TreeSet<>();

			for (Set<Integer> provenance : valueSlots.values())
				mergedValueProvenance.addAll(provenance);
		} else if (calleeReturnProvenance != null)
			mergedValueProvenance = calleeReturnProvenance;

		boolean valueEscaped = !valueSlots.isEmpty() && (!analyzedCallee || calleeValueEscapes);

		return new InterproceduralOutcome(valueEscaped, mergedValueProvenance,
				mergedValueProvenance == null ? calleeReturnDescriptor : null);
	}

	/** Colors {@code value} with value taint carrying {@code provenance} (value dominates shape) and enqueues on change. */
	private static void colorValue(int value, Set<Integer> provenance, Map<Integer, Set<Integer>> valueProvenance,
			Map<Integer, AxisRead> shapeDescriptors, Deque<Integer> worklist) {
		Set<Integer> incoming = provenance == null ? Set.of() : provenance;

		shapeDescriptors.remove(value);
		Set<Integer> existing = valueProvenance.get(value);

		if (existing == null) {
			valueProvenance.put(value, incoming);
			worklist.push(value);
		} else if (!existing.containsAll(incoming)) {
			Set<Integer> merged = new TreeSet<>(existing);
			merged.addAll(incoming);
			valueProvenance.put(value, merged);
			worklist.push(value);
		}
	}

	/** Colors {@code value} with shape taint carrying {@code descriptor}, joining by union, unless it is value-tainted. */
	private static void colorShape(int value, AxisRead descriptor, Map<Integer, Set<Integer>> valueProvenance,
			Map<Integer, AxisRead> shapeDescriptors, Deque<Integer> worklist) {
		if (valueProvenance.containsKey(value))
			return;

		AxisRead existing = shapeDescriptors.get(value);
		AxisRead joined = AxisRead.join(existing, descriptor);

		if (!joined.equals(existing)) {
			shapeDescriptors.put(value, joined);
			worklist.push(value);
		}
	}

	/** True iff {@code binary}'s operator is integer arithmetic (see {@link #ARITHMETIC_OPERATOR_NAMES}). */
	private static boolean isArithmeticOperator(SSABinaryOpInstruction binary) {
		return ARITHMETIC_OPERATOR_NAMES.contains(String.valueOf(binary.getOperator()).toLowerCase());
	}

	/** True iff {@code invoke}'s callee is a property read of {@code memberName} (e.g. {@code self.add_weight}, {@code K.int_shape}). */
	private static boolean isMemberCall(PythonInvokeInstruction invoke, DefUse defUse, String memberName, CGNode node,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		SSAInstruction def = defUse.getDef(invoke.getUse(0));

		if (!(def instanceof PythonPropertyRead read))
			return false;

		return memberName.equals(Util.resolveStringConstant(node, read.getMemberRef(), pointerAnalysis));
	}

	/** True iff {@code invoke} calls the built-in {@code slice} constructor (how a Python {@code x[a:b:c]} subscript is modeled). */
	private static boolean invokesSliceBuiltin(PythonInvokeInstruction invoke, DefUse defUse) {
		SSAInstruction def = defUse.getDef(invoke.getUse(0));

		if (!(def instanceof AstLexicalRead lexical))
			return false;

		Access[] accesses = lexical.getAccesses();
		return accesses.length > 0 && SLICE_BUILTIN_NAME.equals(accesses[0].getName().fst);
	}

	/**
	 * The axes covered by the slice in {@code invoke}: rank-free prefix ({@code [:k]}) and suffix ({@code [-k:]}) forms resolve to absolute
	 * and negative indices respectively; anything else (unknown bounds, non-unit stride) is {@code null} (unknown coverage), which the
	 * consumer treats conservatively.
	 */
	private Set<Integer> resolveSliceAxes(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		Integer start = invoke.getNumberOfUses() > 2 ? this.resolveIntConstant(node, invoke.getUse(2), defUse) : null;
		Integer stop = invoke.getNumberOfUses() > 3 ? this.resolveIntConstant(node, invoke.getUse(3), defUse) : null;
		Integer step = invoke.getNumberOfUses() > 4 ? this.resolveIntConstant(node, invoke.getUse(4), defUse) : null;

		if (step != null && step != 1)
			return null;

		Set<Integer> axes = new TreeSet<>();

		// A pure prefix [:k]: absolute indices 0..k-1.
		if ((start == null || start == 0) && stop != null && stop >= 0 && stop <= MAX_SLICE_EXTENT) {
			for (int i = 0; i < stop; i++)
				axes.add(i);

			return axes;
		}

		// A pure suffix [-k:]: negative indices resolved against the spec's rank by the consumer.
		if (start != null && start < 0 && start >= -MAX_SLICE_EXTENT && stop == null) {
			for (int i = start; i < 0; i++)
				axes.add(i);

			return axes;
		}

		return null;
	}

	/** The maximum slice extent modeled precisely; see {@link NumpyParameterFlowAnalysis}'s rationale. */
	private static final int MAX_SLICE_EXTENT = 32;

	/**
	 * The axis selected by subscripting a shape vector already narrowed to {@code covered} at {@code index}: the covered run is ordered
	 * ascending (a slice yields a pure prefix, suffix, or absolute range), and the subscript selects within that run, negative from its
	 * end. {@code null} (unknown coverage) when the subscript falls outside the run, the conservative direction.
	 *
	 * @param covered The covered axes of the narrowed vector.
	 * @param index The subscript into the narrowed vector.
	 * @return The selected axis as a singleton set, or {@code null} when the subscript falls outside the covered run.
	 */
	private static Set<Integer> composeSubscript(Set<Integer> covered, int index) {
		List<Integer> ordered = new ArrayList<>(new TreeSet<>(covered));
		int resolved = index < 0 ? ordered.size() + index : index;
		return resolved >= 0 && resolved < ordered.size() ? Set.of(ordered.get(resolved)) : null;
	}

	/**
	 * Resolves {@code value} in {@code node} to an integer constant, or {@code null} if it cannot be resolved: a symbol-table literal, a
	 * unary negation of a resolvable value, or an unambiguous interprocedural {@link ConstantKey}.
	 */
	private Integer resolveIntConstant(CGNode node, int value, DefUse defUse) {
		SymbolTable symbolTable = node.getIR().getSymbolTable();

		if (symbolTable.isIntegerConstant(value))
			return symbolTable.getIntValue(value);

		SSAInstruction def = defUse.getDef(value);

		if (def instanceof SSAUnaryOpInstruction unary && NEGATION_OPERATOR_NAME.equalsIgnoreCase(String.valueOf(unary.getOpcode()))) {
			Integer operand = this.resolveIntConstant(node, unary.getUse(0), defUse);
			return operand == null ? null : -operand;
		}

		PointerKey pointerKey = this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, value);
		Integer resolved = null;

		for (InstanceKey instanceKey : this.pointerAnalysis.getPointsToSet(pointerKey))
			if (instanceKey instanceof ConstantKey<?> constantKey && constantKey.getValue() instanceof Number number) {
				long asLong = number.longValue();

				if (number.doubleValue() != asLong || asLong < Integer.MIN_VALUE || asLong > Integer.MAX_VALUE)
					return null;

				int candidate = (int) asLong;

				if (resolved != null && resolved != candidate)
					return null;

				resolved = candidate;
			}

		return resolved;
	}

	/** A stable memo key over the node and both seed maps. */
	private String renderMemoKey(CGNode node, Map<Integer, Set<Integer>> valueSeed, Map<Integer, AxisRead> shapeSeed) {
		Map<Integer, String> values = new TreeMap<>();
		valueSeed.forEach((value, provenance) -> values.put(value, String.valueOf(new TreeSet<>(provenance))));
		Map<Integer, String> shapes = new TreeMap<>();
		shapeSeed.forEach((value, descriptor) -> shapes.put(value, renderDescriptor(descriptor)));
		return this.callGraph.getNumber(node) + ":" + values + ":" + shapes;
	}

	/** A stable rendering of {@code descriptor} for the memo key ({@code *} = unknown). */
	private static String renderDescriptor(AxisRead descriptor) {
		return (descriptor.parameterOrdinals() == null ? "*" : new TreeSet<>(descriptor.parameterOrdinals()).toString()) + "@"
				+ (descriptor.axes() == null ? "*" : new TreeSet<>(descriptor.axes()).toString());
	}
}
