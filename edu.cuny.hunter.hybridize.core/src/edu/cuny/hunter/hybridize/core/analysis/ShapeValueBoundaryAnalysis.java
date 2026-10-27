package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Stream;

import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.CallTargets;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;

import edu.cuny.hunter.hybridize.core.analysis.StaticShapeReadAnalysis.ShapeArguments;

/**
 * The analysis behind {@link PreconditionFailure#PASSES_SHAPE_VALUE_ACROSS_BOUNDARY}: whether a shape known only as a tensor <em>value</em>
 * crosses a function's boundary, which converting the function to {@code tf.function} would lose. Measured on TensorFlow 2.9.3: inside one
 * graph, shape inference partially evaluates a shape tensor such as {@code tf.shape(x)[1:]} back to static extents, so
 * {@code tf.reshape(y, tf.concat((b, lost_shape), 0))} gets a static result shape. Across a {@code tf.function} call, a tensor's type
 * crosses but its value does not, so the reshape result's dimensions become {@code None}, and a later static read of them, such as a Keras
 * lazy {@code build} calling {@code add_weight(shape=[input_shape[-1]])}, raises. A bare decorator and an inferred signature fail the same
 * way, since neither carries the value.
 * <p>
 * Either direction refuses:
 * <ul>
 * <li><b>Inbound.</b> A parameter's value flows, through any def-use chain, into the shape operand of an op whose output's static
 * dimensions come from that operand's value ({@link #SHAPE_OPERANDS}), and that op's result reaches a return. The parameter's type alone
 * crosses, so the result's dimensions are lost. A dynamic shape op ({@code tf.shape}, {@code tf.size}, {@code tf.rank}) and a static
 * {@code .shape} or {@code .dtype} read launder the value: they read the parameter's type, which crosses.
 * <li><b>Outbound.</b> A returned value, or an element of a returned list, tuple or dictionary, derives from a dynamic shape op, or from
 * slicing, concatenating, or computing over one. The caller then receives the shape's type but not its value. A value computed from a shape
 * value together with a tensor that is not one, such as a loss divided by a batch size, is data, not a shape, and is not followed; so is a
 * comparison of one, which is a boolean, and so is the result of a shape operand's op, such as {@code tf.reshape(x, s)}, which is the
 * reason the shape was computed, even when its data operand is itself a shape vector.
 * </ul>
 * Ops are identified by the engine: {@link TensorFlowApis} names the TensorFlow APIs a call may invoke, following the call graph's resolved
 * callees, so an alias or a rebinding of an op is identified as the op it is, and a user function named like an op is not. The names are
 * matched against the ones {@link StaticShapeReadAnalysis} catalogs. {@code Tensor.shape} used as a tensor is not a source, since it is a
 * static {@code TensorShape} whose value is part of the type that crosses.
 * <p>
 * TODO: Three names are not yet answered by the engine (https://github.com/ponder-lab/ML/pull/1084). {@code tf.ensure_shape} has no summary
 * and is named only when written as an attribute chain off the module, and {@code Tensor.set_shape} is named only on a parsed feature's
 * tensor. A call of either one is therefore missed, so a parameter's value reaching its shape operand is not refused. Measured on
 * TensorFlow 2.9.3, either accepts a tensor shape eagerly and raises a {@code TypeError} on one under tracing, so such a function breaks
 * when converted, whether or not the shape is later read statically. {@code tf.keras.backend.int_shape} has no summary either; it reads the
 * static shape and would launder a parameter's value, but its unnamed call passes the value on, which over-refuses rather than misses.
 * <p>
 * The walk is interprocedural over the program functions whose parameters receive the call's own arguments, as the engine reports them
 * ({@link CallTargets}), with a bound method's instance or a class method's class bound first, mapping arguments to parameters by position
 * and keyword; a starred argument, and a closure's free variables, are not followed. A container's elements written under constant keys,
 * such as a returned tuple's, are told apart, so a caller unpacking a helper's {@code (tensor, shape)} result colors only the shape; an
 * element written under a computed key colors the whole container.
 * <p>
 * Conservative by design: either direction refuses without proving that a static read of the lost dimensions exists downstream, so a
 * function whose caller never reads them statically, or reads them eagerly, where the returned tensor carries its value, is refused too.
 * Future work: refine the refusal on the caller side, refusing only when a caller, itself traced, consumes the crossed shape where a static
 * extent is required.
 */
class ShapeValueBoundaryAnalysis {

	/** A taint color. */
	private enum Color {
		/** A value derived from a dynamic shape op, the outbound source. */
		SHAPE,

		/** A value derived from a parameter's value, the inbound source. */
		PARAMETER,

		/** The result of an op whose shape operand carries {@link #PARAMETER}, the inbound event. */
		SHAPED_BY_PARAMETER
	}

	/** Which direction a walk checks. */
	private enum Direction {
		/** Whether a {@link Color#SHAPE} value reaches a return. */
		OUTBOUND,

		/** Whether a {@link Color#SHAPED_BY_PARAMETER} value reaches a return. */
		INBOUND
	}

	/**
	 * The shape operands of the ops whose output's static dimensions come from an operand's value: positional indices, from 0 for the first
	 * argument after the callee, and keyword names. Every op is one {@link StaticShapeReadAnalysis} catalogs as taking a shape, or a method
	 * fixing one ({@code set_shape}, {@code ensure_shape}); the operands are the subset of each one's catalogued arguments that give the
	 * output's dimensions, excluding the data and fill operands.
	 */
	private static final Map<String, ShapeArguments> SHAPE_OPERANDS = Map.ofEntries(
			Map.entry("tensorflow.reshape", new ShapeArguments(Set.of(1), Set.of("shape"))),
			Map.entry("tensorflow.range", new ShapeArguments(Set.of(0, 1, 2), Set.of("start", "limit", "delta"))),
			Map.entry("tensorflow.image.resize", new ShapeArguments(Set.of(1), Set.of("size"))),
			Map.entry("tensorflow.ones", new ShapeArguments(Set.of(0), Set.of("shape"))),
			Map.entry("tensorflow.zeros", new ShapeArguments(Set.of(0), Set.of("shape"))),
			Map.entry("tensorflow.fill", new ShapeArguments(Set.of(0), Set.of("dims"))),
			Map.entry("tensorflow.eye", new ShapeArguments(Set.of(0, 1, 2), Set.of("num_rows", "num_columns", "batch_shape"))),
			Map.entry("tensorflow.random.normal", new ShapeArguments(Set.of(0), Set.of("shape"))),
			Map.entry("tensorflow.random.truncated_normal", new ShapeArguments(Set.of(0), Set.of("shape"))),
			Map.entry("tensorflow.random.uniform", new ShapeArguments(Set.of(0), Set.of("shape"))),
			Map.entry("tensorflow.broadcast_to", new ShapeArguments(Set.of(1), Set.of("shape"))),
			Map.entry("tensorflow.tile", new ShapeArguments(Set.of(1), Set.of("multiples"))),
			Map.entry("tensorflow.one_hot", new ShapeArguments(Set.of(1), Set.of("depth"))),
			Map.entry("tensorflow.ensure_shape", new ShapeArguments(Set.of(1), Set.of("shape"))),
			Map.entry("tensorflow.Tensor.set_shape", new ShapeArguments(Set.of(0), Set.of("shape"))));

	/** The ops outside {@link StaticShapeReadAnalysis}'s catalogs that {@link #SHAPE_OPERANDS} adds: methods and ops fixing a shape. */
	private static final Set<String> SHAPE_FIXING_OP_FQNS = Set.of("tensorflow.ensure_shape", "tensorflow.Tensor.set_shape");

	static {
		assert SHAPE_OPERANDS.keySet().stream()
				.allMatch(fqn -> StaticShapeReadAnalysis.SHAPE_TARGET_OP_FQNS.contains(fqn)
						|| StaticShapeReadAnalysis.SHAPE_ARGUMENT_OPS.containsKey(fqn)
						|| SHAPE_FIXING_OP_FQNS.contains(fqn)) : "Every shape-operand op is catalogued.";
	}

	/**
	 * The member reads that read a tensor's type rather than its value: a static {@code .shape}, {@code .get_shape()}, {@code .ndim} and
	 * {@code .dtype}.
	 */
	private static final Set<String> TYPE_MEMBER_NAMES = Set.of("shape", "get_shape", "ndim", "dtype");

	/**
	 * The names of TensorFlow's elementwise comparison ops, whose result is a boolean, not a shape: a shape value compared, as in
	 * {@code tf.cond(tf.equal(tf.shape(x)[-1], 4), ...)}, colors nothing outbound.
	 */
	private static final Set<String> COMPARISON_OP_FQNS = Set.of("tensorflow.equal", "tensorflow.not_equal", "tensorflow.less",
			"tensorflow.less_equal", "tensorflow.greater", "tensorflow.greater_equal");

	/** The method fixing its receiver's static shape, whose receiver, not its result, takes the shape operand's dimensions. */
	private static final String SET_SHAPE_OP_FQN = "tensorflow.Tensor.set_shape";

	/**
	 * What a walk finds reaching a node's returns: the colors of each returned value as a whole, and, for a returned container whose
	 * elements were written under constant keys, the colors of each element, so that a caller unpacking a {@code (tensor, shape)} result
	 * tells the two apart.
	 *
	 * @param whole The colors of the returned values, less those carried only by a container's keyed elements.
	 * @param elements The colors of the returned containers' elements, by key.
	 */
	private record Returned(Set<Color> whole, Map<String, Set<Color>> elements) {

		/** Nothing returned. */
		static final Returned NONE = new Returned(Set.of(), Map.of());

		/**
		 * Every color returned, element or whole.
		 *
		 * @return The union of {@link #whole()} and every element's colors.
		 */
		Set<Color> all() {
			Set<Color> ret = EnumSet.noneOf(Color.class);
			ret.addAll(this.whole);
			this.elements.values().forEach(ret::addAll);
			return ret;
		}
	}

	/**
	 * The state of one walk over one node: every value's colors, the colors a value carries other than through its keyed elements, the
	 * colors of a container's keyed elements, and the worklist.
	 */
	private static final class Walk {

		/** Every colored value's colors, its elements' included. */
		final Map<Integer, Set<Color>> colors = new HashMap<>();

		/** The colors a value carries other than through a keyed element. */
		final Map<Integer, Set<Color>> direct = new HashMap<>();

		/** The colors of a container's elements, by key, for the elements written under a constant key. */
		final Map<Integer, Map<String, Set<Color>>> elements = new HashMap<>();

		/** The values whose colors grew. */
		final Deque<Integer> worklist = new ArrayDeque<>();

		/** Adds {@code newColors} to {@code valueNumber}'s own colors, queueing it when they grow. */
		void color(int valueNumber, Set<Color> newColors) {
			if (valueNumber < 0 || newColors.isEmpty())
				return;

			this.direct.computeIfAbsent(valueNumber, _ -> EnumSet.noneOf(Color.class)).addAll(newColors);
			this.addColors(valueNumber, newColors, false);
		}

		/** Adds {@code newColors} to the element under {@code key} of the container {@code valueNumber}, queueing it when they grow. */
		void colorElement(int valueNumber, String key, Set<Color> newColors) {
			if (valueNumber < 0 || newColors.isEmpty())
				return;

			Set<Color> element = this.elements.computeIfAbsent(valueNumber, _ -> new HashMap<>()).computeIfAbsent(key,
					_ -> EnumSet.noneOf(Color.class));
			boolean grew = !element.containsAll(newColors);
			element.addAll(newColors);
			this.addColors(valueNumber, newColors, grew);
		}

		/** Adds {@code newColors} to each definition of {@code instruction}. */
		void colorDefs(SSAInstruction instruction, Set<Color> newColors) {
			for (int d = 0; d < instruction.getNumberOfDefs(); d++)
				this.color(instruction.getDef(d), newColors);
		}

		/** Colors each definition of {@code instruction} with what a callee returns, keeping a returned container's elements apart. */
		void colorDefs(SSAInstruction instruction, Returned returned) {
			for (int d = 0; d < instruction.getNumberOfDefs(); d++) {
				int def = instruction.getDef(d);
				this.color(def, returned.whole());
				returned.elements().forEach((key, elementColors) -> this.colorElement(def, key, elementColors));
			}
		}

		/** The colors a read of {@code valueNumber}'s element under {@code key} yields; {@code null} means an unknown key. */
		Set<Color> elementRead(int valueNumber, String key) {
			Map<String, Set<Color>> keyed = this.elements.get(valueNumber);

			if (keyed == null || key == null)
				return this.colors.get(valueNumber);

			Set<Color> ret = EnumSet.noneOf(Color.class);
			ret.addAll(this.direct.getOrDefault(valueNumber, Set.of()));
			ret.addAll(keyed.getOrDefault(key, Set.of()));
			return ret;
		}

		/** The union of the colors of {@code uses}. */
		Set<Color> colorsOf(Stream<Integer> uses) {
			Set<Color> ret = EnumSet.noneOf(Color.class);
			uses.map(this.colors::get).filter(c -> c != null).forEach(ret::addAll);
			return ret;
		}

		private void addColors(int valueNumber, Set<Color> newColors, boolean requeue) {
			Set<Color> existing = this.colors.computeIfAbsent(valueNumber, _ -> EnumSet.noneOf(Color.class));

			if (!existing.containsAll(newColors) || requeue) {
				existing.addAll(newColors);
				this.worklist.push(valueNumber);
			}
		}
	}

	/** The call-graph builder, which the engine asks to name the TensorFlow API a call invokes and the program functions it runs. */
	private final PythonSSAPropagationCallGraphBuilder builder;

	/** The pointer analysis, used to resolve member names and to look up tensor typing. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/** The tensor-typed pointer keys, which tell a shape value combined with data apart from one combined with constants. */
	private final Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys;

	/** The walk memo, by (node, direction, seeded parameter colors). */
	private final Map<String, Returned> memo = new HashMap<>();

	ShapeValueBoundaryAnalysis(PythonSSAPropagationCallGraphBuilder builder, Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys) {
		this.builder = builder;
		this.pointerAnalysis = builder.getPointerAnalysis();
		this.tensorTypedKeys = tensorTypedKeys;
	}

	/**
	 * True iff a shape known only as a value crosses the boundary of the function whose node is {@code node}, in either direction.
	 *
	 * @param node The function's call-graph node.
	 * @param method True iff the function is an instance method, whose receiver slot is not a source.
	 * @return {@code TRUE} if a shape value crosses, {@code FALSE} if none does, or {@code null} if the node has no IR.
	 */
	Boolean passesShapeValue(CGNode node, boolean method) {
		IR ir = node.getIR();

		if (ir == null)
			return null;

		if (this.returned(node, Direction.OUTBOUND, Map.of()).all().contains(Color.SHAPE))
			return Boolean.TRUE;

		// Slot 0 is the function object itself; slot 1 is the receiver of an instance method.
		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();
		Map<Integer, Set<Color>> seed = new TreeMap<>();

		for (int i = method ? 2 : 1; i < parameters.length; i++)
			seed.put(i, EnumSet.of(Color.PARAMETER));

		return this.returned(node, Direction.INBOUND, seed).all().contains(Color.SHAPED_BY_PARAMETER);
	}

	/**
	 * What reaches a return of {@code node}, walking in {@code direction} from {@code seed}, the colors of its parameter slots, and,
	 * outbound, from its own dynamic shape ops and the shape values its user callees return of their own accord.
	 */
	private Returned returned(CGNode node, Direction direction, Map<Integer, Set<Color>> seed) {
		String key = node.getGraphNodeId() + ":" + direction + ":" + seed;
		Returned cached = this.memo.get(key);

		if (cached != null)
			return cached;

		// Optimistic cycle guard: a recursive revisit contributes nothing new.
		this.memo.put(key, Returned.NONE);

		IR ir = node.getIR();

		if (ir == null)
			return Returned.NONE;

		DefUse defUse = node.getDU();
		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();
		Walk walk = new Walk();
		Set<Color> whole = EnumSet.noneOf(Color.class);
		Map<String, Set<Color>> elements = new HashMap<>();

		seed.forEach((slot, slotColors) -> {
			if (slot < parameters.length)
				walk.color(parameters[slot], slotColors);
		});

		// Outbound, the sources are the node's own dynamic shape ops and the shape values its user callees return of their own accord,
		// whether or not any argument is colored.
		if (direction == Direction.OUTBOUND)
			for (SSAInstruction instruction : ir.getInstructions())
				if (instruction instanceof PythonInvokeInstruction invoke)
					if (this.invokesDynamicShapeOp(node, invoke))
						walk.colorDefs(invoke, EnumSet.of(Color.SHAPE));
					else
						for (CGNode callee : this.userCallees(node, invoke).keySet())
							walk.colorDefs(invoke, this.returned(callee, direction, Map.of()));

		while (!walk.worklist.isEmpty()) {
			int valueNumber = walk.worklist.pop();
			Set<Color> valueColors = walk.colors.get(valueNumber);

			for (Iterator<SSAInstruction> uses = defUse.getUses(valueNumber); uses.hasNext();) {
				SSAInstruction use = uses.next();

				// A closure's free variables are not followed.
				// TODO: Follow them. A shape built inside a nested scope, such as a list comprehension returning `tf.shape(x)[i]`
				// elements, is missed here, so the function is not refused. The optimistic cycle guard above misses recursion the same
				// way: a recursive revisit reads the memo's NONE and is never revisited once the walk settles, so a shape value
				// returned only through the recursion is missed. The follow-up should follow the lexical-read precedent of
				// https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1061 (NumpyParameterFlowAnalysis and
				// StaticShapeReadAnalysis color a lexical write's variable and scan the nested functions reading it), and iterate a
				// recursive walk to a fixpoint rather than reading NONE.
				if (use instanceof AstLexicalWrite)
					continue;

				if (use instanceof SSAReturnInstruction) {
					Map<String, Set<Color>> keyed = walk.elements.get(valueNumber);

					if (keyed == null)
						whole.addAll(valueColors);
					else {
						whole.addAll(walk.direct.getOrDefault(valueNumber, Set.of()));
						keyed.forEach((k, c) -> elements.computeIfAbsent(k, _ -> EnumSet.noneOf(Color.class)).addAll(c));
					}

					continue;
				}

				if (use instanceof PythonPropertyRead read) {
					// Only the object's color flows into a read; an index's color flows into it inbound, since the result depends on
					// the index's value, but outbound a shape value indexing data yields data.
					if (read.getObjectRef() == valueNumber) {
						String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);

						// A static shape or dtype read reads the type, which crosses.
						if (member != null && TYPE_MEMBER_NAMES.contains(member))
							continue;

						Set<Color> readColors = walk.elementRead(valueNumber, constantKey(ir, read.getMemberRef()));

						if (readColors != null)
							walk.colorDefs(use, readColors);
					} else if (direction == Direction.INBOUND)
						walk.colorDefs(use, valueColors);

					continue;
				}

				// A value written into a container colors the container's element, which a return, a read or an op may then carry.
				if (use instanceof PythonPropertyWrite write) {
					if (write.getValue() == valueNumber) {
						String elementKey = constantKey(ir, write.getMemberRef());

						if (elementKey == null)
							walk.color(write.getObjectRef(), valueColors);
						else
							walk.colorElement(write.getObjectRef(), elementKey, valueColors);
					}

					continue;
				}

				if (use instanceof PythonInvokeInstruction invoke) {
					this.handleInvoke(node, invoke, defUse, direction, walk);
					continue;
				}

				// Outbound, arithmetic over a shape value is a shape value unless it combines one with a tensor that is not one, and a
				// comparison of one is a boolean.
				if (direction == Direction.OUTBOUND && (use instanceof SSABinaryOpInstruction || use instanceof SSAUnaryOpInstruction)
						&& (this.combinesWithData(node, use, walk.colors)
								|| use instanceof SSABinaryOpInstruction binary && StaticShapeReadAnalysis.COMPARISON_OPERATOR_NAMES
										.contains(String.valueOf(binary.getOperator()).toLowerCase())))
					continue;

				walk.colorDefs(use, valueColors);
			}
		}

		Returned ret = new Returned(whole, elements);
		this.memo.put(key, ret);
		return ret;
	}

	/** Colors the result of {@code invoke}, one of whose uses was just colored. */
	private void handleInvoke(CGNode node, PythonInvokeInstruction invoke, DefUse defUse, Direction direction, Walk walk) {
		Set<String> ops = this.resolvedApis(node, invoke);

		// A dynamic shape op reads its argument's type: inbound it launders, and outbound its result is already a source.
		if (ops.stream().anyMatch(StaticShapeReadAnalysis.DYNAMIC_SHAPE_OP_FQNS::contains))
			return;

		// Outbound, a comparison of a shape value is a boolean, not a shape.
		if (direction == Direction.OUTBOUND && ops.stream().anyMatch(COMPARISON_OP_FQNS::contains))
			return;

		ShapeArguments shapeOperands = ops.stream().map(SHAPE_OPERANDS::get).filter(o -> o != null).findFirst().orElse(null);

		if (shapeOperands != null) {
			// Outbound, the op consumes the shape: its result is the data the shape was computed for, whatever its data operand, so it is
			// left uncolored. The cost is reshaping a shape vector itself, which is not followed.
			if (direction == Direction.OUTBOUND)
				return;

			Set<Integer> shapeUses = shapeOperandUses(invoke, shapeOperands);
			Set<Color> shapeColors = walk.colorsOf(shapeUses.stream());
			Set<Color> result = walk.colorsOf(allArgumentUses(invoke));

			// The inbound event: a parameter's value gives the result's static dimensions.
			if (shapeColors.contains(Color.PARAMETER))
				result.add(Color.SHAPED_BY_PARAMETER);

			walk.colorDefs(invoke, result);

			// `x.set_shape(s)` gives its receiver the dimensions rather than its result.
			if (ops.contains(SET_SHAPE_OP_FQN) && defUse.getDef(invoke.getUse(0)) instanceof PythonPropertyRead method)
				walk.color(method.getObjectRef(), result);

			return;
		}

		boolean descended = false;

		for (Map.Entry<CGNode, Boolean> callee : this.userCallees(node, invoke).entrySet()) {
			descended = true;
			Map<Integer, Set<Color>> seed = argumentSlots(invoke, callee.getKey(), callee.getValue(), walk.colors);

			if (seed == null)
				// A starred argument: the slots cannot be aligned, so the call's result takes its arguments' colors.
				walk.colorDefs(invoke, walk.colorsOf(allUses(invoke)));
			else
				walk.colorDefs(invoke, this.returned(callee.getKey(), direction, seed));
		}

		// A library op, a builtin, or an unresolved call: its result takes its arguments' colors, except that outbound a shape value
		// combined with a tensor that is not one is data.
		if (!descended && !(direction == Direction.OUTBOUND && this.combinesWithData(node, invoke, walk.colors)))
			walk.colorDefs(invoke, walk.colorsOf(allUses(invoke)));
	}

	/**
	 * The key of a container element read or written under the constant {@code member}, such as a tuple index, or {@code null} if the
	 * member is not a constant.
	 */
	private static String constantKey(IR ir, int member) {
		return ir.getSymbolTable().isConstant(member) ? String.valueOf(ir.getSymbolTable().getConstantValue(member)) : null;
	}

	/**
	 * The program functions whose parameters receive {@code invoke}'s own arguments, as the engine reports them
	 * ({@link CallTargets#parameterBoundTargets}), each with whether a receiver, the instance of a bound method or the class of a class
	 * method, is bound before the arguments.
	 *
	 * @param node The node whose IR holds {@code invoke}.
	 * @param invoke The call.
	 * @return Each such function's node, mapped to whether a receiver is bound first.
	 */
	private Map<CGNode, Boolean> userCallees(CGNode node, PythonInvokeInstruction invoke) {
		return CallTargets.parameterBoundTargets(this.builder, node, invoke);
	}

	/**
	 * The colors of {@code invoke}'s arguments, by {@code callee}'s parameter slot, or {@code null} if the call spreads a starred argument.
	 * Positional arguments align by position, shifted by one past a bound receiver; keyword arguments align by parameter name.
	 */
	private static Map<Integer, Set<Color>> argumentSlots(PythonInvokeInstruction invoke, CGNode callee, boolean bound,
			Map<Integer, Set<Color>> colors) {
		if (invoke.getStarredPositions().length > 0)
			return null;

		Map<Integer, Set<Color>> slots = new TreeMap<>();
		int shift = bound ? 1 : 0;

		for (int j = 1; j < invoke.getNumberOfPositionalParameters(); j++) {
			Set<Color> argumentColors = colors.get(invoke.getUse(j));

			if (argumentColors != null)
				slots.put(j + shift, argumentColors);
		}

		IR ir = callee.getIR();

		if (ir != null && !invoke.getKeywords().isEmpty()) {
			int[] parameters = ir.getSymbolTable().getParameterValueNumbers();

			for (String keyword : invoke.getKeywords()) {
				Set<Color> argumentColors = colors.get(invoke.getUse(keyword));

				if (argumentColors != null)
					for (int i = 1; i < parameters.length; i++) {
						String[] names = ir.getLocalNames(0, parameters[i]);

						if (names != null && Set.of(names).contains(keyword))
							slots.put(i, argumentColors);
					}
			}
		}

		return slots;
	}

	/**
	 * True iff {@code instruction} combines a colored operand with a tensor-typed operand that is not colored: data, not a shape. Tensor
	 * typing is the engine's; a value it leaves untyped does not count as data, which keeps the walk conservative.
	 */
	private boolean combinesWithData(CGNode node, SSAInstruction instruction, Map<Integer, Set<Color>> colors) {
		int first = instruction instanceof PythonInvokeInstruction ? 1 : 0;

		for (int j = first; j < instruction.getNumberOfUses(); j++) {
			int use = instruction.getUse(j);

			if (use >= 0 && !colors.containsKey(use)
					&& this.tensorTypedKeys.containsKey(this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, use)))
				return true;
		}

		return false;
	}

	/** True iff {@code invoke}'s resolved callee is a dynamic shape op. */
	private boolean invokesDynamicShapeOp(CGNode node, PythonInvokeInstruction invoke) {
		return this.resolvedApis(node, invoke).stream().anyMatch(StaticShapeReadAnalysis.DYNAMIC_SHAPE_OP_FQNS::contains);
	}

	/**
	 * The fully-qualified names of the TensorFlow APIs {@code invoke} may invoke, as the engine names them ({@link TensorFlowApis}).
	 *
	 * @param node The node whose IR holds {@code invoke}.
	 * @param invoke The call.
	 * @return The APIs' names, such as {@code tensorflow.reshape}; empty when the call invokes no API the engine names.
	 */
	private Set<String> resolvedApis(CGNode node, PythonInvokeInstruction invoke) {
		return TensorFlowApis.of(this.builder, node, invoke);
	}

	/** The value numbers {@code invoke} passes in {@code operands}. */
	private static Set<Integer> shapeOperandUses(PythonInvokeInstruction invoke, ShapeArguments operands) {
		Set<Integer> uses = new HashSet<>();

		// The callee is the invoke's first use, so the argument at index i is use i + 1.
		for (int position : operands.positions())
			if (position + 1 < invoke.getNumberOfPositionalParameters())
				uses.add(invoke.getUse(position + 1));

		for (String keyword : operands.keywords())
			if (invoke.getKeywords().contains(keyword))
				uses.add(invoke.getUse(keyword));

		return uses;
	}

	/** The value numbers of {@code invoke}'s arguments, the callee excluded. */
	private static Stream<Integer> allArgumentUses(PythonInvokeInstruction invoke) {
		return Stream.iterate(1, j -> j < invoke.getNumberOfUses(), j -> j + 1).map(invoke::getUse);
	}

	/** The value numbers of {@code invoke}'s uses, the callee included. */
	private static Stream<Integer> allUses(PythonInvokeInstruction invoke) {
		return Stream.iterate(0, j -> j < invoke.getNumberOfUses(), j -> j + 1).map(invoke::getUse);
	}
}
