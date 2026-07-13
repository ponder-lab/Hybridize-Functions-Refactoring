package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.collections.Pair;

/**
 * The parameter-flow numpy precondition: an interprocedural, two-color taint analysis deciding whether a function (transitively through
 * user-defined callees) applies a numpy/scipy API to a value flowing from its parameters, which raises under {@code @tf.function} tracing
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/740). The scan distinguishes a tensor's <em>value</em> (numpy over
 * it always raises) from its <em>shape</em> metadata, whose staticness is decided per-dimension from the source tensor's inferred
 * {@link TensorType} (issue 747), with the shape's provenance propagated across call boundaries (issue 756) and shape slices narrowed
 * rank-independently where possible (issue 761). An instance is created per function check and carries the analyses the scan consults: the
 * call graph, the pointer analysis, the (node, value number) tensor-type index, and the scan memo, which is shared across the function's
 * call-graph nodes. Extracted from {@link Util} (issue 759).
 */
class NumpyParameterFlowAnalysis {

	/** The call graph, used to follow user-defined callees. */
	private final CallGraph callGraph;

	/** The pointer analysis, used to resolve attribute names, module roots, and interprocedural integer constants. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/**
	 * The (node, value number) index of the tensor-type analysis, consulted for the per-dimension shape staticness at a numpy-over-shape
	 * sink.
	 */
	private final Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex;

	/** The scan memo (see {@link #scanForTaintedNumpySinks}), shared across the checked function's call-graph nodes. */
	private final Map<String, NumpyScanResult> memo = new HashMap<>();

	/**
	 * Creates an analysis over the given call graph, pointer analysis, and tensor-type analysis, whose local pointer keys are indexed once
	 * by (node, value number) for constant-time lookups at the shape sinks.
	 *
	 * @param callGraph The call graph, used to follow user-defined callees.
	 * @param pointerAnalysis The pointer analysis, used to resolve attribute names and module roots.
	 * @param tensorTypeAnalysis The tensor-type analysis, consulted for the per-dimension shape staticness.
	 */
	NumpyParameterFlowAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis, TensorTypeAnalysis tensorTypeAnalysis) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
		this.tensorTypeIndex = indexTensorTypes(tensorTypeAnalysis);
	}

	/** The {@code dtype} attribute: a trace-time-constant element type, so numpy over it is always safe (launders taint). */
	private static final String DTYPE_MEMBER_NAME = "dtype";

	/**
	 * The {@code shape} attribute: its read yields shape metadata (a {@link ShapeDescriptor} covering every dimension of the read tensor).
	 * numpy over shape metadata is safe iff the covered dimensions are statically known, which the scan decides per-dimension from the
	 * source tensor's {@link TensorType}. See {@link #scanForTaintedNumpySinks} and
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747.
	 */
	private static final String SHAPE_MEMBER_NAME = "shape";

	/** The built-in {@code slice} constructor, into which a Python {@code x[a:b:c]} subscript is lowered. */
	private static final String SLICE_BUILTIN_NAME = "slice";

	/**
	 * The unary-negation operator, matched by name: the operator enum ({@code IUnaryOpInstruction.Operator}) lives in a non-exported WALA
	 * package, so its type cannot be referenced here, but its {@code toString()} is stable.
	 */
	private static final String NEGATION_OPERATOR_NAME = "neg";

	/**
	 * Global-read names identifying the numpy/scipy modules by import alias, mirroring {@code Util}'s TensorFlow module global names.
	 * Unlike the TensorFlow fallback, whose over-recognition is incompleteness-safe (it only lets an eager function hybridize),
	 * over-recognition here over-blocks: a non-numpy global named {@code np} would fail the precondition. Accepted safety-first; the
	 * evaluation measures the cost. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/740.
	 */
	private static final Set<String> NUMPY_MODULE_GLOBAL_NAMES = Set.of("global numpy", "global np", "global scipy", "global sp");

	/** Type-name prefixes of the numpy/scipy module objects, for the points-to (precise) branch of the module test. */
	private static final Set<String> NUMPY_MODULE_TYPE_NAME_PREFIXES = Set.of("Lnumpy", "Lscipy");

	/**
	 * True iff {@code node} (transitively through user-defined callees) applies a numpy/scipy API to a value flowing from its parameters.
	 * Under {@code @tf.function}, parameter values become symbolic tensors during tracing, and numpy/scipy applied to them raises, so such
	 * a function crashes on first call when hybridized. The gate is an SSA def-use taint slice rather than a points-to intersection: a
	 * points-to-keyed gate silently passes a crasher under any modeling gap that empties a points-to set, and this precondition exists
	 * precisely to hold while upstream modeling is in motion. A {@code dtype} read launders taint (the element type is a trace-time
	 * constant); a {@code shape} read (or {@code tf.shape}/{@code get_shape_list}) yields shape metadata, over which numpy is declined only
	 * when a covered dimension is provably dynamic, decided per-dimension from the source tensor's inferred {@link TensorType} (the
	 * per-dimension shape-aware verdict of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747), with the shape's
	 * provenance propagated across call boundaries so staticness resolves interprocedurally
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756). Known narrowings, each documented on the issue:
	 * positional arguments only cross call sites, field-mediated and subscript-store flows are not tracked, and a method's receiver is
	 * never a source. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/740.
	 *
	 * @param node The call-graph node to check.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a taint source.
	 * @return True iff a numpy/scipy API is applied to a parameter-flowing value reachable from {@code node}.
	 */
	boolean appliesNumpyToParameters(CGNode node, boolean method) {
		IR ir = node.getIR();

		if (ir == null)
			return false;

		// Parameter value numbers: slot 0 is the function object itself; slot 1 is the receiver for an instance method. Neither is a
		// taint source—tainting the receiver would block any method mixing numpy over scalar fields with sanitized tensor use.
		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();
		Set<Integer> sources = new HashSet<>();

		for (int i = method ? 2 : 1; i < parameters.length; i++)
			sources.add(parameters[i]);

		if (sources.isEmpty())
			return false;

		return this.scanForTaintedNumpySinks(node, sources, new HashSet<>(), new HashMap<>()).sink();
	}

	/**
	 * Indexes {@code tensorTypeAnalysis}'s local pointer keys by (node, value number) once, so per-value lookups
	 * ({@link #lookupTensorTypes}) are constant-time rather than a full linear scan of the analysis per call. Build this once per function
	 * (or project) and reuse it across the function's call-graph nodes rather than rebuilding per node.
	 */
	private static Map<CGNode, Map<Integer, Set<TensorType>>> indexTensorTypes(TensorTypeAnalysis tensorTypeAnalysis) {
		Map<CGNode, Map<Integer, Set<TensorType>>> index = new HashMap<>();

		for (Pair<PointerKey, TensorVariable> pair : tensorTypeAnalysis)
			if (pair.fst instanceof LocalPointerKey local && pair.snd != null)
				index.computeIfAbsent(local.getNode(), n -> new HashMap<>()).computeIfAbsent(local.getValueNumber(), v -> new HashSet<>())
						.addAll(pair.snd.getTypes());

		return index;
	}

	/**
	 * The {@link TensorType}s the tensor-type analysis associates with the local {@code valueNumber} in {@code node}, from the (node, value
	 * number) index built by {@link #indexTensorTypes}. Empty when the analysis has no tensor classification for the value (not analyzed,
	 * or classified not-a-tensor).
	 */
	private Set<TensorType> lookupTensorTypes(CGNode node, int valueNumber) {
		return this.tensorTypeIndex.getOrDefault(node, Map.of()).getOrDefault(valueNumber, Set.of());
	}

	/**
	 * A shape-derived value tracked by the scan: the tensor whose shape it came from ({@code sourceTensor}, a value number in
	 * {@code sourceNode}) and which of that tensor's dimensions it covers ({@code dims}; {@code null} means all dimensions). A non-negative
	 * index is absolute; a negative index counts from the end (Python semantics), resolved against each {@link TensorType}'s own rank at
	 * the sink ({@link #numpyOverShapeStaticness}), so a suffix slice stays tracked even when the source's rank is statically unresolvable
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/761). numpy over such a value is safe iff the covered
	 * dimensions are statically known; the descriptor lets the sink consult the source tensor's per-dim {@link TensorType}. The source
	 * frame need not be the frame the sink sits in: the descriptor crosses call boundaries with the shape taint (argument seeding and
	 * return flow, https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756), and the (node, value number) lookup stays
	 * valid because the tensor-type index is global. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747.
	 */
	private record ShapeDescriptor(CGNode sourceNode, int sourceTensor, Set<Integer> dims) {
	}

	/** The staticness verdict for numpy over a shape-derived value: safe, a crash (dynamic dim), or unprovable (⊤/unresolved). */
	private enum ShapeStaticness {
		STATIC, DYNAMIC, TOP
	}

	/**
	 * The verdict for numpy applied to the shape-derived value described by {@code descriptor}. Consults the source tensor's per-dim
	 * {@link TensorType} across all its analysis contexts: {@link ShapeStaticness#DYNAMIC} if any covered dimension is non-numeric in any
	 * context (numpy would crash under tracing); {@link ShapeStaticness#STATIC} if every covered dimension is a {@link NumericDim} in every
	 * context (safe); {@link ShapeStaticness#TOP} if the source is untyped or its shape is ⊤ (staticness cannot be proven either way). A
	 * negative covered index counts from the end of each context's own rank, so a suffix slice resolves per-context without a
	 * statically-known rank (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/761); a covered index beyond a context's
	 * rank contributes no element there (Python clamps a slice to the vector's extent) and is ignored for that context.
	 */
	private ShapeStaticness numpyOverShapeStaticness(ShapeDescriptor descriptor) {
		// An empty covered set (an empty slice such as `[:0]`) consumes no dimension values: vacuously static, even for an untyped
		// source.
		if (descriptor.dims() != null && descriptor.dims().isEmpty())
			return ShapeStaticness.STATIC;

		Set<TensorType> types = this.lookupTensorTypes(descriptor.sourceNode(), descriptor.sourceTensor());

		if (types.isEmpty())
			return ShapeStaticness.TOP;

		boolean top = false;

		for (TensorType type : types) {
			List<Dimension<?>> typeDims = type.getDims();

			if (typeDims == null) {
				top = true;
				continue;
			}

			Set<Integer> covered = descriptor.dims();

			if (covered == null) {
				covered = new TreeSet<>();

				for (int i = 0; i < typeDims.size(); i++)
					covered.add(i);
			}

			for (int i : covered) {
				int index = i < 0 ? typeDims.size() + i : i;

				// Covered dimensions come from slices, and Python clamps a slice to the vector's extent: an index beyond this
				// context's rank contributes no element here, so it is ignored rather than degrading the verdict.
				if (index < 0 || index >= typeDims.size())
					continue;

				Dimension<?> dim = typeDims.get(index);

				if (dim instanceof NumericDim)
					continue; // a statically-known size: numpy over it is safe.

				// An UnresolvedDim is a fixed run-time size the analysis could not compute (e.g. a config-sourced dimension,
				// https://github.com/wala/ML/issues/721): the run-time TensorShape reports a concrete integer, so get_shape_list does
				// not patch it with tf.shape(...) and numpy stays safe. Its staticness is unprovable, so it is treated like ⊤:
				// permitted under the precision-favoring policy, declined under the sound policy
				// (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/751). This is optimism, not a soundness
				// guarantee: for a `tf.shape(x)[i]` extraction whose axis carries no run-time-None evidence either way, Ariadne
				// classifies UnresolvedDim (wala/ML#722), and permitting it lets numpy (e.g. `np.prod`) run over a dimension that a
				// TensorShape may patch to None at run time. The sound policy (#751) declines this case.
				if (dim instanceof UnresolvedDim) {
					top = true;
					continue;
				}

				return ShapeStaticness.DYNAMIC; // a run-time-dynamic covered dim (DynamicDim/symbolic/ragged) crashes numpy.
			}
		}

		return top ? ShapeStaticness.TOP : ShapeStaticness.STATIC;
	}

	/**
	 * Resolves {@code value} in {@code node} to an integer constant, or {@code null} if it cannot be resolved. Handles a literal integer in
	 * the symbol table, a unary negation of a resolvable value, and an interprocedural constant surfaced by the pointer analysis (a
	 * {@link ConstantKey} in the value's points-to set), mirroring how {@link Function#inferPrimitiveParameters} recovers literal arguments
	 * across call sites.
	 */
	private Integer resolveIntConstant(CGNode node, int value, DefUse defUse) {
		SymbolTable symbolTable = node.getIR().getSymbolTable();

		if (symbolTable.isIntegerConstant(value))
			return symbolTable.getIntValue(value);

		SSAInstruction def = defUse.getDef(value);

		// A unary negation (the `-k` in a `[-k:]` slice); matched by name since the operator enum's type cannot be referenced here.
		if (def instanceof SSAUnaryOpInstruction unary && NEGATION_OPERATOR_NAME.equalsIgnoreCase(String.valueOf(unary.getOpcode()))) {
			Integer operand = this.resolveIntConstant(node, unary.getUse(0), defUse);
			return operand == null ? null : -operand;
		}

		PointerKey pointerKey = this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, value);
		Integer resolved = null;

		for (InstanceKey instanceKey : this.pointerAnalysis.getPointsToSet(pointerKey))
			if (instanceKey instanceof ConstantKey<?> constantKey && constantKey.getValue() instanceof Number number) {
				// Only an integral value in int range is a usable slice bound; a non-integral (e.g. float) constant would truncate under
				// intValue() and wrongly "prove" a bound, so treat it as unresolvable.
				long asLong = number.longValue();

				if (number.doubleValue() != asLong || asLong < Integer.MIN_VALUE || asLong > Integer.MAX_VALUE)
					return null;

				int candidate = (int) asLong;

				if (resolved != null && resolved != candidate)
					return null; // ambiguous across contexts; refuse to guess.

				resolved = candidate;
			}

		return resolved;
	}

	/**
	 * The dimension indices selected by a Python slice {@code [start:stop:step]} of a shape vector of the given {@code rank}, following
	 * Python's clamping and negative-index semantics, or {@code null} if the slice cannot be resolved to a concrete set (a non-unit or
	 * unknown step). {@code null} bounds default to the whole extent.
	 */
	private static Set<Integer> resolveSliceDims(Integer start, Integer stop, Integer step, int rank) {
		int stride = step == null ? 1 : step;

		if (stride != 1)
			return null; // only unit-stride slices are modeled precisely.

		int from = start == null ? 0 : start < 0 ? Math.max(0, rank + start) : Math.min(start, rank);
		int to = stop == null ? rank : stop < 0 ? Math.max(0, rank + stop) : Math.min(stop, rank);
		Set<Integer> dims = new TreeSet<>();

		for (int i = from; i < to; i++)
			dims.add(i);

		return dims;
	}

	/** True iff {@code invoke} calls the built-in {@code slice} constructor (how a Python {@code x[a:b:c]} subscript is modeled). */
	private static boolean invokesSliceBuiltin(PythonInvokeInstruction invoke, DefUse defUse) {
		SSAInstruction def = defUse.getDef(invoke.getUse(0));

		if (!(def instanceof AstLexicalRead lexical))
			return false;

		// Read the name via getName() rather than the variableName field, and guard the array: mirrors PythonModRefWithBuiltinFunctions
		// (WALA 1.8.0 encapsulated the field).
		Access[] accesses = lexical.getAccesses();
		return accesses.length > 0 && SLICE_BUILTIN_NAME.equals(accesses[0].getName().fst);
	}

	/**
	 * The result of a two-color taint scan: whether numpy hit a value-tainted argument, whether a value taint escaped this node, and the
	 * {@link ShapeDescriptor} the node's returns carry, so a caller keeps resolving staticness across the return boundary
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756). The descriptor is {@code null} when no shape-tainted
	 * value is returned <em>or</em> any returned shape lacks a descriptor (provenance is dropped rather than partially trusted);
	 * conflicting tracked returns merge to the ambiguous descriptor instead.
	 */
	private record NumpyScanResult(boolean sink, boolean valueEscapes, ShapeDescriptor returnDescriptor) {
	}

	/**
	 * Worklist taint propagation over {@code node}'s def-use chains, tracking two taint colors. A <em>value</em> taint marks the tensor
	 * value itself, over which numpy always raises under {@code tf.function} tracing (a sink, decline). A <em>shape</em> taint marks a
	 * value derived from a tensor's shape (via {@code .shape}/{@code .shape.as_list()}, {@code tf.shape}/{@code size}/{@code rank}, or a
	 * user-defined shape extractor such as {@code get_shape_list}); it carries a {@link ShapeDescriptor} identifying the source tensor and
	 * the dimensions it covers, narrowed as the shape vector is sliced ({@code [-k:]}, with the bound resolved via the pointer analysis).
	 * numpy over a shape-tainted argument is declined only when a covered dimension is provably dynamic in the source tensor's
	 * {@link TensorType} ({@link #numpyOverShapeStaticness}); a proven-static shape is safe, and only a provably-dynamic covered dimension
	 * is a sink. The descriptor crosses call boundaries with the taint
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756): a shape-tainted argument seeds its descriptor onto the
	 * callee's parameter ({@code shapeDescriptorSeed}), and a callee returning a tracked shape carries its (possibly slice-narrowed)
	 * descriptor back to the call-site result, so staticness resolves in whichever frame the sink sits. An unprovable shape - a ⊤ type or
	 * untracked/ambiguous provenance - is permitted, favoring precision. The sound decline-unless-provably-static variant is tracked on
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/751; see
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747 for the per-dimension shape-aware verdict.
	 * <p>
	 * Call sites are tainted conservatively (a value argument taints the call-site result), which follows a comprehension's
	 * element-through-container flow (NLPGNN's {@code TUDataset.cat}, issue 745). The result is colored SHAPE only when the callee is a
	 * pure <em>shape extractor</em> — its value-tainted arguments are consumed solely by shape operations and never escape (e.g.
	 * {@code get_shape_list}), whose result then describes the tensor argument's shape — otherwise the result is value-tainted. A callee's
	 * {@code valueEscapes} bit records whether any value taint reached a non-shape use inside it, and is what distinguishes a shape
	 * extractor from a value carrier. A {@code dtype} read launders taint entirely. Memoized on (node, value seed, shape seed, shape
	 * descriptor seed) with an optimistic cycle guard. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747.
	 */
	private NumpyScanResult scanForTaintedNumpySinks(CGNode node, Set<Integer> valueSeed, Set<Integer> shapeSeed,
			Map<Integer, ShapeDescriptor> shapeDescriptorSeed) {
		Map<Integer, String> descriptorKey = new TreeMap<>();
		shapeDescriptorSeed.forEach((value, descriptor) -> descriptorKey.put(value, this.renderDescriptorForMemoKey(descriptor)));
		String key = this.callGraph.getNumber(node) + ":" + new TreeSet<>(valueSeed) + ":" + new TreeSet<>(shapeSeed) + ":" + descriptorKey;
		NumpyScanResult cached = this.memo.get(key);

		if (cached != null)
			return cached;

		// Optimistic cycle guard: a recursive revisit contributes nothing new.
		this.memo.put(key, new NumpyScanResult(false, false, null));

		IR ir = node.getIR();

		if (ir == null)
			return new NumpyScanResult(false, false, null);

		DefUse defUse = node.getDU();
		Set<Integer> valueTainted = new HashSet<>(valueSeed);
		Set<Integer> shapeTainted = new HashSet<>(shapeSeed);
		// For each shape-tainted value, the tensor+dimensions its shape came from (absent = shape-derived but provenance lost). Seeded
		// interprocedurally: a shape-tainted argument's descriptor arrives on the corresponding parameter (issue 756).
		Map<Integer, ShapeDescriptor> shapeDescriptors = new HashMap<>(shapeDescriptorSeed);
		Deque<Integer> worklist = new ArrayDeque<>();
		worklist.addAll(valueSeed);
		worklist.addAll(shapeSeed);
		boolean sink = false;
		boolean valueEscapes = false;
		// The descriptor this node's returns carry back to the caller: set while all tracked-shape returns agree, poisoned to
		// AMBIGUOUS_DESCRIPTOR on a conflict, and dropped entirely (null) when an untracked shape is returned.
		ShapeDescriptor returnDescriptor = null;
		boolean returnedUntrackedShape = false;

		while (!worklist.isEmpty()) {
			int valueNumber = worklist.pop();
			boolean valueColored = valueTainted.contains(valueNumber);

			for (Iterator<SSAInstruction> uses = defUse.getUses(valueNumber); uses.hasNext();) {
				SSAInstruction use = uses.next();

				// A `dtype` read is a trace-time constant and launders taint entirely; a `shape` read yields shape metadata (SHAPE color)
				// covering every dimension of the read tensor.
				if (use instanceof PythonPropertyRead read && read.getObjectRef() == valueNumber) {
					String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);

					if (DTYPE_MEMBER_NAME.equals(member))
						continue;

					if (SHAPE_MEMBER_NAME.equals(member)) {
						for (int d = 0; d < read.getNumberOfDefs(); d++)
							colorShapeFrom(read.getDef(d), new ShapeDescriptor(node, valueNumber, null), valueTainted, shapeTainted,
									shapeDescriptors, worklist);
						continue;
					}
				}

				if (use instanceof PythonInvokeInstruction invoke) {
					if (this.invokesNumpyApi(node, invoke, defUse)) {
						// A value-tainted argument is a sink (a value escape). A shape-derived argument is a sink only on a
						// provably-dynamic covered dimension; an unprovable shape (⊤ or untracked/ambiguous provenance) is permitted,
						// favoring precision. The sound decline-unless-provably-static variant is tracked on
						// https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/751.
						if (valueColored) {
							sink = true;
							valueEscapes = true;
						} else {
							ShapeDescriptor descriptor = shapeDescriptors.get(valueNumber);

							if (descriptor != null && this.numpyOverShapeStaticness(descriptor) == ShapeStaticness.DYNAMIC)
								sink = true;
						}

						continue;
					}

					// A Python `x[a:b:c]` subscript is modeled as `slice(x, a, b, c)`. Only a slice of a shape-tainted vector stays shape
					// metadata (covering the sliced dimensions); slicing a tensor VALUE yields a sub-tensor value (e.g. `boxes1[..., :2]`),
					// which must remain value-tainted so numpy over it is still a sink. Guard on the receiver being shape-, not value-,
					// tainted.
					if (!valueColored && invokesSliceBuiltin(invoke, defUse) && invoke.getNumberOfUses() > 1
							&& invoke.getUse(1) == valueNumber) {
						ShapeDescriptor base = shapeDescriptors.get(valueNumber);
						ShapeDescriptor narrowed = base == null ? null : this.narrowBySlice(base, invoke, node, defUse);

						for (int d = 0; d < invoke.getNumberOfDefs(); d++)
							if (narrowed != null)
								colorShapeFrom(invoke.getDef(d), narrowed, valueTainted, shapeTainted, shapeDescriptors, worklist);
							else
								colorShape(invoke.getDef(d), valueTainted, shapeTainted, worklist);
						continue;
					}

					// tf.shape/size/rank consume a tensor and return shape metadata covering every dimension of the argument tensor.
					if (this.invokesShapeMetadataOp(node, invoke, defUse)) {
						ShapeDescriptor descriptor = invoke.getNumberOfUses() > 1 ? new ShapeDescriptor(node, invoke.getUse(1), null)
								: null;

						for (int d = 0; d < invoke.getNumberOfDefs(); d++)
							if (descriptor != null)
								colorShapeFrom(invoke.getDef(d), descriptor, valueTainted, shapeTainted, shapeDescriptors, worklist);
							else
								colorShape(invoke.getDef(d), valueTainted, shapeTainted, worklist);
						continue;
					}

					// Cross into user-defined callees, carrying each tainted argument's color to the corresponding parameter slot.
					Set<Integer> valueSlots = new TreeSet<>();
					Set<Integer> shapeSlots = new TreeSet<>();

					for (int j = 1; j < invoke.getNumberOfUses(); j++) {
						int arg = invoke.getUse(j);

						if (valueTainted.contains(arg))
							valueSlots.add(j);
						else if (shapeTainted.contains(arg))
							shapeSlots.add(j);
					}

					boolean calleeValueEscapes = false;
					boolean analyzedCallee = false;
					// The descriptor the callees' returns carry to this call site: all analyzed targets must agree on a tracked one; a
					// target returning an untracked shape (or none) drops it, and disagreeing targets poison it. Dropping forfeits only
					// the return-flow descriptor: a shape-extractor call site still falls back to the all-dimensions seed anchored to
					// the argument tensor (below), which over-approximates any shape metadata the extractor derives from that tensor.
					ShapeDescriptor calleeReturnDescriptor = null;
					boolean calleeReturnDescriptorDropped = false;

					if (!valueSlots.isEmpty() || !shapeSlots.isEmpty())
						for (CGNode target : this.callGraph.getPossibleTargets(node, invoke.getCallSite())) {
							if (Util.isTensorFlowNode(target))
								continue;

							IR targetIr = target.getIR();

							if (targetIr == null)
								continue;

							int[] targetParameters = targetIr.getSymbolTable().getParameterValueNumbers();
							Set<Integer> targetValueSeed = new HashSet<>();
							Set<Integer> targetShapeSeed = new HashSet<>();
							Map<Integer, ShapeDescriptor> targetDescriptorSeed = new HashMap<>();

							for (int slot : valueSlots)
								if (slot < targetParameters.length)
									targetValueSeed.add(targetParameters[slot]);

							for (int slot : shapeSlots)
								if (slot < targetParameters.length) {
									targetShapeSeed.add(targetParameters[slot]);

									// Thread the argument's shape provenance into the callee's frame, so a numpy sink there still
									// resolves staticness against the source tensor (issue 756).
									ShapeDescriptor argumentDescriptor = shapeDescriptors.get(invoke.getUse(slot));

									if (argumentDescriptor != null)
										targetDescriptorSeed.put(targetParameters[slot], argumentDescriptor);
								}

							if (targetValueSeed.isEmpty() && targetShapeSeed.isEmpty())
								continue;

							analyzedCallee = true;

							NumpyScanResult r = this.scanForTaintedNumpySinks(target, targetValueSeed, targetShapeSeed,
									targetDescriptorSeed);

							if (r.sink())
								sink = true;
							if (r.valueEscapes())
								calleeValueEscapes = true;

							if (r.returnDescriptor() == null)
								calleeReturnDescriptorDropped = true;
							else if (calleeReturnDescriptor == null)
								calleeReturnDescriptor = r.returnDescriptor();
							else if (!calleeReturnDescriptor.equals(r.returnDescriptor()))
								calleeReturnDescriptor = AMBIGUOUS_DESCRIPTOR;
						}

					if (calleeReturnDescriptorDropped)
						calleeReturnDescriptor = null;

					// A callee is a shape extractor iff it was analyzed and no value taint escaped inside it: then its result is shape
					// metadata (SHAPE). Otherwise the call-site result is conservatively value-tainted (a value argument reached a
					// non-shape use, directly or via a library callee), and that value has escaped this node.
					boolean hasValueArg = !valueSlots.isEmpty();
					boolean shapeExtractor = analyzedCallee && !calleeValueEscapes;
					boolean resultValue = hasValueArg && !shapeExtractor;
					boolean resultShape = !resultValue && (hasValueArg || !shapeSlots.isEmpty());

					if (resultValue)
						valueEscapes = true;

					// A shape-extractor result (e.g. `get_shape_list(t)`) is the shape vector of the tensor argument it consumed, covering
					// all of that tensor's dimensions: seed the descriptor from the (first) value-tainted argument. A descriptor carried
					// by the callee's return is preferred over that all-dimensions seed - it reflects any slice narrowing inside the
					// callee (issue 756).
					ShapeDescriptor extractorDescriptor = null;

					if (resultShape && shapeExtractor && !valueSlots.isEmpty())
						extractorDescriptor = new ShapeDescriptor(node, invoke.getUse(valueSlots.iterator().next()), null);

					ShapeDescriptor resultDescriptor = calleeReturnDescriptor != null ? calleeReturnDescriptor : extractorDescriptor;

					for (int d = 0; d < invoke.getNumberOfDefs(); d++) {
						int def = invoke.getDef(d);

						if (resultValue)
							colorValue(def, valueTainted, shapeTainted, worklist);
						else if (resultShape)
							if (resultDescriptor != null)
								colorShapeFrom(def, resultDescriptor, valueTainted, shapeTainted, shapeDescriptors, worklist);
							else
								colorShape(def, valueTainted, shapeTainted, worklist);
					}

					continue;
				}

				if (use instanceof SSAReturnInstruction) {
					if (valueColored)
						valueEscapes = true;
					else {
						// A returned shape-tainted value carries its descriptor back to the caller (issue 756). Conflicting descriptors
						// across returns (or a re-scan after poisoning) merge to AMBIGUOUS; an untracked shape return drops the
						// descriptor entirely.
						ShapeDescriptor descriptor = shapeDescriptors.get(valueNumber);

						if (descriptor == null)
							returnedUntrackedShape = true;
						else if (returnDescriptor == null)
							returnDescriptor = descriptor;
						else if (!returnDescriptor.equals(descriptor))
							returnDescriptor = AMBIGUOUS_DESCRIPTOR;
					}
					continue;
				}

				// Any other instruction is a non-shape use: a value-tainted operand escapes (its value flows onward), and the definitions
				// inherit the color (value dominates).
				if (valueColored)
					valueEscapes = true;

				for (int d = 0; d < use.getNumberOfDefs(); d++) {
					int def = use.getDef(d);

					if (valueColored)
						colorValue(def, valueTainted, shapeTainted, worklist);
					else
						colorShape(def, valueTainted, shapeTainted, worklist);
				}
			}
		}

		NumpyScanResult result = new NumpyScanResult(sink, valueEscapes, returnedUntrackedShape ? null : returnDescriptor);
		this.memo.put(key, result);
		return result;
	}

	/**
	 * A stable rendering of {@code descriptor} for the scan's memo key: the source node's call-graph number, the source tensor's value
	 * number, and the covered dimensions ({@code *} = all dimensions; {@code !} = ambiguous provenance).
	 */
	private String renderDescriptorForMemoKey(ShapeDescriptor descriptor) {
		if (descriptor.sourceNode() == null)
			return "!";

		return this.callGraph.getNumber(descriptor.sourceNode()) + "@" + descriptor.sourceTensor()
				+ (descriptor.dims() == null ? ":*" : ":" + new TreeSet<>(descriptor.dims()));
	}

	/** Colors {@code value} with the value taint (which dominates any shape taint) and enqueues it when newly tainted. */
	private static void colorValue(int value, Set<Integer> valueTainted, Set<Integer> shapeTainted, Deque<Integer> worklist) {
		shapeTainted.remove(value);

		if (valueTainted.add(value))
			worklist.push(value);
	}

	/**
	 * Colors {@code value} with the shape taint and enqueues it when newly tainted, unless it is already value-tainted (value dominates).
	 */
	private static void colorShape(int value, Set<Integer> valueTainted, Set<Integer> shapeTainted, Deque<Integer> worklist) {
		if (valueTainted.contains(value))
			return;

		if (shapeTainted.add(value))
			worklist.push(value);
	}

	/**
	 * A poisoned shape descriptor: recorded when a value receives shape provenance from two conflicting sources (e.g. a phi merge), so the
	 * scan cannot rely on a single source's staticness. Its {@code null} source node makes {@link #numpyOverShapeStaticness} report
	 * {@link ShapeStaticness#TOP} (unprovable), the conservative outcome.
	 */
	private static final ShapeDescriptor AMBIGUOUS_DESCRIPTOR = new ShapeDescriptor(null, -1, null);

	/**
	 * Colors {@code value} with the shape taint and records the {@code descriptor} of the tensor dimensions it covers, so a downstream
	 * numpy sink can consult the source tensor's per-dim {@link TensorType}. Unless {@code value} is already value-tainted (value
	 * dominates). If {@code value} already carries a different descriptor, its provenance is ambiguous (two shape sources merged into it),
	 * so it is poisoned to {@link #AMBIGUOUS_DESCRIPTOR} rather than letting the last writer win.
	 */
	private static void colorShapeFrom(int value, ShapeDescriptor descriptor, Set<Integer> valueTainted, Set<Integer> shapeTainted,
			Map<Integer, ShapeDescriptor> shapeDescriptors, Deque<Integer> worklist) {
		if (valueTainted.contains(value))
			return;

		ShapeDescriptor existing = shapeDescriptors.get(value);

		// Poison on conflicting provenance (two shape sources merged into this value); an already-poisoned value stays poisoned.
		ShapeDescriptor updated = existing == AMBIGUOUS_DESCRIPTOR || existing != null && !existing.equals(descriptor)
				? AMBIGUOUS_DESCRIPTOR
				: descriptor;

		shapeDescriptors.put(value, updated);

		// Re-enqueue on a first taint OR a descriptor change, so a value already popped is re-scanned with the new (possibly poisoned)
		// descriptor rather than leaving a sink decided on the stale one. Descriptors are monotone (unset -> concrete -> ambiguous), so
		// this terminates.
		boolean newlyTainted = shapeTainted.add(value);

		if (newlyTainted || !updated.equals(existing))
			worklist.push(value);
	}

	/**
	 * Narrows {@code base} (a shape vector covering {@code base}'s dimensions) by the slice {@code slice(x, start, stop, step)} in
	 * {@code invoke}, resolving the bounds to integer constants and applying Python slice semantics against the source tensor's rank. When
	 * the rank is statically unresolvable (a ⊤ context or contexts of disagreeing rank), a pure prefix ({@code [:k]}) or suffix
	 * ({@code [-k:]}) slice is still narrowed rank-independently ({@link #resolveRankFreeSliceDims}), deferring resolution to the sink's
	 * per-context verdict - the recovery https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/761 needs, since the corpus
	 * {@code einsum_via_matmul} suffix slice otherwise drops the descriptor. Returns {@code null} when a bound cannot be resolved, the rank
	 * is needed but unresolvable, or {@code base} already covers a proper subset (nested slicing is not composed), so the caller falls back
	 * to an untracked shape taint.
	 */
	private ShapeDescriptor narrowBySlice(ShapeDescriptor base, PythonInvokeInstruction invoke, CGNode node, DefUse defUse) {
		if (base.dims() != null)
			return null; // only slices of a full shape vector are modeled; nested slicing drops the descriptor (unprovable, so permitted
							// under the current precision-favoring policy).

		Integer start = invoke.getNumberOfUses() > 2 ? this.resolveIntConstant(node, invoke.getUse(2), defUse) : null;
		Integer stop = invoke.getNumberOfUses() > 3 ? this.resolveIntConstant(node, invoke.getUse(3), defUse) : null;
		Integer step = invoke.getNumberOfUses() > 4 ? this.resolveIntConstant(node, invoke.getUse(4), defUse) : null;

		// A no-op slice ([:], the Python copy idiom) covers every dimension regardless of rank: preserve the base descriptor.
		if ((start == null || start == 0) && stop == null && (step == null || step == 1))
			return base;

		int rank = this.sourceRank(base);
		Set<Integer> dims = rank < 0 ? resolveRankFreeSliceDims(start, stop, step) : resolveSliceDims(start, stop, step, rank);

		return dims == null ? null : new ShapeDescriptor(base.sourceNode(), base.sourceTensor(), dims);
	}

	/**
	 * The maximum number of dimensions a rank-free slice may cover. Tensor ranks are tiny in practice; without a rank to clamp against, a
	 * pathologically large resolved bound would otherwise enumerate an index per unit of the bound, so anything beyond this cap is treated
	 * as unresolvable instead.
	 */
	private static final int MAX_RANK_FREE_SLICE_EXTENT = 32;

	/**
	 * The dimension indices selected by a Python slice {@code [start:stop:step]} of a shape vector whose rank is statically unresolvable,
	 * or {@code null} if the slice's coverage depends on the rank (or exceeds {@link #MAX_RANK_FREE_SLICE_EXTENT}). Only two
	 * rank-independent forms exist: a pure prefix {@code [:k]} ({@code k >= 0}; {@code [:0]} covers nothing) covers the absolute indices
	 * {@code 0..k-1}, and a pure suffix {@code [-k:]} covers the last {@code k} dimensions, encoded as the negative indices {@code -k..-1}
	 * that {@link #numpyOverShapeStaticness} resolves against each context's own rank. An index beyond a context's rank contributes no
	 * element there and is ignored by the verdict, mirroring Python's slice clamping.
	 */
	private static Set<Integer> resolveRankFreeSliceDims(Integer start, Integer stop, Integer step) {
		if (step != null && step != 1)
			return null; // only unit-stride slices are modeled precisely.

		Set<Integer> dims = new TreeSet<>();

		// A pure prefix [:k]: absolute indices 0..k-1 (none for [:0]), no rank needed.
		if ((start == null || start == 0) && stop != null && stop >= 0 && stop <= MAX_RANK_FREE_SLICE_EXTENT) {
			for (int i = 0; i < stop; i++)
				dims.add(i);

			return dims;
		}

		// A pure suffix [-k:]: the last k dimensions, encoded as negative indices resolved per-context at the sink.
		if (start != null && start < 0 && start >= -MAX_RANK_FREE_SLICE_EXTENT && stop == null) {
			for (int i = start; i < 0; i++)
				dims.add(i);

			return dims;
		}

		return null;
	}

	/**
	 * The rank of the source tensor of {@code descriptor} if all its analysis contexts agree on a concrete (non-⊤) rank, else {@code -1}.
	 */
	private int sourceRank(ShapeDescriptor descriptor) {
		Set<TensorType> types = this.lookupTensorTypes(descriptor.sourceNode(), descriptor.sourceTensor());
		int rank = -1;

		for (TensorType type : types) {
			if (type.getDims() == null)
				return -1;

			if (rank == -1)
				rank = type.getDims().size();
			else if (rank != type.getDims().size())
				return -1;
		}

		return rank;
	}

	/** True iff {@code invoke}'s callee resolves to the numpy/scipy namespace by attribute-chain walk or import-alias fallback. */
	private boolean invokesNumpyApi(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		return this.isNumpyRooted(node, invoke.getUse(0), defUse);
	}

	/** Fully-qualified names of the TensorFlow shape-metadata ops whose results are shape (not value) tainted. */
	private static final Set<String> SHAPE_METADATA_FQNS = Set.of("tensorflow.shape", "tensorflow.size", "tensorflow.rank");

	/** True iff {@code invoke}'s callee is a TensorFlow shape-metadata op ({@code tf.shape}/{@code size}/{@code rank}). */
	private boolean invokesShapeMetadataOp(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		String fqn = Util.resolveCalleeFullyQualifiedName(node, invoke.getUse(0), defUse, this.pointerAnalysis);
		return fqn != null && SHAPE_METADATA_FQNS.contains(fqn);
	}

	/** True iff {@code use}'s attribute chain roots at the numpy/scipy module (points-to preferred, import alias as fallback). */
	private boolean isNumpyRooted(CGNode node, int use, DefUse defUse) {
		if (this.isNumpyModule(node, use, defUse))
			return true;

		SSAInstruction def = defUse.getDef(use);

		if (def instanceof PythonPropertyRead read)
			return this.isNumpyRooted(node, read.getObjectRef(), defUse);

		return false;
	}

	/** True iff {@code use} refers to the numpy/scipy module. Prefers points-to; falls back to the import alias on a global read. */
	private boolean isNumpyModule(CGNode node, int use, DefUse defUse) {
		for (String prefix : NUMPY_MODULE_TYPE_NAME_PREFIXES)
			if (Util.pointsToType(node, use, this.pointerAnalysis, prefix, false))
				return true;

		return defUse.getDef(use) instanceof AstGlobalRead global && NUMPY_MODULE_GLOBAL_NAMES.contains(global.getGlobalName());
	}
}
