package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.WeakHashMap;

import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.CallTargets;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
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
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAConditionalBranchInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAPhiInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.graph.dominators.Dominators;

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

	/** The call-graph builder, which the engine asks for the program functions a call runs ({@link CallTargets}). */
	private final PythonSSAPropagationCallGraphBuilder builder;

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
	 * Creates an analysis over the given call graph, call-graph builder, and tensor-type analysis, whose local pointer keys are indexed
	 * once by (node, value number) for constant-time lookups at the shape sinks.
	 *
	 * @param callGraph The call graph, used to follow user-defined callees.
	 * @param builder The call-graph builder, which the engine asks for a call's program targets, and whose pointer analysis resolves
	 *        attribute names and module roots.
	 * @param tensorTypeAnalysis The tensor-type analysis, consulted for the per-dimension shape staticness.
	 */
	NumpyParameterFlowAnalysis(CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder, TensorTypeAnalysis tensorTypeAnalysis) {
		this.callGraph = callGraph;
		this.builder = builder;
		this.pointerAnalysis = builder.getPointerAnalysis();
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
	private static final String SLICE_BUILTIN_API = "builtins.slice";

	/** The prefix of the NumPy APIs' names. */
	private static final String NUMPY_API_PREFIX = "numpy.";

	/** The APIs each call may invoke, as the engine names them ({@link TensorFlowApis}), by node and then by call. */
	private final Map<CGNode, Map<SSAInstruction, Set<String>>> apis = new HashMap<>();

	/**
	 * The unary-negation operator, matched by name: the operator enum ({@code IUnaryOpInstruction.Operator}) lives in a non-exported WALA
	 * package, so its type cannot be referenced here, but its {@code toString()} is stable.
	 */
	private static final String NEGATION_OPERATOR_NAME = "neg";

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
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756). A value stored into a container or an object (a tuple,
	 * list, or dict element, a subscript store, or an attribute store) taints the whole object, so its reads and the calls it is passed to
	 * are followed. An attribute store thus taints the object's other attributes too, including those of {@code self}. Tainting only the
	 * stored attribute waits for the engine to report the store's kind rather than this analysis decoding it: the engine marks a subscript
	 * write (wala/ML#1100), but a tuple, list, or dict literal's element write is no subscript either, so that mark alone does not yet tell
	 * an attribute store from a literal's element. Known narrowings, each documented on the issue: positional arguments only cross call
	 * sites, a store taints only the object as this function refers to it, not its other references, an attribute written in one method and
	 * read in another method called on the same object is not followed (a call such as {@code self.read()} passes the receiver in no
	 * argument slot, only the method read from it, so the callee is never scanned with a tainted receiver, though a call passing the object
	 * as an argument, such as {@code read(self)}, is followed), and a method's receiver is never a source. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/740.
	 *
	 * @param node The call-graph node to check.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a taint source.
	 * @return True iff a numpy/scipy API is applied to a parameter-flowing value reachable from {@code node}.
	 */
	boolean appliesNumpyToParameters(CGNode node, boolean method) {
		NumpyScanResult result = this.scanParameters(node, method);
		return result != null && result.sink();
	}

	/**
	 * True iff {@code node} (transitively through the user-defined callees and closures {@link #appliesNumpyToParameters}'s walk reaches)
	 * applies Python to a tensor in a way tracing cannot carry (issues 1074 and 1083): {@code bool} of it, or a branch on it whose paths
	 * disagree about {@code None}. That the condition or the argument is a tensor is read from the tensor-type analysis, not from the
	 * parameter flow: a value derived from a parameter may be a Python value, such as the {@code bool} an {@code is None} test yields,
	 * which tracing decides. Shares {@link #appliesNumpyToParameters}'s walk and memo.
	 *
	 * @param node The call-graph node to check.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a taint source.
	 * @return True iff such a use is reachable from {@code node}.
	 */
	boolean appliesUntraceablePythonToParameters(CGNode node, boolean method) {
		NumpyScanResult result = this.scanParameters(node, method);
		return result != null && result.untraceable();
	}

	/** The scan of {@code node} seeded with its parameters, or {@code null} when it has no IR or no parameter to seed. */
	private NumpyScanResult scanParameters(CGNode node, boolean method) {
		IR ir = node.getIR();

		if (ir == null)
			return null;

		// Parameter value numbers: slot 0 is the function object itself; slot 1 is the receiver for an instance method. Neither is a
		// taint source—tainting the receiver would block any method mixing numpy over scalar fields with sanitized tensor use.
		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();
		Set<Integer> sources = new HashSet<>();

		for (int i = method ? 2 : 1; i < parameters.length; i++)
			sources.add(parameters[i]);

		if (sources.isEmpty())
			return null;

		return this.scanForTaintedNumpySinks(node, sources, new HashSet<>(), new HashMap<>(), new HashSet<>(), new HashMap<>());
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
				// A None member is no tensor here; it matters only to the signature (wala/ML#1040).
				pair.snd.getTypes().stream().filter(type -> !type.isNone())
						.forEach(type -> index.computeIfAbsent(local.getNode(), n -> new HashMap<>())
								.computeIfAbsent(local.getValueNumber(), v -> new HashSet<>()).add(type));

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
	private boolean invokesSliceBuiltin(CGNode node, PythonInvokeInstruction invoke) {
		return this.apis(node, invoke).contains(SLICE_BUILTIN_API);
	}

	/**
	 * The APIs {@code invoke} may invoke, as the engine names them.
	 *
	 * @param node The node whose IR holds the call.
	 * @param invoke The call.
	 * @return The APIs' fully qualified names.
	 */
	private Set<String> apis(CGNode node, PythonInvokeInstruction invoke) {
		return this.apis.computeIfAbsent(node, n -> new HashMap<>()).computeIfAbsent(invoke,
				i -> TensorFlowApis.of(this.builder, node, invoke));
	}

	/**
	 * The Python builtins whose application to a symbolic tensor tracing cannot evaluate (issue 1074). Only {@code bool}: AutoGraph
	 * converts {@code int} and {@code float} of a tensor to casts, measured on TensorFlow 2.9.3, so those trace.
	 */
	private static final Set<String> UNTRACEABLE_CONVERSION_BUILTIN_NAMES = Set.of("bool");

	/**
	 * True iff {@code node}'s own body applies {@code bool} to a tensor, or branches on a tensor with paths that disagree about
	 * {@code None}, where a tensor is a value the tensor-type analysis types (issues 1074 and 1083).
	 *
	 * @param node The call-graph node whose body is read.
	 * @param ir The node's IR.
	 * @param defUse The node's def-use chains.
	 * @return True iff such a use appears in the body.
	 */
	private boolean appliesUntraceablePythonToTensors(CGNode node, IR ir, DefUse defUse) {
		Set<SSAConditionalBranchInstruction> tensorBranches = new HashSet<>();

		for (SSAInstruction instruction : ir.getInstructions())
			if (instruction instanceof SSAConditionalBranchInstruction branch) {
				if (this.usesTensor(node, branch, 0))
					tensorBranches.add(branch);
			} else if (instruction instanceof PythonInvokeInstruction invoke && invokesUntraceableConversion(invoke, defUse)
					&& this.usesTensor(node, invoke, 1))
				// Slot 0 of an invoke is the callee itself; the arguments start at 1.
				return true;

		return !tensorBranches.isEmpty() && pathsDisagreeOnNone(ir, tensorBranches);
	}

	/** True iff a use of {@code instruction} at or after {@code from} is a value the tensor-type analysis types in {@code node}. */
	private boolean usesTensor(CGNode node, SSAInstruction instruction, int from) {
		for (int i = from; i < instruction.getNumberOfUses(); i++)
			if (instruction.getUse(i) >= 0 && !this.lookupTensorTypes(node, instruction.getUse(i)).isEmpty())
				return true;

		return false;
	}

	/** True iff {@code invoke} calls one of {@link #UNTRACEABLE_CONVERSION_BUILTIN_NAMES}, read lexically from the script's scope. */
	private static boolean invokesUntraceableConversion(PythonInvokeInstruction invoke, DefUse defUse) {
		SSAInstruction def = defUse.getDef(invoke.getUse(0));

		if (!(def instanceof AstLexicalRead lexical))
			return false;

		Access[] accesses = lexical.getAccesses();
		return accesses.length > 0 && UNTRACEABLE_CONVERSION_BUILTIN_NAMES.contains(accesses[0].getName().fst);
	}

	/**
	 * True iff the paths out of one of {@code branches} disagree about {@code None} (issue 1074): one successor's region returns
	 * {@code None}, explicitly or by falling off the end, while the other's returns a value, or the block where the two paths rejoin merges
	 * {@code None} with a value. AutoGraph converts a branch on a tensor to {@code tf.cond}, which requires both paths to produce the same
	 * structure, so either shape raises under tracing. Returns after the paths rejoin are not compared, since they follow the branch on
	 * every path.
	 *
	 * @param ir The IR containing the branches.
	 * @param branches Conditional branches on a value derived from the function's parameters.
	 * @return Whether the paths of some branch disagree about None.
	 */
	private static boolean pathsDisagreeOnNone(IR ir, Set<SSAConditionalBranchInstruction> branches) {
		var cfg = ir.getControlFlowGraph();
		Dominators<ISSABasicBlock> dominators = Dominators.make(cfg, cfg.entry());
		SymbolTable symbols = ir.getSymbolTable();

		for (SSAConditionalBranchInstruction branch : branches) {
			ISSABasicBlock block = cfg.getBlockForInstruction(branch.iIndex());
			List<ISSABasicBlock> successors = new ArrayList<>(cfg.getNormalSuccessors(block));

			if (successors.size() == 2) {
				Set<Boolean> first = returnKinds(ir, dominators, successors.get(0), symbols);
				Set<Boolean> second = returnKinds(ir, dominators, successors.get(1), symbols);

				if (first.contains(true) && second.contains(false) || first.contains(false) && second.contains(true))
					return true;
			}

			for (ISSABasicBlock merge : cfg)
				if (dominators.getIdom(merge) == block)
					for (Iterator<SSAPhiInstruction> phis = merge.iteratePhis(); phis.hasNext();) {
						SSAPhiInstruction phi = phis.next();
						boolean none = false;
						boolean value = false;

						for (int i = 0; i < phi.getNumberOfUses(); i++)
							if (symbols.isNullConstant(phi.getUse(i)))
								none = true;
							else
								value = true;

						if (none && value)
							return true;
					}
		}

		return false;
	}

	/**
	 * Whether the returns in the region {@code head} dominates yield {@code None} (true) or a value (false). A return without a value is
	 * {@code None}.
	 */
	private static Set<Boolean> returnKinds(IR ir, Dominators<ISSABasicBlock> dominators, ISSABasicBlock head, SymbolTable symbols) {
		Set<Boolean> kinds = new HashSet<>();

		for (SSAInstruction instruction : ir.getInstructions())
			if (instruction instanceof SSAReturnInstruction ret
					&& dominators.isDominatedBy(ir.getControlFlowGraph().getBlockForInstruction(ret.iIndex()), head))
				kinds.add(ret.returnsVoid() || ret.getResult() < 0 || symbols.isNullConstant(ret.getResult()));

		return kinds;
	}

	/**
	 * The result of a two-color taint scan: whether numpy hit a value-tainted argument, whether a value taint escaped this node, and the
	 * {@link ShapeDescriptor} the node's returns carry, so a caller keeps resolving staticness across the return boundary
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756). The descriptor is {@code null} when no shape-tainted
	 * value is returned <em>or</em> any returned shape lacks a descriptor (provenance is dropped rather than partially trusted);
	 * conflicting tracked returns merge to the ambiguous descriptor instead.
	 */
	private record NumpyScanResult(boolean sink, boolean valueEscapes, ShapeDescriptor returnDescriptor, boolean untraceable) {
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
	 * descriptor seed, lexical seeds) with an optimistic cycle guard. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747.
	 * <p>
	 * Taint also crosses into nested functions through their closures
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1061). A nested function (a comprehension, a lambda, or a
	 * nested {@code def}) reads an enclosing function's variable lexically rather than receiving it as an argument, and the enclosing
	 * function publishes the variable with a lexical write, including a prologue write for each parameter a nested function reads. A
	 * tainted value written lexically taints the variable, identified by its name and defining scope, and a lexical read of a tainted
	 * variable, in this node or in any function scanned from it, inherits the taint. Since a nested function need not receive any tainted
	 * argument, every function nested in a tainted variable's defining scope that this node calls is also scanned with the lexical taint,
	 * and its call's result is value-tainted when a tainted value escapes it. A nested function's own write to an enclosing variable
	 * ({@code nonlocal}) is not followed.
	 *
	 * @param node The node to scan.
	 * @param valueSeed The value-tainted value numbers.
	 * @param shapeSeed The shape-tainted value numbers.
	 * @param shapeDescriptorSeed The descriptors of the shape-tainted value numbers whose provenance is tracked.
	 * @param lexicalValueSeed The value-tainted lexical variables, as (name, defining scope).
	 * @param lexicalShapeSeed The shape-tainted lexical variables, each with its {@link ShapeDescriptor} ({@code null} when untracked).
	 * @return The scan's result.
	 */
	private NumpyScanResult scanForTaintedNumpySinks(CGNode node, Set<Integer> valueSeed, Set<Integer> shapeSeed,
			Map<Integer, ShapeDescriptor> shapeDescriptorSeed, Set<Pair<String, String>> lexicalValueSeed,
			Map<Pair<String, String>, ShapeDescriptor> lexicalShapeSeed) {
		Map<Integer, String> descriptorKey = new TreeMap<>();
		shapeDescriptorSeed.forEach((value, descriptor) -> descriptorKey.put(value, this.renderDescriptorForMemoKey(descriptor)));
		Map<String, String> lexicalShapeKey = new TreeMap<>();
		lexicalShapeSeed.forEach((name, descriptor) -> lexicalShapeKey.put(name.toString(),
				descriptor == null ? "?" : this.renderDescriptorForMemoKey(descriptor)));
		String key = this.callGraph.getNumber(node) + ":" + new TreeSet<>(valueSeed) + ":" + new TreeSet<>(shapeSeed) + ":" + descriptorKey
				+ ":" + new TreeSet<>(lexicalValueSeed.stream().map(Pair::toString).toList()) + ":" + lexicalShapeKey;
		NumpyScanResult cached = this.memo.get(key);

		if (cached != null)
			return cached;

		// Optimistic cycle guard: a recursive revisit contributes nothing new.
		this.memo.put(key, new NumpyScanResult(false, false, null, false));

		IR ir = node.getIR();

		if (ir == null)
			return new NumpyScanResult(false, false, null, false);

		DefUse defUse = node.getDU();
		Set<Integer> valueTainted = new HashSet<>(valueSeed);
		Set<Integer> shapeTainted = new HashSet<>(shapeSeed);
		// For each shape-tainted value, the tensor+dimensions its shape came from (absent = shape-derived but provenance lost). Seeded
		// interprocedurally: a shape-tainted argument's descriptor arrives on the corresponding parameter (issue 756).
		Map<Integer, ShapeDescriptor> shapeDescriptors = new HashMap<>(shapeDescriptorSeed);
		Deque<Integer> worklist = new ArrayDeque<>();
		worklist.addAll(valueSeed);
		worklist.addAll(shapeSeed);
		// The tainted lexical variables: those inherited from the caller, and those this node writes from a tainted value (issue 1061).
		Set<Pair<String, String>> lexicalValues = new HashSet<>(lexicalValueSeed);
		Map<Pair<String, String>, ShapeDescriptor> lexicalShapes = new HashMap<>(lexicalShapeSeed);

		for (Pair<String, String> name : lexicalValues)
			colorLexicalReads(ir, name, true, null, valueTainted, shapeTainted, shapeDescriptors, worklist);

		lexicalShapes.forEach((name, descriptor) -> colorLexicalReads(ir, name, false, descriptor, valueTainted, shapeTainted,
				shapeDescriptors, worklist));

		boolean sink = false;
		boolean valueEscapes = false;
		// Python applied to a tensor in a way tracing cannot carry, from a callee or a closure (issue 1074); this node's own uses are read
		// after the walk.
		boolean untraceable = false;
		// The descriptor this node's returns carry back to the caller: set while all tracked-shape returns agree, poisoned to
		// AMBIGUOUS_DESCRIPTOR on a conflict, and dropped entirely (null) when an untracked shape is returned.
		ShapeDescriptor returnDescriptor = null;
		boolean returnedUntrackedShape = false;

		while (true) {
			// A nested function reading a tainted variable through its closure need receive no tainted argument, so the walk may never
			// reach its call. Once the walk settles, scan each function this node calls that is nested in a tainted variable's defining
			// scope, with the lexical taint alone; a value escaping it taints the call's result, which resumes the walk (issue 1061).
			if (worklist.isEmpty()) {
				if (!lexicalValues.isEmpty() || !lexicalShapes.isEmpty())
					for (SSAInstruction instruction : ir.getInstructions()) {
						if (!(instruction instanceof PythonInvokeInstruction invoke))
							continue;

						for (CGNode target : CallTargets.programTargets(this.builder, node, invoke)) {
							if (!isNestedInAny(target, lexicalValues) && !isNestedInAny(target, lexicalShapes.keySet()))
								continue;

							NumpyScanResult r = this.scanForTaintedNumpySinks(target, new HashSet<>(), new HashSet<>(), new HashMap<>(),
									enclosingNames(target, lexicalValues), enclosingNames(target, lexicalShapes));

							if (r.sink())
								sink = true;

							if (r.untraceable())
								untraceable = true;

							if (r.valueEscapes()) {
								valueEscapes = true;

								for (int d = 0; d < invoke.getNumberOfDefs(); d++)
									colorValue(invoke.getDef(d), valueTainted, shapeTainted, worklist);
							}
						}
					}

				// A nested function this node does not call itself, such as a lambda a helper calls or `tf.map_fn` runs, still runs
				// within the function's trace and reads the tainted variables, so it is scanned for sinks too, from the function defining
				// them; with no call site here, its result colors nothing.
				for (CGNode nested : nestedNodes(this.callGraph, node, lexicalValues, lexicalShapes.keySet())) {
					NumpyScanResult r = this.scanForTaintedNumpySinks(nested, new HashSet<>(), new HashSet<>(), new HashMap<>(),
							enclosingNames(nested, lexicalValues), enclosingNames(nested, lexicalShapes));

					if (r.sink())
						sink = true;

					if (r.untraceable())
						untraceable = true;
				}

				if (worklist.isEmpty())
					break;
			}

			int valueNumber = worklist.pop();
			boolean valueColored = valueTainted.contains(valueNumber);

			// The two colors are disjoint by construction: colorValue() evicts the value from shapeTainted, and colorShape() and
			// colorShapeFrom() both decline a value already in valueTainted. The two slice branches below partition on exactly this,
			// so a dual-colored value would send a shape-tainted receiver down the value path and silently skip the dimension-aware
			// narrowing. Assert it rather than rely on reading the three coloring methods together.
			assert !(valueColored && shapeTainted.contains(valueNumber)) : "value " + valueNumber + " is both value- and shape-tainted";

			for (Iterator<SSAInstruction> uses = defUse.getUses(valueNumber); uses.hasNext();) {
				SSAInstruction use = uses.next();

				// A lexical write publishes the value to the nested functions that read the variable (issue 1061), so the variable
				// takes the value's color, and its reads in this node inherit it.
				if (use instanceof AstLexicalWrite write) {
					for (Access access : write.getAccesses())
						if (access.valueNumber() == valueNumber) {
							Pair<String, String> name = access.getName();

							if (valueColored) {
								lexicalShapes.remove(name);

								if (lexicalValues.add(name))
									colorLexicalReads(ir, name, true, null, valueTainted, shapeTainted, shapeDescriptors, worklist);
							} else if (!lexicalValues.contains(name) && !lexicalShapes.containsKey(name)) {
								ShapeDescriptor descriptor = shapeDescriptors.get(valueNumber);
								lexicalShapes.put(name, descriptor);
								colorLexicalReads(ir, name, false, descriptor, valueTainted, shapeTainted, shapeDescriptors, worklist);
							}
						}

					continue;
				}

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
					if (this.invokesNumpyApi(node, invoke)) {
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

					// A slice DERIVES its result from its receiver, and that derivation is asserted here rather than inferred from the
					// callee. Until Ariadne 0.52.102 the `slice` builtin's summary body returned its receiver, so this walk carried
					// the taint out through the callee's SSAReturnInstruction for free. wala/ML#916 made the body return nothing,
					// supplying an identical points-to result through a builder constraint instead, which leaves the points-to set
					// unchanged and DELETES the def-use edge this analysis walks. The derivation is a property of slicing, not of how
					// the builtin happens to be summarized, so it belongs here
					// (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/967).
					//
					// Both shapes regressed and both are pinned: `np.maximum(boxes[:2], 0.0)`, where the slice is a direct operand of
					// the numpy call, and `h = bboxA[:,3] - bboxA[:,1]; np.maximum(w, h)`, where the slice is consumed by arithmetic
					// first and no slice appears at the call site. The second does not follow from the first, so each has its own fixture.
					if (valueColored && this.invokesSliceBuiltin(node, invoke) && invoke.getNumberOfUses() > 1
							&& invoke.getUse(1) == valueNumber) {
						for (int d = 0; d < invoke.getNumberOfDefs(); d++)
							colorValue(invoke.getDef(d), valueTainted, shapeTainted, worklist);

						continue;
					}

					// A Python `x[a:b:c]` subscript is modeled as `slice(x, a, b, c)`. Only a slice of a shape-tainted vector stays shape
					// metadata (covering the sliced dimensions); slicing a tensor VALUE yields a sub-tensor value (e.g. `boxes1[..., :2]`),
					// which must remain value-tainted so numpy over it is still a sink. Guard on the receiver being shape-, not value-,
					// tainted.
					if (!valueColored && this.invokesSliceBuiltin(node, invoke) && invoke.getNumberOfUses() > 1
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
					if (this.invokesShapeMetadataOp(node, invoke)) {
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

					// The engine reports the program functions whose parameters receive this call's arguments, each with whether a
					// receiver, the instance of a bound method or the class of a class method, is bound before them, which shifts the
					// arguments one parameter along.
					if (!valueSlots.isEmpty() || !shapeSlots.isEmpty())
						for (Map.Entry<CGNode, Boolean> entry : CallTargets.parameterBoundTargets(this.builder, node, invoke).entrySet()) {
							CGNode target = entry.getKey();
							IR targetIr = target.getIR();

							if (targetIr == null)
								continue;

							int shift = entry.getValue() ? 1 : 0;
							int[] targetParameters = targetIr.getSymbolTable().getParameterValueNumbers();
							Set<Integer> targetValueSeed = new HashSet<>();
							Set<Integer> targetShapeSeed = new HashSet<>();
							Map<Integer, ShapeDescriptor> targetDescriptorSeed = new HashMap<>();

							for (int slot : valueSlots)
								if (slot + shift < targetParameters.length)
									targetValueSeed.add(targetParameters[slot + shift]);

							for (int slot : shapeSlots)
								if (slot + shift < targetParameters.length) {
									targetShapeSeed.add(targetParameters[slot + shift]);

									// Thread the argument's shape provenance into the callee's frame, so a numpy sink there still
									// resolves staticness against the source tensor (issue 756).
									ShapeDescriptor argumentDescriptor = shapeDescriptors.get(invoke.getUse(slot));

									if (argumentDescriptor != null)
										targetDescriptorSeed.put(targetParameters[slot + shift], argumentDescriptor);
								}

							if (targetValueSeed.isEmpty() && targetShapeSeed.isEmpty())
								continue;

							analyzedCallee = true;

							NumpyScanResult r = this.scanForTaintedNumpySinks(target, targetValueSeed, targetShapeSeed,
									targetDescriptorSeed, enclosingNames(target, lexicalValues), enclosingNames(target, lexicalShapes));

							if (r.sink())
								sink = true;
							if (r.untraceable())
								untraceable = true;
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

				// A value stored into a container or an object: a tuple, list, or dict literal's element, a subscript store
				// (`d[k] = x`, `l[i] = x`), or an attribute store (`obj.attr = x`), each of which the Python front end lowers to a
				// property write. The write defines nothing, so without this the def-use walk would never reach a read of the value
				// back out of the container, nor the container passed to a callee, such as a model called on `(x, w)`. A value-tainted
				// stored value taints the whole object, whose every read is then value-tainted, and escapes this node. Tainting only the
				// stored attribute of an attribute store waits for the engine to mark a literal's element writes as it marks subscript
				// writes (wala/ML#1100), since neither the member nor the subscript mark alone tells an attribute store from a literal's
				// element. Only the value taint is carried: a container colored with the shape taint would
				// carry no provenance, and numpy over a shape without provenance is permitted at every sink (issue 751), so it would add no
				// decline, while a returned untracked shape drops the descriptor the node's other returns carry (issue 756), which could
				// lose one.
				if (use instanceof AstPropertyWrite write && write.getValue() == valueNumber) {
					if (valueColored) {
						valueEscapes = true;
						colorValue(write.getObjectRef(), valueTainted, shapeTainted, worklist);
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

		// `bool` of a tensor cannot be evaluated while tracing, and a branch on a tensor is converted by AutoGraph to `tf.cond`, which
		// rejects
		// paths that disagree about None (issue 1074). Both are decided by whether the tensor-type analysis types the argument or the
		// condition as a tensor (issue 1083), so a parameter's `is None` test, a Python `bool`, is left alone. `int` and `float` of a
		// tensor
		// are left alone too, since AutoGraph converts them to casts.
		if (!untraceable)
			untraceable = this.appliesUntraceablePythonToTensors(node, ir, defUse);

		NumpyScanResult result = new NumpyScanResult(sink, valueEscapes, returnedUntrackedShape ? null : returnDescriptor, untraceable);
		this.memo.put(key, result);
		return result;
	}

	/**
	 * Colors the definitions of this IR's lexical reads of {@code name} (issue 1061): with the value taint when {@code value}, and
	 * otherwise with the shape taint, carrying {@code descriptor} when it is tracked.
	 */
	private static void colorLexicalReads(IR ir, Pair<String, String> name, boolean value, ShapeDescriptor descriptor,
			Set<Integer> valueTainted, Set<Integer> shapeTainted, Map<Integer, ShapeDescriptor> shapeDescriptors, Deque<Integer> worklist) {
		for (SSAInstruction instruction : ir.getInstructions())
			if (instruction instanceof AstLexicalRead read)
				for (Access access : read.getAccesses())
					if (access.getName().equals(name))
						if (value)
							colorValue(access.valueNumber(), valueTainted, shapeTainted, worklist);
						else if (descriptor != null)
							colorShapeFrom(access.valueNumber(), descriptor, valueTainted, shapeTainted, shapeDescriptors, worklist);
						else
							colorShape(access.valueNumber(), valueTainted, shapeTainted, worklist);
	}

	/**
	 * True iff {@code target} is a function nested in the defining scope of one of {@code names}, so it may read that variable through its
	 * closure (issue 1061). A nested function's type name extends its enclosing function's with a {@code /}-separated segment.
	 *
	 * @param target The call-graph node of a possible callee.
	 * @param names Lexical variables, as (name, defining scope).
	 * @return True iff {@code target} is nested in one of the defining scopes.
	 */
	static boolean isNestedInAny(CGNode target, Set<Pair<String, String>> names) {
		String type = stripTypePrefix(target.getMethod().getDeclaringClass().getName().toString());

		for (Pair<String, String> name : names)
			if (name.snd != null && type.startsWith(stripTypePrefix(name.snd) + "/"))
				return true;

		return false;
	}

	/**
	 * Per call graph, its user-code nodes with an IR, keyed by their type names with the {@code L} prefix removed, so the functions nested
	 * in a scope are a range lookup rather than a pass over the whole graph (issue 1061).
	 */
	private static final Map<CallGraph, NavigableMap<String, Set<CGNode>>> NODES_BY_TYPE = Collections.synchronizedMap(new WeakHashMap<>());

	/**
	 * Clears the per-call-graph index. The map's weak keys do not release a call graph on their own, since each indexed node refers back to
	 * its call graph, so the index is cleared with the other analysis caches between projects.
	 */
	static void clearCaches() {
		NODES_BY_TYPE.clear();
	}

	/**
	 * The call-graph nodes nested in {@code definer}'s own scope, wherever they are called from, when one of the given lexical variables is
	 * defined there (issue 1061). Only the function defining a scope scans the functions nested in it: a function nested in a scope reads
	 * that scope's variables too, but it is itself scanned from the scope's definer, along with every other function nested there, so
	 * scanning the scope again from each of them would only recurse from sibling to sibling, as deep as the scope has functions.
	 *
	 * @param callGraph The call graph.
	 * @param definer The node whose own scope's nested functions are wanted.
	 * @param values The value-tainted lexical variables.
	 * @param shapes The shape-tainted lexical variables.
	 * @return The user-code nodes with an IR nested in {@code definer}'s scope, or none when no given variable is defined there.
	 */
	static Set<CGNode> nestedNodes(CallGraph callGraph, CGNode definer, Set<Pair<String, String>> values,
			Set<Pair<String, String>> shapes) {
		Set<CGNode> nodes = new HashSet<>();

		if (values.isEmpty() && shapes.isEmpty())
			return nodes;

		NavigableMap<String, Set<CGNode>> index = NODES_BY_TYPE.computeIfAbsent(callGraph, k -> {
			NavigableMap<String, Set<CGNode>> byType = new TreeMap<>();

			for (CGNode node : k)
				if (CallTargets.isProgramCode(node))
					byType.computeIfAbsent(stripTypePrefix(node.getMethod().getDeclaringClass().getName().toString()), t -> new HashSet<>())
							.add(node);

			return byType;
		});

		String own = stripTypePrefix(definer.getMethod().getDeclaringClass().getName().toString());
		Set<String> scopes = new TreeSet<>();

		for (Set<Pair<String, String>> names : List.of(values, shapes))
			for (Pair<String, String> name : names)
				if (name.snd != null && stripTypePrefix(name.snd).equals(own))
					scopes.add(own);

		// The types nested in a scope are exactly those that sort between "scope/" and "scope0", since '0' follows '/'.
		for (String scope : scopes)
			index.subMap(scope + "/", true, scope + "0", false).values().forEach(nodes::addAll);

		return nodes;
	}

	/**
	 * The lexical variables {@code target} can read: those whose defining scope encloses it. A function outside a scope, such as a
	 * top-level helper a closure calls, cannot read that scope's variables, and neither can anything it calls that is not itself nested in
	 * the scope, which is scanned from the scope's own node. Passing only these keeps the scan's seeds, and so its memo keys, bounded by
	 * the target's enclosing scopes (issue 1061); passing every tainted variable made sibling closures rescan one another under
	 * ever-growing seeds. The filter preserves every result: a lexical read is matched on its name and defining scope together, and a
	 * function outside a scope contains no read of that scope's variables, so a dropped variable could never have colored anything there.
	 *
	 * @param target The call-graph node to be scanned.
	 * @param names Lexical variables, as (name, defining scope).
	 * @return The variables of {@code names} defined in a scope enclosing {@code target}.
	 */
	static Set<Pair<String, String>> enclosingNames(CGNode target, Set<Pair<String, String>> names) {
		Set<Pair<String, String>> ret = new HashSet<>();

		for (Pair<String, String> name : names)
			if (isNestedInAny(target, Set.of(name)))
				ret.add(name);

		return ret;
	}

	/**
	 * The entries of {@code names} whose variable {@code target} can read; see {@link #enclosingNames(CGNode, Set)}.
	 *
	 * @param <V> The type of the taint each variable carries.
	 * @param target The call-graph node to be scanned.
	 * @param names Lexical variables, as (name, defining scope), with their taint.
	 * @return The entries of {@code names} defined in a scope enclosing {@code target}.
	 */
	static <V> Map<Pair<String, String>, V> enclosingNames(CGNode target, Map<Pair<String, String>, V> names) {
		Map<Pair<String, String>, V> ret = new HashMap<>();
		names.forEach((name, taint) -> {
			if (isNestedInAny(target, Set.of(name)))
				ret.put(name, taint);
		});
		return ret;
	}

	/** The name with a leading {@code L} type-descriptor prefix removed, so a scope's entity name and its type name compare equal. */
	private static String stripTypePrefix(String name) {
		return name.startsWith("L") ? name.substring(1) : name;
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

	/** True iff {@code invoke} may invoke a NumPy API, as the engine names it. */
	private boolean invokesNumpyApi(CGNode node, PythonInvokeInstruction invoke) {
		return this.apis(node, invoke).stream().anyMatch(api -> api.startsWith(NUMPY_API_PREFIX));
	}

	/** Fully-qualified names of the TensorFlow shape-metadata ops whose results are shape (not value) tainted. */
	private static final Set<String> SHAPE_METADATA_FQNS = Set.of("tensorflow.shape", "tensorflow.size", "tensorflow.rank");

	/** True iff {@code invoke} may invoke a TensorFlow shape-metadata op ({@code tf.shape}/{@code size}/{@code rank}). */
	private boolean invokesShapeMetadataOp(CGNode node, PythonInvokeInstruction invoke) {
		return !Collections.disjoint(this.apis(node, invoke), SHAPE_METADATA_FQNS);
	}
}
