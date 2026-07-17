package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static edu.cuny.hunter.hybridize.core.analysis.Information.TYPE_INFERENCING;
import static edu.cuny.hunter.hybridize.core.analysis.Util.getFullyQualifiedName;
import static edu.cuny.hunter.hybridize.core.analysis.Util.getSelection;
import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;
import static java.util.Collections.unmodifiableSet;
import static org.eclipse.core.runtime.Platform.getLog;
import static org.eclipse.core.runtime.SubMonitor.convert;
import static org.python.pydev.parser.visitors.NodeUtils.getFullRepresentationString;
import static org.python.pydev.parser.visitors.NodeUtils.getRepresentationString;
import static org.python.pydev.parser.visitors.NodeUtils.getTypeForParameterFromAST;

import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.jface.text.IDocument;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.VisitorBase;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.parser.visitors.TypeInfo;

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.ipa.callgraph.ScopeMappingInstanceKeys.ScopeMappingInstanceKey;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.CompoundDim;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.CallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Pair;

/**
 * A representation of a Python function parameter.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public final class Parameter {

	private static final ILog LOG = getLog(Parameter.class);

	/**
	 * Conventional name of the implicit first parameter of an instance method in Python.
	 */
	private static final String SELF_PARAMETER_NAME = "self";

	/**
	 * Fully-qualified name of TensorFlow's tensor type, used to recognize tensor-typed type hints in {@link #hasTensorTypeHint}.
	 */
	private static final String TF_TENSOR_FQN = "tensorflow.python.framework.ops.Tensor";

	/**
	 * Fully-qualified name of TensorFlow's {@code RaggedTensor} type, a tensor-equivalent subtype recognized in type hints.
	 */
	private static final String TF_RAGGED_TENSOR_FQN = "tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor";

	/**
	 * Fully-qualified name of TensorFlow's {@code SparseTensor} type, a tensor-equivalent subtype recognized in type hints.
	 */
	private static final String TF_SPARSE_TENSOR_FQN = "tensorflow.python.framework.sparse_tensor.SparseTensor";

	/**
	 * Fully-qualified name of TensorFlow's {@code Variable} type, a tensor-equivalent subtype recognized in type hints.
	 */
	private static final String TF_VARIABLE_FQN = "tensorflow.python.ops.variables.Variable";

	/**
	 * Fully-qualified name of TensorFlow's {@code IndexedSlices} type, a tensor-equivalent subtype recognized in type hints.
	 */
	private static final String TF_INDEXED_SLICES_FQN = "tensorflow.python.framework.indexed_slices.IndexedSlices";

	/**
	 * Fully-qualified names of the TensorFlow types recognized as tensor-typed in a type hint: {@link #TF_TENSOR_FQN} and its
	 * tensor-equivalent subtypes ({@code RaggedTensor}, {@code SparseTensor}, {@code Variable}, {@code IndexedSlices}). The {@code *Spec}
	 * descriptors ({@code TensorSpec}, {@code RaggedTensorSpec}, …) are deliberately excluded: they describe tensors but are not tensors
	 * themselves. See <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/434">#434</a>.
	 */
	private static final Set<String> TF_TENSOR_TYPE_HINT_FQNS = Set.of(TF_TENSOR_FQN, TF_RAGGED_TENSOR_FQN, TF_SPARSE_TENSOR_FQN,
			TF_VARIABLE_FQN, TF_INDEXED_SLICES_FQN);

	/**
	 * Parent Jython AST node carrying every positional name expression (in {@link argumentsType#args}) and per-position annotation (in
	 * {@link argumentsType#annotation}) of the owning function, plus the sibling keyword-only arrays ({@link argumentsType#kwonlyargs} and
	 * {@link argumentsType#kwonlyargannotation}). Shared across all {@link Parameter}s of the same function.
	 */
	private final argumentsType arguments;

	/**
	 * Zero-based position of this parameter within {@link #arguments}{@code .args} (positional) or {@link #arguments}{@code .kwonlyargs}
	 * (keyword-only, see {@link #keywordOnly}). Bounded at construction time to be a valid index for the relevant array.
	 */
	private final int index;

	/**
	 * True iff this parameter is keyword-only (declared after a bare {@code *}), indexing into {@link #arguments}{@code .kwonlyargs} rather
	 * than {@code .args} (#607).
	 */
	private final boolean keywordOnly;

	/**
	 * The possible {@link TensorType}s of this {@link Parameter}.
	 */
	private Set<TensorType> tensorTypes;

	/**
	 * Cached classification of whether this parameter is a tensor container (e.g., a list/tuple/dict whose elements are tensors). Populated
	 * by {@link #classifyAsTensor} only when Phase 3 ({@link #hasTensorContainer}) executes—i.e., when classification falls through Phase 1
	 * (type hints) and the Phase 2 (Ariadne) cache is empty. Earlier-returning phases (self, type-hint hit, non-empty Phase 2 result, or
	 * empty call-graph nodes) leave this field at its default. {@code null} therefore means either classification has not run or it ran but
	 * did not reach Phase 3.
	 */
	private Boolean tensorContainer;

	/**
	 * Cached classification of whether this parameter is tensor-typed. {@code null} until {@link #classifyAsTensor} has run; otherwise
	 * {@code TRUE} or {@code FALSE}.
	 */
	private Boolean tensor;

	/**
	 * Cached per-position element types of a sequence-container parameter, populated by {@link #extractContainerElements} when Phase 3
	 * classifies this parameter as a tensor container and the container form is one the nested-spec reduction models: every value reaching
	 * the parameter is a list or tuple known to the tensor analysis, every container's element structure is a contiguous run of constant
	 * indices with the same arity, and every position carries tensor evidence (#781). Position {@code j} holds the union of the element
	 * types at index {@code j} across the reaching containers. {@code null} when extraction did not run or the form is unsupported;
	 * distinguish the arity-disagreement case via {@link #getContainerAritiesDisagree()}.
	 */
	private List<Set<TensorType>> containerElementTypes;

	/**
	 * {@code TRUE} iff Phase 3 found sequence containers reaching this parameter whose element counts disagree, which is its own blocking
	 * reason (no wildcard arity exists; TensorFlow enforces the declared length) rather than an unsupported form (#781). {@code null} when
	 * extraction did not run or did not reach the arity check.
	 */
	private Boolean containerAritiesDisagree;

	/**
	 * Cached answer to whether any call site supplies an argument for this parameter, positionally or by keyword. Populated by
	 * {@link Function#inferSuppliedParameters}. {@code null} until that runs, and deliberately left {@code null} when the answer cannot be
	 * established (no call-graph node for the owning function, an unresolvable predecessor, or a non-Python invoke), so that
	 * {@link Function#inferInputSignature()} treats ignorance as "supplied" rather than as "not supplied" (#787). Meaningful only for a
	 * parameter with a default: one without a default must always be covered by the signature regardless.
	 */
	private Boolean suppliedAtCallSite;

	/**
	 * Owning {@link Function} back-reference.
	 */
	private final Function function;

	/**
	 * Cache of tensor-container element evidence for each tensor type analysis. Keyed by the analysis object to ensure that cached results
	 * are discarded when the analysis is re-run and a new object is returned. Each value maps a container instance key (a list/tuple/dict
	 * whose elements the analysis associates with tensors) to its per-field element types: field name (the stringified catalog constant,
	 * e.g. {@code "0"}) to the {@link TensorType}s of the {@link InstanceFieldPointerKey} for that field. The key set alone answers the
	 * boolean container question (Phase 3 of {@link #classifyAsTensor}); the field map is what the nested-spec reduction consumes (#781).
	 */
	private static Map<TensorTypeAnalysis, Map<InstanceKey, Map<String, Set<TensorType>>>> tensorContainersCache = Maps.newConcurrentMap();

	/**
	 * Package-private because {@link Parameter}s are only ever constructed inside {@link Function}'s constructor (same package).
	 *
	 * @param arguments The parent {@link argumentsType} node. Non-null.
	 * @param index The zero-based positional index within {@code arguments.args}. Must be in {@code [0, arguments.args.length)}.
	 * @param function The owning {@link Function}. Non-null.
	 * @throws IndexOutOfBoundsException If {@code index} is out of range for {@code arguments.args}.
	 */
	Parameter(argumentsType arguments, int index, Function function) {
		this(arguments, index, function, false);
	}

	/**
	 * Package-private constructor allowing keyword-only parameters (#607). Positional parameters index into {@code arguments.args};
	 * keyword-only parameters (declared after a bare {@code *}) index into {@code arguments.kwonlyargs}.
	 *
	 * @param arguments The parent {@link argumentsType} node. Non-null.
	 * @param index The zero-based index within the relevant name array ({@code args} or {@code kwonlyargs}).
	 * @param function The owning {@link Function}. Non-null.
	 * @param keywordOnly True iff this is a keyword-only parameter (indexing into {@code arguments.kwonlyargs}).
	 * @throws IndexOutOfBoundsException If {@code index} is out of range for the relevant name array.
	 */
	Parameter(argumentsType arguments, int index, Function function, boolean keywordOnly) {
		this.arguments = Objects.requireNonNull(arguments);
		this.function = Objects.requireNonNull(function);
		this.keywordOnly = keywordOnly;
		exprType[] names = keywordOnly ? arguments.kwonlyargs : arguments.args;
		if (index < 0 || names == null || index >= names.length)
			throw new IndexOutOfBoundsException("Parameter index " + index + " out of bounds for " + (keywordOnly ? "kwonlyargs" : "args")
					+ " of length " + (names == null ? 0 : names.length) + ".");
		this.index = index;
	}

	/**
	 * Returns this parameter's zero-based positional index within the owning function's positional-argument list.
	 *
	 * @return The zero-based index.
	 */
	public int getIndex() {
		return this.index;
	}

	/**
	 * Returns the identifier text of this parameter as declared (e.g. {@code "x"}, {@code "self"}). Derived from the underlying Jython name
	 * expression via {@link NodeUtils#getRepresentationString}.
	 *
	 * @return The parameter name.
	 */
	public String getName() {
		return getRepresentationString(this.getNameExpr());
	}

	/**
	 * Returns true iff this parameter is the implicit first parameter of an instance method: positional index {@code 0} and named
	 * {@code self} (the conventional name).
	 *
	 * @return True iff this parameter is at index {@code 0} and is named {@code self}.
	 */
	public boolean isSelf() {
		return !this.keywordOnly && this.getIndex() == 0 && SELF_PARAMETER_NAME.equals(this.getName());
	}

	/**
	 * Returns true iff this parameter declares a default value (e.g. the {@code training} of {@code def call(self, x, training=True)}).
	 * <p>
	 * PyDev's {@link argumentsType#defaults} is parallel to {@link argumentsType#args} with a {@code null} at each position lacking a
	 * default, rather than the shorter trailing list CPython's {@code ast} uses: the grammar's tree builder appends {@code node.value} for
	 * every regular parameter, and that value is {@code null} when none was written. {@link argumentsType#kw_defaults} is parallel to
	 * {@link argumentsType#kwonlyargs} the same way. This mirrors {@link argumentsType#annotation}, which {@link #getTypeInfo()} already
	 * indexes positionally.
	 *
	 * @return True iff a default value is declared for this parameter.
	 */
	public boolean hasDefault() {
		exprType[] defaults = this.keywordOnly ? this.arguments.kw_defaults : this.arguments.defaults;
		return defaults != null && this.index < defaults.length && defaults[this.index] != null;
	}

	/**
	 * Returns PyDev's AST-derived type information for this parameter (i.e. the type-hint annotation in the function's signature), or
	 * {@code null} if no type hint is declared.
	 *
	 * @return The {@link TypeInfo} for this parameter, or {@code null} if no type hint is present.
	 */
	protected TypeInfo getTypeInfo() {
		// PyDev's by-name resolver (NodeUtils.getTypeForParameterFromStaticTyping) scans only `args.args`/`args.annotation`, so it never
		// finds a keyword-only parameter's annotation (ponder-lab/Pydev#11). Read `kwonlyargannotation` directly for keyword-only
		// parameters (#607).
		if (this.keywordOnly) {
			exprType[] kwonlyAnnotations = this.arguments.kwonlyargannotation;
			if (kwonlyAnnotations != null && this.index < kwonlyAnnotations.length && kwonlyAnnotations[this.index] != null)
				return new TypeInfo(kwonlyAnnotations[this.index]);
			return null;
		}
		return getTypeForParameterFromAST(this.getName(), this.function.getFunctionDefinition().getFunctionDef());
	}

	/**
	 * Returns the set of possible {@link TensorType}s for this {@link Parameter}.
	 *
	 * @return The set of possible {@link TensorType}s for this {@link Parameter}.
	 */
	public Set<TensorType> getTensorTypes() {
		return this.tensorTypes;
	}

	protected void setTensorTypes(Set<TensorType> tensorTypes) {
		this.tensorTypes = tensorTypes;
	}

	/**
	 * Returns the cached classification of whether this parameter is a tensor container (e.g., a list/tuple/dict whose elements are
	 * tensors). Populated only when {@link #classifyAsTensor}'s Phase 3 ({@link #hasTensorContainer}) executes; earlier-returning phases
	 * (self, type-hint hit, non-empty Phase 2 result, or empty call-graph nodes) leave it at the default. Returns {@code null} when
	 * classification has not run or did not reach Phase 3. Distinct from {@link #getTensorTypes()}: a tensor container is recognized as
	 * tensor-typed by Phase 3's container detection, but Ariadne does not emit a single {@link TensorType} for the container itself.
	 * Consumers that distinguish "container of tensors" from "direct tensor" should use this method.
	 *
	 * @return {@code TRUE} if Phase 3 classified this parameter as a tensor container, {@code FALSE} if Phase 3 ran and did not classify
	 *         the parameter as a tensor container, or {@code null} if Phase 3 did not run (classification not started, or returned
	 *         earlier).
	 */
	public Boolean isTensorContainer() {
		return this.tensorContainer;
	}

	/**
	 * Returns the per-position element types of this sequence-container parameter, or {@code null} when the container form is not one the
	 * nested-spec reduction models (see {@link #getContainerAritiesDisagree()} for the arity-disagreement case) or classification did not
	 * reach the extraction. Position {@code j} holds the union of the element types at index {@code j} across the container values reaching
	 * the parameter; the reduction reduces each position independently (#781).
	 *
	 * @return One {@link TensorType} set per element position, or {@code null}.
	 */
	public List<Set<TensorType>> getContainerElementTypes() {
		return this.containerElementTypes;
	}

	/**
	 * Returns whether the sequence containers reaching this parameter disagree on element count. {@code TRUE} is its own blocking reason
	 * (no wildcard arity exists; TensorFlow rejects a sequence of a different length than declared), distinct from the unsupported-form
	 * case where {@link #getContainerElementTypes()} is {@code null} with this {@code null} too (#781).
	 *
	 * @return {@code TRUE} on arity disagreement, {@code FALSE} when extraction succeeded, or {@code null} otherwise.
	 */
	public Boolean getContainerAritiesDisagree() {
		return this.containerAritiesDisagree;
	}

	/**
	 * One diagnostic row of the raw inferred tensor lattice for a parameter, at per-dimension granularity (#780). Where an
	 * {@link InputSignature} entry collapses every non-constant dimension to a single wildcard, this preserves the underlying
	 * {@link Dimension} class so a wildcard that is a static constant or an unresolved-but-fixed size is separable from a genuinely dynamic
	 * one. There is one row per dimension of each inferred {@link TensorType}; a shape-⊤ type (unknown rank) and a scalar (rank zero) each
	 * contribute a single summary row with a {@code null} {@code dimIndex}.
	 *
	 * @param containerPosition The element position for a sequence-container parameter, or {@code null} for a direct tensor parameter.
	 * @param typeOrdinal The index of the {@link TensorType} within the parameter's (sorted) type set, disambiguating multi-context types.
	 * @param rank The rank as a string: the dimension count, {@code "TOP"} for a shape-⊤ type, or {@code "0"} for a scalar.
	 * @param dimIndex The zero-based dimension position, or {@code null} for the shape-⊤ and scalar summary rows.
	 * @param dimClass The {@link Dimension} class: {@code "Constant,<n>"}, {@code "Symbolic,<name>"}, {@code "Unresolved"},
	 *        {@code "Dynamic"}, {@code "Ragged"}, {@code "Compound"}, {@code "TOP"} (shape-⊤), or {@code "scalar"} (rank zero).
	 * @param dtype The dtype name (e.g. {@code "FLOAT32"}, {@code "UNKNOWN"}).
	 * @param dtypeTop True iff the dtype is ⊤ ({@link DType#UNKNOWN}).
	 */
	public record TensorTypeDimensionRow(Integer containerPosition, int typeOrdinal, String rank, Integer dimIndex, String dimClass,
			String dtype, boolean dtypeTop) {
	}

	/**
	 * Returns the raw inferred tensor lattice for this parameter as diagnostic rows at per-dimension granularity (#780), reading the cached
	 * {@link #getTensorTypes()} and {@link #getContainerElementTypes()} without recomputing anything. Empty when the parameter carries no
	 * inferred {@link TensorType} (e.g. a non-tensor parameter, or one classified only by type hint or bare container detection).
	 *
	 * @return The per-dimension diagnostic rows, in declaration order (direct types first, then container positions).
	 */
	public List<TensorTypeDimensionRow> getTensorTypeDiagnostics() {
		return computeTensorTypeDiagnostics(this.getTensorTypes(), this.getContainerElementTypes());
	}

	/**
	 * Computes the per-dimension diagnostic rows (#780) for a parameter's direct tensor types and, when present, its sequence-container
	 * element types. Package-visible and static so it can be exercised directly on synthesized {@link TensorType}s, mirroring
	 * {@link Function#inferSpec}.
	 *
	 * @param directTypes The parameter's direct inferred {@link TensorType}s ({@link #getTensorTypes()}), possibly {@code null} or empty.
	 * @param containerElementTypes The per-position element types ({@link #getContainerElementTypes()}), or {@code null} when not a modeled
	 *        container.
	 * @return The diagnostic rows: direct types (with a {@code null} container position) followed by each container position.
	 */
	public static List<TensorTypeDimensionRow> computeTensorTypeDiagnostics(Set<TensorType> directTypes,
			List<Set<TensorType>> containerElementTypes) {
		List<TensorTypeDimensionRow> rows = new ArrayList<>();
		addDimensionRows(rows, directTypes, null);

		if (containerElementTypes != null)
			for (int position = 0; position < containerElementTypes.size(); position++)
				addDimensionRows(rows, containerElementTypes.get(position), position);

		return rows;
	}

	private static void addDimensionRows(List<TensorTypeDimensionRow> rows, Set<TensorType> types, Integer containerPosition) {
		if (types == null)
			return;

		// The type set is unordered; sort by rendering so the type ordinal (and the CSV) is reproducible across runs.
		List<TensorType> sorted = types.stream().sorted(Comparator.comparing(Object::toString)).toList();

		for (int typeOrdinal = 0; typeOrdinal < sorted.size(); typeOrdinal++) {
			TensorType type = sorted.get(typeOrdinal);
			String dtype = type.getDType().name();
			boolean dtypeTop = type.getDType() == DType.UNKNOWN;
			List<Dimension<?>> dims = type.getDims();

			if (dims == null)
				rows.add(new TensorTypeDimensionRow(containerPosition, typeOrdinal, "TOP", null, "TOP", dtype, dtypeTop));
			else if (dims.isEmpty())
				rows.add(new TensorTypeDimensionRow(containerPosition, typeOrdinal, "0", null, "scalar", dtype, dtypeTop));
			else
				for (int dimIndex = 0; dimIndex < dims.size(); dimIndex++)
					rows.add(new TensorTypeDimensionRow(containerPosition, typeOrdinal, Integer.toString(dims.size()), dimIndex,
							renderDimensionClass(dims.get(dimIndex)), dtype, dtypeTop));
		}
	}

	/**
	 * Renders a single {@link Dimension}'s class for the #780 diagnostic, reading the raw subtype directly rather than through
	 * {@link Function#inferSpec}, which lossily collapses every non-constant dimension to a wildcard.
	 *
	 * @param dimension The dimension to render.
	 * @return The class string: {@code "Constant,<n>"}, {@code "Symbolic,<name>"}, {@code "Unresolved"}, {@code "Dynamic"},
	 *         {@code "Ragged"}, or {@code "Compound"}.
	 */
	private static String renderDimensionClass(Dimension<?> dimension) {
		if (dimension instanceof NumericDim numeric)
			return "Constant," + numeric.value();
		if (dimension instanceof SymbolicDim symbolic)
			return "Symbolic," + symbolic.value();
		if (dimension instanceof UnresolvedDim)
			return "Unresolved";
		if (dimension instanceof DynamicDim)
			return "Dynamic";
		if (dimension instanceof RaggedDim)
			return "Ragged";
		if (dimension instanceof CompoundDim)
			return "Compound";
		return dimension.getClass().getSimpleName();
	}

	/**
	 * Returns the fully-qualified name of this parameter's declared type hint, or {@code null} if no type hint is declared. Only supports
	 * simple type hints that are directly {@link Attribute} expressions; more complex hints (e.g. subscripted generics) are not supported
	 * and may trigger an exception.
	 *
	 * @return The fully-qualified name of the declared type hint, or {@code null} if no type hint is present.
	 * @throws IllegalStateException If a type hint is declared but its AST node is not an {@link Attribute} (e.g., a subscripted generic
	 *         like {@code List[Tensor]}).
	 */
	public String getTypeHintName() {
		// get the type hint.
		TypeInfo typeInfo = this.getTypeInfo();

		if (typeInfo == null)
			// no type hint declared.
			return null;

		exprType node = typeInfo.getNode();

		if (node instanceof Attribute attribute)
			return getFullRepresentationString(attribute);

		throw new IllegalStateException("Unexpected type hint node type: " + node.getClass() + " for parameter: " + this + ".");
	}

	/**
	 * Returns true iff this parameter's declared type hint (if any) names a tensor type. Combines {@link #getTypeInfo()} with the owning
	 * {@link Function}'s tensor-name attribute check.
	 *
	 * @param monitor Progress monitor for the attribute-resolution sub-work.
	 * @return True iff a tensor-typed type hint is declared for this parameter.
	 * @throws Exception If the underlying AST traversal or attribute resolution fails.
	 */
	public boolean hasTensorTypeHint(IProgressMonitor monitor) throws Exception {
		TypeInfo argTypeInfo = this.getTypeInfo();
		if (argTypeInfo == null)
			return false;

		Set<Attribute> attributes = getAllAttributes(argTypeInfo.getNode());
		SubMonitor subMonitor = convert(monitor, "Examining type hints.", attributes.size() * 2);

		for (Attribute typeHintExpr : attributes) {
			IDocument document = this.function.getContainingDocument();

			String fqn;
			PySelection selection = null;
			try {
				selection = getSelection(typeHintExpr.attr, document);
				fqn = getFullyQualifiedName(typeHintExpr, this.function.getContainingModuleName(), this.function.getContainingFile(),
						selection, this.function.getNature(), subMonitor.split(1));
			} catch (AmbiguousDeclaringModuleException | NoDeclaringModuleException | NoTextSelectionException e) {
				LOG.warn(String.format(
						"Can't determine FQN of type hint expression: %s in selection: %s, module: %s, file: %s, and project: %s.",
						typeHintExpr, selection == null ? "null" : selection.getSelectedText(), this.function.getContainingModuleName(),
						this.function.getContainingFile().getName(), this.function.getProject()), e);

				subMonitor.worked(1);
				continue; // next attribute.
			}

			LOG.info("Found FQN: " + fqn + ".");

			if (TF_TENSOR_TYPE_HINT_FQNS.contains(fqn)) {
				subMonitor.done();
				return true;
			}

			subMonitor.worked(1);
		}

		subMonitor.done();
		return false;
	}

	private static Set<Attribute> getAllAttributes(exprType node) throws Exception {
		Set<Attribute> ret = Sets.newHashSet();

		if (node instanceof Attribute)
			ret.add((Attribute) node);

		if (node != null)
			node.traverse(new VisitorBase() {

				@Override
				public void traverse(SimpleNode node) throws Exception {
					node.traverse(this);
				}

				@Override
				protected Object unhandled_node(SimpleNode node) throws Exception {
					return null;
				}

				@Override
				public Object visitAttribute(Attribute node) throws Exception {
					ret.add(node);
					return super.visitAttribute(node);
				}
			});

		return ret;
	}

	/**
	 * Returns true if the given {@link InstanceKey} is contained in the given {@link Set} of tensor container {@link InstanceKey}s. Also
	 * returns true if the given {@link InstanceKey} represents a container whose constituent elements are contained in the given
	 * {@link Set}.
	 *
	 * @param instanceKey The {@link InstanceKey} in question.
	 * @param tensorContainers A {@link Set} of {@link InstanceKey}s representing containers of tensors.
	 * @param builder The {@link PythonSSAPropagationCallGraphBuilder}.
	 * @return True iff either the given {@link InstanceKey} is a member of the given {@link Set} or the given {@link InstanceKey} is itself
	 *         a container whose elements are (ultimately) contained in the given {@link Set}.
	 */
	private static boolean isTensorContainer(InstanceKey instanceKey, Set<InstanceKey> tensorContainers,
			PythonSSAPropagationCallGraphBuilder builder) {
		return isTensorContainer(instanceKey, tensorContainers, builder, new HashSet<>());
	}

	private static boolean isTensorContainer(InstanceKey instanceKey, Set<InstanceKey> tensorContainers,
			PythonSSAPropagationCallGraphBuilder builder, Set<InstanceKey> seen) {
		if (tensorContainers.contains(instanceKey))
			return true;

		seen.add(instanceKey);

		if (Util.isContainerType(instanceKey.getConcreteType().getReference())) {
			PointerKey catalogPointerKey = ((AstPointerKeyFactory) builder.getPointerKeyFactory())
					.getPointerKeyForObjectCatalog(instanceKey);
			Iterable<InstanceKey> catalogPointsToSet = builder.getPointerAnalysis().getPointsToSet(catalogPointerKey);

			for (InstanceKey catalogInstanceKey : catalogPointsToSet)
				if (catalogInstanceKey instanceof ConstantKey<?>) {
					ConstantKey<?> constantKey = (ConstantKey<?>) catalogInstanceKey;
					Object value = constantKey.getValue();

					if (value != null) {
						IClass concreteType = instanceKey.getConcreteType();
						IField field = concreteType.getField(Atom.findOrCreateAsciiAtom(value.toString()));
						PointerKey pointerKeyForField = builder.getPointerKeyForInstanceField(instanceKey, field);
						Iterable<InstanceKey> fieldPointsToSet = builder.getPointerAnalysis().getPointsToSet(pointerKeyForField);

						for (InstanceKey fieldInstanceKey : fieldPointsToSet)
							if (!seen.contains(fieldInstanceKey) && isTensorContainer(fieldInstanceKey, tensorContainers, builder, seen))
								return true;
					}
				} else if (catalogInstanceKey instanceof AllocationSiteInNode || catalogInstanceKey instanceof ScopeMappingInstanceKey) {
					AllocationSiteInNode asin = getAllocationSiteInNode(catalogInstanceKey);

					if (!seen.contains(asin))
						return isTensorContainer(asin, tensorContainers, builder, seen);
				} else
					throw new IllegalArgumentException(
							"Not expecting a catalog instance of " + instanceKey + " to be: " + catalogInstanceKey.getClass());
		}

		return false;
	}

	private static TypeReference getTypeReference(InstanceKey instanceKey) {
		if (instanceKey instanceof AllocationSiteInNode || instanceKey instanceof ScopeMappingInstanceKey) {
			AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
			return asin.getConcreteType().getReference();
		} else if (instanceKey instanceof ConstantKey<?>) {
			ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
			return constantKey.getConcreteType().getReference();
		} else
			throw new IllegalStateException("Not expecting: " + instanceKey.getClass());
	}

	/**
	 * Returns, for each container of tensors the given analysis knows, its per-field element types: field name (the stringified catalog
	 * constant, e.g. {@code "0"}) to the {@link TensorType}s the analysis associates with that field's {@link InstanceFieldPointerKey}. The
	 * key set answers the boolean container question; the field maps feed the nested-spec reduction (#781).
	 *
	 * @param tensorAnalysis The {@link TensorTypeAnalysis}.
	 * @param monitor Progress.
	 * @return Container instance keys mapped to their per-field element {@link TensorType}s.
	 */
	private static Map<InstanceKey, Map<String, Set<TensorType>>> getTensorContainerElements(TensorTypeAnalysis tensorAnalysis,
			IProgressMonitor monitor) {
		SubMonitor progress = SubMonitor.convert(monitor, tensorAnalysis.getNumberOfEvaluations());

		Map<InstanceKey, Map<String, Set<TensorType>>> result = tensorContainersCache.computeIfAbsent(tensorAnalysis, k -> {
			Map<InstanceKey, Map<String, Set<TensorType>>> tensorContainers = new HashMap<>();

			for (Pair<PointerKey, TensorVariable> pair : k) {
				PointerKey pointerKey = pair.fst;

				if (pointerKey instanceof InstanceFieldPointerKey) {
					InstanceFieldPointerKey ifpk = (InstanceFieldPointerKey) pointerKey;
					InstanceKey instanceKey = ifpk.getInstanceKey();
					TypeReference reference = getTypeReference(instanceKey);

					if (reference != null && Util.isContainerType(reference)) {
						// We have a match.
						// check the existence of the tensor variable.
						assert pair.snd != null : "Tensor variable should be non-null if there is a PK.";
						Map<String, Set<TensorType>> fields = tensorContainers.computeIfAbsent(instanceKey, x -> new HashMap<>());

						// The field name lives on the concrete key; other implementors still register the container for the boolean
						// question, only without element evidence, which the reduction reports as an unsupported form.
						if (ifpk instanceof InstanceFieldKey fieldKey)
							fields.computeIfAbsent(fieldKey.getField().getName().toString(), x -> new HashSet<>())
									.addAll(pair.snd.getTypes());
					}
				}

				progress.worked(1);
			}

			return tensorContainers;
		});

		progress.done();
		return result;
	}

	/**
	 * Returns a {@link Set} of {@link InstanceKey}s representing containers of tensors.
	 *
	 * @param tensorAnalysis The {@link TensorTypeAnalysis}.
	 * @param monitor Progress.
	 * @return A {@link Set} of {@link InstanceKey}s representing containers of tensors.
	 */
	private static Set<InstanceKey> getTensorContainers(TensorTypeAnalysis tensorAnalysis, IProgressMonitor monitor) {
		return getTensorContainerElements(tensorAnalysis, monitor).keySet();
	}

	/**
	 * Returns true iff the given parameter represents a container in the given {@link TensorTypeAnalysis}.
	 *
	 * @param tensorAnalysis The {@link TensorTypeAnalysis}.
	 * @param paramInx The index of the parameter under question.
	 * @param nodes The call graph nodes corresponding to the parameter's owning function.
	 * @param builder The {@link CallGraphBuilder}.
	 * @param monitor For progress.
	 * @return True iff the given {@link TensorTypeAnalysis} includes a container corresponding to the given parameter index.
	 */
	protected static boolean tensorAnalysisIncludesParameterContainer(TensorTypeAnalysis tensorAnalysis, int paramInx, Set<CGNode> nodes,
			PythonSSAPropagationCallGraphBuilder builder, IProgressMonitor monitor) {
		SubMonitor progress = SubMonitor.convert(monitor, "Checking tensor analysis for containers of tensors sent as arguments.", 100);
		Set<InstanceKey> tensorContainers = getTensorContainers(tensorAnalysis, progress.split(30));

		SubMonitor loopProgress = progress.split(70).setWorkRemaining(nodes.size());

		for (CGNode node : nodes) {
			IR ir = node.getIR();
			int i = paramInx + 1;

			if (i >= ir.getNumberOfParameters()) {
				LOG.warn("Parameter index (" + i + ") must be inbounds (" + ir.getNumberOfParameters() + "). Skipping: "
						+ ir.getMethod().getSignature());
				continue;
			}

			int param = ir.getParameter(i); // the first argument is the function being invoked.

			PointerKey paramePointerKey = builder.getPointerKeyForLocal(node, param);
			Iterable<InstanceKey> paramPointsToSet = builder.getPointerAnalysis().getPointsToSet(paramePointerKey);

			for (InstanceKey instanceKey : paramPointsToSet)
				if (isTensorContainer(instanceKey, tensorContainers, builder)) {
					progress.done();
					return true;
				}

			loopProgress.worked(1);
		}

		return false;
	}

	/**
	 * Returns true iff Ariadne's tensor analysis associates a tensor-container instance key with this parameter's slot in the call graph
	 * (i.e. the parameter receives a list/tuple/dict whose elements are tensors).
	 *
	 * @param tensorAnalysis Ariadne's analysis result.
	 * @param callGraph The call graph being queried.
	 * @param builder The propagation-call-graph builder for the project.
	 * @param monitor Progress monitor for the sub-work.
	 * @return True iff the analysis associates a tensor-container with this parameter.
	 * @throws CoreException If the underlying analysis fails.
	 */
	public boolean hasTensorContainer(TensorTypeAnalysis tensorAnalysis, CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder,
			IProgressMonitor monitor) throws CoreException {
		return this.hasTensorContainer(tensorAnalysis, this.function.getNodes(callGraph), builder, monitor);
	}

	boolean hasTensorContainer(TensorTypeAnalysis tensorAnalysis, Set<CGNode> nodes, PythonSSAPropagationCallGraphBuilder builder,
			IProgressMonitor monitor) {
		return tensorAnalysisIncludesParameterContainer(tensorAnalysis, this.getIndex(), nodes, builder, monitor);
	}

	/**
	 * Extracts the per-position element types of this parameter's sequence containers, populating {@link #getContainerElementTypes()} (and
	 * {@link #getContainerAritiesDisagree()} for the arity-disagreement case). Runs only after Phase 3 classified this parameter as a
	 * tensor container. The form is modeled iff every instance key the parameter points to is a list or tuple the tensor analysis knows as
	 * a container of tensors, every such container's object catalog is a set of constant indices forming a contiguous run {@code 0..n-1},
	 * the containers agree on {@code n}, and every position carries tensor evidence. Anything else leaves
	 * {@link #getContainerElementTypes()} {@code null}, which the reduction reports as an unsupported container form (#781).
	 *
	 * @param tensorAnalysis Ariadne's analysis result.
	 * @param nodes The call graph nodes corresponding to the owning function.
	 * @param builder The propagation-call-graph builder for the project.
	 * @param monitor Progress monitor for the sub-work.
	 */
	private void extractContainerElements(TensorTypeAnalysis tensorAnalysis, Set<CGNode> nodes,
			PythonSSAPropagationCallGraphBuilder builder, IProgressMonitor monitor) {
		Map<InstanceKey, Map<String, Set<TensorType>>> containers = getTensorContainerElements(tensorAnalysis, monitor);

		// The container values reaching this parameter, matched by key identity exactly as the boolean Phase 3 matches them.
		Set<InstanceKey> reaching = new LinkedHashSet<>();

		for (CGNode node : nodes) {
			IR ir = node.getIR();
			int paramInx = this.getIndex() + 1; // the first argument is the function being invoked.

			if (paramInx >= ir.getNumberOfParameters())
				continue;

			PointerKey pointerKey = builder.getPointerKeyForLocal(node, ir.getParameter(paramInx));

			for (InstanceKey instanceKey : builder.getPointerAnalysis().getPointsToSet(pointerKey)) {
				TypeReference reference = getTypeReference(instanceKey);

				if (reference == null || !Util.isSequenceType(reference) || !containers.containsKey(instanceKey)) {
					// A non-sequence value, or a sequence the analysis has no element evidence for, reaches the parameter alongside the
					// containers. TensorFlow enforces the declared nesting, so a nested spec would reject that value; the form is
					// unsupported.
					LOG.info("Parameter " + this + " mixes containers with unmodeled values; not reducing: " + instanceKey + ".");
					return;
				}

				reaching.add(instanceKey);
			}
		}

		if (reaching.isEmpty())
			return;

		// Arity per container, from the object catalog: a contiguous run of constant indices 0..n-1, or the form is unsupported.
		int arity = -1;

		for (InstanceKey container : reaching) {
			Set<String> fieldNames = new HashSet<>();
			PointerKey catalogPointerKey = ((AstPointerKeyFactory) builder.getPointerKeyFactory()).getPointerKeyForObjectCatalog(container);

			for (InstanceKey catalogInstanceKey : builder.getPointerAnalysis().getPointsToSet(catalogPointerKey)) {
				if (!(catalogInstanceKey instanceof ConstantKey<?> constantKey) || constantKey.getValue() == null) {
					// E.g. a list grown by an append loop the analysis cannot enumerate.
					LOG.info("Parameter " + this + " has a container with a non-constant element structure; not reducing: " + container
							+ ".");
					return;
				}

				fieldNames.add(constantKey.getValue().toString());
			}

			for (int j = 0; j < fieldNames.size(); j++)
				if (!fieldNames.contains(String.valueOf(j))) {
					LOG.info("Parameter " + this + " has a container with non-contiguous indices; not reducing: " + container + ".");
					return;
				}

			if (arity == -1)
				arity = fieldNames.size();
			else if (arity != fieldNames.size()) {
				LOG.info("Parameter " + this + " receives sequences of disagreeing arities (" + arity + " vs " + fieldNames.size() + ").");
				this.containerAritiesDisagree = TRUE;
				return;
			}
		}

		if (arity <= 0)
			return;

		// Union the element types per position across the reaching containers; a position with no evidence leaves the form unsupported.
		List<Set<TensorType>> elements = new ArrayList<>(arity);

		for (int j = 0; j < arity; j++) {
			Set<TensorType> union = new HashSet<>();

			for (InstanceKey container : reaching) {
				Set<TensorType> types = containers.get(container).get(String.valueOf(j));

				if (types == null || types.isEmpty()) {
					LOG.info("Parameter " + this + " has no tensor evidence for element " + j + " of: " + container + "; not reducing.");
					return;
				}

				union.addAll(types);
			}

			elements.add(unmodifiableSet(union));
		}

		this.containerAritiesDisagree = FALSE;
		this.containerElementTypes = elements;
	}

	/**
	 * Infers the {@link TensorType}s the given {@link TensorTypeAnalysis} associates with this parameter. The lattice contract (see
	 * {@code Tensor Type Generators} in {@code com.ibm.wala.cast.python.ml/CONTRIBUTING.md}) lives at the <em>per-shape and per-dtype</em>
	 * level <em>inside</em> each {@link TensorType}: {@link TensorType#getDims()} {@code == null} is shape-⊤, {@link TensorType#getDType()}
	 * {@code == DType.UNKNOWN} is dtype-⊤; the two axes are orthogonal. Empty-set outputs from a generator's {@code getDefaultShapes} or
	 * {@code getDefaultDTypes} mean ⊥ at that generator's call site (per the contract tables). {@code TensorVariable.state} (the
	 * accumulated {@code Set<TensorType>} the iterator surfaces) is <em>not</em> itself a lattice point. Generators that classify a
	 * variable as a tensor but cannot determine shape/dtype emit a placeholder {@code TensorType(UNKNOWN, null)} per the orthogonality
	 * rule, so a non-empty state means "Ariadne has at least one tensor classification (possibly ⊤ on either axis)" and an empty state
	 * means "no generator emitted a TensorType for this variable" (effectively ⊥ at the variable level, under contract-compliant
	 * generators). {@link TensorTypeAnalysis#iterator} filters its output to {@code state != null && !state.isEmpty()}, which collapses the
	 * iterator-side "variable not analyzed" case with the "Ariadne classified as not-a-tensor" case—both surface as no entry. What this
	 * method exposes to callers:
	 * <ul>
	 * <li>Empty {@code Set<TensorType>}: no iterator entry for this parameter. Equivalent to "Hybridize has no information"—either Ariadne
	 * didn't analyze the variable or it classified the variable as not-a-tensor. Both behave identically for downstream consumers (e.g.,
	 * {@code Function.inferInputSignature} excludes the parameter from the inferred signature).
	 * <li>Non-empty {@code Set<TensorType>}: Ariadne emitted at least one {@code TensorType}. Individual entries may carry shape-⊤ (null
	 * dims) or dtype-⊤ ({@code DType.UNKNOWN}). Callers that need the lattice signal should inspect each {@code TensorType}'s dims and
	 * dtype directly.
	 * </ul>
	 * The {@code null} {@code TensorVariable} branch is a defensive contract assertion: {@link TensorTypeAnalysis#iterator} filters its
	 * output to non-null {@code TensorVariable}s, so the {@code IllegalStateException} would only fire on a future Ariadne contract
	 * violation rather than under any current path.
	 *
	 * @param analysis The {@link TensorTypeAnalysis} to query.
	 */
	void inferTensorTypes(TensorTypeAnalysis analysis) {
		Set<TensorType> result = new HashSet<>();

		for (Pair<PointerKey, TensorVariable> pair : analysis) {
			PointerKey pointerKey = pair.fst;
			if (pointerKey instanceof LocalPointerKey) {
				LocalPointerKey localPointerKey = (LocalPointerKey) pointerKey;
				if (localPointerKey.isParameter() && this.matches(localPointerKey)) {
					TensorVariable tensorVariable = pair.snd;
					if (tensorVariable == null)
						throw new IllegalStateException("Tensor variable was null even though the matching PointerKey is present.");
					result.addAll(tensorVariable.getTypes());
				}
			}
		}

		this.setTensorTypes(unmodifiableSet(result));
	}

	private exprType getNameExpr() {
		return (this.keywordOnly ? this.arguments.kwonlyargs : this.arguments.args)[this.index];
	}

	/**
	 * Returns true iff the given pointer key corresponds to this parameter in Ariadne's IR. The comparison is by source-position equality
	 * (same containing file and same begin-line/begin-column on the parameter declaration). Ariadne's parameter-position metadata is the
	 * only stable correspondence between Jython AST nodes and WALA pointer keys.
	 *
	 * @param rhsPointerKey A parameter pointer key from a {@link TensorTypeAnalysis} entry.
	 * @return True iff the pointer key represents this parameter.
	 */
	private boolean matches(LocalPointerKey rhsPointerKey) {
		File containingFile = this.function.getContainingFile();
		CGNode node = rhsPointerKey.getNode();
		IMethod nodeMethod = node.getMethod();

		if (nodeMethod instanceof AstMethod) {
			AstMethod astMethod = (AstMethod) nodeMethod;
			String sourceFileName = astMethod.getDeclaringClass().getSourceFileName();

			if (containingFile.getAbsolutePath().equals(sourceFileName)) {
				assert rhsPointerKey.isParameter();

				exprType lhsParamExpr = this.getNameExpr();
				int lhsBeginColumn = lhsParamExpr.beginColumn;
				int lhsBeginLine = lhsParamExpr.beginLine;

				int paramIndex = rhsPointerKey.getValueNumber() - 1;
				Position parameterPosition = astMethod.getParameterPosition(paramIndex);

				if (parameterPosition != null) {
					int rhsBeginColumn = parameterPosition.getFirstCol() + 1; // workaround https://github.com/jython/jython3/issues/48.
					int rhsBeginLine = parameterPosition.getFirstLine();

					return lhsBeginColumn == rhsBeginColumn && lhsBeginLine == rhsBeginLine;
				}
			}
		}

		return false;
	}

	/**
	 * Classifies this parameter as tensor-typed (or not) by combining type-hint detection, Ariadne's tensor-type analysis, and
	 * tensor-container detection. Populates the {@link #isTensor()} cache and returns the same verdict for caller convenience.
	 *
	 * @param tensorAnalysis Ariadne's tensor type analysis for the project.
	 * @param callGraph The call graph for the project.
	 * @param builder The propagation-call-graph builder for the project.
	 * @param monitor Progress monitor for the sub-work.
	 * @return True iff this parameter is classified as tensor-typed.
	 * @throws Exception If the underlying analysis or AST traversal fails.
	 */
	public boolean classifyAsTensor(TensorTypeAnalysis tensorAnalysis, CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder,
			IProgressMonitor monitor) throws Exception {
		return this.classifyAsTensor(tensorAnalysis, this.function.getNodes(callGraph), builder, monitor);
	}

	/**
	 * Classifies this parameter as tensor-typed (or not) by combining type-hint detection, Ariadne's tensor-type analysis, and
	 * tensor-container detection. Populates the {@link #isTensor()} cache and returns the same verdict for caller convenience.
	 *
	 * @param tensorAnalysis Ariadne's tensor type analysis for the project.
	 * @param nodes The call graph nodes corresponding to the owning function.
	 * @param builder The propagation-call-graph builder for the project.
	 * @param monitor Progress monitor for the sub-work.
	 * @return True iff this parameter is classified as tensor-typed.
	 * @throws Exception If the underlying analysis or AST traversal fails.
	 */
	boolean classifyAsTensor(TensorTypeAnalysis tensorAnalysis, Set<CGNode> nodes, PythonSSAPropagationCallGraphBuilder builder,
			IProgressMonitor monitor) throws Exception {
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Checking if parameter: " + this + " is tensor-typed...", 4);

		try {
			// don't consider `self` as a tensor.
			if (this.isSelf())
				return this.tensor = FALSE;

			// Populate the tensor-types cache up front whenever Ariadne has anything to say about this parameter, so subsequent reads via
			// `getTensorTypes()` (no-arg) see a consistent value regardless of which classification phase below fires. In particular, the
			// type-hint shortcut (Phase 1) `return`s before reaching the Ariadne query; populating here keeps the cache correct for
			// type-hint parameters that Ariadne also classified from the call site.
			if (!nodes.isEmpty())
				this.inferTensorTypes(tensorAnalysis);

			// check a special case where we consider type hints.
			boolean followTypeHints = this.function.getAlwaysFollowTypeHints() || this.function.getHybridizationParameters() != null
					// TODO: Actually get the value here (#111).
					&& this.function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam();

			// Phase 1: type hints.
			if (followTypeHints) {
				LOG.info("Following type hints for: " + this.function + " and parameter: " + this.getName() + ".");

				if (this.hasTensorTypeHint(subMonitor.split(1))) {
					LOG.info(this.function + " likely has a tensor parameter: " + this.getName() + " due to a type hint.");
					this.function.addInfo(TYPE_INFERENCING, "Used a type hint to infer tensor type for parameter: " + this.getName() + ".");
					subMonitor.worked(2);
					return this.tensor = TRUE;
				}
			} else
				subMonitor.worked(1);

			// If this function is in the call graph.
			if (!nodes.isEmpty()) {
				// Phase 2: read the tensor-types cache populated above. If Ariadne associated any tensor type with this parameter, treat it
				// as tensor-typed.
				//
				// Under contract-compliant generators (per wala/ML CONTRIBUTING.md, "Tensor Type Generators"), a non-empty
				// `getTensorTypes()` corresponds to "Ariadne classifies this as a tensor (possibly with ⊤ on shape or dtype)"; empty
				// corresponds to ⊥ (not a tensor). The lattice is per-axis inside individual `TensorType`s; the aggregation in
				// `TensorGenerator.getTensorTypes` upstream of us folds any ⊥-on-an-axis into "no TensorType emitted," so a non-empty
				// state here means at least one (shape, dtype) combination survived without either axis being ⊥. We trust the iterator
				// output rather than re-validating lattice consistency on our side.
				if (!this.getTensorTypes().isEmpty()) {
					LOG.info(this.function + " likely has a tensor parameter: " + this.getName() + " due to tensor analysis.");
					this.function.addInfo(TYPE_INFERENCING,
							"Used tensor type analysis to infer tensor type for parameter: " + this.getName() + ".");
					subMonitor.worked(2);
					return this.tensor = TRUE;
				}

				subMonitor.worked(1);

				// Phase 3: check for containers of tensors.
				boolean isContainer = this.hasTensorContainer(tensorAnalysis, nodes, builder, subMonitor.split(1));
				this.tensorContainer = isContainer;
				if (isContainer) {
					// Surface the elements' types for the nested-spec reduction (#781); the boolean verdict stands regardless of whether
					// the container form is one the reduction models.
					this.extractContainerElements(tensorAnalysis, nodes, builder, subMonitor.split(1));
					LOG.info(this.function + " likely has a tensor-like parameter: " + this.getName() + " due to tensor analysis.");
					this.function.addInfo(TYPE_INFERENCING,
							"Used tensor type analysis to infer tensor container type for parameter: " + this.getName() + ".");
					return this.tensor = TRUE;
				}
			} else
				subMonitor.worked(2);

			return this.tensor = FALSE;
		} finally {
			subMonitor.done();
		}
	}

	/**
	 * Returns the cached tensor-typed classification produced by {@link #classifyAsTensor}.
	 *
	 * @return {@code TRUE} if tensor-typed, {@code FALSE} if not, or {@code null} if {@link #classifyAsTensor} has not yet run.
	 */
	public Boolean isTensor() {
		return this.tensor;
	}

	/**
	 * Returns whether any call site supplies an argument for this parameter, as computed by {@link Function#inferSuppliedParameters}.
	 *
	 * @return {@code TRUE} if some call site supplies it, {@code FALSE} if none does, or {@code null} if the question was not asked or
	 *         could not be answered. Callers must treat {@code null} as "cannot omit this parameter"; see the field's documentation.
	 */
	public Boolean isSuppliedAtCallSite() {
		return this.suppliedAtCallSite;
	}

	void setSuppliedAtCallSite(Boolean suppliedAtCallSite) {
		this.suppliedAtCallSite = suppliedAtCallSite;
	}

	@Override
	public int hashCode() {
		return Objects.hash(this.function, Integer.valueOf(this.index), this.getName());
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null || getClass() != obj.getClass())
			return false;
		Parameter other = (Parameter) obj;
		return this.index == other.index && Objects.equals(this.function, other.function)
				&& Objects.equals(this.getName(), other.getName());
	}

	@Override
	public String toString() {
		return this.getName() + "@" + this.index + " of " + this.function;
	}

	/**
	 * Clears any cached analysis results.
	 */
	public static void clearCaches() {
		tensorContainersCache.clear();
	}
}
