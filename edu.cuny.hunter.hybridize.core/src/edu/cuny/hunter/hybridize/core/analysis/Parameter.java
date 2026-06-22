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
import java.util.HashSet;
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
import com.ibm.wala.cast.python.ml.types.TensorType;
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
	 * Owning {@link Function} back-reference.
	 */
	private final Function function;

	/**
	 * Cache of tensor-container instance keys for each tensor type analysis. Keyed by the analysis object to ensure that cached results are
	 * discarded when the analysis is re-run and a new object is returned. Each value is a set of instance keys that the analysis associates
	 * with containers of tensors (e.g. lists/tuples/dicts whose elements are tensors).
	 */
	private static Map<TensorTypeAnalysis, Set<InstanceKey>> tensorContainersCache = Maps.newConcurrentMap();

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
	 * Returns a {@link Set} of {@link InstanceKey}s representing containers of tensors.
	 *
	 * @param tensorAnalysis The {@link TensorTypeAnalysis}.
	 * @param monitor Progress.
	 * @return A {@link Set} of {@link InstanceKey}s representing containers of tensors.
	 */
	private static Set<InstanceKey> getTensorContainers(TensorTypeAnalysis tensorAnalysis, IProgressMonitor monitor) {
		SubMonitor progress = SubMonitor.convert(monitor, tensorAnalysis.getNumberOfEvaluations());

		Set<InstanceKey> result = tensorContainersCache.computeIfAbsent(tensorAnalysis, k -> {
			Set<InstanceKey> tensorContainers = new HashSet<>();

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
						tensorContainers.add(instanceKey);
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
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Checking if parameter: " + this + " is tensor-typed...", 3);

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
