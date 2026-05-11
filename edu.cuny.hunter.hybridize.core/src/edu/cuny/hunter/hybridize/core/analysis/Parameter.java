package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

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

import com.google.common.collect.Sets;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.collections.Pair;

/**
 * Analytical wrapper around a single positional Python function parameter. Carries enough context to identify the parameter
 * ({@code argumentsType} parent + positional index + owning {@link Function}) and hosts the per-parameter classification queries that would
 * otherwise live on {@link Function}: type-hint detection ({@link #hasTensorTypeHint(IProgressMonitor)}), Ariadne tensor-type lookup
 * ({@link #getTensorTypes(TensorTypeAnalysis)}), and tensor-container detection
 * ({@link #hasTensorContainer(TensorTypeAnalysis, CallGraph, PythonSSAPropagationCallGraphBuilder, IProgressMonitor)}).
 * <p>
 * Intentionally narrow public surface: {@link #getIndex()}, {@link #getName()}, {@link #isSelf()}, {@link #getTypeInfo()},
 * {@link #hasTensorTypeHint(IProgressMonitor)}, {@link #getTensorTypes(TensorTypeAnalysis)},
 * {@link #hasTensorContainer(TensorTypeAnalysis, CallGraph, PythonSSAPropagationCallGraphBuilder, IProgressMonitor)}, plus
 * {@code equals}/{@code hashCode}/{@code toString}. Constructed only by {@link Function} (package-private constructor).
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
	 * Parent Jython AST node carrying every positional name expression (in {@link argumentsType#args}) and per-position annotation (in
	 * {@link argumentsType#annotation}) of the owning function. Shared across all {@link Parameter}s of the same function.
	 */
	private final argumentsType arguments;

	/**
	 * Zero-based position of this parameter within {@link #arguments}{@code .args}. Bounded at construction time to be a valid index.
	 */
	private final int index;

	/**
	 * Owning {@link Function} back-reference. Reached by every analytical method on this class for project-context state and helpers:
	 * {@link #getTypeInfo()} via {@link Function#getFunctionDefinition()}, {@link #hasTensorTypeHint(IProgressMonitor)} via
	 * {@link Function#getContainingDocument()}/{@code getContainingModuleName}/{@code getContainingFile}/{@code getNature}/{@code getProject},
	 * {@link #hasTensorContainer(TensorTypeAnalysis, CallGraph, PythonSSAPropagationCallGraphBuilder, IProgressMonitor)} via
	 * {@code Function.tensorAnalysisIncludesParameterContainer}, and {@link #matches(LocalPointerKey)} via
	 * {@link Function#getContainingFile()}.
	 */
	private final Function function;

	/**
	 * Package-private because {@link Parameter}s are only ever constructed inside {@link Function}'s constructor (same package).
	 *
	 * @param arguments The parent {@link argumentsType} node. Non-null.
	 * @param index The zero-based positional index within {@code arguments.args}. Must be in {@code [0, arguments.args.length)}.
	 * @param function The owning {@link Function}. Non-null.
	 * @throws IndexOutOfBoundsException If {@code index} is out of range for {@code arguments.args}.
	 */
	Parameter(argumentsType arguments, int index, Function function) {
		this.arguments = Objects.requireNonNull(arguments);
		this.function = Objects.requireNonNull(function);
		if (index < 0 || arguments.args == null || index >= arguments.args.length)
			throw new IndexOutOfBoundsException("Parameter index " + index + " out of bounds for arguments of length "
					+ (arguments.args == null ? 0 : arguments.args.length) + ".");
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
		return NodeUtils.getRepresentationString(this.getNameExpr());
	}

	/**
	 * Returns true iff this parameter is the implicit first parameter of an instance method: positional index {@code 0} and named
	 * {@code self} (the conventional name).
	 *
	 * @return True iff this parameter is at index {@code 0} and is named {@code self}.
	 */
	public boolean isSelf() {
		return this.getIndex() == 0 && SELF_PARAMETER_NAME.equals(this.getName());
	}

	/**
	 * Returns PyDev's AST-derived type information for this parameter (i.e. the type-hint annotation in the function's signature), or
	 * {@code null} if no type hint is declared.
	 *
	 * @return The {@link TypeInfo} for this parameter, or {@code null} if no type hint is present.
	 */
	public TypeInfo getTypeInfo() {
		return NodeUtils.getTypeForParameterFromAST(this.getName(), this.function.getFunctionDefinition().getFunctionDef());
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
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Examining type hints.", attributes.size() * 2);

		for (Attribute typeHintExpr : attributes) {
			IDocument document = this.function.getContainingDocument();

			String fqn;
			PySelection selection = null;
			try {
				selection = Util.getSelection(typeHintExpr.attr, document);
				fqn = Util.getFullyQualifiedName(typeHintExpr, this.function.getContainingModuleName(), this.function.getContainingFile(),
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

			if (fqn.equals(TF_TENSOR_FQN)) { // TODO: Also check for subtypes (RaggedTensor, SparseTensor, Variable, IndexedSlices) (#434).
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
	 * Returns true iff Ariadne's tensor analysis associates a tensor-container instance key with this parameter's slot in the call graph
	 * (i.e. the parameter receives a list/tuple/dict whose elements are tensors).
	 *
	 * @param tensorAnalysis Ariadne's analysis result.
	 * @param callGraph The call graph being queried.
	 * @param builder The propagation-call-graph builder for the project.
	 * @param monitor Progress monitor for the sub-work.
	 * @return True iff the analysis associates a tensor-container with this parameter.
	 * @throws org.eclipse.core.runtime.CoreException If the underlying analysis fails.
	 */
	public boolean hasTensorContainer(TensorTypeAnalysis tensorAnalysis, CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder,
			IProgressMonitor monitor) throws org.eclipse.core.runtime.CoreException {
		return this.function.tensorAnalysisIncludesParameterContainer(tensorAnalysis, this.getIndex(), callGraph, builder, monitor);
	}

	/**
	 * Returns the {@link TensorType}s the given {@link TensorTypeAnalysis} associates with this parameter. Computed fresh on each call (no
	 * caching) by iterating {@code analysis}.
	 * <p>
	 * Returns an empty (but non-null) set when the analysis associated no entries with this parameter. Note that with the current
	 * {@link TensorTypeAnalysis#iterator()} contract, "tensor with unknown types" (i.e. a {@code TensorVariable} with empty state) and "not
	 * a tensor" (no {@code TensorVariable} bound to the matching pointer key) are indistinguishable, so an empty result means one of those
	 * two cases without telling them apart. Honoring the wala/ML lattice distinction would require a richer Ariadne-side query; left for a
	 * future enhancement.
	 *
	 * @param analysis The {@link TensorTypeAnalysis} to query. Non-null.
	 * @return Unmodifiable, possibly-empty set of inferred tensor types. Never {@code null}.
	 */
	public Set<TensorType> getTensorTypes(TensorTypeAnalysis analysis) {
		Objects.requireNonNull(analysis);
		Set<TensorType> result = new HashSet<>();
		for (Pair<PointerKey, TensorVariable> pair : analysis) {
			PointerKey pointerKey = pair.fst;
			if (pointerKey instanceof LocalPointerKey) {
				LocalPointerKey localPointerKey = (LocalPointerKey) pointerKey;
				if (localPointerKey.isParameter() && this.matches(localPointerKey)) {
					TensorVariable tensorVariable = pair.snd;
					if (tensorVariable != null)
						result.addAll(tensorVariable.getTypes());
				}
			}
		}
		return Collections.unmodifiableSet(result);
	}

	private exprType getNameExpr() {
		return this.arguments.args[this.index];
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
}
