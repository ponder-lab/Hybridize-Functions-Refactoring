package edu.cuny.hunter.hybridize.core.analysis;

import java.io.File;
import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;

import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.collections.Pair;

/**
 * Analytical wrapper around a single positional Python function parameter. Holds the minimum context needed to identify the parameter
 * ({@code argumentsType} parent + positional index + owning {@link Function}) and exposes the per-parameter Ariadne tensor-type query as
 * {@link #getTensorTypes()}.
 * <p>
 * Intentionally narrow public surface: {@link #getIndex()}, {@link #getName()}, {@link #getTensorTypes()}, plus
 * {@code equals}/{@code hashCode}/{@code toString}. No Jython AST types leak through the public API. Constructed only by {@link Function}
 * (package-private constructor).
 */
public final class Parameter {

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
	 * Owning {@link Function} back-reference. Reached for {@link Function#getContainingFile()} and
	 * {@link Function#getTensorTypeAnalysis()}.
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
	 * Returns the {@link TensorType}s Ariadne's tensor analysis associates with this parameter. Computed fresh on each call (no caching)
	 * against the {@link TensorTypeAnalysis} the owning {@link Function} has recorded.
	 * <p>
	 * Precondition: the owning {@link Function} has already had {@link Function#inferTensorTensorParameters} run on it (which records the
	 * analysis); calling this beforehand is a programming error and throws.
	 * <p>
	 * Returns an empty (but non-null) set when the analysis has run but associated no entries with this parameter — note that with the
	 * current {@link TensorTypeAnalysis#iterator()} contract, "tensor with unknown types" (i.e. a {@code TensorVariable} with empty state)
	 * and "not a tensor" (no {@code TensorVariable} bound to the matching pointer key) are indistinguishable, so an empty result means one
	 * of those two cases without telling them apart. Honoring the wala/ML lattice distinction would require a richer Ariadne-side query;
	 * left for a future enhancement.
	 *
	 * @return Unmodifiable, possibly-empty set of inferred tensor types. Never {@code null}.
	 * @throws IllegalStateException If the owning {@link Function} has no recorded {@link TensorTypeAnalysis} yet.
	 */
	public Set<TensorType> getTensorTypes() {
		TensorTypeAnalysis analysis = this.function.getTensorTypeAnalysis();
		if (analysis == null)
			throw new IllegalStateException(
					"Tensor analysis has not been recorded on " + this.function + "; call Function.inferTensorTensorParameters first.");

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
	 * (same containing file and same begin-line/begin-column on the parameter declaration) — Ariadne's parameter-position metadata is the
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
