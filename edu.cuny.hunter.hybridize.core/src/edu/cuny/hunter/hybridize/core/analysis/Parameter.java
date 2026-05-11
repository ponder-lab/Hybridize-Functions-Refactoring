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

	private final argumentsType arguments;

	private final int index;

	private final Function function;

	Parameter(argumentsType arguments, int index, Function function) {
		this.arguments = Objects.requireNonNull(arguments);
		this.function = Objects.requireNonNull(function);
		if (index < 0 || arguments.args == null || index >= arguments.args.length)
			throw new IndexOutOfBoundsException("Parameter index " + index + " out of bounds for arguments of length "
					+ (arguments.args == null ? 0 : arguments.args.length) + ".");
		this.index = index;
	}

	public int getIndex() {
		return this.index;
	}

	public String getName() {
		return NodeUtils.getRepresentationString(this.getNameExpr());
	}

	/**
	 * Returns the {@link TensorType}s Ariadne's tensor analysis associates with this parameter. Computed fresh on each call (no caching)
	 * against the {@link TensorTypeAnalysis} stashed on the owning {@link Function} by {@link Function#inferTensorTensorParameters}; before
	 * that runs, the owning {@code Function} has no analysis and this returns the empty set.
	 *
	 * @return Unmodifiable, possibly-empty set of inferred tensor types. Never {@code null}.
	 */
	public Set<TensorType> getTensorTypes() {
		TensorTypeAnalysis analysis = this.function.getTensorTypeAnalysis();
		if (analysis == null)
			return Collections.emptySet();

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
