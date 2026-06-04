package edu.cuny.hunter.hybridize.core.analysis;

import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.StringJoiner;
import java.util.stream.Collectors;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;

/**
 * The inferred input signature of a hybridizable function: an ordered tuple of {@link TensorType}s in declaration order, one per
 * non-{@code self} parameter. Produced by {@link Function#inferInputSignature}. `InputSignature` is total over the function's
 * non-{@code self} parameters: every non-{@code self} parameter contributes a {@link TensorType}, or {@link Function#inferInputSignature}
 * returns {@link java.util.Optional#empty} instead of producing a partial signature. The per-parameter dispatch
 * (tensor-classified-with-evidence vs tensor-classified-without-evidence vs truly non-tensor) and its diagnostics live in
 * {@link Function#inferInputSignature}; see that method's Javadoc and #508 for the rules.
 */
public record InputSignature(List<TensorType> parameterTypes) {

	/**
	 * Format this signature as a Python source-code expression suitable for the {@code input_signature=} keyword argument of
	 * {@code @tf.function(...)}. Each parameter's {@link TensorType} renders as {@code tfPrefix + "TensorSpec(shape=(...), dtype=" +
	 * tfPrefix + "<dtype>)"}; shape dims render as concrete integers for {@link NumericDim} and {@code None} for every other
	 * {@link Dimension} subtype (dynamic, ragged, symbolic—all encoded the same way on the {@code TensorSpec} surface). The whole thing is
	 * wrapped in {@code [...]} so it can drop straight into {@code @tf.function(input_signature=...)}.
	 * <p>
	 * Examples (with {@code tfPrefix = "tf."}):
	 * <ul>
	 * <li>Single rank-2 tensor {@code (FLOAT32, [DynamicDim, NumericDim(32)])} → {@code [tf.TensorSpec(shape=(None, 32),
	 * dtype=tf.float32)]}
	 * <li>Two tensors → {@code [tf.TensorSpec(...), tf.TensorSpec(...)]}
	 * <li>Scalar tensor (empty dims) → {@code [tf.TensorSpec(shape=(), dtype=tf.int32)]}
	 * </ul>
	 * <p>
	 * The {@code tfPrefix} argument carries the user's existing import shape so the emitted source matches what the user wrote:
	 * {@code "tf."} for {@code import tensorflow as tf}, {@code "tensorflow."} for {@code import tensorflow}, or {@code ""} when
	 * {@code TensorSpec} and the dtype constants are already in scope via {@code from tensorflow import *} or similar.
	 *
	 * @param tfPrefix The TensorFlow module prefix (e.g., {@code "tf."}), including the trailing dot. May be empty.
	 * @return The Python source-code list expression.
	 */
	public String toTensorSpecList(String tfPrefix) {
		StringJoiner specs = new StringJoiner(", ", "[", "]");

		for (TensorType t : parameterTypes) {
			List<Dimension<?>> dims = t.getDims();
			String shape;
			if (dims == null) {
				// Shape-⊤ from `Function.inferSpec` Step 2 (rank disagrees across contexts, or any context has null dims). TF's
				// `tf.TensorSpec(shape=None, ...)` accepts any shape at runtime—the appropriate encoding when rank itself is unknown.
				shape = "None";
			} else {
				StringJoiner shapeDims = new StringJoiner(", ", "(", dims.size() == 1 ? ",)" : ")");
				for (Dimension<?> d : dims)
					shapeDims.add(d instanceof NumericDim ? d.value().toString() : "None");
				shape = shapeDims.toString();
			}

			String dtype = tfPrefix + dtypeName(t);
			specs.add(tfPrefix + "TensorSpec(shape=" + shape + ", dtype=" + dtype + ")");
		}

		return specs.toString();
	}

	/**
	 * The set of dtype constant names this signature references (e.g., {@code "float32"}, {@code "int32"}), as the bare Python identifiers
	 * emitted by {@link #toTensorSpecList(String)} (without any module prefix). The source-write uses this to verify each required dtype
	 * constant is reachable under the chosen import prefix before emitting an unqualified signature: on the
	 * {@code from tensorflow import ...} named-import path, {@code TensorSpec} being in scope does not imply the dtype constants are too,
	 * so an unguarded emission would produce a {@code NameError}-raising decorator.
	 *
	 * @return The referenced dtype constant names.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/585">Issue 585</a>
	 */
	public Set<String> requiredDTypeNames() {
		return parameterTypes.stream().map(InputSignature::dtypeName).collect(Collectors.toUnmodifiableSet());
	}

	/**
	 * The bare Python dtype constant identifier for a {@link TensorType} (e.g., {@code "float32"}), without any module prefix.
	 * <p>
	 * {@code Locale.ROOT} so dtype identifiers stay ASCII regardless of the JVM default (Turkish locale lower-cases {@code I} to a
	 * non-ASCII dotless-i, which would corrupt {@code INT32} → {@code ınt32}, {@code STRING} → {@code strıng}).
	 *
	 * @param t The tensor type whose dtype constant name to render.
	 * @return The lower-cased dtype constant identifier.
	 */
	private static String dtypeName(TensorType t) {
		return t.getDType().name().toLowerCase(Locale.ROOT);
	}
}
