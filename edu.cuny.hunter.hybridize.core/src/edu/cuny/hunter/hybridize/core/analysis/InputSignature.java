package edu.cuny.hunter.hybridize.core.analysis;

import java.util.List;
import java.util.StringJoiner;

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
			StringJoiner shapeDims = new StringJoiner(", ", "(", t.getDims().size() == 1 ? ",)" : ")");
			for (Dimension<?> d : t.getDims())
				shapeDims.add(d instanceof NumericDim ? d.value().toString() : "None");

			String dtype = tfPrefix + t.getDType().name().toLowerCase();
			specs.add(tfPrefix + "TensorSpec(shape=" + shapeDims + ", dtype=" + dtype + ")");
		}

		return specs.toString();
	}
}
