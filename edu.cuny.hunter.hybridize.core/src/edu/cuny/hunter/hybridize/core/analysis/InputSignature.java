package edu.cuny.hunter.hybridize.core.analysis;

import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.StringJoiner;
import java.util.stream.Collectors;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
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
	 * How a supplied (developer-written) signature relates to an inferred one under the per-parameter, per-axis partial order on the
	 * tensor-type lattice. On each axis, a wildcard ({@code None}/dynamic/symbolic dimension, an unknown-rank shape, or an {@code UNKNOWN}
	 * dtype) is the most general value; a concrete value (a fixed dimension, a known-rank shape, or a concrete dtype) is more specific; two
	 * distinct concrete values (and two shapes of different rank) are incomparable. A signature is at least as specific as another iff it
	 * is at least as specific on every parameter and axis.
	 */
	public enum Relation {
		/** The supplied and inferred signatures are identical. */
		AGREEMENT,

		/**
		 * The supplied signature is strictly more specific than the inferred one (it would reject inputs the call-site evidence shows the
		 * function receives). The inferred signature should replace it.
		 */
		SUPPLIED_TIGHTER,

		/**
		 * The supplied signature is strictly more general than the inferred one (it admits more inputs than the evidence requires). The
		 * supplied signature should be preserved, as the broader contract may be intentional and invisible to the static analysis.
		 */
		SUPPLIED_BROADER,

		/**
		 * The supplied and inferred signatures are incomparable—each is more specific than the other on some axis (or they differ in shape
		 * rank, parameter count, or concrete dtype). The inferred signature should replace it.
		 */
		INCOMPARABLE
	}

	/**
	 * Relates this signature, taken as the developer-supplied one, to {@code inferred}, the signature derived from call-site evidence by
	 * {@link Function#inferInputSignature}. See {@link Relation} for the partial order. Used by {@link Function#check()} to decide whether
	 * an existing {@code input_signature} should be overwritten ({@link Relation#SUPPLIED_TIGHTER}, {@link Relation#INCOMPARABLE}),
	 * preserved ({@link Relation#SUPPLIED_BROADER}), or left untouched ({@link Relation#AGREEMENT}).
	 *
	 * @param inferred The signature inferred from call-site evidence.
	 * @return How this (supplied) signature relates to {@code inferred}.
	 */
	public Relation relate(InputSignature inferred) {
		List<TensorType> supplied = this.parameterTypes();
		List<TensorType> evidence = inferred.parameterTypes();

		// A parameter-count mismatch has no meaningful per-parameter order; treat it as incomparable so the inferred signature wins.
		if (supplied.size() != evidence.size())
			return Relation.INCOMPARABLE;

		Relation result = Relation.AGREEMENT;
		for (int i = 0; i < supplied.size(); i++)
			result = combine(result, relate(supplied.get(i), evidence.get(i)));

		return result;
	}

	/**
	 * Combines two axis/parameter relations into the relation of the whole. {@link Relation#AGREEMENT} is the identity (a fully-agreeing
	 * part imposes nothing); {@link Relation#INCOMPARABLE} is absorbing; and a part where the supplied side is tighter combined with one
	 * where it is broader is itself incomparable (neither side is uniformly at least as specific).
	 *
	 * @param a One relation.
	 * @param b The other relation.
	 * @return Their combination.
	 */
	private static Relation combine(Relation a, Relation b) {
		if (a == b)
			return a;
		if (a == Relation.AGREEMENT)
			return b;
		if (b == Relation.AGREEMENT)
			return a;
		// a and b are two different non-AGREEMENT relations: any pairing of {TIGHTER, BROADER, INCOMPARABLE} that isn't equal is
		// incomparable overall.
		return Relation.INCOMPARABLE;
	}

	/**
	 * Relates a supplied {@link TensorType} to an inferred one by combining the dtype-axis and shape-axis relations.
	 *
	 * @param supplied The developer-supplied tensor type.
	 * @param inferred The inferred tensor type.
	 * @return How the supplied tensor type relates to the inferred one.
	 */
	private static Relation relate(TensorType supplied, TensorType inferred) {
		return combine(relateDType(supplied.getDType(), inferred.getDType()), relateShape(supplied.getDims(), inferred.getDims()));
	}

	/**
	 * Relates a supplied dtype to an inferred one. {@code UNKNOWN} is the lattice top (most general); two distinct concrete dtypes are
	 * incomparable.
	 *
	 * @param supplied The supplied dtype.
	 * @param inferred The inferred dtype.
	 * @return How the supplied dtype relates to the inferred one.
	 */
	private static Relation relateDType(DType supplied, DType inferred) {
		if (supplied == inferred)
			return Relation.AGREEMENT;
		if (supplied == DType.UNKNOWN)
			return Relation.SUPPLIED_BROADER;
		if (inferred == DType.UNKNOWN)
			return Relation.SUPPLIED_TIGHTER;
		return Relation.INCOMPARABLE; // two distinct concrete dtypes.
	}

	/**
	 * Relates a supplied shape to an inferred one. A {@code null} dimension list is unknown rank, the lattice top (most general). Two
	 * shapes of different rank are incomparable; otherwise the relation is the combination of the per-dimension relations.
	 *
	 * @param supplied The supplied dimension list, or {@code null} for unknown rank.
	 * @param inferred The inferred dimension list, or {@code null} for unknown rank.
	 * @return How the supplied shape relates to the inferred one.
	 */
	private static Relation relateShape(List<Dimension<?>> supplied, List<Dimension<?>> inferred) {
		if (supplied == null && inferred == null)
			return Relation.AGREEMENT;
		if (supplied == null)
			return Relation.SUPPLIED_BROADER; // unknown rank is more general than any known rank.
		if (inferred == null)
			return Relation.SUPPLIED_TIGHTER;
		if (supplied.size() != inferred.size())
			return Relation.INCOMPARABLE; // different ranks.

		Relation result = Relation.AGREEMENT;
		for (int i = 0; i < supplied.size(); i++)
			result = combine(result, relateDim(supplied.get(i), inferred.get(i)));

		return result;
	}

	/**
	 * Relates a supplied dimension to an inferred one. Any non-{@link NumericDim} (dynamic {@code None}, symbolic {@code ?}, ragged) is a
	 * wildcard—the most general dimension—so all wildcards relate as equal; a concrete {@link NumericDim} is more specific than a wildcard;
	 * two distinct concrete dimensions are incomparable.
	 *
	 * @param supplied The supplied dimension.
	 * @param inferred The inferred dimension.
	 * @return How the supplied dimension relates to the inferred one.
	 */
	private static Relation relateDim(Dimension<?> supplied, Dimension<?> inferred) {
		boolean suppliedConcrete = supplied instanceof NumericDim;
		boolean inferredConcrete = inferred instanceof NumericDim;

		if (!suppliedConcrete && !inferredConcrete)
			return Relation.AGREEMENT; // both wildcards.
		if (suppliedConcrete && !inferredConcrete)
			return Relation.SUPPLIED_TIGHTER;
		if (!suppliedConcrete && inferredConcrete)
			return Relation.SUPPLIED_BROADER;
		return supplied.value().equals(inferred.value()) ? Relation.AGREEMENT : Relation.INCOMPARABLE;
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
