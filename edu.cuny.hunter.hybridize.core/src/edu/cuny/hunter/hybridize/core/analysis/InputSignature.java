package edu.cuny.hunter.hybridize.core.analysis;

import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.StringJoiner;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;

/**
 * The inferred input signature of a hybridizable function: an ordered tuple of {@link SpecEntry}s in declaration order, one per
 * non-{@code self} parameter. Produced by {@link Function#inferInputSignature}. An entry is either a {@link Single} (the parameter is one
 * tensor, rendering as one {@code TensorSpec}) or a {@link Sequence} (the parameter is a Python list or tuple of tensors, rendering as a
 * nested spec list, which {@code tf.function} accepts and enforces structurally; #781). {@code InputSignature} is total over the function's
 * non-{@code self} parameters: every non-{@code self} parameter contributes an entry, or {@link Function#inferInputSignature} returns an
 * absent result instead of producing a partial signature. The per-parameter dispatch (tensor-classified-with-evidence vs
 * tensor-classified-without-evidence vs truly non-tensor) and its diagnostics live in {@link Function#inferInputSignature}; see that
 * method's Javadoc and #508 for the rules.
 *
 * @param entries The per-parameter spec entries, in declaration order.
 */
public record InputSignature(List<SpecEntry> entries) {

	/** The {@code tf.TensorSpec} constructor name, emitted for a dense tensor parameter. */
	private static final String TENSOR_SPEC = "TensorSpec";

	/** The {@code tf.RaggedTensorSpec} constructor name, emitted for a ragged tensor parameter (#524). */
	private static final String RAGGED_TENSOR_SPEC = "RaggedTensorSpec";

	/** The {@code tf.SparseTensorSpec} constructor name, emitted for a sparse tensor parameter (#533). */
	private static final String SPARSE_TENSOR_SPEC = "SparseTensorSpec";

	/**
	 * One parameter's contribution to an {@link InputSignature}: a {@link Single} tensor spec or a {@link Sequence} of them. Sealed so the
	 * dispatch sites (rendering, name collection, {@link InputSignature#relate}) stay exhaustive as forms are added (a mapping form for
	 * dict-valued parameters is a candidate future member; see #781).
	 */
	public sealed interface SpecEntry permits Single, Sequence {
	}

	/**
	 * A parameter that is one tensor, rendering as one {@code TensorSpec} (or its ragged/sparse variant).
	 *
	 * @param type The parameter's reduced {@link TensorType}.
	 */
	public record Single(TensorType type) implements SpecEntry {
	}

	/**
	 * A parameter that is a Python list or tuple of tensors, rendering as a spec list nested inside the outer {@code input_signature} list
	 * (e.g. {@code [tf.TensorSpec(shape=(2, 2), dtype=tf.int32)]}). TensorFlow enforces the structure: a caller passing a bare tensor where
	 * a sequence is declared raises {@code TypeError}, and a sequence of a different length raises {@code ValueError}, so this entry is
	 * only sound for a parameter whose every call-site value is a sequence of exactly this arity (#781).
	 *
	 * @param elementTypes The reduced {@link TensorType} of each element, in positional order; one per element of the sequence.
	 */
	public record Sequence(List<TensorType> elementTypes) implements SpecEntry {
	}

	/**
	 * Wraps a flat list of per-parameter {@link TensorType}s into an all-{@link Single} signature: the pre-#781 shape, still what the
	 * supplied-signature parser produces (it models flat signatures only) and what most reductions yield. A named factory rather than a
	 * constructor overload because {@code List<TensorType>} and {@code List<SpecEntry>} erase to the same signature.
	 *
	 * @param parameterTypes One {@link TensorType} per non-{@code self} parameter, in declaration order.
	 * @return The signature with each type wrapped as a {@link Single}.
	 */
	public static InputSignature ofSingles(List<TensorType> parameterTypes) {
		return new InputSignature(parameterTypes.stream().<SpecEntry>map(Single::new).toList());
	}

	/**
	 * The flat projection of an all-{@link Single} signature: one {@link TensorType} per parameter. The inverse of {@link #ofSingles} for
	 * signatures with no {@link Sequence} entry, serving consumers and assertions that predate nesting.
	 *
	 * @return The per-parameter {@link TensorType}s.
	 * @throws IllegalStateException If any entry is a {@link Sequence}; a nested signature has no flat projection.
	 */
	public List<TensorType> singleTypes() {
		return entries.stream().map(e -> switch (e) {
		case Single s -> s.type();
		case Sequence q -> throw new IllegalStateException("A nested signature has no flat projection: " + q + ".");
		}).toList();
	}

	/**
	 * Every {@link TensorType} this signature references, across {@link Single} entries and {@link Sequence} elements alike. The traversal
	 * behind {@link #requiredSpecTypeNames()} and {@link #requiredDTypeNames()}: a nested ragged element needs {@code RaggedTensorSpec} in
	 * scope exactly as a top-level one does.
	 *
	 * @return The leaf tensor types, in entry order.
	 */
	private Stream<TensorType> leaves() {
		return entries.stream().flatMap(e -> switch (e) {
		case Single s -> Stream.of(s.type());
		case Sequence q -> q.elementTypes().stream();
		});
	}

	/**
	 * Format this signature as a Python source-code expression suitable for the {@code input_signature=} keyword argument of
	 * {@code @tf.function(...)}. A {@link Single} renders as {@code tfPrefix + "<SpecType>(shape=(...), dtype=" + tfPrefix + "<dtype>)"},
	 * where {@code <SpecType>} is {@code SparseTensorSpec} for a sparse parameter, {@code RaggedTensorSpec} for a parameter with a ragged
	 * dimension, and {@code TensorSpec} otherwise ({@link #specTypeName}); shape dims render as concrete integers for {@link NumericDim}
	 * and {@code None} for every other {@link Dimension} subtype (dynamic, ragged, symbolic—all encoded the same way on the spec surface).
	 * A {@link Sequence} renders its element specs inside a further {@code [...]}, the nested form {@code tf.function} accepts for a
	 * list-valued parameter. The whole thing is wrapped in {@code [...]} so it can drop straight into
	 * {@code @tf.function(input_signature=...)}.
	 * <p>
	 * Examples (with {@code tfPrefix = "tf."}):
	 * <ul>
	 * <li>Single rank-2 tensor {@code (FLOAT32, [DynamicDim, NumericDim(32)])} → {@code [tf.TensorSpec(shape=(None, 32),
	 * dtype=tf.float32)]}
	 * <li>Two tensors → {@code [tf.TensorSpec(...), tf.TensorSpec(...)]}
	 * <li>A tensor and a singleton list of tensors → {@code [tf.TensorSpec(...), [tf.TensorSpec(shape=(2, 2), dtype=tf.int32)]]}
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

		for (SpecEntry e : entries)
			switch (e) {
			case Single s -> specs.add(specString(s.type(), tfPrefix));
			case Sequence q -> {
				StringJoiner inner = new StringJoiner(", ", "[", "]");
				for (TensorType t : q.elementTypes())
					inner.add(specString(t, tfPrefix));
				specs.add(inner.toString());
			}
			}

		return specs.toString();
	}

	/**
	 * One {@code TensorSpec}-constructor call as Python source.
	 *
	 * @param t The tensor type to render.
	 * @param tfPrefix The TensorFlow module prefix, including the trailing dot; may be empty.
	 * @return E.g. {@code "tf.TensorSpec(shape=(None, 32), dtype=tf.float32)"}.
	 */
	private static String specString(TensorType t, String tfPrefix) {
		return tfPrefix + specTypeName(t) + "(shape=" + shapeString(t) + ", dtype=" + tfPrefix + dtypeName(t) + ")";
	}

	/**
	 * The raw shape rendering for a {@link TensorType}: {@code "None"} when the rank is unknown (null dims—shape-⊤, which
	 * {@code tf.TensorSpec(shape=None, ...)} accepts at runtime), otherwise a tuple of the per-dimension renderings, the concrete size for
	 * a {@link NumericDim} and {@code "None"} for every other {@link Dimension} subtype (dynamic, ragged, symbolic).
	 *
	 * @param t The tensor type.
	 * @return The raw shape rendering (e.g. {@code "(None, 128)"}, {@code "(20,)"}, {@code "()"}, or {@code "None"}).
	 */
	private static String shapeString(TensorType t) {
		List<Dimension<?>> dims = t.getDims();
		if (dims == null)
			return "None";

		StringJoiner shapeDims = new StringJoiner(", ", "(", dims.size() == 1 ? ",)" : ")");
		for (Dimension<?> d : dims)
			shapeDims.add(d instanceof NumericDim ? d.value().toString() : "None");

		return shapeDims.toString();
	}

	/**
	 * This signature's per-parameter characteristics in declaration order (one {@link ParameterSpec} per non-{@code self} parameter), so
	 * downstream analysis reads each parameter's dtype and shape as columns rather than parsing {@link #toTensorSpecList(String)}. A
	 * {@link Sequence} entry stays one row—the row-to-parameter pairing is positional in the consumer—rendering its shape as the bracketed
	 * element shapes (e.g. {@code "[(2, 2)]"}) and its dtype as the distinct element dtypes joined with {@code "|"} (a single name for the
	 * common singleton case).
	 *
	 * @return The per-parameter (dtype, shape) rows.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/665">Issue 665</a>
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/786">Issue 786</a>
	 */
	public List<ParameterSpec> getParameterSpecs() {
		return entries.stream().map(e -> switch (e) {
		case Single s -> new ParameterSpec(dtypeName(s.type()), shapeString(s.type()));
		case Sequence q -> new ParameterSpec(
				q.elementTypes().stream().map(InputSignature::dtypeName).distinct().collect(Collectors.joining("|")),
				q.elementTypes().stream().map(InputSignature::shapeString).collect(Collectors.joining(", ", "[", "]")));
		}).toList();
	}

	/**
	 * A parameter's inferred tensor characteristics at per-{@code TensorSpec} granularity: the bare dtype constant name and the raw shape
	 * rendering this signature emits, without the spec-type wrapping of {@link #toTensorSpecList(String)}.
	 *
	 * @param dtype The bare dtype constant name (e.g. {@code "float32"}).
	 * @param shape The raw shape rendering (e.g. {@code "(None, 128)"}, {@code "None"}, or {@code "[(2, 2)]"} for a sequence parameter).
	 */
	public record ParameterSpec(String dtype, String shape) {
	}

	/**
	 * The TensorFlow spec-type constructor for a tensor type: {@code SparseTensorSpec} when it is sparse (a {@code tf.SparseTensor}, #533),
	 * {@code RaggedTensorSpec} when it has a ragged dimension (a {@code tf.RaggedTensor}, #524), otherwise {@code TensorSpec}. The bare
	 * name without any module prefix.
	 *
	 * @param t The {@link TensorType}.
	 * @return {@code "SparseTensorSpec"}, {@code "RaggedTensorSpec"}, or {@code "TensorSpec"}.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/524">Issue 524</a>
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/533">Issue 533</a>
	 */
	private static String specTypeName(TensorType t) {
		if (t.isSparse())
			return SPARSE_TENSOR_SPEC;

		List<Dimension<?>> dims = t.getDims();
		boolean ragged = dims != null && dims.stream().anyMatch(RaggedDim.class::isInstance);
		return ragged ? RAGGED_TENSOR_SPEC : TENSOR_SPEC;
	}

	/**
	 * The set of spec-type constructor names this signature references (e.g., {@code "TensorSpec"}, {@code "RaggedTensorSpec"},
	 * {@code "SparseTensorSpec"}), as the bare Python identifiers emitted by {@link #toTensorSpecList(String)} (without any module prefix),
	 * collected across {@link Single} entries and {@link Sequence} elements alike. The source-write verifies each is reachable under the
	 * chosen import prefix before emitting an unqualified signature: on the {@code from tensorflow import ...} named-import path a
	 * signature with a sparse or ragged parameter needs {@code SparseTensorSpec} or {@code RaggedTensorSpec} in scope, which
	 * {@code TensorSpec} being imported does not imply, so an unguarded emission would produce a {@code NameError}-raising decorator.
	 *
	 * @return The referenced spec-type constructor names.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/524">Issue 524</a>
	 */
	public Set<String> requiredSpecTypeNames() {
		return leaves().map(InputSignature::specTypeName).collect(Collectors.toUnmodifiableSet());
	}

	/**
	 * The set of dtype constant names this signature references (e.g., {@code "float32"}, {@code "int32"}), as the bare Python identifiers
	 * emitted by {@link #toTensorSpecList(String)} (without any module prefix), collected across {@link Single} entries and
	 * {@link Sequence} elements alike. The source-write uses this to verify each required dtype constant is reachable under the chosen
	 * import prefix before emitting an unqualified signature: on the {@code from tensorflow import ...} named-import path,
	 * {@code TensorSpec} being in scope does not imply the dtype constants are too, so an unguarded emission would produce a
	 * {@code NameError}-raising decorator.
	 *
	 * @return The referenced dtype constant names.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/585">Issue 585</a>
	 */
	public Set<String> requiredDTypeNames() {
		return leaves().map(InputSignature::dtypeName).collect(Collectors.toUnmodifiableSet());
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
		 * rank, parameter count, entry structure, or concrete dtype). The inferred signature should replace it.
		 */
		INCOMPARABLE
	}

	/**
	 * Relates this signature, taken as the developer-supplied one, to {@code inferred}, the signature derived from call-site evidence by
	 * {@link Function#inferInputSignature}. See {@link Relation} for the partial order. A {@link Single} against a {@link Sequence} (in
	 * either direction) is {@link Relation#INCOMPARABLE}: the two admit disjoint call structures, since TensorFlow enforces the declared
	 * nesting. Two {@link Sequence}s of different arity are likewise incomparable (no wildcard arity exists); at equal arity they relate by
	 * folding the element relations. Used by {@link Function#check()} to decide whether an existing {@code input_signature} should be
	 * overwritten ({@link Relation#SUPPLIED_TIGHTER}, {@link Relation#INCOMPARABLE}), preserved ({@link Relation#SUPPLIED_BROADER}), or
	 * left untouched ({@link Relation#AGREEMENT}).
	 *
	 * @param inferred The signature inferred from call-site evidence.
	 * @return How this (supplied) signature relates to {@code inferred}.
	 */
	public Relation relate(InputSignature inferred) {
		List<SpecEntry> supplied = this.entries();
		List<SpecEntry> evidence = inferred.entries();

		// A parameter-count mismatch has no meaningful per-parameter order; treat it as incomparable so the inferred signature wins.
		if (supplied.size() != evidence.size())
			return Relation.INCOMPARABLE;

		Relation result = Relation.AGREEMENT;
		for (int i = 0; i < supplied.size(); i++)
			result = combine(result, relate(supplied.get(i), evidence.get(i)));

		return result;
	}

	/**
	 * Relates one supplied {@link SpecEntry} to one inferred entry: matching structures relate by their contents; mismatched structures are
	 * incomparable.
	 *
	 * @param supplied The developer-supplied entry.
	 * @param inferred The inferred entry.
	 * @return How the supplied entry relates to the inferred one.
	 */
	private static Relation relate(SpecEntry supplied, SpecEntry inferred) {
		if (supplied instanceof Single s && inferred instanceof Single i)
			return relate(s.type(), i.type());

		if (supplied instanceof Sequence s && inferred instanceof Sequence i) {
			if (s.elementTypes().size() != i.elementTypes().size())
				return Relation.INCOMPARABLE; // no wildcard arity exists; TensorFlow enforces the length.

			Relation result = Relation.AGREEMENT;
			for (int j = 0; j < s.elementTypes().size(); j++)
				result = combine(result, relate(s.elementTypes().get(j), i.elementTypes().get(j)));
			return result;
		}

		return Relation.INCOMPARABLE; // a tensor where a sequence is declared (or vice versa) admits a disjoint call structure.
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
