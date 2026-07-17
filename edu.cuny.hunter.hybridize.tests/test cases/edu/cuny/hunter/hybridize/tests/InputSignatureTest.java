package edu.cuny.hunter.hybridize.tests;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.BOOL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.UNKNOWN;
import static org.junit.Assert.assertEquals;

import java.util.List;
import java.util.Locale;
import java.util.Set;

import org.junit.Test;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;

import edu.cuny.hunter.hybridize.core.analysis.InputSignature;
import edu.cuny.hunter.hybridize.core.analysis.InputSignature.ParameterSpec;
import edu.cuny.hunter.hybridize.core.analysis.InputSignature.Relation;
import edu.cuny.hunter.hybridize.core.analysis.InputSignature.Sequence;
import edu.cuny.hunter.hybridize.core.analysis.InputSignature.Single;

/**
 * Tests for {@link InputSignature#toTensorSpecList(String)}.
 *
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
 */
@SuppressWarnings("static-method")
public class InputSignatureTest {

	/**
	 * Single rank-2 dense tensor with a dynamic batch dim: the keras.Input shape.
	 */
	@Test
	public void testSingleRank2WithDynamicBatch() {
		InputSignature sig = InputSignature.ofSingles(List.of(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32)))));
		assertEquals("[tf.TensorSpec(shape=(None, 32), dtype=tf.float32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * Two parameters, fully concrete shapes, mixed dtypes.
	 */
	@Test
	public void testTwoConcreteParameters() {
		InputSignature sig = InputSignature.ofSingles(List.of(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))),
				new TensorType(INT32, List.of(new NumericDim(5)))));
		assertEquals("[tf.TensorSpec(shape=(2, 3), dtype=tf.float32), tf.TensorSpec(shape=(5,), dtype=tf.int32)]",
				sig.toTensorSpecList("tf."));
	}

	/**
	 * Single-dim tensor renders with a trailing-comma tuple, matching Python's one-element tuple syntax.
	 */
	@Test
	public void testRank1TupleHasTrailingComma() {
		InputSignature sig = InputSignature.ofSingles(List.of(new TensorType(INT32, List.of(new NumericDim(4)))));
		assertEquals("[tf.TensorSpec(shape=(4,), dtype=tf.int32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * Scalar (rank-0) tensor renders as an empty shape tuple.
	 */
	@Test
	public void testScalarRendersAsEmptyTuple() {
		InputSignature sig = InputSignature.ofSingles(List.of(new TensorType(INT32, List.of())));
		assertEquals("[tf.TensorSpec(shape=(), dtype=tf.int32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * {@link SymbolicDim}, {@link RaggedDim}, and {@link DynamicDim} all render as {@code None} on the shape surface. A {@link RaggedDim}
	 * additionally selects {@code RaggedTensorSpec} over {@code TensorSpec}, since a ragged dimension marks the parameter as a
	 * {@code tf.RaggedTensor} (#524).
	 */
	@Test
	public void testSymbolicAndRaggedDimsCollapseToNone() {
		InputSignature sig = InputSignature.ofSingles(
				List.of(new TensorType(INT32, List.of(new NumericDim(3), new SymbolicDim("?"), RaggedDim.INSTANCE, DynamicDim.INSTANCE))));
		assertEquals("[tf.RaggedTensorSpec(shape=(3, None, None, None), dtype=tf.int32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * A sparse {@link TensorType} selects {@code SparseTensorSpec} over {@code TensorSpec}, since sparseness marks the parameter as a
	 * {@code tf.SparseTensor} (#533). The shape renders the same as a dense tensor of the same dimensions.
	 */
	@Test
	public void testSparseRendersAsSparseTensorSpec() {
		InputSignature sig = InputSignature
				.ofSingles(List.of(new TensorType(INT32, List.of(new NumericDim(3), new NumericDim(4))).asSparse()));
		assertEquals("[tf.SparseTensorSpec(shape=(3, 4), dtype=tf.int32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * {@link InputSignature#requiredSpecTypeNames} reports {@code TensorSpec} for a dense parameter, {@code RaggedTensorSpec} for a
	 * parameter with a ragged dimension, {@code SparseTensorSpec} for a sparse parameter, and the union for a mixed signature. The
	 * source-write gates emission on these being reachable (#524, #533).
	 */
	@Test
	public void testRequiredSpecTypeNames() {
		TensorType dense = new TensorType(INT32, List.of(new NumericDim(3)));
		TensorType ragged = new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE));
		TensorType sparse = new TensorType(INT32, List.of(new NumericDim(3))).asSparse();

		assertEquals(Set.of("TensorSpec"), InputSignature.ofSingles(List.of(dense)).requiredSpecTypeNames());
		assertEquals(Set.of("RaggedTensorSpec"), InputSignature.ofSingles(List.of(ragged)).requiredSpecTypeNames());
		assertEquals(Set.of("SparseTensorSpec"), InputSignature.ofSingles(List.of(sparse)).requiredSpecTypeNames());
		assertEquals(Set.of("TensorSpec", "RaggedTensorSpec", "SparseTensorSpec"),
				InputSignature.ofSingles(List.of(dense, ragged, sparse)).requiredSpecTypeNames());
	}

	/**
	 * Non-numeric dtypes render with their lowercase name (e.g., {@code tf.string}, {@code tf.bool}).
	 */
	@Test
	public void testNonNumericDtypes() {
		InputSignature sig = InputSignature
				.ofSingles(List.of(new TensorType(STRING, List.of(new NumericDim(2))), new TensorType(BOOL, List.of(new NumericDim(2)))));
		assertEquals("[tf.TensorSpec(shape=(2,), dtype=tf.string), tf.TensorSpec(shape=(2,), dtype=tf.bool)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * A {@code "tensorflow."} prefix (from {@code import tensorflow}) flows through both the constructor and the dtype reference.
	 */
	@Test
	public void testFullModulePrefix() {
		InputSignature sig = InputSignature.ofSingles(List.of(new TensorType(FLOAT32, List.of(new NumericDim(2)))));
		assertEquals("[tensorflow.TensorSpec(shape=(2,), dtype=tensorflow.float32)]", sig.toTensorSpecList("tensorflow."));
	}

	/**
	 * An empty prefix (from {@code from tensorflow import *} or similar) emits bare {@code TensorSpec(...)} with bare dtype references. The
	 * caller is responsible for ensuring those names are in scope; this method only formats.
	 */
	@Test
	public void testEmptyPrefix() {
		InputSignature sig = InputSignature.ofSingles(List.of(new TensorType(FLOAT32, List.of(new NumericDim(2)))));
		assertEquals("[TensorSpec(shape=(2,), dtype=float32)]", sig.toTensorSpecList(""));
	}

	/**
	 * Shape-⊤ tensors (rank disagrees across contexts, or any context has unknown rank): {@code Function.inferSpec} returns
	 * {@code TensorType(dtype, null)}. The formatter must encode this as {@code shape=None}—TF's {@code tf.TensorSpec(shape=None, ...)}
	 * accepts any shape at runtime.
	 */
	@Test
	public void testShapeTopRendersAsNone() {
		InputSignature sig = InputSignature.ofSingles(List.of(new TensorType(FLOAT32, null)));
		assertEquals("[tf.TensorSpec(shape=None, dtype=tf.float32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * Locale-independence: dtype identifiers must lower-case under {@link Locale#ROOT}, not the JVM default. Under Turkish locale,
	 * {@code "INT32".toLowerCase()} produces {@code "ınt32"} (dotless-i, non-ASCII), which would corrupt the emitted Python.
	 */
	@Test
	public void testDtypeLowerCaseIsLocaleIndependent() {
		Locale prior = Locale.getDefault();
		Locale.setDefault(Locale.of("tr", "TR"));
		try {
			InputSignature sig = InputSignature.ofSingles(
					List.of(new TensorType(INT32, List.of(new NumericDim(2))), new TensorType(STRING, List.of(new NumericDim(2)))));
			assertEquals("[tf.TensorSpec(shape=(2,), dtype=tf.int32), tf.TensorSpec(shape=(2,), dtype=tf.string)]",
					sig.toTensorSpecList("tf."));
		} finally {
			Locale.setDefault(prior);
		}
	}

	/**
	 * {@link InputSignature#getParameterSpecs()} exposes each parameter's dtype and raw shape in declaration order, the per-parameter view
	 * the {@code input_signatures.csv} emitter consumes, matching the rendering {@link InputSignature#toTensorSpecList(String)} joins
	 * (#665).
	 */
	@Test
	public void testParameterSpecs() {
		InputSignature sig = sig(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))),
				new TensorType(INT32, List.of(new NumericDim(5))));
		assertEquals(List.of(new ParameterSpec("float32", "(2, 3)"), new ParameterSpec("int32", "(5,)")), sig.getParameterSpecs());
	}

	/**
	 * {@link InputSignature#getParameterSpecs()} renders a rank-0 tensor's shape as {@code "()"} and a shape-⊤ ({@code null} dims) tensor's
	 * shape as {@code "None"}, matching {@link InputSignature#toTensorSpecList(String)}.
	 */
	@Test
	public void testParameterSpecsScalarAndShapeTop() {
		assertEquals(List.of(new ParameterSpec("int32", "()")), sig(new TensorType(INT32, List.of())).getParameterSpecs());
		assertEquals(List.of(new ParameterSpec("float32", "None")), sig(new TensorType(FLOAT32, null)).getParameterSpecs());
	}

	/**
	 * {@link InputSignature#getParameterSpecs()} collapses every non-numeric dimension to {@code None} on the shape surface, as
	 * {@link InputSignature#toTensorSpecList(String)} does.
	 */
	@Test
	public void testParameterSpecsWildcardDims() {
		assertEquals(List.of(new ParameterSpec("float32", "(None, 32)")),
				sig(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32)))).getParameterSpecs());
	}

	private static InputSignature sig(TensorType... types) {
		return InputSignature.ofSingles(List.of(types));
	}

	/**
	 * A {@link Sequence} entry renders its element specs inside a further list, beside a {@link Single} (#781).
	 */
	@Test
	public void testSequenceRendering() {
		InputSignature sig = new InputSignature(List.of(new Single(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(1)))),
				new Sequence(List.of(new TensorType(INT32, List.of(new NumericDim(2), new NumericDim(2)))))));
		assertEquals("[tf.TensorSpec(shape=(2, 1), dtype=tf.float32), [tf.TensorSpec(shape=(2, 2), dtype=tf.int32)]]",
				sig.toTensorSpecList("tf."));
	}

	/**
	 * A multi-element {@link Sequence} renders each element in order (#781).
	 */
	@Test
	public void testSequenceRenderingMultiElement() {
		InputSignature sig = new InputSignature(List.of(new Sequence(
				List.of(new TensorType(INT32, List.of(new NumericDim(2))), new TensorType(FLOAT32, List.of(new NumericDim(3)))))));
		assertEquals("[[tf.TensorSpec(shape=(2,), dtype=tf.int32), tf.TensorSpec(shape=(3,), dtype=tf.float32)]]",
				sig.toTensorSpecList("tf."));
	}

	/**
	 * Required names are collected across {@link Sequence} elements: a nested ragged element needs {@code RaggedTensorSpec} in scope
	 * exactly as a top-level one does (#524, #781), and every leaf's dtype constant is required.
	 */
	@Test
	public void testSequenceRequiredNames() {
		InputSignature sig = new InputSignature(List.of(new Single(new TensorType(FLOAT32, List.of(new NumericDim(2)))),
				new Sequence(List.of(new TensorType(INT32, List.of(RaggedDim.INSTANCE, new NumericDim(2)))))));
		assertEquals(Set.of("TensorSpec", "RaggedTensorSpec"), sig.requiredSpecTypeNames());
		assertEquals(Set.of("float32", "int32"), sig.requiredDTypeNames());
	}

	/**
	 * {@link InputSignature#getParameterSpecs()} keeps one row per parameter for a {@link Sequence}, bracketing the element shapes and
	 * joining distinct element dtypes (#781, #786).
	 */
	@Test
	public void testParameterSpecsSequence() {
		InputSignature sig = new InputSignature(
				List.of(new Sequence(List.of(new TensorType(INT32, List.of(new NumericDim(2), new NumericDim(2)))))));
		assertEquals(List.of(new ParameterSpec("int32", "[(2, 2)]")), sig.getParameterSpecs());
	}

	/**
	 * A flat entry against a nested one is incomparable in either direction: TensorFlow enforces the declared structure, so the two admit
	 * disjoint call shapes (#781).
	 */
	@Test
	public void testRelateSingleVsSequenceIncomparable() {
		TensorType t = new TensorType(FLOAT32, List.of(new NumericDim(2)));
		InputSignature flat = sig(t);
		InputSignature nested = new InputSignature(List.of(new Sequence(List.of(t))));
		assertEquals(Relation.INCOMPARABLE, flat.relate(nested));
		assertEquals(Relation.INCOMPARABLE, nested.relate(flat));
	}

	/**
	 * Two {@link Sequence}s of different arity are incomparable (no wildcard arity exists); at equal arity they relate by their elements
	 * (#781).
	 */
	@Test
	public void testRelateSequences() {
		TensorType concrete = new TensorType(FLOAT32, List.of(new NumericDim(2)));
		TensorType wild = new TensorType(FLOAT32, null);

		InputSignature one = new InputSignature(List.of(new Sequence(List.of(concrete))));
		InputSignature two = new InputSignature(List.of(new Sequence(List.of(concrete, concrete))));
		assertEquals(Relation.INCOMPARABLE, one.relate(two));

		InputSignature broad = new InputSignature(List.of(new Sequence(List.of(wild))));
		assertEquals(Relation.SUPPLIED_BROADER, broad.relate(one));
		assertEquals(Relation.SUPPLIED_TIGHTER, one.relate(broad));
		assertEquals(Relation.AGREEMENT, one.relate(new InputSignature(List.of(new Sequence(List.of(concrete))))));
	}

	/**
	 * Identical signatures agree.
	 */
	@Test
	public void testRelateAgreement() {
		InputSignature a = sig(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))));
		InputSignature b = sig(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))));
		assertEquals(Relation.AGREEMENT, a.relate(b));
	}

	/**
	 * A concrete supplied dimension where the inferred one is a wildcard makes the supplied signature strictly tighter.
	 */
	@Test
	public void testRelateSuppliedTighterByDim() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(new NumericDim(32), new NumericDim(784))));
		InputSignature inferred = sig(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(784))));
		assertEquals(Relation.SUPPLIED_TIGHTER, supplied.relate(inferred));
	}

	/**
	 * A known-rank supplied shape against an unknown-rank ({@code null} dims) inferred shape is strictly tighter.
	 */
	@Test
	public void testRelateSuppliedTighterByShapeTop() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(new NumericDim(2))));
		InputSignature inferred = sig(new TensorType(FLOAT32, null));
		assertEquals(Relation.SUPPLIED_TIGHTER, supplied.relate(inferred));
	}

	/**
	 * A concrete supplied dtype against an {@code UNKNOWN} inferred dtype is strictly tighter (defensive: inference drops {@code UNKNOWN}
	 * upstream, but the lattice still orders it as the top).
	 */
	@Test
	public void testRelateSuppliedTighterByDtype() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(new NumericDim(2))));
		InputSignature inferred = sig(new TensorType(UNKNOWN, List.of(new NumericDim(2))));
		assertEquals(Relation.SUPPLIED_TIGHTER, supplied.relate(inferred));
	}

	/**
	 * A wildcard supplied dimension where the inferred one is concrete makes the supplied signature strictly broader.
	 */
	@Test
	public void testRelateSuppliedBroaderByDim() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(784))));
		InputSignature inferred = sig(new TensorType(FLOAT32, List.of(new NumericDim(32), new NumericDim(784))));
		assertEquals(Relation.SUPPLIED_BROADER, supplied.relate(inferred));
	}

	/**
	 * An unknown-rank ({@code null} dims) supplied shape against a known-rank inferred shape is strictly broader.
	 */
	@Test
	public void testRelateSuppliedBroaderByShapeTop() {
		InputSignature supplied = sig(new TensorType(FLOAT32, null));
		InputSignature inferred = sig(new TensorType(FLOAT32, List.of(new NumericDim(2))));
		assertEquals(Relation.SUPPLIED_BROADER, supplied.relate(inferred));
	}

	/**
	 * Two distinct concrete dimensions at the same position are incomparable.
	 */
	@Test
	public void testRelateIncomparableByDim() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(new NumericDim(32), new NumericDim(784))));
		InputSignature inferred = sig(new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(784))));
		assertEquals(Relation.INCOMPARABLE, supplied.relate(inferred));
	}

	/**
	 * Two distinct concrete dtypes are incomparable.
	 */
	@Test
	public void testRelateIncomparableByDtype() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(new NumericDim(2))));
		InputSignature inferred = sig(new TensorType(INT32, List.of(new NumericDim(2))));
		assertEquals(Relation.INCOMPARABLE, supplied.relate(inferred));
	}

	/**
	 * Two shapes of different rank are incomparable.
	 */
	@Test
	public void testRelateIncomparableByRank() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(new NumericDim(2))));
		InputSignature inferred = sig(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2))));
		assertEquals(Relation.INCOMPARABLE, supplied.relate(inferred));
	}

	/**
	 * A signature that is tighter on one parameter and broader on another is incomparable overall (neither is uniformly at least as
	 * specific).
	 */
	@Test
	public void testRelateIncomparableMixedParameters() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(new NumericDim(32))),
				new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE)));
		InputSignature inferred = sig(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE)),
				new TensorType(FLOAT32, List.of(new NumericDim(32))));
		assertEquals(Relation.INCOMPARABLE, supplied.relate(inferred));
	}

	/**
	 * Different parameter counts are incomparable.
	 */
	@Test
	public void testRelateIncomparableParameterCount() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(new NumericDim(2))));
		InputSignature inferred = sig(new TensorType(FLOAT32, List.of(new NumericDim(2))),
				new TensorType(FLOAT32, List.of(new NumericDim(2))));
		assertEquals(Relation.INCOMPARABLE, supplied.relate(inferred));
	}

	/**
	 * All wildcard kinds ({@link DynamicDim}, {@link SymbolicDim}) are the lattice top and relate as equal, so a supplied {@code None}
	 * agrees with an inferred symbolic wildcard.
	 */
	@Test
	public void testRelateWildcardKindsAgree() {
		InputSignature supplied = sig(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(784))));
		InputSignature inferred = sig(new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(784))));
		assertEquals(Relation.AGREEMENT, supplied.relate(inferred));
	}
}
