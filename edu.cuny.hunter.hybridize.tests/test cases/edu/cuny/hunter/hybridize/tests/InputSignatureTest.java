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
import edu.cuny.hunter.hybridize.core.analysis.InputSignature.Relation;

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
		InputSignature sig = new InputSignature(List.of(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32)))));
		assertEquals("[tf.TensorSpec(shape=(None, 32), dtype=tf.float32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * Two parameters, fully concrete shapes, mixed dtypes.
	 */
	@Test
	public void testTwoConcreteParameters() {
		InputSignature sig = new InputSignature(List.of(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))),
				new TensorType(INT32, List.of(new NumericDim(5)))));
		assertEquals("[tf.TensorSpec(shape=(2, 3), dtype=tf.float32), tf.TensorSpec(shape=(5,), dtype=tf.int32)]",
				sig.toTensorSpecList("tf."));
	}

	/**
	 * Single-dim tensor renders with a trailing-comma tuple, matching Python's one-element tuple syntax.
	 */
	@Test
	public void testRank1TupleHasTrailingComma() {
		InputSignature sig = new InputSignature(List.of(new TensorType(INT32, List.of(new NumericDim(4)))));
		assertEquals("[tf.TensorSpec(shape=(4,), dtype=tf.int32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * Scalar (rank-0) tensor renders as an empty shape tuple.
	 */
	@Test
	public void testScalarRendersAsEmptyTuple() {
		InputSignature sig = new InputSignature(List.of(new TensorType(INT32, List.of())));
		assertEquals("[tf.TensorSpec(shape=(), dtype=tf.int32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * {@link SymbolicDim}, {@link RaggedDim}, and {@link DynamicDim} all render as {@code None} on the shape surface. A {@link RaggedDim}
	 * additionally selects {@code RaggedTensorSpec} over {@code TensorSpec}, since a ragged dimension marks the parameter as a
	 * {@code tf.RaggedTensor} (#524).
	 */
	@Test
	public void testSymbolicAndRaggedDimsCollapseToNone() {
		InputSignature sig = new InputSignature(
				List.of(new TensorType(INT32, List.of(new NumericDim(3), new SymbolicDim("?"), RaggedDim.INSTANCE, DynamicDim.INSTANCE))));
		assertEquals("[tf.RaggedTensorSpec(shape=(3, None, None, None), dtype=tf.int32)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * A sparse {@link TensorType} selects {@code SparseTensorSpec} over {@code TensorSpec}, since sparseness marks the parameter as a
	 * {@code tf.SparseTensor} (#533). The shape renders the same as a dense tensor of the same dimensions.
	 */
	@Test
	public void testSparseRendersAsSparseTensorSpec() {
		InputSignature sig = new InputSignature(List.of(new TensorType(INT32, List.of(new NumericDim(3), new NumericDim(4))).asSparse()));
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

		assertEquals(Set.of("TensorSpec"), new InputSignature(List.of(dense)).requiredSpecTypeNames());
		assertEquals(Set.of("RaggedTensorSpec"), new InputSignature(List.of(ragged)).requiredSpecTypeNames());
		assertEquals(Set.of("SparseTensorSpec"), new InputSignature(List.of(sparse)).requiredSpecTypeNames());
		assertEquals(Set.of("TensorSpec", "RaggedTensorSpec", "SparseTensorSpec"),
				new InputSignature(List.of(dense, ragged, sparse)).requiredSpecTypeNames());
	}

	/**
	 * Non-numeric dtypes render with their lowercase name (e.g., {@code tf.string}, {@code tf.bool}).
	 */
	@Test
	public void testNonNumericDtypes() {
		InputSignature sig = new InputSignature(
				List.of(new TensorType(STRING, List.of(new NumericDim(2))), new TensorType(BOOL, List.of(new NumericDim(2)))));
		assertEquals("[tf.TensorSpec(shape=(2,), dtype=tf.string), tf.TensorSpec(shape=(2,), dtype=tf.bool)]", sig.toTensorSpecList("tf."));
	}

	/**
	 * A {@code "tensorflow."} prefix (from {@code import tensorflow}) flows through both the constructor and the dtype reference.
	 */
	@Test
	public void testFullModulePrefix() {
		InputSignature sig = new InputSignature(List.of(new TensorType(FLOAT32, List.of(new NumericDim(2)))));
		assertEquals("[tensorflow.TensorSpec(shape=(2,), dtype=tensorflow.float32)]", sig.toTensorSpecList("tensorflow."));
	}

	/**
	 * An empty prefix (from {@code from tensorflow import *} or similar) emits bare {@code TensorSpec(...)} with bare dtype references. The
	 * caller is responsible for ensuring those names are in scope; this method only formats.
	 */
	@Test
	public void testEmptyPrefix() {
		InputSignature sig = new InputSignature(List.of(new TensorType(FLOAT32, List.of(new NumericDim(2)))));
		assertEquals("[TensorSpec(shape=(2,), dtype=float32)]", sig.toTensorSpecList(""));
	}

	/**
	 * Shape-⊤ tensors (rank disagrees across contexts, or any context has unknown rank): {@code Function.inferSpec} returns
	 * {@code TensorType(dtype, null)}. The formatter must encode this as {@code shape=None}—TF's {@code tf.TensorSpec(shape=None, ...)}
	 * accepts any shape at runtime.
	 */
	@Test
	public void testShapeTopRendersAsNone() {
		InputSignature sig = new InputSignature(List.of(new TensorType(FLOAT32, null)));
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
			InputSignature sig = new InputSignature(
					List.of(new TensorType(INT32, List.of(new NumericDim(2))), new TensorType(STRING, List.of(new NumericDim(2)))));
			assertEquals("[tf.TensorSpec(shape=(2,), dtype=tf.int32), tf.TensorSpec(shape=(2,), dtype=tf.string)]",
					sig.toTensorSpecList("tf."));
		} finally {
			Locale.setDefault(prior);
		}
	}

	private static InputSignature sig(TensorType... types) {
		return new InputSignature(List.of(types));
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
