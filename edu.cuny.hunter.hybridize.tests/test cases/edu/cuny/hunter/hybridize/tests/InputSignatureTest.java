package edu.cuny.hunter.hybridize.tests;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.BOOL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static org.junit.Assert.assertEquals;

import java.util.List;
import java.util.Locale;

import org.junit.Test;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;

import edu.cuny.hunter.hybridize.core.analysis.InputSignature;

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
	 * {@link SymbolicDim} and {@link RaggedDim} render as {@code None}, the same as {@link DynamicDim}. The {@code TensorSpec} surface
	 * cannot encode raggedness; the {@code RaggedTensorSpec} flip is tracked separately at #524.
	 */
	@Test
	public void testSymbolicAndRaggedDimsCollapseToNone() {
		InputSignature sig = new InputSignature(
				List.of(new TensorType(INT32, List.of(new NumericDim(3), new SymbolicDim("?"), RaggedDim.INSTANCE, DynamicDim.INSTANCE))));
		assertEquals("[tf.TensorSpec(shape=(3, None, None, None), dtype=tf.int32)]", sig.toTensorSpecList("tf."));
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
		Locale.setDefault(new Locale("tr", "TR"));
		try {
			InputSignature sig = new InputSignature(
					List.of(new TensorType(INT32, List.of(new NumericDim(2))), new TensorType(STRING, List.of(new NumericDim(2)))));
			assertEquals("[tf.TensorSpec(shape=(2,), dtype=tf.int32), tf.TensorSpec(shape=(2,), dtype=tf.string)]",
					sig.toTensorSpecList("tf."));
		} finally {
			Locale.setDefault(prior);
		}
	}
}
