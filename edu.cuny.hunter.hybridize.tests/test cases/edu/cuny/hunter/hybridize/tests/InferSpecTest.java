package edu.cuny.hunter.hybridize.tests;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.UNKNOWN;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Optional;
import java.util.Set;

import org.junit.Test;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;

import edu.cuny.hunter.hybridize.core.analysis.Function;

/**
 * Synthesized-input tests for {@link Function#inferSpec(Set)}, the per-parameter multi-context reduction. Inputs are hand-built
 * {@link TensorType} sets so the reduction is exercised in isolation from upstream tensor-type precision.
 *
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/494">Issue 494</a>
 */
@SuppressWarnings("static-method")
public class InferSpecTest {

	/**
	 * A single context whose dtype is {@code UNKNOWN} (dtype-⊤) drops the spec. {@code tf.UNKNOWN} is not a valid runtime dtype for
	 * {@code tf.function(input_signature=...)}, so the conservative reduction is to bottom. This branch is fixture-unreachable (upstream no
	 * longer emits a {@code DType.UNKNOWN} singleton), so the synthesized input is the only way to exercise it.
	 */
	@Test
	public void testDtypeTopSingletonDrops() {
		Optional<TensorType> spec = Function.inferSpec(Set.of(new TensorType(UNKNOWN, List.of(new NumericDim(3)))));
		assertTrue("A dtype-⊤ (UNKNOWN) singleton should reduce to bottom.", spec.isEmpty());
	}

	/**
	 * Witness that the drop above is caused by the {@code UNKNOWN} dtype specifically, not by the singleton shape: an otherwise-identical
	 * context with a concrete dtype reduces to that same concrete type.
	 */
	@Test
	public void testConcreteDtypeSingletonReduces() {
		TensorType concrete = new TensorType(FLOAT32, List.of(new NumericDim(3)));
		Optional<TensorType> spec = Function.inferSpec(Set.of(concrete));
		assertFalse("A concrete-dtype singleton should reduce to a spec.", spec.isEmpty());
		assertEquals(concrete, spec.get());
	}
}
