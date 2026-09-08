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
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;

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

	/**
	 * A parameter that is sparse at one call site and dense at another (same dtype, same shape) abandons to bottom. The dtype consensus
	 * holds but the sparseness consensus fails, and no single spec is sound: a dense {@code TensorSpec} rejects the {@code SparseTensor}
	 * the function accepts at the sparse call site, while a {@code SparseTensorSpec} rejects the dense tensor it accepts at the dense call
	 * site. So the reduction drops (#642). This is the inversion of the former dense fall-through, which #650 pinned with a TODO.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/642">Issue 642</a>
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/533">Issue 533</a>
	 */
	@Test
	public void testMixedSparseDenseDrops() {
		TensorType dense = new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(3)));
		TensorType sparse = new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(3))).asSparse();
		Optional<TensorType> spec = Function.inferSpec(Set.of(dense, sparse));

		assertTrue("A mixed sparse/dense parameter has no sound single spec, so it abandons to bottom (#642).", spec.isEmpty());
	}

	/**
	 * A concrete extent at a position loses to a symbolic twin at that same position. The reduction takes per-position CONSENSUS, not the
	 * more precise member: a {@link NumericDim} beside a {@link SymbolicDim} is a disagreement, and a disagreement wildcards the axis.
	 * <p>
	 * This is the contract an upstream precision fix has to satisfy to reach the specification surface. A fix that ADDS the concrete member
	 * beside an existing placeholder changes nothing here, because the placeholder still disagrees with it; only a fix that ELIMINATES the
	 * placeholder moves the emitted axis. Pinning it so the distinction is not rediscovered from a whole-project run.
	 *
	 * @see <a href="https://github.com/wala/ML/issues/875">wala/ML issue 875</a>
	 */
	@Test
	public void testConcreteDimLosesToSymbolicTwin() {
		TensorType concrete = new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new NumericDim(46)));
		TensorType twin = new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new SymbolicDim("?")));
		Optional<TensorType> spec = Function.inferSpec(Set.of(concrete, twin));

		assertFalse("Dtype and rank agree, so the reduction yields a spec rather than bottom.", spec.isEmpty());
		assertEquals("The agreeing leading axes survive and the disputed trailing axis wildcards.",
				new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new SymbolicDim("?"))), spec.get());
	}

	/**
	 * The witness for the test above: with the symbolic twin removed, the same concrete extents survive the reduction. Without this, a
	 * wildcarded trailing axis is equally consistent with the reduction never keeping any extent, and the pin above would assert nothing
	 * about the twin specifically.
	 *
	 * @see <a href="https://github.com/wala/ML/issues/875">wala/ML issue 875</a>
	 */
	@Test
	public void testAgreeingConcreteDimsSurvive() {
		TensorType concrete = new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new NumericDim(46)));
		Optional<TensorType> spec = Function.inferSpec(Set.of(concrete));

		assertFalse("A concrete singleton reduces to a spec.", spec.isEmpty());
		assertEquals("Every axis is concrete and agreed, so every axis survives.", concrete, spec.get());
	}

	/**
	 * Contexts that disagree on a concrete extent wildcard only the disputed position. Distinguishes the twin case above from ordinary
	 * multi-context disagreement: both wildcard, but this one has no placeholder involved, so a fold that removes a placeholder cannot
	 * recover it. The trailing axis agrees across both contexts and survives, which is what makes an upstream fold worth making.
	 *
	 * @see <a href="https://github.com/wala/ML/issues/875">wala/ML issue 875</a>
	 */
	@Test
	public void testDisagreeingConcreteDimsWildcardOnlyThatAxis() {
		TensorType first = new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new NumericDim(46)));
		TensorType second = new TensorType(FLOAT32, List.of(new NumericDim(8), new NumericDim(10), new NumericDim(46)));
		Optional<TensorType> spec = Function.inferSpec(Set.of(first, second));

		assertFalse("Dtype and rank agree, so the reduction yields a spec rather than bottom.", spec.isEmpty());
		assertEquals("The disputed leading axes wildcard and the agreed trailing extent survives.",
				new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new SymbolicDim("?"), new NumericDim(46))), spec.get());
	}
}
