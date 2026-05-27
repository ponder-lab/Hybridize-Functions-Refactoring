package edu.cuny.hunter.hybridize.tests;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;

/**
 * Tests for the {@code inferInputSignatures} flag API on {@link HybridizeFunctionRefactoringProcessor}: the 8-arg no-{@code Set}
 * constructor, the {@link HybridizeFunctionRefactoringProcessor#getInferInputSignatures()} accessor, and the propagating
 * {@link HybridizeFunctionRefactoringProcessor#setInferInputSignatures(boolean)} mutator (the API surface that issue 481's UI checkbox /
 * eval wiring will call).
 *
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/481">Issue 481</a>
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
 */
@SuppressWarnings("static-method")
public class HybridizeFunctionRefactoringProcessorTest {

	/**
	 * The 8-arg no-{@code Set} constructor records the {@code inferInputSignatures} value, and the accessor returns it.
	 */
	@Test
	public void testConstructorSetsInferInputSignatures() {
		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(false, false, false, true, false, false,
				false, true);
		assertTrue(processor.getInferInputSignatures());
	}

	/**
	 * Default for the 7-arg overload is {@code inferInputSignatures = false}.
	 */
	@Test
	public void testSevenArgOverloadDefaultsToFalse() {
		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(false, false, false, true, false, false,
				false);
		assertFalse(processor.getInferInputSignatures());
	}

	/**
	 * Setter flips the value and the accessor reflects it. The propagation loop runs over an empty function set without throwing—exercises
	 * the loop's no-iteration branch.
	 */
	@Test
	public void testSetterFlipsValue() {
		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(false, false, false, true, false, false,
				false);
		assertFalse(processor.getInferInputSignatures());
		processor.setInferInputSignatures(true);
		assertTrue(processor.getInferInputSignatures());
		processor.setInferInputSignatures(false);
		assertFalse(processor.getInferInputSignatures());
	}
}
