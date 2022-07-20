package edu.cuny.hunter.hybridize.tests;

import edu.cuny.citytech.refactoring.common.tests.RefactoringTest;

public class HybridizeFunctionRefactoringTest extends RefactoringTest {

	private static final String REFACTORING_PATH = "HybridizeFunction/";

	@Override
	protected String getRefactoringPath() {
		return REFACTORING_PATH;
	}

	/**
	 * Test #5. This simply tests whether the annotation is present for now. It's
	 * probably not a "candidate," however, since it doesn't have a Tensor argument.
	 * NOTE: This may wind up failing at some point since it doesn't have a Tensor
	 * argument.
	 */
	public void testIsHybrid() {

	}
}
