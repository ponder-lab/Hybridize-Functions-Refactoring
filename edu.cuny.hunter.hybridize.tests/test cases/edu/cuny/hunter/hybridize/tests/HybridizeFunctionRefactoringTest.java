package edu.cuny.hunter.hybridize.tests;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import org.junit.Test;
import org.python.pydev.navigator.elements.PythonNode;

import edu.cuny.citytech.refactoring.common.tests.RefactoringTest;

public class HybridizeFunctionRefactoringTest extends RefactoringTest {

	private static final String REFACTORING_PATH = "HybridizeFunction/";

	private static final String TEST_FILE_EXTENION = "py";

	@Override
	protected String getRefactoringPath() {
		return REFACTORING_PATH;
	}

	/**
	 * Runs a single analysis test.
	 */
	private void helper() throws IOException {
		PythonNode pythonNode = createPythonNodeFromTestFile("A");
	}

	private PythonNode createPythonNodeFromTestFile(String fileName) throws IOException {
		return createPythonNodeFromTestFile(fileName, true);
	}

	private PythonNode createPythonNodeFromTestFile(String fileName, boolean input) throws IOException {
		String contents = input ? getFileContents(getInputTestFileName(fileName))
				: getFileContents(getOutputTestFileName(fileName));

		return createPythonNode(fileName + ".py", contents);
	}

	private PythonNode createPythonNode(String name, String contents) {
		// TODO: Look at bookmarks in PyDev.
		return null;
	}

	/**
	 * Test #5. This simply tests whether the annotation is present for now. It's
	 * probably not a "candidate," however, since it doesn't have a Tensor argument.
	 * NOTE: This may wind up failing at some point since it doesn't have a Tensor
	 * argument.
	 */
	@Test
	public void testIsHybrid() throws IOException {
		System.out.println("Hi");
		assertTrue(true);
		this.helper();
	}

	@Override
	protected String getName() {
		// TODO Auto-generated method stub
		return super.getName();
	}

	@Override
	protected String getTestPath() {
		// TODO Auto-generated method stub
		return super.getTestPath();
	}

	@Override
	protected String getTestFileExtension() {
		return TEST_FILE_EXTENION;
	}

	@Override
	protected String getInputTestFileName(String cuName) {
		// TODO Auto-generated method stub
		return super.getInputTestFileName(cuName);
	}

	@Override
	protected String getInputTestFileName(String cuName, String subDirName) {
		// TODO Auto-generated method stub
		return super.getInputTestFileName(cuName, subDirName);
	}

	@Override
	protected String getOutputTestFileName(String cuName) {
		// TODO Auto-generated method stub
		return super.getOutputTestFileName(cuName);
	}

	@Override
	protected String getOutputTestFileName(String cuName, String subDirName) {
		// TODO Auto-generated method stub
		return super.getOutputTestFileName(cuName, subDirName);
	}
}
