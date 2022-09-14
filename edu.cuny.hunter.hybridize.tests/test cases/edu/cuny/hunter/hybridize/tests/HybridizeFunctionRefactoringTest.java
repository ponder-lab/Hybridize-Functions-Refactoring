package edu.cuny.hunter.hybridize.tests;

import static org.eclipse.core.runtime.Platform.getLog;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.eclipse.core.runtime.ILog;
import org.eclipse.jface.text.Document;
import org.eclipse.jface.text.IDocument;
import org.eclipse.ltk.core.refactoring.RefactoringCore;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.junit.Test;
import org.python.pydev.core.IGrammarVersionProvider;
import org.python.pydev.core.MisconfigurationException;
import org.python.pydev.parser.PyParser;
import org.python.pydev.parser.PyParser.ParserInfo;
import org.python.pydev.parser.jython.ParseException;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.Token;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.shared_core.parsing.BaseParser.ParseOutput;

import edu.cuny.citytech.refactoring.common.tests.RefactoringTest;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.FunctionExtractor;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;
import edu.cuny.hunter.hybridize.core.utils.RefactoringAvailabilityTester;

@SuppressWarnings("restriction")
public class HybridizeFunctionRefactoringTest extends RefactoringTest {

	private static final ILog LOG = getLog(HybridizeFunctionRefactoringTest.class);

	private static final String REFACTORING_PATH = "HybridizeFunction/";

	private static final String TEST_FILE_EXTENION = "py";

	private static SimpleNode createPythonNode(String moduleName, String fileName, String contents) throws MisconfigurationException {
		LOG.info("Creating PythonNode for " + fileName + " in " + fileName);
		LOG.info("Contents: " + contents);

		IDocument document = new Document(contents);

		IGrammarVersionProvider provider = new IGrammarVersionProvider() {

			@Override
			public AdditionalGrammarVersionsToCheck getAdditionalGrammarVersions() throws MisconfigurationException {
				return null;
			}

			@Override
			public int getGrammarVersion() throws MisconfigurationException {
				return IGrammarVersionProvider.LATEST_GRAMMAR_PY3_VERSION;
			}
		};

		ParserInfo parserInfo = new ParserInfo(document, provider, moduleName, new File(fileName));

		// Parsing.
		ParseOutput parseOutput = PyParser.reparseDocument(parserInfo);

		// Check for parsing errors.
		Object err = parseOutput.error;
		if (err != null) {
			String s = "";
			if (err instanceof ParseException) {
				ParseException parseErr = (ParseException) err;
				parseErr.printStackTrace();

				Token token = parseErr.currentToken;
				if (token != null)
					fail("Expected no error, received: " + parseErr.getMessage() + "\n" + s + "\nline:" + token.beginLine + "\ncol:"
							+ token.beginColumn);
			}

			fail("Expected no error, received:\n" + err + "\n" + s);
		}

		// Check for AST failure.
		assertNotNull("Failed to generate AST.", parseOutput.ast);

		return (SimpleNode) parseOutput.ast;
	}

	/**
	 * Installs the required packages for running an input test file. Assumes that requirements.txt is located in the given path.
	 *
	 * @param path The {@link Path} containing the requirements.txt file.
	 */
	private static void installRequirements(Path path) throws IOException, InterruptedException {
		Path requirements = path.resolve("requirements.txt");

		// install requirements.
		runCommand("pip3", "install", "-r", requirements.toString());
	}

	private static void runCommand(String... command) throws IOException, InterruptedException {
		ProcessBuilder processBuilder = new ProcessBuilder(command);

		LOG.info("Executing: " + processBuilder.command().stream().collect(Collectors.joining(" ")));

		Process process = processBuilder.start();
		int exitCode = process.waitFor();
		String errorOutput = null;

		if (exitCode != 0) { // there's a problem.
			// retrieve the error output.
			BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
			errorOutput = errorReader.lines().collect(Collectors.joining("\n"));
		}

		assertEquals("Error code should be 0. Error was:\n" + errorOutput + ".", 0, exitCode);
	}

	/**
	 * Runs python on the file presented by the given {@link Path}.
	 *
	 * @param path The {@link Path} of the file to interpret.
	 */
	private static void runPython(Path path) throws IOException, InterruptedException {
		// run the code.
		runCommand("python3", path.toString());
	}

	private SimpleNode createPythonNodeFromTestFile(String fileName) throws IOException, MisconfigurationException {
		return this.createPythonNodeFromTestFile(fileName, true);
	}

	private SimpleNode createPythonNodeFromTestFile(String fileName, boolean input) throws IOException, MisconfigurationException {
		String contents = input ? this.getFileContents(this.getInputTestFileName(fileName))
				: this.getFileContents(this.getOutputTestFileName(fileName));

		return createPythonNode(fileName, fileName + '.' + TEST_FILE_EXTENION, contents);
	}

	@Override
	public void genericafter() throws Exception {
	}

	@Override
	public void genericbefore() throws Exception {
		if (this.fIsVerbose) {
			System.out.println("\n---------------------------------------------");
			System.out.println("\nTest:" + this.getClass() + "." + this.getName());
		}

		RefactoringCore.getUndoManager().flush();

		String inputTestFileName = this.getInputTestFileName("A");
		Path inputTestFileAbsolutionPath = getAbsolutionPath(inputTestFileName);

		installRequirements(inputTestFileAbsolutionPath.getParent());
		runPython(inputTestFileAbsolutionPath);
	}

	/**
	 * Runs a single analysis test.
	 *
	 * @return The set of {@link Function}s analyzed.
	 */
	private Set<Function> getFunctions() throws Exception {
		SimpleNode pythonNode = this.createPythonNodeFromTestFile("A");

		// extract function definitions.
		FunctionExtractor functionExtractor = new FunctionExtractor();
		pythonNode.accept(functionExtractor);
		Set<FunctionDef> availableFunctions = functionExtractor.getDefinitions().stream()
				.filter(RefactoringAvailabilityTester::isHybridizationAvailable).collect(Collectors.toSet());

		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(
				availableFunctions.toArray(FunctionDef[]::new));
		ProcessorBasedRefactoring refactoring = new ProcessorBasedRefactoring(processor);

		RefactoringStatus status = this.performRefactoringWithStatus(refactoring);
		assertTrue(status.isOK());

		return processor.getFunctions();
	}

	@Override
	protected String getRefactoringPath() {
		return REFACTORING_PATH;
	}

	@Override
	protected String getTestFileExtension() {
		return TEST_FILE_EXTENION;
	}

	/**
	 * Test for #30. This simply tests whether we can parse the tf.function argument input_signature.
	 */
	@Test
	public void testComputeParameters() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();
		assertNotNull(args);

		assertTrue(args.getInputSignatureParamExists() & !args.getAutoGraphParamExists() && !args.getJitCompileParamExists()
				&& !args.getReduceRetracingParamExists() && !args.getExpImplementsParamExists() && !args.getExpAutographOptParamExists()
				&& !args.getExpTypeHintsParamExists());
	}

	/**
	 * Test for #30. Test custom decorator with the same parameter names as tf.function.
	 */
	@Test
	public void testComputeParameters10() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();

		// This test is with a custom decorator `@custom.decorator` that contains a parameter `input_signature`
		// like `tf.function`. With this test, we want to verify that we only parse through the arguments
		// if the function is hybrid. Since this test is not with `tf.function` we are expecting the method
		// to return False.

		assertNull(args);
	}

	/**
	 * Test for #30. Test custom decorator with the same parameter names as tf.function and a tf.function (total of two decorators) and only
	 * count the parameters from the tf.function decorator.
	 */
	@Test
	public void testComputeParameters11() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();
		// This test is with a custom decorator `@custom.decorator` that contains a parameter `input_signature`
		// like `tf.function`. But it also has a tf.function decorator, therefore args should not be Null.
		assertNotNull(args);

		assertTrue(!args.getInputSignatureParamExists() & args.getAutoGraphParamExists() && !args.getJitCompileParamExists()
				&& !args.getReduceRetracingParamExists() && !args.getExpImplementsParamExists() && !args.getExpAutographOptParamExists()
				&& !args.getExpTypeHintsParamExists());

	}

	/**
	 * Test for #30. Tests two different tf.functions. Should only count the parameters of the last one.
	 */
	@Test
	public void testComputeParameters12() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();
		assertNotNull(args);

		// We need put `!` on args.getAutoGraphParamExists() once we fix #55
		assertTrue(!args.getInputSignatureParamExists() & args.getAutoGraphParamExists() && args.getJitCompileParamExists()
				&& !args.getReduceRetracingParamExists() && !args.getExpImplementsParamExists() && !args.getExpAutographOptParamExists()
				&& !args.getExpTypeHintsParamExists());

	}

	/**
	 * Test for #30. This simply tests whether we can parse the tf.function argument experimental_autograph_options
	 */
	@Test
	public void testComputeParameters2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.new HybridizationParameters();
		assertNotNull(args);

		assertTrue(args.getExpAutographOptParamExists() && !args.getAutoGraphParamExists() && !args.getJitCompileParamExists()
				&& !args.getReduceRetracingParamExists() && !args.getExpImplementsParamExists() && !args.getInputSignatureParamExists()
				&& !args.getExpTypeHintsParamExists() && !args.getFuncParamExists());
	}

	/**
	 * Test for #30. This simply tests whether we can parse the tf.function argument experimental_follow_type_hints.
	 */
	@Test
	public void testComputeParameters3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();
		assertNotNull(args);

		assertTrue(!args.getExpAutographOptParamExists() && !args.getAutoGraphParamExists() && !args.getJitCompileParamExists()
				&& !args.getReduceRetracingParamExists() && !args.getExpImplementsParamExists() && !args.getInputSignatureParamExists()
				&& args.getExpTypeHintsParamExists() && !args.getFuncParamExists());
	}

	/**
	 * Test for #30. This simply tests whether we can parse the tf.function argument experimental_implements.
	 */
	@Test
	public void testComputeParameters4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();
		assertNotNull(args);

		assertTrue(!args.getExpAutographOptParamExists() && !args.getAutoGraphParamExists() && !args.getJitCompileParamExists()
				&& !args.getReduceRetracingParamExists() && args.getExpImplementsParamExists() && !args.getInputSignatureParamExists()
				&& !args.getExpTypeHintsParamExists() && !args.getFuncParamExists());
	}

	/**
	 * Test for #30. This simply tests whether we can parse the tf.function argument jit_compile.
	 */
	@Test
	public void testComputeParameters5() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();
		assertNotNull(args);

		assertTrue(!args.getExpAutographOptParamExists() && !args.getAutoGraphParamExists() && args.getJitCompileParamExists()
				&& !args.getReduceRetracingParamExists() && !args.getExpImplementsParamExists() && !args.getInputSignatureParamExists()
				&& !args.getExpTypeHintsParamExists() && !args.getFuncParamExists());
	}

	/**
	 * Test for #30. This simply tests whether we can parse the tf.function argument reduce_retracing.
	 */
	@Test
	public void testComputeParameters6() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();
		assertNotNull(args);

		assertTrue(!args.getExpAutographOptParamExists() && !args.getAutoGraphParamExists() && !args.getJitCompileParamExists()
				&& args.getReduceRetracingParamExists() && !args.getExpImplementsParamExists() && !args.getInputSignatureParamExists()
				&& !args.getExpTypeHintsParamExists() && !args.getFuncParamExists());
	}

	/**
	 * Test for #30. This simply tests whether we can parse the tf.function argument autograph.
	 */
	@Test
	public void testComputeParameters7() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();
		assertNotNull(args);

		assertTrue(!args.getExpAutographOptParamExists() && args.getAutoGraphParamExists() && !args.getJitCompileParamExists()
				&& !args.getReduceRetracingParamExists() && !args.getExpImplementsParamExists() && !args.getInputSignatureParamExists()
				&& !args.getExpTypeHintsParamExists() && !args.getFuncParamExists());
	}

	/**
	 * Test for #30. This simply tests whether we can identify when there are no tf.function args.
	 */
	@Test
	public void testComputeParameters8() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		Function.HybridizationParameters args = function.getArgs();
		assertNotNull(args);

		assertTrue(!args.getInputSignatureParamExists() && !args.getAutoGraphParamExists() && !args.getJitCompileParamExists()
				&& !args.getReduceRetracingParamExists() && !args.getExpImplementsParamExists() && !args.getExpAutographOptParamExists()
				&& !args.getExpTypeHintsParamExists() && !args.getFuncParamExists());
	}

	/**
	 * Test for #30. This simply tests whether we can parse tf.function arguments when we have multiple.
	 */
	@Test
	public void testComputeParameters9() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		// Needs to be an ArrayList because we are testing a tf.function with multiple
		// parameters.
		Map<String, ArrayList<String>> funcParameters = new HashMap<>();

		ArrayList<String> paramNames = new ArrayList<>();
		paramNames.add("input_signature");
		paramNames.add("autograph");
		funcParameters.put("func", paramNames);

		for (Function func : functions) {
			assertNotNull(func);
			Function.HybridizationParameters args = func.getArgs();
			assertNotNull(args);

			String actualFunctionName = NodeUtils.getFullRepresentationString(func.getFunctionDef());
			ArrayList<String> functionParameters = funcParameters.get(actualFunctionName);
			for (String param : functionParameters) {
				if (param == "input_signature")
					assertTrue(args.getInputSignatureParamExists());
				if (param == "autograph")
					assertTrue(args.getAutoGraphParamExists());
			}
		}
	}

	/**
	 * This simply tests whether we have the correct fully qualified name.
	 */
	@Test
	public void testFQN() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(7, functions.size());

		Map<String, String> funcSimpleNameToExpectedSignature = new HashMap<>();

		funcSimpleNameToExpectedSignature.put("func", "func");
		funcSimpleNameToExpectedSignature.put("func1", "func1");
		funcSimpleNameToExpectedSignature.put("func2", "func1.func2");
		funcSimpleNameToExpectedSignature.put("func_class1", "Class1.func_class1");
		funcSimpleNameToExpectedSignature.put("func_class2", "Class1.Class2.func_class2");
		funcSimpleNameToExpectedSignature.put("func_class3", "Class1.func_class3");
		funcSimpleNameToExpectedSignature.put("func_class4", "Class1.Class2.func_class4");

		for (Function func : functions) {
			LOG.info("Checking: " + func);

			assertNotNull(func);

			String simpleName = NodeUtils.getFullRepresentationString(func.getFunctionDef());

			LOG.info("Function simple name: " + simpleName);

			String expectedSignature = funcSimpleNameToExpectedSignature.get(simpleName);

			LOG.info("Expected signature: " + expectedSignature);

			String actualSignature = func.getIdentifer();

			LOG.info("Actual signature: " + actualSignature);

			assertEquals(expectedSignature, actualSignature);
		}
	}

	/**
	 * Test for #47. Here, we test using an alias.
	 */
	@Test
	public void testIsHybrid() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		// TODO: Set to assertTrue() after fixing #47.
		assertFalse(function.isHybrid());
	}

	/**
	 * Test for #47. No alias used here.
	 */
	@Test
	public void testIsHybrid2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		// TODO: Set to assertTrue() after fixing #47.
		assertFalse(function.isHybrid());
	}

	/**
	 * Test for #47. This function is not from TensorFlow.
	 */
	@Test
	public void testIsHybrid3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size()); // one function is for the decorator.
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());
	}

	/**
	 * Test for #47. This function is not from TensorFlow.
	 */
	@Test
	public void testIsHybrid4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size()); // The decorator is in another file.
		Function function = functions.iterator().next();
		assertNotNull(function);
		// TODO: Set to assertFalse() after fixing #47.
		assertTrue(function.isHybrid());
	}

	/**
	 * Test for #47. This function is not from TensorFlow.
	 */
	@Test
	public void testIsHybrid5() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size()); // The decorator is in another file.
		Function function = functions.iterator().next();
		assertNotNull(function);

		// NOTE: There's not much we can do about this since it has the same signature as the real TF. Even PyDev has
		// trouble with this.
		assertTrue(function.isHybrid());
	}

	/**
	 * Test for #47. This function is not from TensorFlow. Same as 4, but uses "from tf import function."
	 */
	@Test
	public void testIsHybrid6() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size()); // The decorator is in another file.
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());
	}

	/**
	 * Same as testIsHybridTrue except that we use "from" in the import statement.
	 */
	@Test
	public void testIsHybrid7() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		// TODO: Change to assertTrue once #20 is fixed.
		assertFalse(function.isHybrid());
	}

	/**
	 * Same as testIsHybrid7 except that we use "from *" in the import statement.
	 */
	@Test
	public void testIsHybrid8() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		// TODO: Change to assertTrue once #20 is fixed.
		assertFalse(function.isHybrid());
	}

	/**
	 * This simply tests whether the annotation is present for now. Case: not hybrid
	 */
	@Test
	public void testIsHybridFalse() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(3, functions.size());

		for (Function func : functions) {
			assertNotNull(func);
			assertFalse(func.isHybrid());
		}
	}

	/**
	 * Test #23. This simply tests whether this tool does not crash with decorators with multiple dots Case: not hybrid
	 */
	@Test
	public void testIsHybridMultipleAttributes() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(3, functions.size());

		for (Function func : functions) {
			assertNotNull(func);
			assertFalse(func.isHybrid());
		}
	}

	/**
	 * Test #17. This simply tests whether this tool looks at multiple decorator. Case: Hybrid
	 */
	@Test
	public void testIsHybridMultipleDecorators() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		for (Function func : functions) {
			assertNotNull(func);
			assertTrue(func.isHybrid());
		}
	}

	/**
	 * Test #5. This simply tests whether the annotation is present for now. It's probably not a "candidate," however, since it doesn't have
	 * a Tensor argument. NOTE: This may wind up failing at some point since it doesn't have a Tensor argument. Case: Hybrid
	 */
	@Test
	public void testIsHybridTrue() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());
	}

	/**
	 * Test for #19. This simply tests whether a decorator with parameters is correctly identified as hybrid. Case: hybrid
	 */
	@Test
	public void testIsHybridWithParameters() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(3, functions.size());

		for (Function func : functions) {
			assertNotNull(func);
			assertTrue(func.isHybrid());
		}
	}

	/**
	 * This simply tests whether we can process the decorator that has a decorator of type Name.
	 */
	@Test
	public void testProcessDecorator() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		// NOTE: This should actually be assertTrue() instead of assertFalse().
		// TODO: Change it to assertTrue() after we fix #20.
		assertFalse(function.isHybrid());
	}

	/**
	 * Test #38. This simply tests whether two functions with the same names in a file are processed individually.
	 */
	@Test
	public void testSameFileSameName() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		// TODO: Change to 2 after #41 is fixed.
		assertEquals(1, functions.size());

		Set<String> functionNames = new HashSet<>();

		for (Function func : functions) {
			assertNotNull(func);
			functionNames.add(func.getIdentifer());
		}

		// TODO: Change to 2 after #41 is fixed.
		assertEquals(1, functionNames.size());
	}

	/**
	 * Test #38. This simply tests whether two functions with the same names in a file are processed individually.
	 */
	@Test
	public void testSameFileSameName2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		// TODO: Change to 2 when #41 is fixed.
		assertEquals(1, functions.size());

		Set<String> functionNames = new HashSet<>();

		for (Function func : functions) {
			assertNotNull(func);
			functionNames.add(func.getIdentifer());
		}

		// NOTE: Both of these functions have the same qualified name.
		assertEquals(1, functionNames.size());
	}
}
