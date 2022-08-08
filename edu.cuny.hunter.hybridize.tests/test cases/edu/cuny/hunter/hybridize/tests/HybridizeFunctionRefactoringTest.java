package edu.cuny.hunter.hybridize.tests;

import static org.eclipse.core.runtime.Platform.getLog;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
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

	private static final String REFACTORING_PATH = "HybridizeFunction/";

	private static final String TEST_FILE_EXTENION = "py";

	private static final ILog LOG = getLog(HybridizeFunctionRefactoringTest.class);

	@Override
	protected String getRefactoringPath() {
		return REFACTORING_PATH;
	}

	/**
	 * Runs a single analysis test.
	 * 
	 * @return The set of {@link Function}s analyzed.
	 */
	private Set<Function> getFunctions() throws Exception {
		SimpleNode pythonNode = createPythonNodeFromTestFile("A");

		// extract function definitions.
		FunctionExtractor functionExtractor = new FunctionExtractor();
		pythonNode.accept(functionExtractor);
		Set<FunctionDef> availableFunctions = functionExtractor.getDefinitions().stream()
				.filter(RefactoringAvailabilityTester::isHybridizationAvailable).collect(Collectors.toSet());

		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(
				availableFunctions.toArray(FunctionDef[]::new));
		ProcessorBasedRefactoring refactoring = new ProcessorBasedRefactoring(processor);

		RefactoringStatus status = performRefactoringWithStatus(refactoring);
		assertTrue(status.isOK());

		return processor.getFunctions();
	}

	private SimpleNode createPythonNodeFromTestFile(String fileName) throws IOException, MisconfigurationException {
		return createPythonNodeFromTestFile(fileName, true);
	}

	private SimpleNode createPythonNodeFromTestFile(String fileName, boolean input)
			throws IOException, MisconfigurationException {
		String contents = input ? getFileContents(getInputTestFileName(fileName))
				: getFileContents(getOutputTestFileName(fileName));

		return createPythonNode(fileName, fileName + '.' + TEST_FILE_EXTENION, contents);
	}

	private SimpleNode createPythonNode(String moduleName, String fileName, String contents)
			throws MisconfigurationException {
		LOG.info("Creating PythonNode for " + fileName + " in " + fileName);
		LOG.info("Contents: " + contents);

		IDocument document = new Document(contents);

		IGrammarVersionProvider provider = new IGrammarVersionProvider() {

			@Override
			public int getGrammarVersion() throws MisconfigurationException {
				return IGrammarVersionProvider.LATEST_GRAMMAR_PY3_VERSION;
			}

			@Override
			public AdditionalGrammarVersionsToCheck getAdditionalGrammarVersions() throws MisconfigurationException {
				return null;
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
				if (token != null) {
					fail("Expected no error, received: " + parseErr.getMessage() + "\n" + s + "\nline:"
							+ token.beginLine + "\ncol:" + token.beginColumn);
				}
			}

			fail("Expected no error, received:\n" + err + "\n" + s);
		}

		// Check for AST failure.
		assertNotNull("Failed to generate AST.", parseOutput.ast);

		return (SimpleNode) parseOutput.ast;
	}

	/**
	 * Test #5. This simply tests whether the annotation is present for now.
	 * It's probably not a "candidate," however, since it doesn't have a Tensor
	 * argument. NOTE: This may wind up failing at some point since it doesn't
	 * have a Tensor argument. Case: Hybrid
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
	 * This simply tests whether the annotation is present for now. Case: not
	 * hybrid
	 */
	@Test
	public void testIsHybridFalse() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		for (Function func : functions) {
			assertNotNull(func);
			assertFalse(func.isHybrid());
		}
	}
	
	/**
	 * Test #17. This simply tests whether this tool looks at multiple decorator.
	 * Case: Hybrid
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
	 * Test for #19. This simply tests whether a decorator with parameters is
	 * correctly identified as hybrid. Case: Hybrid
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
	 * This simply tests whether we have the correct fully qualified name.
	 */
	@Test
	public void testFQN() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(5, functions.size());

		Map<String, String> funcSimpleNameToExpectedSignature = new HashMap<String, String>();

		funcSimpleNameToExpectedSignature.put("func", "func");
		funcSimpleNameToExpectedSignature.put("func1", "func1");
		funcSimpleNameToExpectedSignature.put("func2", "func1.func2");
		funcSimpleNameToExpectedSignature.put("func_class1", "Class1.func_class1");
		funcSimpleNameToExpectedSignature.put("func_class2", "Class1.Class2.func_class2");

		for (Function func : functions) {
			assertNotNull(func);
			String actualFunctionDefFullRepresentationString = NodeUtils
					.getFullRepresentationString(func.getFunctionDef());
			assertEquals(funcSimpleNameToExpectedSignature.get(actualFunctionDefFullRepresentationString),
					func.getIdentifer());
		}
	}

	/**
	 * This simply tests whether we can process the decorator that has a
	 * decorator of type Name.
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

	@Override
	public void genericbefore() throws Exception {
		if (fIsVerbose) {
			System.out.println("\n---------------------------------------------");
			System.out.println("\nTest:" + getClass() + "." + getName());
		}
		RefactoringCore.getUndoManager().flush();
	}

	@Override
	public void genericafter() throws Exception {
	}
}
