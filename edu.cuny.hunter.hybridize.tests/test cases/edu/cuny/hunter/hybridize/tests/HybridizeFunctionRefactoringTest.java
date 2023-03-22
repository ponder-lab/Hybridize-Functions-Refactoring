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
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.stream.Collectors;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.jface.text.Document;
import org.eclipse.jface.text.IDocument;
import org.eclipse.ltk.core.refactoring.RefactoringCore;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.python.pydev.ast.codecompletion.revisited.ASTManager;
import org.python.pydev.ast.codecompletion.revisited.ModulesManager;
import org.python.pydev.ast.codecompletion.revisited.ModulesManagerWithBuild;
import org.python.pydev.ast.codecompletion.revisited.ProjectStub;
import org.python.pydev.ast.codecompletion.revisited.PythonPathHelper;
import org.python.pydev.ast.codecompletion.revisited.modules.AbstractModule;
import org.python.pydev.ast.codecompletion.revisited.modules.CompiledModule;
import org.python.pydev.ast.codecompletion.revisited.modules.SourceModule;
import org.python.pydev.ast.interpreter_managers.InterpreterInfo;
import org.python.pydev.ast.interpreter_managers.InterpreterManagersAPI;
import org.python.pydev.ast.interpreter_managers.PythonInterpreterManager;
import org.python.pydev.core.CorePlugin;
import org.python.pydev.core.IGrammarVersionProvider;
import org.python.pydev.core.IInterpreterInfo;
import org.python.pydev.core.IInterpreterManager;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.MisconfigurationException;
import org.python.pydev.core.ModulesKey;
import org.python.pydev.core.TestDependent;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.core.preferences.InterpreterGeneralPreferences;
import org.python.pydev.core.proposals.CompletionProposalFactory;
import org.python.pydev.editor.codecompletion.proposals.DefaultCompletionProposalFactory;
import org.python.pydev.parser.PyParser;
import org.python.pydev.parser.PyParser.ParserInfo;
import org.python.pydev.parser.jython.ParseException;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.Token;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.plugin.PydevPlugin;
import org.python.pydev.plugin.PydevTestUtils;
import org.python.pydev.plugin.nature.PythonNature;
import org.python.pydev.refactoring.ast.PythonModuleManager;
import org.python.pydev.shared_core.io.FileUtils;
import org.python.pydev.shared_core.parsing.BaseParser.ParseOutput;
import org.python.pydev.shared_core.preferences.InMemoryEclipsePreferences;
import org.python.pydev.shared_core.string.CoreTextSelection;
import org.python.pydev.shared_core.string.StringUtils;
import org.python.pydev.ui.BundleInfoStub;

import com.python.pydev.analysis.additionalinfo.AbstractAdditionalDependencyInfo;
import com.python.pydev.analysis.additionalinfo.AdditionalProjectInterpreterInfo;

import edu.cuny.citytech.refactoring.common.tests.RefactoringTest;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.analysis.FunctionExtractor;
import edu.cuny.hunter.hybridize.core.analysis.Util;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;
import edu.cuny.hunter.hybridize.core.utils.RefactoringAvailabilityTester;

@SuppressWarnings("restriction")
public class HybridizeFunctionRefactoringTest extends RefactoringTest {

	protected static final boolean ADD_MX_TO_FORCED_BUILTINS = true;

	protected static final boolean ADD_NUMPY_TO_FORCED_BUILTINS = true;

	protected static final int GRAMMAR_TO_USE_FOR_PARSING = IGrammarVersionProvider.LATEST_GRAMMAR_PY3_VERSION;

	private static final ILog LOG = getLog(HybridizeFunctionRefactoringTest.class);

	/**
	 * The {@link PythonNature} to be used for the tests.
	 */
	private static PythonNature nature = new PythonNature() {
		@Override
		public AdditionalGrammarVersionsToCheck getAdditionalGrammarVersions() throws MisconfigurationException {
			return null;
		}

		@Override
		public int getGrammarVersion() {
			return GRAMMAR_TO_USE_FOR_PARSING;
		}

		@Override
		public int getInterpreterType() throws CoreException {
			return IInterpreterManager.INTERPRETER_TYPE_PYTHON;
		}
	};

	private static final String REFACTORING_PATH = "HybridizeFunction/";

	private static final String TEST_FILE_EXTENION = "py";

	private static final String TF_FUNCTION_FQN = "tensorflow.python.eager.def_function.function";

	/**
	 * Add a module to the given {@link IPythonNature}.
	 *
	 * @param ast the ast that defines the module
	 * @param modName the module name
	 * @param natureToAdd the nature where the module should be added
	 * @throws MisconfigurationException on project's misconfiguration.
	 */
	@SuppressWarnings("unused")
	private static void addModuleToNature(final SimpleNode ast, String modName, IPythonNature natureToAdd, File f)
			throws MisconfigurationException {
		// this is to add the info from the module that we just created...
		AbstractAdditionalDependencyInfo additionalInfo;
		additionalInfo = AdditionalProjectInterpreterInfo.getAdditionalInfoForProject(natureToAdd);
		additionalInfo.addAstInfo(ast, new ModulesKey(modName, f), false);
		ModulesManager modulesManager = (ModulesManager) natureToAdd.getAstManager().getModulesManager();
		SourceModule mod = (SourceModule) AbstractModule.createModule(ast, f, modName, natureToAdd);
		modulesManager.doAddSingleModule(new ModulesKey(modName, f), mod);
	}

	/**
	 * Checks if the size of the system modules manager and the project module manager are coherent (we must have more modules in the system
	 * than in the project).
	 */
	protected static void checkSize() throws MisconfigurationException {
		IInterpreterManager interpreterManager = getInterpreterManager();
		InterpreterInfo info = (InterpreterInfo) interpreterManager.getDefaultInterpreterInfo(false);
		assertTrue(info.getModulesManager().getSize(true) > 0);

		int size = ((ASTManager) nature.getAstManager()).getSize();
		assertTrue("Interpreter size:" + info.getModulesManager().getSize(true) + " should be smaller than project size:" + size + " "
				+ "(because it contains system+project info)", info.getModulesManager().getSize(true) < size);
	}

	private static Entry<SimpleNode, IDocument> createPythonNode(String moduleName, File file, String contents)
			throws MisconfigurationException {
		LOG.info("Creating PythonNode for " + moduleName + " in " + file);
		LOG.info("Contents: " + contents);

		IDocument document = new Document(contents);

		assertTrue("Test file: " + file + " must exist.", file.exists());
		ParserInfo parserInfo = new ParserInfo(document, nature, moduleName, file);

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

		SimpleNode simpleNode = (SimpleNode) parseOutput.ast;
		return Map.entry(simpleNode, document);
	}

	protected static String getCompletePythonLib(boolean addSitePackages, boolean isPython3) {
		StringBuilder ret = new StringBuilder();

		if (isPython3) {
			ret.append(TestDependent.PYTHON_30_LIB);

			if (TestDependent.PYTHON3_LIB_DYNLOAD != null)
				ret.append("|" + TestDependent.PYTHON3_LIB_DYNLOAD);
		} else { // Python 2.
			ret.append(TestDependent.PYTHON2_LIB);

			if (TestDependent.PYTHON2_LIB_DYNLOAD != null)
				ret.append("|" + TestDependent.PYTHON2_LIB_DYNLOAD);
		}

		if (addSitePackages)
			if (isPython3) {
				String sitePackages = TestDependent.PYTHON3_SITE_PACKAGES;

				if (!TestDependent.isWindows())
					// replace ~ with the actual directory. See https://bit.ly/3gPN8O4.
					sitePackages = sitePackages.replaceFirst("^~", Matcher.quoteReplacement(System.getProperty("user.home")));

				ret.append("|" + sitePackages);
			} else
				ret.append("|" + TestDependent.PYTHON2_SITE_PACKAGES);

		if (TestDependent.isWindows() && !isPython3) // NOTE: No DLLs for Python 3.
			ret.append("|" + TestDependent.PYTHON2_DLLS);

		return ret.toString();
	}

	/**
	 * Get the default {@link InterpreterInfo}.
	 *
	 * @return The default interpreter info for the current manager.
	 * @throws MisconfigurationException when the interpreter has a misconfiguration.
	 */
	protected static InterpreterInfo getDefaultInterpreterInfo() throws MisconfigurationException {
		IInterpreterManager interpreterManager = getInterpreterManager();
		InterpreterInfo info;
		info = (InterpreterInfo) interpreterManager.getDefaultInterpreterInfo(false);
		return info;
	}

	/**
	 * Get the {@link IInterpreterManager} we are testing.
	 *
	 * @return The PyDev interpreter manager we are testing.
	 */
	protected static IInterpreterManager getInterpreterManager() {
		return InterpreterManagersAPI.getPythonInterpreterManager();
	}

	protected static String getSystemPythonpathPaths() {
		StringBuilder ret = new StringBuilder();

		String completePythonLib = getCompletePythonLib(true, isPython3Test());
		ret.append(completePythonLib);

		if (TestDependent.PYTHON38_QT5_PACKAGES != null) {
			String str = "|" + TestDependent.PYTHON38_QT5_PACKAGES;
			ret.append(str);
		}

		if (TestDependent.PYTHON3_DIST_PACKAGES != null) {
			String str = "|" + TestDependent.PYTHON3_DIST_PACKAGES;
			ret.append(str);
		}

		return ret.toString();
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

	protected static boolean isPython3Test() {
		return true;
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

	/**
	 * This method sets the ast manager for a nature and restores the pythonpath with the path passed.
	 *
	 * @param path the pythonpath that should be set for this nature.
	 * @param projectStub the project where the nature should be set.
	 */
	protected static void setAstManager(String path, ProjectStub projectStub) {
		nature.setProject(projectStub);
		projectStub.setNature(nature);
		ASTManager astManager = new ASTManager();
		nature.setAstManager(astManager);

		astManager.setNature(nature);
		astManager.setProject(projectStub, nature, false);
		astManager.changePythonPath(path, projectStub, null);
	}

	/**
	 * Sets the interpreter manager we should use.
	 *
	 * @param path The path to use.
	 */
	protected static void setInterpreterManager(String path) {
		PythonInterpreterManager interpreterManager = new PythonInterpreterManager(new InMemoryEclipsePreferences());

		InterpreterInfo info;
		if (isPython3Test()) {
			info = (InterpreterInfo) interpreterManager.createInterpreterInfo(TestDependent.PYTHON_30_EXE, new NullProgressMonitor(),
					false);
			TestDependent.PYTHON_30_EXE = info.executableOrJar;
		} else {
			info = (InterpreterInfo) interpreterManager.createInterpreterInfo(TestDependent.PYTHON2_EXE, new NullProgressMonitor(), false);
			TestDependent.PYTHON2_EXE = info.executableOrJar;
		}
		if (path != null)
			info = new InterpreterInfo(info.getVersion(), info.executableOrJar,
					PythonPathHelper.parsePythonPathFromStr(path, new ArrayList<String>()));

		interpreterManager.setInfos(new IInterpreterInfo[] { info }, null, null);
		InterpreterManagersAPI.setPythonInterpreterManager(interpreterManager);
	}

	@BeforeClass
	public static void setUp() throws MisconfigurationException {
		CompiledModule.COMPILED_MODULES_ENABLED = true;
		SourceModule.TESTING = true;
		CompletionProposalFactory.set(new DefaultCompletionProposalFactory());
		PydevPlugin.setBundleInfo(new BundleInfoStub());
		CorePlugin.setBundleInfo(new BundleInfoStub());
		ModulesManagerWithBuild.IN_TESTS = true;
		FileUtils.IN_TESTS = true;
		PydevTestUtils.setTestPlatformStateLocation();
		AbstractAdditionalDependencyInfo.TESTING = true;
		InterpreterGeneralPreferences.FORCE_USE_TYPESHED = Boolean.TRUE;
		PythonNature.IN_TESTS = true;
		PythonModuleManager.setTesting(true);

		String paths = getSystemPythonpathPaths();
		paths = StringUtils.replaceAllSlashes(paths);
		final Set<String> s = new HashSet<>(Arrays.asList(paths.split("\\|")));
		InterpreterInfo.configurePathsCallback = arg -> {
			List<String> toAsk = arg.o1;
			List<String> l = arg.o2;

			for (String t : toAsk)
				if (s.contains(StringUtils.replaceAllSlashes(t.toLowerCase())))
					l.add(t);
			// System.out.println("Added:"+t);
			return Boolean.TRUE;
		};

		// System Python paths.

		setInterpreterManager(paths);

		InterpreterInfo info = getDefaultInterpreterInfo();
		info.restoreCompiledLibs(null);

		if (ADD_MX_TO_FORCED_BUILTINS)
			info.addForcedLib("mx");

		if (ADD_NUMPY_TO_FORCED_BUILTINS)
			info.addForcedLib("numpy");
	}

	@AfterClass
	public static void tearDown() {
		CompletionProposalFactory.set(null);
		PydevPlugin.setBundleInfo(null);
		CorePlugin.setBundleInfo(null);
		ModulesManagerWithBuild.IN_TESTS = false;
		FileUtils.IN_TESTS = false;
		AbstractAdditionalDependencyInfo.TESTING = false;
		CompiledModule.COMPILED_MODULES_ENABLED = false;
		SourceModule.TESTING = false;
		InterpreterGeneralPreferences.FORCE_USE_TYPESHED = null;
		PythonNature.IN_TESTS = false;
		PythonModuleManager.setTesting(false);
	}

	private Entry<SimpleNode, IDocument> createPythonNodeFromTestFile(String fileName) throws IOException, MisconfigurationException {
		return this.createPythonNodeFromTestFile(fileName, true);
	}

	private Entry<SimpleNode, IDocument> createPythonNodeFromTestFile(String fileName, boolean input)
			throws IOException, MisconfigurationException {
		String inputTestFileName = this.getInputTestFileName(fileName);

		String contents = input ? this.getFileContents(inputTestFileName) : this.getFileContents(this.getOutputTestFileName(fileName));

		Path path = getAbsolutionPath(inputTestFileName);
		File file = path.toFile();

		return createPythonNode(fileName, file, contents);
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

		boolean validSourceFile = PythonPathHelper.isValidSourceFile(inputTestFileAbsolutionPath.toString());
		assertTrue("Source file must be valid.", validSourceFile);

		Path inputTestFileDirectoryAbsolutePath = inputTestFileAbsolutionPath.getParent();

		// Run the Python test file.
		installRequirements(inputTestFileDirectoryAbsolutePath);
		runPython(inputTestFileAbsolutionPath);

		// Project Python path.
		String projectPath = inputTestFileDirectoryAbsolutePath.toString();

		ProjectStub projectStub = new ProjectStub("TestProject", projectPath, new IProject[0], new IProject[0]);

		setAstManager(projectPath, projectStub);

		AdditionalProjectInterpreterInfo.getAdditionalInfo(nature);

		checkSize();

		// NOTE (RK): Adding the test module to the nature. I think this already done anyway from the project path
		// above.
		// SimpleNode ast = request.getAST();
		// addModuleToNature(ast, modName, nature, file);
	}

	/**
	 * Returns the refactoring available {@link FunctionDef}s found in the test file A.py. The {@link IDocument} represents the contents of
	 * A.py.
	 *
	 * @return The refactoring available {@link FunctionDef}s in A.py represented by the {@link IDocument}.
	 */
	private Entry<IDocument, Collection<FunctionDef>> getDocumentToAvailableFunctionDefinitions() throws Exception {
		Entry<SimpleNode, IDocument> pythonNodeToDocument = this.createPythonNodeFromTestFile("A");

		// extract function definitions.
		FunctionExtractor functionExtractor = new FunctionExtractor();
		SimpleNode node = pythonNodeToDocument.getKey();
		node.accept(functionExtractor);

		// filter out the unavailable ones.
		Collection<FunctionDef> availableFunctionDefinitions = functionExtractor.getDefinitions().stream()
				.filter(RefactoringAvailabilityTester::isHybridizationAvailable).collect(Collectors.toList());

		IDocument document = pythonNodeToDocument.getValue();

		return Map.entry(document, availableFunctionDefinitions);
	}

	/**
	 * Returns the {@link Function}s in the test file.
	 *
	 * @return The set of {@link Function}s analyzed.
	 */
	private Set<Function> getFunctions() throws Exception {
		NullProgressMonitor monitor = new NullProgressMonitor();
		File inputTestFile = this.getInputTestFile();

		Entry<IDocument, Collection<FunctionDef>> documentToAvailableFunctionDefs = this.getDocumentToAvailableFunctionDefinitions();

		IDocument document = documentToAvailableFunctionDefs.getKey();
		Collection<FunctionDef> availableFunctionDefs = documentToAvailableFunctionDefs.getValue();

		Set<FunctionDefinition> inputFunctionDefinitions = availableFunctionDefs.stream()
				.map(f -> new FunctionDefinition(f, "A", inputTestFile, document, nature)).collect(Collectors.toSet());

		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(inputFunctionDefinitions, monitor);

		ProcessorBasedRefactoring refactoring = new ProcessorBasedRefactoring(processor);

		RefactoringStatus status = this.performRefactoringWithStatus(refactoring);
		assertTrue(status.isOK());

		return processor.getFunctions();
	}

	/**
	 * Return the {@link File} representing A.py.
	 *
	 * @return The {@link File} representing A.py.
	 */
	private File getInputTestFile() {
		String fileName = this.getInputTestFileName("A");
		Path path = getAbsolutionPath(fileName);
		File file = path.toFile();
		assertTrue("Test file must exist.", file.exists());
		return file;
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
	 * Test for #106. Contains ambiguous definitions using a property decorator for methods getter and setter.
	 */
	@Test
	public void testAmbiguousDefinition() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(3, functions.size());

		for (Function function : functions) {
			assertNotNull(function);
			assertFalse(function.isHybrid());
			assertFalse(function.likelyHasTensorParameter());

		}
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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && args.hasInputSignatureParam() & !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() & !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() & !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& args.hasExperimentalFollowTypeHintsParam());
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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() & !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() & !args.hasAutoGraphParam() && args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
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

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() & !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() & args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() & !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && args.hasInputSignatureParam() & args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
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

		assertFalse(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();

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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		// This test is with a custom decorator `@custom.decorator` that contains a parameter `input_signature`
		// like `tf.function`. But it also has a tf.function decorator, therefore args should not be Null.
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() & args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());

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

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() & !args.hasAutoGraphParam() && args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptParam()
				&& !args.hasExperimentalFollowTypeHintsParam());

	}

	/**
	 * This simply tests whether we have the correct qualified name.
	 */
	@Test
	public void testQN() throws Exception {
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

			String simpleName = func.getSimpleName();

			LOG.info("Function simple name: " + simpleName);

			String expectedSignature = funcSimpleNameToExpectedSignature.get(simpleName);

			LOG.info("Expected signature: " + expectedSignature);

			String actualSignature = func.getIdentifer();

			LOG.info("Actual signature: " + actualSignature);

			assertEquals(expectedSignature, actualSignature);
		}
	}

	/**
	 * Test for #47. Attribute case.
	 */
	@Test
	public void testGetDecoratorFQN() throws Exception {
		this.testGetDecoratorFQNInternal();
	}

	/**
	 * Test for #47. Call case.
	 */
	@Test
	public void testGetDecoratorFQN2() throws Exception {
		this.testGetDecoratorFQNInternal();
	}

	private void testGetDecoratorFQNInternal() throws Exception {
		Entry<IDocument, Collection<FunctionDef>> documentToAvailableFunctionDefinitions = this.getDocumentToAvailableFunctionDefinitions();

		Collection<FunctionDef> functionDefinitions = documentToAvailableFunctionDefinitions.getValue();
		assertNotNull(functionDefinitions);
		assertEquals(1, functionDefinitions.size());

		FunctionDef functionDef = functionDefinitions.iterator().next();
		assertNotNull(functionDef);

		decoratorsType[] decoratorArray = functionDef.decs;
		assertNotNull(decoratorArray);
		assertEquals(1, decoratorArray.length);

		decoratorsType decorator = decoratorArray[0];
		assertNotNull(decorator);

		exprType decoratorFunction = decorator.func;
		assertNotNull(decoratorFunction);

		String representationString = NodeUtils.getFullRepresentationString(decoratorFunction);
		assertEquals("tf.function", representationString);

		File inputTestFile = this.getInputTestFile();

		IDocument document = documentToAvailableFunctionDefinitions.getKey();

		int offset = NodeUtils.getOffset(document, decoratorFunction);
		String representationString2 = NodeUtils.getRepresentationString(decoratorFunction);

		CoreTextSelection coreTextSelection = new CoreTextSelection(document, offset, representationString2.length());

		PySelection selection = new PySelection(document, coreTextSelection);

		String fullyQualifiedName = Util.getFullyQualifiedName(decorator, "A", inputTestFile, selection, nature, new NullProgressMonitor());

		assertEquals(TF_FUNCTION_FQN, fullyQualifiedName);
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
		assertTrue(function.isHybrid());
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
		assertTrue(function.isHybrid());
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
		assertFalse(function.isHybrid());
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
		assertFalse(function.isHybrid());
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
		assertTrue(function.isHybrid());
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
		assertTrue(function.isHybrid());
	}

	/**
	 * Call case.
	 */
	@Test
	public void testIsHybrid9() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());
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
		assertTrue(function.isHybrid());
	}

	/**
	 * Test #38. This simply tests whether two functions with the same names in a file are processed individually.
	 */
	@Test
	public void testSameFileSameName() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		assertEquals(2, functions.size());

		Set<String> functionNames = new HashSet<>();

		for (Function func : functions) {
			assertNotNull(func);
			functionNames.add(func.getIdentifer());
		}

		assertEquals(2, functionNames.size());
	}

	/**
	 * Test #38. This simply tests whether two functions with the same names in a file are processed individually.
	 */
	@Test
	public void testSameFileSameName2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		assertEquals(2, functions.size());

		Set<String> functionNames = new HashSet<>();

		for (Function func : functions) {
			assertNotNull(func);
			functionNames.add(func.getIdentifer());
		}

		// NOTE: Both of these functions have the same qualified name.
		assertEquals(1, functionNames.size());
	}

	@Test
	public void testFunctionEquality() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		for (Function func : functions) {
			assertNotNull(func);
			String id = func.getIdentifer();
			assertNotNull(id);
			assertTrue(id.equals("a") || id.equals("b"));
		}

		Iterator<Function> iterator = functions.iterator();
		assertNotNull(iterator);
		assertTrue(iterator.hasNext());

		Function func1 = iterator.next();
		assertNotNull(func1);

		String identifer1 = func1.getIdentifer();
		assertNotNull(identifer1);

		assertTrue(iterator.hasNext());

		Function func2 = iterator.next();
		assertNotNull(func2);

		String identifer2 = func2.getIdentifer();
		assertNotNull(identifer2);

		assertTrue(!identifer1.equals("a") || identifer2.equals("b"));
		assertTrue(!identifer1.equals("b") || identifer2.equals("a"));

		assertTrue(!func1.equals(func2));
		assertTrue(func1.hashCode() != func2.hashCode());

		assertTrue(!func2.equals(func1));
		assertTrue(func2.hashCode() != func1.hashCode());
	}

	@Test
	public void testFunctionEquality2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		for (Function func : functions) {
			assertNotNull(func);
			String id = func.getIdentifer();
			assertNotNull(id);
			assertTrue(id.equals("a"));
		}

		Iterator<Function> iterator = functions.iterator();
		assertNotNull(iterator);
		assertTrue(iterator.hasNext());

		Function func1 = iterator.next();
		assertNotNull(func1);

		String identifer1 = func1.getIdentifer();
		assertNotNull(identifer1);

		assertTrue(iterator.hasNext());

		Function func2 = iterator.next();
		assertNotNull(func2);

		String identifer2 = func2.getIdentifer();
		assertNotNull(identifer2);

		assertTrue(!func1.equals(func2));
		assertTrue(func1.hashCode() != func2.hashCode());

		assertTrue(!func2.equals(func1));
		assertTrue(func2.hashCode() != func1.hashCode());
	}

	@Test
	public void testFunctionEquality3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());

		Iterator<Function> iterator = functions.iterator();
		assertNotNull(iterator);
		assertTrue(iterator.hasNext());

		Function func = iterator.next();
		assertNotNull(func);

		String id = func.getIdentifer();
		assertNotNull(id);
		assertTrue(id.equals("a"));

		assertTrue(func.equals(func));
	}

	/**
	 * Test for #2. Here, the function has no parameters and is not hybrid. Thus, it's not likely to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());

		argumentsType params = function.getParameters();

		// no params.
		assertEquals(0, params.args.length);

		assertFalse(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has one parameter and is not hybrid. Thus, it's not likely to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());

		argumentsType params = function.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("x", paramName);

		assertFalse(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has one parameter with a default value and is not hybrid. The default value is not being used. Thus,
	 * it's not likely to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());

		argumentsType params = function.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("x", paramName);

		assertFalse(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has one parameter with a default value and is not hybrid. The default value is being used. Thus, it's
	 * not likely to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());

		argumentsType params = function.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("x", paramName);

		assertFalse(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has no parameters and is hybrid. Thus, it's not likely to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter5() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());

		argumentsType params = function.getParameters();

		// no params.
		assertEquals(0, params.args.length);

		assertFalse(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has no parameters, is hybrid, and considers type hints. Thus, it's not likely to have a tensor
	 * parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter6() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());

		// TODO: Need to check the value (#111).
		assertTrue(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());

		argumentsType params = function.getParameters();

		// no params.
		assertEquals(0, params.args.length);

		assertFalse(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has one parameters, is hybrid, and considers type hints. But, no type hint is supplied. Thus, it's
	 * not likely to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter7() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());
		assertTrue(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());
		// TODO: And the value is true (#111).

		argumentsType params = function.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("x", paramName);

		// get the type hint.
		exprType[] annotations = params.annotation;
		assertNotNull(annotations);

		// no type hint.
		assertEquals(1, annotations.length);
		exprType annotationExpr = annotations[0];
		assertNull(annotationExpr);

		assertFalse(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has one parameters, is hybrid, and does not consider type hints. But, a type hint is supplied. In
	 * other words, a type hint supplied but we don't use it. Thus, it's not likely to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter8() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());
		assertFalse(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());

		argumentsType params = function.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("x", paramName);

		// get the type hint.
		exprType[] annotations = params.annotation;
		assertNotNull(annotations);

		// Tensor type hint.
		assertEquals(1, annotations.length);
		exprType annotationExpr = annotations[0];
		assertNotNull(annotationExpr);

		assertTrue(annotationExpr instanceof Attribute);
		Attribute typeHint = (Attribute) annotationExpr;

		String attributeName = NodeUtils.getFullRepresentationString(typeHint);
		assertEquals("tf.Tensor", attributeName);

		assertFalse(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has one parameter, is hybrid and considers type hints. And, a tf.Tensor type hint is supplied. Thus,
	 * is likely to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter9() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());

		// TODO: Need to check the value (#111).
		assertTrue(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());

		argumentsType params = function.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("x", paramName);

		// get the type hint.
		exprType[] annotations = params.annotation;
		assertNotNull(annotations);

		// Tensor type hint.
		assertEquals(1, annotations.length);
		exprType annotationExpr = annotations[0];
		assertNotNull(annotationExpr);

		assertTrue(annotationExpr instanceof Attribute);
		Attribute typeHint = (Attribute) annotationExpr;

		String attributeName = NodeUtils.getFullRepresentationString(typeHint);
		assertEquals("tf.Tensor", attributeName);

		assertTrue(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has one parameter, is hybrid, but does not consider type hints by setting the flag to False. Thus,
	 * it's not likely to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter10() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());

		// The flag is there.
		assertTrue(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());

		// But, it's set to False.
		// TODO: assert that the experimental type hints param is set to false (#111).

		argumentsType params = function.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("x", paramName);

		// get the type hint.
		exprType[] annotations = params.annotation;
		assertNotNull(annotations);

		// Tensor type hint.
		assertEquals(1, annotations.length);
		exprType annotationExpr = annotations[0];
		assertNotNull(annotationExpr);

		assertTrue(annotationExpr instanceof Attribute);
		Attribute typeHint = (Attribute) annotationExpr;

		String attributeName = NodeUtils.getFullRepresentationString(typeHint);
		assertEquals("tf.Tensor", attributeName);

		// TODO: Set to assertFalse() when #111 is fixed.
		assertTrue(function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. From https://tensorflow.org/guide/function#usage.
	 */
	@Test
	public void testHasLikelyTensorParameter11() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());

		argumentsType params = function.getParameters();

		// two params.
		exprType[] actualParams = params.args;
		assertEquals(2, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("a", paramName);

		actualParameter = actualParams[1];
		assertNotNull(actualParameter);

		paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("b", paramName);

		assertTrue("Expecting function with likely tensor parameter.", function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. From https://tensorflow.org/guide/function#usage. <code>tf.Variable</code>s are similar to <code>tf.Tensor</code>s,
	 * thus, can we say that it's a likely tensor parameter? Why not? The first parameter is a <code>tf.Variable</code>.
	 */
	@Test
	public void testHasLikelyTensorParameter12() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());

		argumentsType params = function.getParameters();

		// two params.
		exprType[] actualParams = params.args;
		assertEquals(2, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("a", paramName);

		actualParameter = actualParams[1];
		assertNotNull(actualParameter);

		paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("b", paramName);

		assertTrue("Expecting function with likely tensor parameter.", function.likelyHasTensorParameter());
	}

	/**
	 * Test for #2. From https://tensorflow.org/guide/function#usage.
	 */
	@Test
	public void testHasLikelyTensorParameter13() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		FunctionUnderTest add = new FunctionUnderTest("add");
		add.addParameters("a", "b");
		functionsToTest.add(add);

		FunctionUnderTest denseLayer = new FunctionUnderTest("dense_layer");
		denseLayer.addParameters("x", "w", "b");
		functionsToTest.add(denseLayer);

		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(functionsToTest.size(), functions.size());

		Map<String, List<Function>> nameToFunctions = functions.stream().collect(Collectors.groupingBy(Function::getSimpleName));
		assertEquals(functionsToTest.size(), nameToFunctions.size());

		for (FunctionUnderTest fut : functionsToTest) {
			List<Function> functionList = nameToFunctions.get(fut.getName());
			assertEquals(1, functionList.size());

			Function function = functionList.iterator().next();
			assertNotNull(function);
			assertEquals(fut.isHybrid(), function.isHybrid());

			argumentsType params = function.getParameters();

			exprType[] actualParams = params.args;
			List<String> expectedParameters = fut.getParameters();
			assertEquals(expectedParameters.size(), actualParams.length);

			for (int i = 0; i < actualParams.length; i++) {
				exprType actualParameter = actualParams[i];
				assertNotNull(actualParameter);

				String paramName = NodeUtils.getRepresentationString(actualParameter);
				assertEquals(expectedParameters.get(i), paramName);
			}

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.likelyHasTensorParameter());
		}
	}

	/**
	 * Test for #2. From https://tensorflow.org/guide/function#usage.
	 */
	@Test
	public void testHasLikelyTensorParameter14() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		FunctionUnderTest functionToTest = new FunctionUnderTest("conv_fn");
		functionToTest.addParameters("image");
		functionsToTest.add(functionToTest);

		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(functionsToTest.size(), functions.size());

		Map<String, List<Function>> nameToFunctions = functions.stream().collect(Collectors.groupingBy(Function::getSimpleName));
		assertEquals(functionsToTest.size(), nameToFunctions.size());

		for (FunctionUnderTest fut : functionsToTest) {
			List<Function> functionList = nameToFunctions.get(fut.getName());
			assertEquals(1, functionList.size());

			Function function = functionList.iterator().next();
			assertNotNull(function);
			assertEquals(fut.isHybrid(), function.isHybrid());

			argumentsType params = function.getParameters();

			exprType[] actualParams = params.args;
			List<String> expectedParameters = fut.getParameters();
			assertEquals(expectedParameters.size(), actualParams.length);

			for (int i = 0; i < actualParams.length; i++) {
				exprType actualParameter = actualParams[i];
				assertNotNull(actualParameter);

				String paramName = NodeUtils.getRepresentationString(actualParameter);
				assertEquals(expectedParameters.get(i), paramName);
			}

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.likelyHasTensorParameter());
		}
	}

	/**
	 * Test for #2. From https://www.tensorflow.org/guide/function#what_is_tracing.
	 */
	@Test
	public void testHasLikelyTensorParameter15() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		FunctionUnderTest functionToTest = new FunctionUnderTest("double");
		functionToTest.addParameters("a");
		functionsToTest.add(functionToTest);

		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(functionsToTest.size(), functions.size());

		Map<String, List<Function>> nameToFunctions = functions.stream().collect(Collectors.groupingBy(Function::getSimpleName));
		assertEquals(functionsToTest.size(), nameToFunctions.size());

		for (FunctionUnderTest fut : functionsToTest) {
			List<Function> functionList = nameToFunctions.get(fut.getName());
			assertEquals(1, functionList.size());

			Function function = functionList.iterator().next();
			assertNotNull(function);
			assertEquals(fut.isHybrid(), function.isHybrid());

			argumentsType params = function.getParameters();

			exprType[] actualParams = params.args;
			List<String> expectedParameters = fut.getParameters();
			assertEquals(expectedParameters.size(), actualParams.length);

			for (int i = 0; i < actualParams.length; i++) {
				exprType actualParameter = actualParams[i];
				assertNotNull(actualParameter);

				String paramName = NodeUtils.getRepresentationString(actualParameter);
				assertEquals(expectedParameters.get(i), paramName);
			}

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.likelyHasTensorParameter());
		}
	}

	// TODO: Left off at https://www.tensorflow.org/guide/function#controlling_retracing
	// TODO: Test arbitrary expression.
	// TODO: Test cast/assert statements?
	// TODO: https://www.tensorflow.org/guide/function#pass_tensors_instead_of_python_literals. How do we deal with union types? Do we want
	// those to be refactored?

	/**
	 * Test a model. No tf.function in this one.
	 */
	@Test
	public void testModel() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));
	}

	/**
	 * Test a model. No tf.function in this one. Use call instead of __call__. Ariadne doesn't support __call__. See
	 * https://github.com/wala/ML/issues/24.
	 */
	@Test
	public void testModel2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));
	}

	/**
	 * Test a model. No tf.function in this one. Explicit call method.
	 */
	@Test
	public void testModel3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));
	}

	/**
	 * Test a model. No tf.function in this one. Explicit call method.
	 */
	@Test
	public void testModel4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));
	}

	// TODO: Test models that have tf.functions.
}
