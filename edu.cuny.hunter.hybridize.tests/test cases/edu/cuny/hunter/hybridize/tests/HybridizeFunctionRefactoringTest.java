package edu.cuny.hunter.hybridize.tests;

import static edu.cuny.hunter.hybridize.core.analysis.Function.PLUGIN_ID;
import static edu.cuny.hunter.hybridize.core.analysis.Information.SPECULATIVE_ANALYSIS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.CANT_APPROXIMATE_RECURSION;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_NO_TENSOR_PARAMETERS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_PRIMITIVE_PARAMETERS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.IS_RECURSIVE;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.UNDETERMINABLE_SIDE_EFFECTS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.UNDETERMINABLE_TENSOR_PARAMETER;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P1;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P2;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P3;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.OPTIMIZE_HYBRID_FUNCTION;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.CONVERT_TO_EAGER;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.CONVERT_TO_HYBRID;
import static java.lang.Integer.MAX_VALUE;
import static java.nio.file.FileVisitOption.FOLLOW_LINKS;
import static java.util.Collections.singleton;
import static java.util.stream.Collectors.toSet;
import static org.eclipse.core.resources.ResourceAttributes.fromFile;
import static org.eclipse.core.runtime.Platform.getLog;
import static org.eclipse.ltk.core.refactoring.RefactoringStatus.INFO;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.stream.Collectors;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.ResourceAttributes;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IPath;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.jface.text.Document;
import org.eclipse.jface.text.IDocument;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.RefactoringCore;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.RefactoringStatusEntry;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Ignore;
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
import org.python.pydev.plugin.FileStub2;
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
import edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure;
import edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess;
import edu.cuny.hunter.hybridize.core.analysis.Refactoring;
import edu.cuny.hunter.hybridize.core.analysis.Transformation;
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

	private static final String TEST_FILE_EXTENSION = "py";

	private static final String TF_FUNCTION_FQN = "tensorflow.python.eager.def_function.function";

	/**
	 * Check Python side-effects regardless if it's a candidate.
	 */
	private static final boolean ALWAYS_CHECK_PYTHON_SIDE_EFFECTS = true;

	private static final boolean ALWAYS_CHECK_RECURSION = true;

	private static final boolean USE_TEST_ENTRYPOINTS = true;

	private static final boolean ALWAYS_FOLLOW_TYPE_HINTS = true;

	private static final boolean USE_SPECULATIVE_ANALYSIS = true;

	/**
	 * Whether we should run the function processing in parallel. Running in parallel makes the logs difficult to read and doesn't offer
	 * much in way of speedup since each test has only a few {@link Function}s.
	 */
	private static final boolean PROCESS_FUNCTIONS_IN_PARALLEL = false;

	private static final String RUN_INPUT_TEST_FILE_KEY = "edu.cuny.hunter.hybridize.tests.runInput";

	private static final String COMPARE_OUTPUT_TEST_FILE_KEY = "edu.cuny.hunter.hybridize.tests.compareOutput";

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
	 * @return The exit code.
	 */
	private static int installRequirements(Path path) throws IOException, InterruptedException {
		Path requirements = path.resolve("requirements.txt");
		assertTrue("Requirements file must be present.", requirements.toFile().exists());

		// install requirements.
		return runCommand("python3.10", "-m", "pip", "install", "-r", requirements.toString());
	}

	protected static boolean isPython3Test() {
		return true;
	}

	private static int runCommand(String... command) throws IOException, InterruptedException {
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
		return exitCode;
	}

	/**
	 * Runs python on the file presented by the given {@link Path}.
	 *
	 * @param path The {@link Path} of the file to interpret.
	 * @return The exist code.
	 */
	private static int runPython(Path path) throws IOException, InterruptedException {
		// run the code.
		return runCommand("python3.10", path.toString());
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

	/**
	 * True iff the input test Python file should be executed.
	 */
	protected boolean runInputTestFile = Boolean.getBoolean(RUN_INPUT_TEST_FILE_KEY);

	/**
	 * True iff the output test Python file should be compared.
	 */
	protected boolean compareOutputTestFile = Boolean.getBoolean(COMPARE_OUTPUT_TEST_FILE_KEY);

	private Entry<SimpleNode, IDocument> createPythonNodeFromTestFile(String fileNameWithoutExtension)
			throws IOException, MisconfigurationException {
		return this.createPythonNodeFromTestFile(fileNameWithoutExtension, true);
	}

	private Entry<SimpleNode, IDocument> createPythonNodeFromTestFile(String fileNameWithoutExtension, boolean input)
			throws IOException, MisconfigurationException {
		String inputTestFileName = this.getInputTestFileName(fileNameWithoutExtension);

		String contents = input ? this.getFileContents(inputTestFileName)
				: this.getFileContents(this.getOutputTestFileName(fileNameWithoutExtension));

		Path path = getAbsolutePath(inputTestFileName);
		File file = path.toFile();

		return createPythonNode(fileNameWithoutExtension, file, contents);
	}

	/**
	 * True iff the input test Python file should be executed.
	 */
	public boolean getRunInputTestFile() {
		return this.runInputTestFile;
	}

	/**
	 * True iff the output test Python file should be compared.
	 */
	public boolean getCompareOutputTestFile() {
		return this.compareOutputTestFile;
	}

	@Override
	public void genericafter() throws Exception {
		// Do nothing.
	}

	@Override
	public void genericbefore() throws Exception {
		if (this.fIsVerbose) {
			System.out.println("\n---------------------------------------------");
			System.out.println("\nTest:" + this.getClass() + "." + this.getName());
		}

		RefactoringCore.getUndoManager().flush();

		String inputTestFileName = this.getInputTestFileName("A"); // There must at least be an A.py file.
		Path inputTestFileAbsolutePath = getAbsolutePath(inputTestFileName);
		Path inputTestFileDirectoryAbsolutePath = inputTestFileAbsolutePath.getParent();

		if (this.getRunInputTestFile()) {
			LOG.info("Running input test file(s).");

			// install dependencies.
			int rc = installRequirements(inputTestFileDirectoryAbsolutePath);
			LOG.info("Installing requirements was " + (rc == 0 ? "successful." : "unsuccessful."));

			// the number of Python files executed.
			int filesRun = 0;

			// for each Python file in the test file directory, recursively.
			Set<Path> pythonFilesInTestFileDirectory = Files.find(inputTestFileDirectoryAbsolutePath, MAX_VALUE,
					(path, attr) -> path.toFile().getName().endsWith(".py"), FOLLOW_LINKS).collect(toSet());

			for (Path path : pythonFilesInTestFileDirectory) {
				boolean validSourceFile = PythonPathHelper.isValidSourceFile(path.toString());
				assertTrue("Source file must be valid.", validSourceFile);

				// Run the Python test file.
				rc = runPython(path);
				LOG.info("Running the test file was " + (rc == 0 ? "successful." : "unsuccesful."));
				++filesRun;
			}

			assertTrue("Must have executed at least A.py.", filesRun > 0);
		}

		// Project Python path.
		String projectPath = inputTestFileDirectoryAbsolutePath.toString();

		ProjectStub projectStub = new ProjectStub("TestProject", projectPath, new IProject[0], new IProject[0]);

		setAstManager(projectPath, projectStub);

		AdditionalProjectInterpreterInfo.getAdditionalInfo(nature);

		checkSize();
	}

	/**
	 * Returns the refactoring available {@link FunctionDef}s found in the test file X.py, where X is fileNameWithoutExtension. The
	 * {@link IDocument} represents the contents of X.py.
	 *
	 * @param fileNameWithoutExtension The name of the test file excluding the file extension.
	 * @return The refactoring available {@link FunctionDef}s in X.py, where X is fileNameWithoutExtension, represented by the
	 *         {@link IDocument}.
	 */
	private Entry<IDocument, Collection<FunctionDef>> getDocumentToAvailableFunctionDefinitions(String fileNameWithoutExtension)
			throws Exception {
		Entry<SimpleNode, IDocument> pythonNodeToDocument = this.createPythonNodeFromTestFile(fileNameWithoutExtension);

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
	 * @param fileNameWithoutExtension The name of the test file excluding the file extension.
	 * @return The set of {@link Function}s analyzed.
	 */
	private Set<Function> getFunctions(String fileNameWithoutExtension) throws Exception {
		File inputTestFile = this.getInputTestFile(fileNameWithoutExtension);

		Entry<IDocument, Collection<FunctionDef>> documentToAvailableFunctionDefs = this
				.getDocumentToAvailableFunctionDefinitions(fileNameWithoutExtension);

		IDocument document = documentToAvailableFunctionDefs.getKey();
		Collection<FunctionDef> availableFunctionDefs = documentToAvailableFunctionDefs.getValue();

		IFile actualInputTestFile = new FileStub2(fileNameWithoutExtension + ".py") {
			@Override
			public String getFileExtension() {
				return TEST_FILE_EXTENSION;
			}

			@Override
			public IPath getFullPath() {
				// NOTE: This is incorrect when implemenng https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/359.
				return org.eclipse.core.runtime.Path.fromOSString(inputTestFile.getAbsolutePath());
			}

			@Override
			public String getCharset(boolean checkImplicit) throws CoreException {
				return System.getProperty("file.encoding");
			}

			@Override
			public long getModificationStamp() {
				return 0;
			}

			@Override
			public ResourceAttributes getResourceAttributes() {
				return fromFile(inputTestFile);
			}
		};

		Set<FunctionDefinition> inputFunctionDefinitions = availableFunctionDefs.stream()
				.map(f -> new FunctionDefinition(f, fileNameWithoutExtension, inputTestFile, actualInputTestFile, document, nature))
				.collect(Collectors.toSet());

		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(inputFunctionDefinitions,
				ALWAYS_CHECK_PYTHON_SIDE_EFFECTS, PROCESS_FUNCTIONS_IN_PARALLEL, ALWAYS_CHECK_RECURSION, USE_TEST_ENTRYPOINTS,
				ALWAYS_FOLLOW_TYPE_HINTS, USE_SPECULATIVE_ANALYSIS);

		ProcessorBasedRefactoring refactoring = new ProcessorBasedRefactoring(processor);

		RefactoringStatus status = refactoring.checkAllConditions(new NullProgressMonitor());

		if (processor.getFunctions().stream().map(Function::getStatus).allMatch(RefactoringStatus::isOK))
			assertTrue(status.isOK());
		else
			assertFalse(status.isOK());

		// NOTE: Fix https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/359 first.
		if (this.getCompareOutputTestFile()) {
			// check if there's an expected output.
			File outputTestFile = this.getOutputTestFile(fileNameWithoutExtension);

			if (outputTestFile.exists()) {
				Change change = refactoring.createChange(new NullProgressMonitor());
				this.performChange(change);

				String expected = this.getFileContents(this.getOutputTestFileName(fileNameWithoutExtension));
				String actual = document.get();
				assertEqualLines(expected, actual);
			}
		}

		return processor.getFunctions();
	}

	/**
	 * Returns the {@link Function}s in the test file A.py.
	 *
	 * @return The set of {@link Function}s analyzed.
	 */
	private Set<Function> getFunctions() throws Exception {
		return getFunctions("A");
	}

	/**
	 * Returns the first {@link Function} in the default test file with the given identifier.
	 *
	 * @param functionIndentifier The {@link Function} to return.
	 * @return The first {@link Function} in the default test file with the given identifier.
	 */
	private Function getFunction(String functionIndentifier) throws Exception {
		Set<Function> functions = this.getFunctions();
		return functions.stream().filter(f -> f.getIdentifier().equals(functionIndentifier)).findFirst().orElseThrow();
	}

	/**
	 * Return the {@link File} representing X.py, where X is fileNameWithoutExtension.
	 *
	 * @param fileNameWithoutExtension The filename not including the file extension.
	 * @return The {@link File} representing X.py, where X is fileNameWithoutExtension.
	 */
	private File getInputTestFile(String fileNameWithoutExtension) {
		String fileName = this.getInputTestFileName(fileNameWithoutExtension);
		File file = getFile(fileName);
		assertTrue("Test file: " + file.getName() + " must exist at path: " + file.getPath() + ".", file.exists());
		return file;
	}

	/**
	 * Return the {@link File} representing X.py, where X is fileNameWithoutExtension.
	 *
	 * @param fileNameWithoutExtension The filename not including the file extension.
	 * @return The {@link File} representing X.py, where X is fileNameWithoutExtension.
	 */
	private File getOutputTestFile(String fileNameWithoutExtension) {
		String fileName = this.getOutputTestFileName(fileNameWithoutExtension);
		return getFile(fileName);
	}

	private static File getFile(String fileName) {
		return getAbsolutePath(fileName).toFile();
	}

	@Override
	protected String getRefactoringPath() {
		return REFACTORING_PATH;
	}

	@Override
	protected String getTestFileExtension() {
		return TEST_FILE_EXTENSION;
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

			switch (function.getIdentifier()) {
			case "Test.name":
				assertNull(function.getHasTensorParameter());
				break;
			case "Test.value":
				assertFalse(function.getHasTensorParameter());
				break;
			case "Test.__init__":
				assertFalse(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Unknown function: " + function + ".");
			}

			switch (function.getIdentifier()) {
			case "Test.value":
			case "Test.name":
				checkSideEffectStatus(function);
				break;
			default:
				break;
			}
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

		assertTrue(!args.isFuncParamExists() && args.isInputSignatureParamExists() & !args.isAutoGraphParamExists()
				&& !args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());
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

		assertTrue(!args.isFuncParamExists() && !args.isInputSignatureParamExists() & !args.isAutoGraphParamExists()
				&& !args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());
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

		assertTrue(!args.isFuncParamExists() && !args.isInputSignatureParamExists() & !args.isAutoGraphParamExists()
				&& !args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && args.isExperimentalFollowTypeHintsParamExists());
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

		assertTrue(!args.isFuncParamExists() && !args.isInputSignatureParamExists() & !args.isAutoGraphParamExists()
				&& !args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());
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

		assertTrue(!args.isFuncParamExists() && !args.isInputSignatureParamExists() & !args.isAutoGraphParamExists()
				&& args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());
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

		assertTrue(!args.isFuncParamExists() && !args.isInputSignatureParamExists() & !args.isAutoGraphParamExists()
				&& !args.isJitCompileParamExists() && args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());
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

		assertTrue(!args.isFuncParamExists() && !args.isInputSignatureParamExists() & args.isAutoGraphParamExists()
				&& !args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());
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

		assertTrue(!args.isFuncParamExists() && !args.isInputSignatureParamExists() & !args.isAutoGraphParamExists()
				&& !args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());
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

		assertTrue(!args.isFuncParamExists() && args.isInputSignatureParamExists() & args.isAutoGraphParamExists()
				&& !args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());
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
		checkSideEffectStatus(function);
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

		assertTrue(!args.isFuncParamExists() && !args.isInputSignatureParamExists() & args.isAutoGraphParamExists()
				&& !args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());

		checkSideEffectStatus(function);
	}

	private static void checkSideEffectStatus(Function function) {
		RefactoringStatus status = function.getStatus();
		assertTrue("Should fail due to a call graph issue, either a decorated function or missing function invocation.", status.hasError());
		assertNull(function.getHasPythonSideEffects());
		RefactoringStatusEntry entry = status.getEntryMatchingCode(Function.PLUGIN_ID,
				PreconditionFailure.UNDETERMINABLE_SIDE_EFFECTS.getCode());
		assertNotNull(entry);
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

		assertTrue(!args.isFuncParamExists() && !args.isInputSignatureParamExists() & !args.isAutoGraphParamExists()
				&& args.isJitCompileParamExists() && !args.isReduceRetracingParamExists() && !args.isExperimentalImplementsParamExists()
				&& !args.isExperimentalAutographOptParamExists() && !args.isExperimentalFollowTypeHintsParamExists());
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

			String actualSignature = func.getIdentifier();

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
		Entry<IDocument, Collection<FunctionDef>> documentToAvailableFunctionDefinitions = this
				.getDocumentToAvailableFunctionDefinitions("A");

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

		File inputTestFile = this.getInputTestFile("A");

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

		functions.stream().forEach(f -> {
			assertNotNull(f);
			assertFalse(f.isHybrid());
		});
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
	@Ignore("This test is flaky.")
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
		assertFalse(function.getStatus().hasError());
		assertFalse(function.getHasPythonSideEffects());
		RefactoringStatusEntry entry = function.getStatus().getEntryMatchingCode(Function.PLUGIN_ID,
				PreconditionFailure.UNDETERMINABLE_SIDE_EFFECTS.getCode());
		assertNull(entry);
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
			assertFalse(func.getHasPythonSideEffects());
			RefactoringStatus status = func.getStatus();
			RefactoringStatusEntry entry = status.getEntryMatchingCode(Function.PLUGIN_ID,
					PreconditionFailure.UNDETERMINABLE_SIDE_EFFECTS.getCode());
			assertNull(entry);
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
			checkSideEffectStatus(func);
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
		checkSideEffectStatus(function);
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
			functionNames.add(func.getIdentifier());
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
			functionNames.add(func.getIdentifier());
		}

		// NOTE: Both of these functions have the same qualified name.
		assertEquals(1, functionNames.size());
	}

	private void testDifferentFileSameNameHelper(int expectedNumberOfFunctions, int expectedNumberOfFunctionNames, boolean expectedIsHybrid,
			boolean expectedHasTensorParameter) throws Exception {
		final String[] testFileNamesWithoutExtensions = { "A", "B" };
		Set<Function> functions = new HashSet<>();

		for (String fileName : testFileNamesWithoutExtensions) {
			Set<Function> functionsFromFile = this.getFunctions(fileName);
			assertNotNull(functionsFromFile);
			assertEquals(1, functionsFromFile.size());

			functions.addAll(functionsFromFile);
		}

		assertEquals(expectedNumberOfFunctions, functions.size());

		for (Function func : functions) {
			assertNotNull(func);

			assertEquals(expectedIsHybrid, func.isHybrid());
			assertEquals(expectedHasTensorParameter, func.getHasTensorParameter());
		}

		Set<String> functionNames = new HashSet<>();

		for (Function func : functions) {
			assertNotNull(func);
			functionNames.add(func.getIdentifier());
		}

		assertEquals(expectedNumberOfFunctionNames, functionNames.size());
	}

	private void testDifferentFileSameNameHelper(Set<FunctionUnderTest> functionsToTest, int expectedNumberOfFunctionNames)
			throws Exception {
		final String[] testFileNamesWithoutExtensions = { "A", "B" };
		Set<Function> functions = new HashSet<>();

		for (String fileName : testFileNamesWithoutExtensions) {
			Set<Function> functionsFromFile = this.getFunctions(fileName);
			assertNotNull(functionsFromFile);
			assertEquals(1, functionsFromFile.size());

			functions.addAll(functionsFromFile);
		}

		assertEquals(functionsToTest.size(), functions.size());

		for (Function func : functions) {
			assertNotNull(func);

			// find the corresponding FUT.
			FunctionUnderTest fut = null;
			int foundCount = 0;

			for (FunctionUnderTest funcUnderTest : functionsToTest) {
				if (funcUnderTest.getName().equals(func.getIdentifier())
						&& (funcUnderTest.getModuleName() == null || funcUnderTest.getModuleName().equals(func.getContainingModuleName()))
						&& funcUnderTest.getParameters().size() == func.getNumberOfParameters()) {
					// found it.
					fut = funcUnderTest;
					++foundCount;
				}
			}

			assertNotNull("Can't finding matching fuction test specification for: " + func + ".", fut);
			assertEquals("Ambiguous FUTs.", 1, foundCount);

			fut.compareTo(func);

			assertEquals(fut.isHybrid(), func.isHybrid());
			assertEquals(fut.getLikelyHasTensorParameter(), func.getHasTensorParameter());
		}

		Set<String> functionNames = new HashSet<>();

		for (Function func : functions) {
			assertNotNull(func);
			functionNames.add(func.getIdentifier());
		}

		assertEquals(expectedNumberOfFunctionNames, functionNames.size());
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName() throws Exception {
		testDifferentFileSameNameHelper(2, 2, false, false);
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName2() throws Exception {
		// NOTE: Both of these functions have the same qualified name.
		testDifferentFileSameNameHelper(2, 1, false, false);
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName3() throws Exception {
		// NOTE: Both of these functions have the same qualified name.
		testDifferentFileSameNameHelper(2, 1, false, false);
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName4() throws Exception {
		testDifferentFileSameNameHelper(2, 2, true, false);
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName5() throws Exception {
		testDifferentFileSameNameHelper(2, 2, true, true);
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName6() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		functionsToTest.add(new FunctionUnderTest("Test.b", true, false, "self"));
		functionsToTest.add(new FunctionUnderTest("Test2.b", false, false, "self"));

		testDifferentFileSameNameHelper(functionsToTest, 2);
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName7() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		functionsToTest.add(new FunctionUnderTest("Test.b", true, true, "self", "a"));
		functionsToTest.add(new FunctionUnderTest("Test2.b", true, false, "self", "a"));

		testDifferentFileSameNameHelper(functionsToTest, 2);
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName8() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		functionsToTest.add(new FunctionUnderTest("b", true, false, "a"));
		functionsToTest.add(new FunctionUnderTest("b", true, false));

		// NOTE: Both of these functions have the same qualified name.
		testDifferentFileSameNameHelper(functionsToTest, 1);
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName9() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		functionsToTest.add(new FunctionUnderTest("b", true, false, "a"));
		functionsToTest.add(new FunctionUnderTest("b", false, false));

		// NOTE: Both of these functions have the same qualified name.
		testDifferentFileSameNameHelper(functionsToTest, 1);
	}

	/**
	 * Tests #104. This simply tests whether two functions with the same names in different files are processed individually.
	 */
	@Test
	public void testDifferentFileSameName10() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		functionsToTest.add(new FunctionUnderTest("b", "A", true, true, "a"));
		functionsToTest.add(new FunctionUnderTest("b", "B", true, false, "a"));

		// NOTE: Both of these functions have the same qualified name.
		testDifferentFileSameNameHelper(functionsToTest, 1);
	}

	@Test
	public void testFunctionEquality() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		for (Function func : functions) {
			assertNotNull(func);
			String id = func.getIdentifier();
			assertNotNull(id);
			assertTrue(id.equals("a") || id.equals("b"));
		}

		Iterator<Function> iterator = functions.iterator();
		assertNotNull(iterator);
		assertTrue(iterator.hasNext());

		Function func1 = iterator.next();
		assertNotNull(func1);

		String identifer1 = func1.getIdentifier();
		assertNotNull(identifer1);

		assertTrue(iterator.hasNext());

		Function func2 = iterator.next();
		assertNotNull(func2);

		String identifer2 = func2.getIdentifier();
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
			String id = func.getIdentifier();
			assertNotNull(id);
			assertTrue(id.equals("a"));
		}

		Iterator<Function> iterator = functions.iterator();
		assertNotNull(iterator);
		assertTrue(iterator.hasNext());

		Function func1 = iterator.next();
		assertNotNull(func1);

		String identifer1 = func1.getIdentifier();
		assertNotNull(identifer1);

		assertTrue(iterator.hasNext());

		Function func2 = iterator.next();
		assertNotNull(func2);

		String identifer2 = func2.getIdentifier();
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

		String id = func.getIdentifier();
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

		assertFalse(function.getHasTensorParameter());
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

		assertFalse(function.getHasTensorParameter());
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

		assertFalse(function.getHasTensorParameter());
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

		assertFalse(function.getHasTensorParameter());
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

		assertFalse(function.getHasTensorParameter());
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
		assertTrue(function.getHybridizationParameters().isExperimentalFollowTypeHintsParamExists());

		argumentsType params = function.getParameters();

		// no params.
		assertEquals(0, params.args.length);

		assertFalse(function.getHasTensorParameter());
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
		assertTrue(function.getHybridizationParameters().isExperimentalFollowTypeHintsParamExists());
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

		assertFalse(function.getHasTensorParameter());
	}

	/**
	 * Test for #2. Here, the function has one parameters, is hybrid, and does not consider type hints. But, a type hint is supplied. In
	 * other words, a type hint supplied. We use it because of our option but not because of any hybridization parameters. Thus, it's likely
	 * to have a tensor parameter.
	 */
	@Test
	public void testHasLikelyTensorParameter8() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());
		assertFalse(function.getHybridizationParameters().isExperimentalFollowTypeHintsParamExists());

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

		assertTrue(function.getHasTensorParameter());
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
		assertTrue(function.getHybridizationParameters().isExperimentalFollowTypeHintsParamExists());

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

		assertTrue(function.getHasTensorParameter());
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
		assertTrue(function.getHybridizationParameters().isExperimentalFollowTypeHintsParamExists());

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

		// NOTE: Set to assertFalse() when #111 is fixed.
		assertTrue(function.getHasTensorParameter());
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
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

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.getHasTensorParameter());
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

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.getHasTensorParameter());
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

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.getHasTensorParameter());
		}
	}

	/**
	 * Test for #2. From https://tensorflow.org/guide/function#autograph_transformations.
	 */
	@Test
	public void testHasLikelyTensorParameter16() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		FunctionUnderTest functionToTest = new FunctionUnderTest("f", "x");
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
			fut.compareTo(function);

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.getHasTensorParameter());
		}
	}

	/**
	 * Test for #2. From https://www.tensorflow.org/guide/function#executing_python_side_effects. The parameters here are ints and not
	 * tensor-like.
	 */
	@Test
	public void testHasLikelyTensorParameter17() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		FunctionUnderTest functionToTest = new FunctionUnderTest("f", "x");
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
			fut.compareTo(function);

			assertFalse("Expecting " + function + " to not likely have a tensor-like parameter.", function.getHasTensorParameter());
		}
	}

	/**
	 * Test for #2. From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#features. Example with closures.
	 */
	@Test
	public void testHasLikelyTensorParameter18() throws Exception {
		Set<FunctionUnderTest> functionsToTest = new LinkedHashSet<>();

		FunctionUnderTest functionToTest = new FunctionUnderTest("f");
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
			fut.compareTo(function);

			// TODO: Not sure about this. Does WALA find closures? What really is the difference between having explicit parameters and
			// implicit ones? We still need to examine the calling contexts to get any info. Even though this function doesn't have a tensor
			// parameter, it probably should be hybridized because its free variables are tensors.
			assertFalse("Expecting " + function + " to not likely have a tensor-like parameter.", function.getHasTensorParameter());
		}
	}

	/**
	 * Test for #2. From https://tensorflow.org/guide/function#usage. The first parameter is not a tensor type.
	 */
	@Test
	public void testHasLikelyTensorParameter19() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());

		argumentsType params = function.getParameters();

		// three params.
		exprType[] actualParams = params.args;
		assertEquals(3, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("z", paramName);

		actualParameter = actualParams[1];
		assertNotNull(actualParameter);

		paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("a", paramName);

		actualParameter = actualParams[2];
		assertNotNull(actualParameter);

		paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("b", paramName);

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.zeros`.
	 */
	@Test
	public void testHasLikelyTensorParameter20() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter21() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter22() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.random.uniform`.
	 */
	@Test
	public void testHasLikelyTensorParameter23() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.SparseTensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter24() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.Tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter25() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.fill`.
	 */
	@Test
	public void testHasLikelyTensorParameter26() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter27() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.zero_likes`.
	 */
	@Test
	public void testHasLikelyTensorParameter28() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.one_hot`.
	 */
	@Test
	public void testHasLikelyTensorParameter29() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.convert_to_tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter30() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter31() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `ones`.
	 */
	@Test
	public void testHasLikelyTensorParameter32() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `ones`.
	 */
	@Test
	public void testHasLikelyTensorParameter33() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `random.uniform`.
	 */
	@Test
	public void testHasLikelyTensorParameter34() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `random.uniform`.
	 */
	@Test
	public void testHasLikelyTensorParameter35() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `random.uniform`.
	 */
	@Test
	public void testHasLikelyTensorParameter36() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter37() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter38() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter39() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter40() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter41() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter42() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter43() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter44() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `zeros`.
	 */
	@Test
	public void testHasLikelyTensorParameter45() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `SparseTensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter46() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `fill`.
	 */
	@Test
	public void testHasLikelyTensorParameter47() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `zeros_like`.
	 */
	@Test
	public void testHasLikelyTensorParameter48() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `one_hot`.
	 */
	@Test
	public void testHasLikelyTensorParameter49() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `convert_tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter50() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `range`.
	 */
	@Test
	public void testHasLikelyTensorParameter51() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter52() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Function functionToBeEvaluated = null;

		for (Function func : functions) {
			if (Objects.equals(func.getSimpleName(), "func2"))
				functionToBeEvaluated = func;
		}

		assertNotNull(functionToBeEvaluated);

		argumentsType params = functionToBeEvaluated.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("t", paramName);

		assertTrue("Expecting function with unlikely tensor parameter.", functionToBeEvaluated.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter53() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Function functionToBeEvaluated = null;

		for (Function func : functions) {
			if (Objects.equals(func.getSimpleName(), "func2"))
				functionToBeEvaluated = func;
		}

		assertNotNull(functionToBeEvaluated);

		argumentsType params = functionToBeEvaluated.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("t", paramName);

		assertTrue("Expecting function with unlikely tensor parameter.", functionToBeEvaluated.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter54() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Function functionToBeEvaluated = null;

		for (Function func : functions) {
			if (Objects.equals(func.getSimpleName(), "func2"))
				functionToBeEvaluated = func;
		}

		assertNotNull(functionToBeEvaluated);

		argumentsType params = functionToBeEvaluated.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("t", paramName);

		assertTrue("Expecting function with unlikely tensor parameter.", functionToBeEvaluated.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter55() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `Tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter56() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `SparseTensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter57() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `ones`.
	 */
	@Test
	public void testHasLikelyTensorParameter58() throws Exception {
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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	private void testHasLikelyTensorParameterHelper(boolean expectingHybridFunction, boolean expectingTensorParameter) throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertEquals(expectingHybridFunction, function.isHybrid());

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

		assertEquals(expectingTensorParameter, function.getHasTensorParameter());
	}

	private void testHasLikelyTensorParameterHelper() throws Exception {
		testHasLikelyTensorParameterHelper(false, true);
	}

	private void testHasLikelyTensorParameterHelper(boolean expectingHybridFunction) throws Exception {
		testHasLikelyTensorParameterHelper(expectingHybridFunction, true);
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter59() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter60() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter61() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter62() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter63() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter64() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter65() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter66() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter67() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter68() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter69() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter70() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter71() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter72() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter73() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_limits`.
	 */
	@Test
	public void testHasLikelyTensorParameter74() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_limits`.
	 */
	@Test
	public void testHasLikelyTensorParameter75() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_limits`.
	 */
	@Test
	public void testHasLikelyTensorParameter76() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_limits`.
	 */
	@Test
	public void testHasLikelyTensorParameter77() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_starts`.
	 */
	@Test
	public void testHasLikelyTensorParameter78() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_starts`.
	 */
	@Test
	public void testHasLikelyTensorParameter79() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter80() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter81() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter82() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter83() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter84() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter85() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter86() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.normal`.
	 */
	@Test
	public void testHasLikelyTensorParameter87() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter88() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter89() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.poisson`.
	 */
	@Test
	public void testHasLikelyTensorParameter90() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.poisson`.
	 */
	@Test
	public void testHasLikelyTensorParameter91() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.truncated_normal`.
	 */
	@Test
	public void testHasLikelyTensorParameter92() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `random.truncated_normal`.
	 */
	@Test
	public void testHasLikelyTensorParameter93() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `sparse.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter94() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `sparse.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter95() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `sparse.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter96() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `linalg.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter97() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `linalg.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter98() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `linalg.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter99() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter100() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter101() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter102() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter103() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter104() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter105() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter106() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter107() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter108() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `sparse.SparseTensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter109() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `sparse.SparseTensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter110() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `sparse.SparseTensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter111() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter112() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter113() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter114() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter115() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter116() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter117() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter118() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter119() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter120() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter121() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter122() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter123() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());

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

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.keras.layers.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter124() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `tf.keras.layers.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter125() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter126() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `tf.keras.layers.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter127() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter128() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter129() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter130() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Function functionToBeEvaluated = null;

		for (Function func : functions) {
			if (Objects.equals(func.getSimpleName(), "func2"))
				functionToBeEvaluated = func;
		}

		assertNotNull(functionToBeEvaluated);

		argumentsType params = functionToBeEvaluated.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("t", paramName);

		assertTrue("Expecting function with unlikely tensor parameter.", functionToBeEvaluated.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter131() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Function functionToBeEvaluated = null;

		for (Function func : functions) {
			if (Objects.equals(func.getSimpleName(), "func2"))
				functionToBeEvaluated = func;
		}

		assertNotNull(functionToBeEvaluated);

		argumentsType params = functionToBeEvaluated.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("t", paramName);

		assertTrue("Expecting function with unlikely tensor parameter.", functionToBeEvaluated.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter132() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Function functionToBeEvaluated = null;

		for (Function func : functions) {
			if (Objects.equals(func.getSimpleName(), "func2"))
				functionToBeEvaluated = func;
		}

		assertNotNull(functionToBeEvaluated);

		argumentsType params = functionToBeEvaluated.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("t", paramName);

		assertTrue("Expecting function with unlikely tensor parameter.", functionToBeEvaluated.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter133() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Function functionToBeEvaluated = null;

		for (Function func : functions) {
			if (Objects.equals(func.getSimpleName(), "func2"))
				functionToBeEvaluated = func;
		}

		assertNotNull(functionToBeEvaluated);

		argumentsType params = functionToBeEvaluated.getParameters();

		// one param.
		exprType[] actualParams = params.args;
		assertEquals(1, actualParams.length);

		exprType actualParameter = actualParams[0];
		assertNotNull(actualParameter);

		String paramName = NodeUtils.getRepresentationString(actualParameter);
		assertEquals("t", paramName);

		assertTrue("Expecting function with unlikely tensor parameter.", functionToBeEvaluated.getHasTensorParameter());
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter134() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter135() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but the tf.function has a parenthesis.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter136() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter137() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter138() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter139() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter140() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter141() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter142() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter143() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true);
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/265.
	 */
	@Test
	public void testHasLikelyTensorParameter144() throws Exception {
		testHasLikelyTensorParameterHelper(false, true);
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/265.
	 */
	@Test
	public void testHasLikelyTensorParameter145() throws Exception {
		testHasLikelyTensorParameterHelper();
	}

	/**
	 * Test lists.
	 */
	@Test
	public void testHasLikelyTensorParameter146() throws Exception {
		testHasLikelyTensorParameterHelper(false, true);
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/265.
	 */
	@Test
	public void testHasLikelyTensorParameter147() throws Exception {
		testHasLikelyTensorParameterHelper(false, false);
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/294. Control case.
	 */
	@Test
	public void testHasLikelyTensorParameter148() throws Exception {
		Function function = getFunction("add");

		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNull(function.getPassingPrecondition());
		assertTrue(function.getTransformations().isEmpty());
		assertTrue(function.getStatus().hasError());
		assertNotNull(function.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/294. No call.
	 */
	@Test
	public void testHasLikelyTensorParameter149() throws Exception {
		Function function = getFunction("add");

		assertTrue(function.isHybrid());
		assertNull(function.getHasTensorParameter());
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNull(function.getPassingPrecondition());
		assertTrue(function.getTransformations().isEmpty());
		assertTrue(function.getStatus().hasError());
		assertNotNull(function.getEntryMatchingFailure(UNDETERMINABLE_SIDE_EFFECTS));
		assertNotNull(function.getEntryMatchingFailure(CANT_APPROXIMATE_RECURSION));
		assertNotNull(function.getEntryMatchingFailure(UNDETERMINABLE_TENSOR_PARAMETER));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/283.
	 */
	@Test
	public void testHasLikelyTensorParameter150() throws Exception {
		Function function = getFunction("add");

		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNull(function.getPassingPrecondition());
		assertTrue(function.getTransformations().isEmpty());
		assertTrue(function.getStatus().hasError());
		assertNotNull(function.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/283.
	 */
	@Test
	public void testHasLikelyTensorParameter151() throws Exception {
		Function function = getFunction("add");

		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNotNull(function.getPassingPrecondition());
		assertEquals(P3, function.getPassingPrecondition());
		assertFalse(function.getTransformations().isEmpty());
		assertEquals(singleton(Transformation.CONVERT_TO_EAGER), function.getTransformations());
		assertFalse(function.getStatus().hasError());
		assertNull(function.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/283.
	 */
	@Test
	public void testHasLikelyTensorParameter152() throws Exception {
		Function function = getFunction("add");

		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNotNull(function.getPassingPrecondition());
		assertEquals(PreconditionSuccess.P3, function.getPassingPrecondition());
		assertFalse(function.getTransformations().isEmpty());
		assertEquals(singleton(CONVERT_TO_EAGER), function.getTransformations());
		assertFalse(function.getStatus().hasError());
		assertNull(function.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/283.
	 */
	@Test
	public void testHasLikelyTensorParameter153() throws Exception {
		Function function = getFunction("add");

		assertTrue(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertEquals(P2, function.getPassingPrecondition());
		assertEquals(1, function.getTransformations().size());
		assertEquals(CONVERT_TO_EAGER, function.getTransformations().iterator().next());
		assertFalse(function.getStatus().hasError());
		assertNull(function.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/283.
	 */
	@Test
	public void testHasLikelyTensorParameter154() throws Exception {
		Function function = getFunction("add");

		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNull(function.getPassingPrecondition());
		assertTrue(function.getTransformations().isEmpty());
		assertTrue(function.getStatus().hasError());
		assertNotNull(function.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/283.
	 */
	@Test
	public void testHasLikelyTensorParameter155() throws Exception {
		Function function = getFunction("add");

		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNull(function.getPassingPrecondition());
		assertTrue(function.getTransformations().isEmpty());
		assertTrue(function.getStatus().hasError());
		assertNotNull(function.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/283.
	 */
	@Test
	public void testHasLikelyTensorParameter156() throws Exception {
		Function function = getFunction("add");

		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertTrue(function.getHasPrimitiveParameter());
		// FIXME: This is a strange case. We use type hints for tf.Tensor on primitives. Since they're cast automatically, we shouldn't
		// consider this parameter as a primitive.
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNotNull(function.getPassingPrecondition());
		assertEquals(P3, function.getPassingPrecondition());
		assertFalse(function.getTransformations().isEmpty());
		assertEquals(singleton(CONVERT_TO_EAGER), function.getTransformations());
		assertFalse(function.getStatus().hasError());
		assertNull(function.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/265.
	 */
	@Test
	public void testHasLikelyTensorParameter157() throws Exception {
		testHasLikelyTensorParameterHelper(false, true);
	}

	/**
	 * Use the type hint here even though experimental_follow_type_hints isn't supplied.
	 */
	@Test
	public void testHasLikelyTensorParameter158() throws Exception {
		Function function = getFunction("add");
		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());

	}

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
		assertEquals("Expecting two functions.", 2, functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));

		// check function parameters.
		functions.forEach(f -> {
			String simpleName = f.getSimpleName();
			switch (simpleName) {
			case "__init__":
				assertFalse("Expecting " + simpleName + " to not have a tensor param.", f.getHasTensorParameter());
				assertFalse(f.isHybrid());
				assertTrue(f.getHasPythonSideEffects());
				checkOptimizationNotAvailableStatus(f);
				break;
			case "__call__":
				assertTrue("Expecting " + simpleName + " to have a tensor param.", f.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting function: " + simpleName + ".");
			}
		});
	}

	/**
	 * Test a model. No tf.function in this one. Use call instead of __call__. Ariadne doesn't support call. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/291.
	 */
	@Test
	public void testModel2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());
		assertEquals("Expecting two functions.", 2, functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));

		// check function parameters.
		functions.forEach(f -> {
			String simpleName = f.getSimpleName();
			switch (simpleName) {
			case "__init__":
				assertFalse("Expecting " + simpleName + " to not have a tensor param.", f.getHasTensorParameter());
				assertFalse(f.isHybrid());
				assertTrue(f.getHasPythonSideEffects());
				checkOptimizationNotAvailableStatus(f);
				break;
			case "call":
				assertTrue("Expecting " + simpleName + " to have a tensor param.", f.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting function: " + simpleName + ".");
			}
		});
	}

	/**
	 * Test a model. No tf.function in this one. Explicit call method.
	 */
	@Test
	public void testModel3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());
		assertEquals("Expecting two functions.", 2, functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));

		// check function parameters.
		functions.forEach(f -> {
			String simpleName = f.getSimpleName();
			switch (simpleName) {
			case "__init__":
				assertFalse("Expecting " + simpleName + " to not have a tensor param.", f.getHasTensorParameter());
				assertFalse(f.isHybrid());
				assertTrue(f.getHasPythonSideEffects());
				checkOptimizationNotAvailableStatus(f);
				break;
			case "call":
				assertTrue("Expecting " + simpleName + " to have a tensor param.", f.getHasTensorParameter());
				assertFalse("Should pass preconditions.", f.getStatus().hasError());
				assertFalse("No Python side-effects.", f.getHasPythonSideEffects());
				break;
			default:
				throw new IllegalStateException("Not expecting function: " + simpleName + ".");
			}
		});
	}

	private static void checkOptimizationNotAvailableStatus(Function f) {
		RefactoringStatus status = f.getStatus();
		assertTrue("Should not be available for optimization.", status.hasError());
		RefactoringStatusEntry noTensorsFailure = f.getEntryMatchingFailure(PreconditionFailure.HAS_NO_TENSOR_PARAMETERS);
		assertTrue(!f.isHybrid() || (noTensorsFailure != null && noTensorsFailure.isError()));
	}

	/**
	 * Test a model. No tf.function in this one. Explicit call method.
	 */
	@Test
	public void testModel4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());
		assertEquals("Expecting two functions.", 2, functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));

		// check function parameters.
		functions.forEach(f -> {
			String simpleName = f.getSimpleName();
			switch (simpleName) {
			case "__init__":
				assertFalse("Expecting " + simpleName + " to not have a tensor param.", f.getHasTensorParameter());
				assertFalse(f.isHybrid());
				assertTrue(f.getHasPythonSideEffects());
				checkOptimizationNotAvailableStatus(f);
				break;
			case "__call__":
				assertTrue("Expecting " + simpleName + " to have a tensor param.", f.getHasTensorParameter());
				assertFalse("Should pass preconditions.", f.getStatus().hasError());
				assertFalse("No Python side-effects.", f.getHasPythonSideEffects());
				break;
			default:
				throw new IllegalStateException("Not expecting function: " + simpleName + ".");
			}
		});
	}

	/**
	 * Test a model w/o client code (use contexts). No tf.function in this one.
	 */
	@Test
	public void testModel5() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());
		assertEquals("Expecting two functions.", 2, functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));

		// check function parameters.
		functions.forEach(f -> {
			String simpleName = f.getSimpleName();
			switch (simpleName) {
			case "__init__":
				assertFalse("Expecting " + simpleName + " to not have a tensor param.", f.getHasTensorParameter());
				assertFalse(f.isHybrid());
				assertTrue(f.getHasPythonSideEffects());
				checkOptimizationNotAvailableStatus(f);
				break;
			case "call":
				assertTrue("Expecting " + simpleName + " to have a tensor param.", f.getHasTensorParameter());
				// Can't infer side-effects here because there's no invocation of this method.
				checkSideEffectStatus(f);
				break;
			default:
				throw new IllegalStateException("Not expecting function: " + simpleName + ".");
			}
		});
	}

	/**
	 * Test a model w/o client code (use contexts). No tf.function in this one.
	 */
	@Test
	public void testModel6() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());
		assertEquals("Expecting two functions.", 2, functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));

		// check function parameters.
		functions.forEach(f -> {
			String simpleName = f.getSimpleName();
			switch (simpleName) {
			case "__init__":
				assertFalse("Expecting " + simpleName + " to not have a tensor param.", f.getHasTensorParameter());
				assertFalse(f.isHybrid());
				assertTrue(f.getHasPythonSideEffects());
				checkOptimizationNotAvailableStatus(f);
				break;
			case "__call__":
				assertTrue("Expecting " + simpleName + " to have a tensor param.", f.getHasTensorParameter());
				// No invocation, so we won't be able to infer side-effects.
				checkSideEffectStatus(f);
				break;
			default:
				throw new IllegalStateException("Not expecting function: " + simpleName + ".");
			}
		});
	}

	/**
	 * Test a model. No tf.function in this one. Explicit call method. Unlike testModel3, there are Python side-effects in
	 * SequentialModel.__init__() and SequentialModel.call().
	 *
	 * @see HybridizeFunctionRefactoringTest#testModel3
	 */
	@Test
	public void testModel7() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());
		assertEquals("Expecting two functions.", 3, functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));

		// check function parameters.
		functions.forEach(f -> {
			String simpleName = f.getSimpleName();
			switch (simpleName) {
			case "__init__":
				assertFalse("Expecting " + simpleName + " to not have a tensor param.", f.getHasTensorParameter());
				assertFalse(f.isHybrid());
				assertTrue(f.getHasPythonSideEffects());
				checkOptimizationNotAvailableStatus(f);
				break;
			case "get_stuff":
				assertFalse("Expecting " + simpleName + " to not have a tensor param.", f.getHasTensorParameter());
				assertFalse(f.isHybrid());
				assertFalse(f.getHasPythonSideEffects());
				checkOptimizationNotAvailableStatus(f);
				break;
			case "call":
				assertTrue("Expecting " + simpleName + " to have a tensor param.", f.getHasTensorParameter());
				assertTrue("Should have python side-effects.", f.getHasPythonSideEffects());
				break;
			default:
				throw new IllegalStateException("Not expecting function: " + simpleName + ".");
			}
		});
	}

	@Test
	public void testModel8() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertFalse(function.isHybrid());
	}

	@Test
	public void testModel9() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertFalse(function.isHybrid());
	}

	@Test
	public void testModel10() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertFalse(function.isHybrid());
	}

	@Test
	public void testModel11() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertFalse(function.isHybrid());
	}

	@Test
	public void testModel12() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertFalse(function.isHybrid());
	}

	@Test
	public void testModel13() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertFalse(function.isHybrid());
	}

	@Test
	public void testModel14() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertTrue(function.isHybrid());
	}

	@Test
	public void testModel15() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertTrue(function.isHybrid());
	}

	@Test
	public void testModel16() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertTrue(function.isHybrid());
	}

	@Test
	public void testModel17() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertTrue(function.isHybrid());
	}

	@Test
	public void testModel18() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertTrue(function.isHybrid());
	}

	@Test
	public void testModel19() throws Exception {
		Function function = this.getSingleFunction();
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
		assertFalse(function.getHasPythonSideEffects());
		assertTrue(function.isHybrid());
	}

	@Test
	public void testModel20() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());

		for (Function f : functions) {
			switch (f.getIdentifier()) {
			case "f":
				assertTrue(f.getHasTensorParameter());
				assertFalse(f.getHasPrimitiveParameter());
				assertFalse(f.getHasPythonSideEffects());
				assertFalse(f.isHybrid());
				break;
			case "g":
				assertTrue(f.getHasTensorParameter());
				assertFalse(f.getHasPrimitiveParameter());
				assertFalse(f.getHasPythonSideEffects());
				assertFalse(f.isHybrid());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + f + ".");
			}
		}
	}

	/**
	 * Test a model. No tf.function in this one. Has two functions that should be hybridized.
	 */
	@Test
	public void testModel21() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);

		LOG.info("Found functions: " + functions.size());
		assertEquals("Expecting three functions.", 3, functions.size());

		// no hybrids.
		assertTrue(functions.stream().map(Function::isHybrid).allMatch(b -> b == false));

		// check function parameters.
		functions.forEach(f -> {
			String simpleName = f.getSimpleName();
			switch (simpleName) {
			case "__init__":
				assertFalse("Expecting " + simpleName + " to not have a tensor param.", f.getHasTensorParameter());
				assertFalse(f.isHybrid());
				assertTrue(f.getHasPythonSideEffects());
				checkOptimizationNotAvailableStatus(f);
				break;
			case "__call__":
			case "call2":
				assertTrue("Expecting " + simpleName + " to have a tensor param.", f.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting function: " + simpleName + ".");
			}
		});
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229.
	 */
	@Test
	public void testModel22() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Set<Function> callFunctions = functions.stream().filter(f -> f.getSimpleName().equals("__call__")).collect(toSet());
		assertEquals(1, callFunctions.size());

		callFunctions.forEach(f -> {
			assertTrue(f.getHasTensorParameter());
			RefactoringStatusEntry entry = f.getStatus().getEntryMatchingCode(PLUGIN_ID, SPECULATIVE_ANALYSIS.getCode());
			assertNotNull(entry);
		});
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229.
	 */
	@Test
	public void testModel23() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Set<Function> callFunctions = functions.stream().filter(f -> f.getSimpleName().equals("call")).collect(toSet());
		assertEquals(1, callFunctions.size());

		callFunctions.forEach(f -> {
			assertTrue(f.getHasTensorParameter());
			RefactoringStatusEntry entry = f.getStatus().getEntryMatchingCode(PLUGIN_ID, SPECULATIVE_ANALYSIS.getCode());
			assertNotNull(entry);
		});
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229.
	 */
	@Test
	public void testModel24() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Set<Function> callFunctions = functions.stream().filter(f -> f.getSimpleName().equals("__call__")).collect(toSet());
		assertEquals(1, callFunctions.size());

		callFunctions.forEach(f -> {
			assertFalse(f.getHasTensorParameter());
			RefactoringStatusEntry entry = f.getStatus().getEntryMatchingCode(PLUGIN_ID, SPECULATIVE_ANALYSIS.getCode());
			assertNull(entry);
		});
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229.
	 */
	@Test
	public void testModel25() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Set<Function> callFunctions = functions.stream().filter(f -> f.getSimpleName().equals("__call__")).collect(toSet());
		assertEquals(1, callFunctions.size());

		callFunctions.forEach(f -> {
			assertTrue(f.getHasTensorParameter());
			RefactoringStatusEntry entry = f.getStatus().getEntryMatchingCode(PLUGIN_ID, SPECULATIVE_ANALYSIS.getCode());
			assertNotNull(entry);
		});
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229.
	 */
	@Test
	public void testModel26() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Set<Function> callFunctions = functions.stream().filter(f -> f.getSimpleName().equals("__call__")).collect(toSet());
		assertEquals(1, callFunctions.size());

		callFunctions.forEach(f -> {
			assertTrue(f.getHasTensorParameter());
			RefactoringStatusEntry entry = f.getStatus().getEntryMatchingCode(PLUGIN_ID, SPECULATIVE_ANALYSIS.getCode());
			assertNotNull(entry);
		});
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229.
	 */
	@Test
	public void testModel27() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Set<Function> callFunctions = functions.stream().filter(f -> f.getSimpleName().equals("__call__")).collect(toSet());
		assertEquals(1, callFunctions.size());

		callFunctions.forEach(f -> {
			assertTrue(f.getHasTensorParameter());
			RefactoringStatusEntry entry = f.getStatus().getEntryMatchingCode(PLUGIN_ID, SPECULATIVE_ANALYSIS.getCode());
			assertNotNull(entry);
		});
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229.
	 */
	@Test
	public void testModel28() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Set<Function> callFunctions = functions.stream().filter(f -> f.getSimpleName().equals("__call__")).collect(toSet());
		assertEquals(1, callFunctions.size());

		callFunctions.forEach(f -> {
			assertTrue(f.getHasTensorParameter());
			RefactoringStatusEntry entry = f.getStatus().getEntryMatchingCode(PLUGIN_ID, SPECULATIVE_ANALYSIS.getCode());
			assertNotNull(entry);
		});
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229.
	 */
	@Test
	public void testModel29() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		Set<Function> callFunctions = functions.stream().filter(f -> f.getSimpleName().equals("__call__")).collect(toSet());
		assertEquals(1, callFunctions.size());

		callFunctions.forEach(f -> {
			assertTrue(f.getHasTensorParameter());
			RefactoringStatusEntry entry = f.getStatus().getEntryMatchingCode(PLUGIN_ID, SPECULATIVE_ANALYSIS.getCode());
			assertNotNull(entry);
		});
	}

	// TODO: Test models that have tf.functions.

	private void testPreconditionCheckingHelper(boolean expectedHybrid, boolean expectedTensorParameter, Refactoring expectedRefactoring,
			Transformation expectedTransformation, PreconditionSuccess precondition) throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		assertEquals(expectedHybrid, function.isHybrid());
		assertEquals(expectedTensorParameter, function.getHasTensorParameter());

		assertEquals(expectedRefactoring, function.getRefactoring());

		Set<Transformation> transformationSet = function.getTransformations();
		assertNotNull(transformationSet);
		assertEquals(1, transformationSet.size());
		Iterator<Transformation> it = transformationSet.iterator();
		assertTrue(it.hasNext());
		Transformation transformation = it.next();
		assertEquals(expectedTransformation, transformation);

		assertEquals(precondition, function.getPassingPrecondition());
		assertFalse(function.getStatus().hasError());
	}

	@Test
	public void testPreconditionChecking() throws Exception {
		testPreconditionCheckingHelper(true, false, OPTIMIZE_HYBRID_FUNCTION, CONVERT_TO_EAGER, P2);
	}

	@Test
	public void testPreconditionChecking2() throws Exception {
		// it's not hybrid but it has a tensor parameter. Let's make it hybrid.
		testPreconditionCheckingHelper(false, true, Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID, Transformation.CONVERT_TO_HYBRID, P1);
	}

	@Test
	public void testPythonSideEffects() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter()); // the example uses a primitive type.
		assertTrue("Expecting a Python side-effect.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects2() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter()); // the example uses a primitive type.
		// there's a call to a TF operation. So, no "Python" side-effects.
		assertFalse("TF operations shouldn't be considered Python side-effects.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects3() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter()); // the example uses a primitive type.
		// there's a transitive Python side-effect.
		assertTrue("Expecting a Python side-effect from a transitive local variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects4() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter()); // the example uses a primitive type.
		// there's a Python statement but no side-effect.
		assertFalse("This Python statement only modifies a local variable, so no side-effects.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects5() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with side-effects.
		assertTrue("This Python statement modifies a global variable, so it has side-effects.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects6() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with side-effects. Multiple calls to the function.
		assertTrue("This Python statement modifies a global variable, so it has side-effects.", function.getHasPythonSideEffects());
	}

	/**
	 * Test transitive side-effects in the same file.
	 */
	@Test
	public void testPythonSideEffects7() throws Exception {
		Set<Function> functionSet = getFunctions();
		assertEquals(2, functionSet.size());
		testPythonSideEffects(functionSet);
	}

	private static void testPythonSideEffects(Set<Function> functionSet) {
		functionSet.forEach(f -> {
			assertFalse(f.isHybrid());
			assertFalse(f.getHasTensorParameter());

			switch (f.getIdentifier()) {
			case "f":
			case "g":
				// there's a Python statement with (transitive) side-effects.
				assertTrue("This Python statement modifies a global variable, so it has side-effects.", f.getHasPythonSideEffects());
				break;

			default:
				fail("Not expecting: " + f.getIdentifier() + ".");
				break;
			}
		});
	}

	private static void testPythonSideEffects(Map<Function, Boolean> functionToHasSideEffects) {
		functionToHasSideEffects.forEach((f, s) -> {
			assertFalse(f.isHybrid());
			assertFalse(f.getHasTensorParameter());
			assertEquals("Function: " + f + " should " + (s ? "" : "not ") + "have side-effects.", s, f.getHasPythonSideEffects());
		});
	}

	/**
	 * Returns the only function defined in the default test file.
	 *
	 * @return The only function defined in the default test file.
	 */
	private Function getSingleFunction() throws Exception {
		return getSingleFunction(this.getFunctions());
	}

	/**
	 * Returns the only function defined in the given test file.
	 *
	 * @param fileNameWithoutExtension The name of the file declaring the function without a file extension.
	 * @return The only function defined in the test file.
	 */
	private Function getSingleFunction(String fileNameWithoutExtension) throws Exception {
		return getSingleFunction(this.getFunctions(fileNameWithoutExtension));
	}

	/**
	 * Returns the only function contained in the given set of functions.
	 *
	 * @param functions The set of functions containing only one function.
	 * @return The sole function contained in the given set of functions.
	 */
	private static Function getSingleFunction(Set<Function> functions) {
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		return function;
	}

	/**
	 * Test transitive side-effects in different files.
	 */
	@Test
	public void testPythonSideEffects8() throws Exception {
		Function functionFromA = this.getSingleFunction("A");
		assertEquals("f", functionFromA.getIdentifier());

		Function functionFromB = this.getSingleFunction("B");
		assertEquals("g", functionFromB.getIdentifier());

		Set<Function> functionSet = new HashSet<>(Arrays.asList(functionFromA, functionFromB));
		testPythonSideEffects(functionSet);
	}

	/**
	 * Like testPythonSideEffects but only a single call. Simplifies the call graph since there seems to be a node for each call to a
	 * function.
	 *
	 * @see HybridizeFunctionRefactoringTest#testPythonSideEffects
	 */
	@Test
	public void testPythonSideEffects9() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter()); // the example uses a primitive type.
		assertTrue("Expecting a Python side-effect.", function.getHasPythonSideEffects());
	}

	/**
	 * Test write().
	 */
	@Test
	public void testPythonSideEffects10() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// NOTE: Switch to asserTrue when https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/273 is fixed.
		assertFalse("Not expecting a Python side-effect.", function.getHasPythonSideEffects());
	}

	/**
	 * Test writelines().
	 */
	@Test
	public void testPythonSideEffects11() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// NOTE: Switch to asserTrue when https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/273 is fixed.
		assertFalse("Not expecting a Python side-effect.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects12() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with no side-effects.
		assertFalse("This Python statement modifies a local variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects13() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with no side-effects.
		assertFalse("This Python statement uses a list comprehension to modify a local variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects14() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with no side-effects.
		assertFalse("This Python statement uses a lambda to modify a local variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects15() throws Exception {
		Function function = getSingleFunction();
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with no side-effects.
		assertFalse("This Python statement uses a loop to modify a local variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects16() throws Exception {
		Function function = getFunction("f");
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with side-effects.
		assertTrue("This Python statement uses a list comprehension to modify a global variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects17() throws Exception {
		Function function = getFunction("f");
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// NOTE: Switch to assertTrue when https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/274 is fixed.
		assertFalse("This Python statement uses a lambda to modify a global variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects18() throws Exception {
		Function f = getFunction("f");
		assertFalse(f.isHybrid());
		assertFalse(f.getHasTensorParameter());
		assertTrue(f.getHasPythonSideEffects());

		Function g = getFunction("g");
		assertTrue(g.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects19() throws Exception {
		Function f = getFunction("f");
		assertFalse(f.isHybrid());
		assertFalse(f.getHasTensorParameter());
		assertFalse(f.getHasPythonSideEffects());

		Function g = getFunction("g");
		assertFalse(g.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects20() throws Exception {
		Function f = getFunction("f");
		assertFalse(f.isHybrid());
		assertFalse(f.getHasTensorParameter());
		assertTrue("Function f() calls g(), which has Python side-effets. Thus, f() also has Python side-effects.",
				f.getHasPythonSideEffects());

		Function g = getFunction("g");
		assertTrue("Function g() modifies a global variable through the global keyword.", g.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects21() throws Exception {
		Function f = getFunction("f");
		assertFalse(f.isHybrid());
		assertFalse(f.getHasTensorParameter());
		assertFalse(f.getHasPythonSideEffects());

		Function g = getFunction("g");
		assertFalse(g.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects22() throws Exception {
		Set<Function> functionSet = getFunctions();

		for (Function f : functionSet) {
			assertFalse(f.isHybrid());
			assertFalse(f.getHasTensorParameter());
			assertFalse("This Python statement (transitively) uses a list comprehension to modify a local variable.",
					f.getHasPythonSideEffects());
		}
	}

	@Test
	public void testPythonSideEffects23() throws Exception {
		Set<Function> functionSet = getFunctions();

		for (Function function : functionSet) {
			switch (function.getIdentifier()) {
			case "f":
			case "g":
			case "fun_with_side_effects":
				assertFalse(function.isHybrid());
				assertFalse(function.getHasTensorParameter());
				// there's a Python statement with side-effects.
				assertTrue("This Python statement (transitively) uses a list comprehension to modify a global variable.",
						function.getHasPythonSideEffects());
				break;
			case "h":
				assertFalse(function.getHasPythonSideEffects());
				break;
			default:
				throw new IllegalStateException("Unknown function: " + function + ".");
			}
		}
	}

	@Test
	public void testPythonSideEffects24() throws Exception {
		Function function = getFunction("f");
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		assertFalse("This Python statement (transitively) uses a lambda to modify a local variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects25() throws Exception {
		Function function = getFunction("f");
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with no side-effects.
		assertFalse("This Python statement (transitively) uses a loop to modify a local variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects26() throws Exception {
		Function function = getFunction("f");
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with side-effects.
		assertTrue("A loop to modifies a global variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects27() throws Exception {
		Function function = getFunction("f");
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// there's a Python statement with side-effects.
		assertTrue("A loop to modifies a global variable.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects28() throws Exception {
		Map<String, Set<Function>> map = this.getFunctions().stream()
				.collect(Collectors.groupingBy(Function::getIdentifier, Collectors.toSet()));

		assertEquals(2, map.size());

		map.get("f").stream().map(Function::getHasPythonSideEffects).forEach(s -> assertFalse(s));
		map.get("g").stream().map(Function::getHasPythonSideEffects).forEach(s -> assertTrue(s));
	}

	@Test
	public void testPythonSideEffects29() throws Exception {
		Function f = getFunction("f");
		assertTrue("Function f() calls g(), which has Python side-effets. Thus, f() also has Python side-effects.",
				f.getHasPythonSideEffects());

		Function g = getFunction("g");
		assertTrue("Function g() modifies a global variable through the global keyword.", g.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects30() throws Exception {
		Function f = getFunction("f");
		assertFalse("Removed the global keyword from g().", f.getHasPythonSideEffects());

		Function g = getFunction("g");
		assertFalse("Function g() modifies a lobal variable (removed the global keyword).", g.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects31() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPythonSideEffects());

		Function g = getFunction("g");
		assertTrue(g.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects32() throws Exception {
		Function f = getFunction("f");
		assertFalse(f.getHasPythonSideEffects());

		Function g = getFunction("g");
		assertFalse(g.getHasPythonSideEffects());

		Function h = this.getFunction("h");
		assertTrue(h.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects33() throws Exception {
		Function g = getFunction("g");
		assertFalse("g() only returns the parameter.", g.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects34() throws Exception {
		Function f = getFunction("f");
		assertFalse(f.getHasPythonSideEffects());

		Function g = getFunction("g");
		assertFalse("g() modifies a copy of a parameter.", g.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects35() throws Exception {
		Function function = getFunction("side_effect");

		assertTrue(function.isHybrid());
		assertFalse("side_effect() is passed an integer (from docs).", function.getHasTensorParameter());
		assertTrue("side_effect() modifies a global list.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects36() throws Exception {
		Function function = getFunction("side_effect");

		assertTrue(function.isHybrid());
		assertFalse("side_effect() is passed an integer (from docs).", function.getHasTensorParameter());
		assertTrue("side_effect() modifies a global list.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects37() throws Exception {
		Function function = getFunction("no_side_effect");

		assertTrue(function.isHybrid());
		assertFalse("no_side_effect() is passed an integer (from docs).", function.getHasTensorParameter());
		assertFalse("no_side_effect() doesn't modifies a global list.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects38() throws Exception {
		Function function = getFunction("Model.__call__");
		assertNotNull(function);

		assertTrue(function.isHybrid());
		assertFalse(function.getHasTensorParameter()); // self isn't a tensor parameter.
		assertTrue(function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects39() throws Exception {
		Function function = getFunction("Model.__call__");
		assertNotNull(function);

		assertTrue(function.isHybrid());
		assertFalse(function.getHasTensorParameter()); // self isn't a tensor parameter.
		assertTrue(function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects40() throws Exception {
		Function function = getFunction("buggy_consume_next");

		assertTrue(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		// TODO: Change to assertTrue() when https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/278 is fixed:
		assertFalse("next() moves the iterator's cursor, and the iterator is over a list.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects41() throws Exception {
		Function function = getFunction("good_consume_next");

		assertTrue(function.isHybrid());
		assertFalse("iterator still isn't a tensor. I wonder if you get speedup from that.", function.getHasTensorParameter());
		assertFalse("next() moves the iterator's cursor, but the iterator is over a dataset.", function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects42() throws Exception {
		Function function = getFunction("leaky_function");

		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertTrue(function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects43() throws Exception {
		Function function = getFunction("leaky_function");

		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects44() throws Exception {
		Function function = getFunction("leaky_function");

		assertFalse(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPythonSideEffects());

		assertFalse(function.getStatus().hasError());
		assertTrue(function.getRefactoring() == Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID);
		assertTrue(function.getPassingPrecondition() == PreconditionSuccess.P1);
		assertEquals(Collections.singleton(Transformation.CONVERT_TO_HYBRID), function.getTransformations());
	}

	@Test
	public void testPythonSideEffects45() throws Exception {
		Function function = getFunction("leaky_function");

		assertTrue(function.isHybrid());
		// This is a hybrid function, so the refactoring should be OPTIMIZE_HYBRID_FUNCTION.
		assertEquals(Refactoring.OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());

		assertTrue(function.getHasTensorParameter());
		// In table 2, we need it not to have a tensor parameter to de-hybridize, so this is a "failure."
		assertTrue(function.getEntryMatchingFailure(PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS).isError());

		assertTrue(function.getHasPythonSideEffects());
		// We also can't de-hybridize if it has Python side-effects. So, that's an error.
		assertTrue(function.getEntryMatchingFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS).isError());
		// Also, we have a hybrid function with Python side-effects. Let's warn about that.
		assertEquals(1, Arrays.stream(function.getStatus().getEntries()).map(RefactoringStatusEntry::getSeverity)
				.filter(s -> s == RefactoringStatus.WARNING).count());

		assertNull(function.getPassingPrecondition());
		assertTrue(function.getTransformations().isEmpty());
	}

	@Test
	public void testPythonSideEffects46() throws Exception {
		Function function = getFunction("leaky_function");

		assertFalse(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertTrue(function.getHasPythonSideEffects());

		RefactoringStatus status = function.getStatus();

		// We have an eager function with a tensor parameter but Python side-effects. Should be a P1 failure.
		assertFalse(status.isOK());
		assertEquals(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS.getCode(), status.getEntryWithHighestSeverity().getCode());
		assertEquals(Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID, function.getRefactoring());
		assertNull(function.getPassingPrecondition());
		assertEquals(Collections.emptySet(), function.getTransformations());
	}

	@Test
	public void testPythonSideEffects47() throws Exception {
		Function leakyFunction = getFunction("leaky_function");

		assertTrue(leakyFunction.isHybrid());
		assertTrue(leakyFunction.getHasTensorParameter());
		assertTrue(leakyFunction.getHasPythonSideEffects());

		Function capturesLeakedTensor = getFunction("captures_leaked_tensor");

		assertTrue(capturesLeakedTensor.isHybrid());
		assertTrue(capturesLeakedTensor.getHasTensorParameter());

		// NOTE: This function doesn't have Python side-effects, but it does capture a "leaky" tensor. See
		// https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/281.
		assertFalse(capturesLeakedTensor.getHasPythonSideEffects());

		assertFalse(capturesLeakedTensor.getStatus().isOK());
		assertTrue(capturesLeakedTensor.getStatus().hasError());
		assertFalse(capturesLeakedTensor.getStatus().hasFatalError());
		RefactoringStatusEntry error = capturesLeakedTensor.getStatus().getEntryMatchingSeverity(RefactoringStatus.ERROR);
		assertTrue(error.isError());
		assertEquals(PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS.getCode(), error.getCode());

		// NOTE: Change to assertEquals(..., 1, ...) once https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/281 is fixed.
		assertEquals("We should warn that the hybrid function is capturing leaked tensors.", 0,
				Arrays.stream(capturesLeakedTensor.getStatus().getEntries()).map(RefactoringStatusEntry::getSeverity)
						.filter(s -> s == RefactoringStatus.WARNING).count());

		assertNotNull(capturesLeakedTensor.getRefactoring());
		assertEquals("P2 \"failure.\"", Refactoring.OPTIMIZE_HYBRID_FUNCTION, capturesLeakedTensor.getRefactoring());
		assertNull(capturesLeakedTensor.getPassingPrecondition());
		assertTrue(capturesLeakedTensor.getTransformations().isEmpty());

		long warningCount = Arrays.stream(capturesLeakedTensor.getStatus().getEntries())
				.filter(e -> e.getSeverity() == RefactoringStatus.WARNING).count();

		// NOTE: Change to assertEquals(..., 1, ...) when https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/281 is fixed.
		// NOTE: Add assertEquals(RefactoringStatus.WARNING, entry.getSeverity()) when
		// https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/281 is fixed.
		assertEquals("Warn about a hybrid function that leaks as a potential tensor.", 0, warningCount);
	}

	@Test
	public void testPythonSideEffects48() throws Exception {
		Function leakyFunction = getFunction("leaky_function");

		assertTrue(leakyFunction.isHybrid());
		assertTrue(leakyFunction.getHasTensorParameter());
		assertTrue(leakyFunction.getHasPythonSideEffects());

		Function capturesLeakedTensor = getFunction("captures_leaked_tensor");

		assertFalse(capturesLeakedTensor.isHybrid());
		assertTrue(capturesLeakedTensor.getHasTensorParameter());

		// NOTE: This function doesn't have Python side-effects, but it does capture a "leaky" tensor. See
		// https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/281.
		assertFalse(capturesLeakedTensor.getHasPythonSideEffects());

		// NOTE: Change to assertFalse once https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/281 is fixed.
		assertFalse("Passes P1.", capturesLeakedTensor.getStatus().hasError());

		assertFalse(capturesLeakedTensor.getStatus().hasWarning());
		// NOTE: Change to assertTrue once https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/281 is fixed.
		assertFalse(capturesLeakedTensor.getStatus().hasError());

		assertNotNull(capturesLeakedTensor.getRefactoring());
		assertEquals("We shouldn't refactor this but we do currently. Nevertheless, the refactoring kind should remain intact.",
				Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID, capturesLeakedTensor.getRefactoring());

		// NOTE: Change to assertNull once https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/281 is fixed.
		assertNotNull(capturesLeakedTensor.getPassingPrecondition());
		assertEquals("We really shouldn't refactor this.", capturesLeakedTensor.getPassingPrecondition(), PreconditionSuccess.P1);

		// NOTE: Change to assertTrue once https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/281 is fixed.
		assertFalse(capturesLeakedTensor.getTransformations().isEmpty());
		assertEquals("We really shouldn't transform this.", Collections.singleton(Transformation.CONVERT_TO_HYBRID),
				capturesLeakedTensor.getTransformations());
	}

	@Test
	public void testPythonSideEffects49() throws Exception {
		Function function = getFunction("leaky_function");

		assertFalse(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertTrue(function.getHasPythonSideEffects());

		// This is a P1 failure.
		RefactoringStatus status = function.getStatus();
		assertFalse(status.isOK());
		assertTrue(status.hasError());
		assertFalse(status.hasFatalError());

		RefactoringStatusEntry entry = status.getEntryWithHighestSeverity();
		assertEquals(RefactoringStatus.ERROR, entry.getSeverity());
		assertEquals(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS.getCode(), entry.getCode());
		assertEquals(function, entry.getData());

		assertEquals(Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID, function.getRefactoring());
		assertNull(function.getPassingPrecondition());
		assertTrue(function.getTransformations().isEmpty());
	}

	@Test
	public void testPythonSideEffects50() throws Exception {
		Function function = getFunction("leaky_function");

		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		assertTrue(function.getHasPythonSideEffects());

		RefactoringStatus status = function.getStatus();
		assertFalse("This is a P1 failure.", status.isOK());
		assertTrue(status.hasError());
		assertFalse(status.hasFatalError());

		RefactoringStatusEntry[] statusEntries = status.getEntries();
		assertEquals(2, statusEntries.length);

		assertTrue(Arrays.stream(statusEntries).map(RefactoringStatusEntry::getSeverity)
				.allMatch(s -> Objects.equals(s, RefactoringStatus.ERROR)));

		assertTrue(Arrays.stream(statusEntries).map(RefactoringStatusEntry::getData).allMatch(d -> Objects.equals(d, function)));

		Map<Integer, List<RefactoringStatusEntry>> codeToEntry = Arrays.stream(statusEntries)
				.collect(Collectors.groupingBy(RefactoringStatusEntry::getCode));

		assertEquals(1, codeToEntry.get(PreconditionFailure.HAS_NO_TENSOR_PARAMETERS.getCode()).size());
		assertEquals(1, codeToEntry.get(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS.getCode()).size());

		assertEquals(Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID, function.getRefactoring());
		assertNull(function.getPassingPrecondition());
		assertTrue(function.getTransformations().isEmpty());
	}

	@Test
	public void testPythonSideEffects51() throws Exception {
		Function function = getFunction("leaky_function");

		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		assertTrue(function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects52() throws Exception {
		Function function = getFunction("leaky_function");

		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		assertTrue(function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects53() throws Exception {
		Function function = getFunction("not_leaky_function");

		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		assertFalse(function.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects54() throws Exception {
		Function function = getFunction("leaky_function");

		assertTrue(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		assertTrue(function.getHasPythonSideEffects());

		RefactoringStatus status = function.getStatus();
		assertTrue("We can't convert something to eager if it has side-effects because that will alter semantics.", status.hasError());
		assertEquals(2, Arrays.stream(status.getEntries()).filter(e -> !e.isInfo()).count());
		assertEquals(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS.getCode(), status.getEntryWithHighestSeverity().getCode());

		assertEquals(Refactoring.OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNull(function.getPassingPrecondition());
		assertTrue(function.getTransformations().isEmpty());
	}

	@Test
	public void testPythonSideEffects55() throws Exception {
		Function function = getFunction("leaky_function");

		assertTrue(function.isHybrid());
		assertFalse(function.getHasTensorParameter());
		assertFalse(function.getHasPythonSideEffects());

		RefactoringStatus status = function.getStatus();
		assertFalse("We can convert something to eager if it does not have side-effects because that will not alter semantics.",
				status.hasError());
		assertEquals(0, Arrays.stream(status.getEntries()).filter(RefactoringStatusEntry::isError).count());

		assertEquals(Refactoring.OPTIMIZE_HYBRID_FUNCTION, function.getRefactoring());
		assertNotNull(function.getPassingPrecondition());
		assertEquals(PreconditionSuccess.P2, function.getPassingPrecondition());
		assertFalse(function.getTransformations().isEmpty());
		assertEquals(Collections.singleton(Transformation.CONVERT_TO_EAGER), function.getTransformations());
	}

	@Test
	public void testPythonSideEffects56() throws Exception {
		Function f = getFunction("f");
		assertFalse("Keyword argument assignments shouldn't be considered as heap writes.", f.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects57() throws Exception {
		Function f = getFunction("f");
		assertFalse("Embedded functions aren't side-effects.", f.getHasPythonSideEffects());
	}

	@Test
	public void testPythonSideEffects58() throws Exception {
		Function f = getFunction("f");
		assertFalse("Decorated embedded functions aren't side-effects.", f.getHasPythonSideEffects());
	}

	/**
	 * Test transitive side-effects in different files. Unlike testPythonSideEffects8, the import is different. This is actually a test of
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	@Ignore("Workaround https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/374.")
	public void testPythonSideEffects59() throws Exception {
		Function functionFromA = this.getSingleFunction("A");
		assertEquals("f", functionFromA.getIdentifier());

		Function functionFromB = this.getSingleFunction("B");
		assertEquals("g", functionFromB.getIdentifier());

		Set<Function> functionSet = new HashSet<>(Arrays.asList(functionFromA, functionFromB));
		Map<Function, Boolean> functionToExpectedSideEffects = new HashMap<>();

		for (Function function : functionSet) {
			switch (function.getIdentifier()) {
			case "f":
			case "g":
				functionToExpectedSideEffects.put(function, true);
				break;
			default:
				fail("Not expecting: " + function + ".");
			}
		}

		testPythonSideEffects(functionToExpectedSideEffects);
	}

	/**
	 * Test transitive side-effects in different files. Unlike testPythonSideEffects8, the import is different. This is actually a test of
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	@Ignore("Workaround https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/374.")
	public void testPythonSideEffects60() throws Exception {
		Function functionFromA = this.getSingleFunction("A");
		assertEquals("f", functionFromA.getIdentifier());

		Function functionFromB = this.getSingleFunction("B");
		assertEquals("C.g", functionFromB.getIdentifier());

		Set<Function> functionSet = new HashSet<>(Arrays.asList(functionFromA, functionFromB));
		Map<Function, Boolean> functionToExpectedSideEffects = new HashMap<>();

		for (Function function : functionSet) {
			switch (function.getIdentifier()) {
			case "f":
			case "C.g":
				functionToExpectedSideEffects.put(function, true);
				break;
			default:
				fail("Not expecting: " + function + ".");
			}
		}

		testPythonSideEffects(functionToExpectedSideEffects);
	}

	/**
	 * Test transitive side-effects in different files. Unlike testPythonSideEffects8, the import is different. This is actually a test of
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	@Ignore("Workaround https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/374.")
	public void testPythonSideEffects61() throws Exception {
		Function functionFromA = this.getSingleFunction("A");
		assertEquals("f", functionFromA.getIdentifier());

		Function functionFromB = this.getSingleFunction("B");
		assertEquals("C.__init__", functionFromB.getIdentifier());

		Set<Function> functionSet = new HashSet<>(Arrays.asList(functionFromA, functionFromB));
		Map<Function, Boolean> functionToExpectedSideEffects = new HashMap<>();

		for (Function function : functionSet) {
			switch (function.getIdentifier()) {
			case "f":
			case "C.__init__":
				functionToExpectedSideEffects.put(function, true);
				break;
			default:
				fail("Not expecting: " + function + ".");
			}
		}

		testPythonSideEffects(functionToExpectedSideEffects);
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	public void testPythonSideEffects62() throws Exception {
		Function function = this.getSingleFunction("A");
		assertEquals("f", function.getIdentifier());
		assertFalse(function.getHasPythonSideEffects());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	public void testPythonSideEffects63() throws Exception {
		Function function = this.getSingleFunction("A");
		assertEquals("f", function.getIdentifier());
		assertFalse(function.getHasPythonSideEffects());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	public void testPythonSideEffects64() throws Exception {
		Function function = this.getSingleFunction("A");
		assertEquals("f", function.getIdentifier());
		assertTrue(function.getHasPythonSideEffects());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	public void testPythonSideEffects65() throws Exception {
		Function function = this.getSingleFunction("A");
		assertEquals("f", function.getIdentifier());
		assertTrue(function.getHasPythonSideEffects());
	}

	@Test
	public void testRecursion() throws Exception {
		Function f = getFunction("recursive_fn");

		assertTrue(f.isRecursive());

		assertFalse(f.isHybrid());
		assertEquals(Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID, f.getRefactoring());

		assertTrue("No recursive functions.", f.getStatus().hasError());
		assertTrue(f.getEntryMatchingFailure(IS_RECURSIVE).isError());
	}

	@Test
	public void testRecursion2() throws Exception {
		Function f = getFunction("not_recursive_fn");

		assertFalse(f.isHybrid()); // eag.
		assertEquals(Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID, f.getRefactoring());

		assertTrue(f.getHasTensorParameter()); // T.
		assertNull(f.getEntryMatchingFailure(PreconditionFailure.HAS_NO_TENSOR_PARAMETERS));

		assertFalse(f.getHasPythonSideEffects()); // F.
		assertNull(f.getEntryMatchingFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS));

		assertFalse(f.isRecursive()); // F.
		assertNull(f.getEntryMatchingFailure(PreconditionFailure.IS_RECURSIVE));

		assertFalse(f.getStatus().hasError());
		assertEquals(P1, f.getPassingPrecondition());
		assertEquals(Collections.singleton(CONVERT_TO_HYBRID), f.getTransformations());
	}

	@Test
	public void testRecursion3() throws Exception {
		Function f = getFunction("recursive_fn");
		assertEquals(Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID, f.getRefactoring());

		assertTrue(f.isRecursive());
		assertTrue("No (transitively) recursive functions.", f.getStatus().hasError());
		assertTrue(f.getEntryMatchingFailure(IS_RECURSIVE).isError());
	}

	@Test
	public void testRecursion4() throws Exception {
		Function f = getFunction("recursive_fn");

		assertTrue(f.isHybrid()); // hyb.
		assertEquals(Refactoring.OPTIMIZE_HYBRID_FUNCTION, f.getRefactoring());

		assertTrue(f.getHasTensorParameter()); // T.
		assertTrue(f.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS).isError());

		assertFalse(f.getHasPythonSideEffects()); // F.
		assertNull(f.getEntryMatchingFailure(HAS_PYTHON_SIDE_EFFECTS));

		assertTrue(f.isRecursive()); // T.
		assertNull(f.getEntryMatchingFailure(IS_RECURSIVE));

		assertEquals("We have a recursive hybrid function with a tensor parameter. Warn.", 1, f.getWarnings().size());

	}

	@Test
	public void testRecursion5() throws Exception {
		Function f = getFunction("not_recursive_fn");

		assertTrue(f.isHybrid());
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, f.getRefactoring());

		assertTrue(f.getHasTensorParameter());
		assertFalse("Already optimal.", f.getStatus().isOK());
		assertTrue(f.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS).isError());

		assertFalse(f.getHasPythonSideEffects()); // F.
		assertNull(f.getEntryMatchingFailure(HAS_PYTHON_SIDE_EFFECTS));

		assertFalse(f.isRecursive()); // F.
		assertNull(f.getEntryMatchingFailure(IS_RECURSIVE));

		assertTrue("We have a non-recursive hybrid function with a tensor parameter. No warning.", f.getWarnings().isEmpty());

	}

	@Test
	public void testRecursion6() throws Exception {
		Function f = getFunction("recursive_fn");

		assertTrue(f.isHybrid()); // hyb.
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, f.getRefactoring());

		assertTrue(f.getHasTensorParameter()); // T.
		assertFalse("Already optimal.", f.getStatus().isOK());
		assertTrue(f.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS).isError());

		assertFalse(f.getHasPythonSideEffects()); // F.
		assertNull(f.getEntryMatchingFailure(HAS_PYTHON_SIDE_EFFECTS));

		assertTrue(f.isRecursive()); // T.
		assertNull(f.getEntryMatchingFailure(IS_RECURSIVE));

		assertEquals("We have a recursive hybrid function with a tensor parameter. Warn.", 1, f.getWarnings().size());
	}

	@Test
	public void testRecursion7() throws Exception {
		Function f = getFunction("recursive_fn");

		assertFalse(f.isHybrid());
		assertEquals(CONVERT_EAGER_FUNCTION_TO_HYBRID, f.getRefactoring());

		assertFalse(f.getHasTensorParameter());
		assertTrue(f.getEntryMatchingFailure(HAS_NO_TENSOR_PARAMETERS).isError());

		assertTrue(f.getHasPythonSideEffects()); // T.
		assertTrue(f.getEntryMatchingFailure(HAS_PYTHON_SIDE_EFFECTS).isError());

		assertTrue(f.isRecursive());
		assertTrue("No recursive functions.", f.getStatus().hasError());
		assertTrue(f.getEntryMatchingFailure(IS_RECURSIVE).isError());

		assertTrue(f.getWarnings().isEmpty());
	}

	@Test
	public void testRecursion8() throws Exception {
		Function f = getFunction("recursive_fn");

		assertTrue(f.isHybrid()); // hyb.
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, f.getRefactoring());

		assertFalse(f.getHasTensorParameter()); // F.
		assertNull(f.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
		assertNull("Not having tensor parameters is not a failure for: " + OPTIMIZE_HYBRID_FUNCTION + ".",
				f.getEntryMatchingFailure(HAS_NO_TENSOR_PARAMETERS));

		assertTrue(f.getHasPythonSideEffects()); // T.
		assertTrue(f.getEntryMatchingFailure(HAS_PYTHON_SIDE_EFFECTS).isError());

		assertTrue(f.isRecursive()); // T.
		assertNull("Because there is no tensor parameter, it doesn't matter if it's recursive or not.",
				f.getEntryMatchingFailure(IS_RECURSIVE));

		assertEquals("No tensor parameter. No warning. The warning currently is from side-effects", 1, f.getWarnings().size());
	}

	@Test
	public void testRecursion9() throws Exception {
		Function f = getFunction("recursive_fn");

		assertFalse(f.isHybrid());
		assertEquals(CONVERT_EAGER_FUNCTION_TO_HYBRID, f.getRefactoring());

		assertFalse(f.getHasTensorParameter());
		assertTrue(f.getEntryMatchingFailure(HAS_NO_TENSOR_PARAMETERS).isError());

		assertFalse(f.getHasPythonSideEffects());
		assertNull(f.getEntryMatchingFailure(HAS_PYTHON_SIDE_EFFECTS));

		assertTrue(f.isRecursive());
		assertTrue("No recursive functions.", f.getStatus().hasError());
		assertTrue(f.getEntryMatchingFailure(IS_RECURSIVE).isError());

		assertTrue(f.getWarnings().isEmpty());
	}

	@Test
	public void testRecursion10() throws Exception {
		Function f = getFunction("recursive_fn");

		assertTrue(f.isHybrid()); // hyb
		assertEquals(OPTIMIZE_HYBRID_FUNCTION, f.getRefactoring());

		assertFalse(f.getHasTensorParameter()); // F.
		assertNull(f.getEntryMatchingFailure(HAS_NO_TENSOR_PARAMETERS));
		assertNull(f.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));

		assertFalse(f.getHasPythonSideEffects()); // F.
		assertNull(f.getEntryMatchingFailure(HAS_PYTHON_SIDE_EFFECTS));

		assertTrue(f.isRecursive());
		assertNull("Because there is no tensor parameter, it doesn't matter if it's recursive or not.",
				f.getEntryMatchingFailure(IS_RECURSIVE));

		assertTrue(f.getWarnings().isEmpty());

		assertEquals(P2, f.getPassingPrecondition());
		assertEquals(Collections.singleton(CONVERT_TO_EAGER), f.getTransformations());
	}

	@Test
	public void testRecursion11() throws Exception {
		Function f = getFunction("recursive_fn");
		assertEquals(Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID, f.getRefactoring());

		assertTrue(f.isRecursive());
		assertTrue("No (transitively) recursive functions.", f.getStatus().hasError());
		assertTrue(f.getEntryMatchingFailure(IS_RECURSIVE).isError());
	}

	// TODO: Left off at https://www.tensorflow.org/guide/function#depending_on_python_global_and_free_variables.

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/280.
	 */
	@Test
	public void testCallback() throws Exception {
		Function f = getFunction("replica_fn");
		assertTrue(f.getHasTensorParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter() throws Exception {
		Function f = getFunction("f");
		assertFalse("This function has no parameters.", f.getHasTensorParameter());
		assertFalse("This function has no parameters.", f.getHasPrimitiveParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter2() throws Exception {
		Function f = getFunction("f");
		assertFalse("This function has one parameter.", f.getHasTensorParameter());
		assertTrue("This function has one parameter.", f.getHasPrimitiveParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter3() throws Exception {
		Function f = getFunction("f");
		assertTrue("This function has one (tensor) parameter.", f.getHasTensorParameter());
		assertFalse("This function has one (tensor) parameter.", f.getHasPrimitiveParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter4() throws Exception {
		Function f = getFunction("f");
		assertTrue("This function has one tensor parameter and one non-tensor parameter.", f.getHasTensorParameter());
		assertTrue("This function has one tensor parameter and one non-tensor parameter.", f.getHasPrimitiveParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter5() throws Exception {
		Function f = getFunction("f");
		assertTrue("This function has one parameter with one tensor argument and one non-tensor argument.", f.getHasTensorParameter());
		assertTrue("This function has one parameter with one tensor argument and one non-tensor argument.", f.getHasPrimitiveParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter6() throws Exception {
		Function f = getFunction("f");
		assertTrue("This function has one parameter with one tensor argument and one non-tensor argument.", f.getHasTensorParameter());
		assertTrue("This function has one parameter with one tensor argument and one non-tensor argument.", f.getHasPrimitiveParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter7() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasTensorParameter());
		assertTrue(f.getHasPrimitiveParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter8() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
		assertFalse(f.getHasTensorParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter9() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
		assertFalse(f.getHasTensorParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter10() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
		assertFalse(f.getHasTensorParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter11() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
		assertFalse(f.getHasTensorParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter12() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
		assertFalse(f.getHasTensorParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter13() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
		assertFalse(f.getHasTensorParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter14() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
		assertFalse(f.getHasTensorParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter15() throws Exception {
		Function f = getFunction("f");
		assertFalse("This is a user-defined class with no fields.", f.getHasPrimitiveParameter());
	}

	@Test
	public void testLikelyHasNonTensorParameter16() throws Exception {
		Function f = getFunction("f");
		assertTrue("This is a user-defined class with a primitive field?", f.getHasPrimitiveParameter());
	}

	@Test
	public void testBooleanParameter() throws Exception {
		Function f = getFunction("f");
		assertFalse(f.getHasPrimitiveParameter());
	}

	@Test
	public void testBooleanParameter2() throws Exception {
		Function f = getFunction("f");
		assertFalse(f.getHasPrimitiveParameter());
	}

	@Test
	public void testBooleanParameter3() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
	}

	@Test
	public void testBooleanParameter4() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
	}

	@Test
	public void testBooleanParameter5() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasPrimitiveParameter());
	}

	/**
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing,
	 */
	@Test
	public void testRetracing() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasTensorParameter());
		assertTrue(f.getHasPrimitiveParameter());
		assertFalse(f.isHybrid());
		assertEquals(CONVERT_EAGER_FUNCTION_TO_HYBRID, f.getRefactoring());
		assertNull(f.getPassingPrecondition());
		assertNotNull(f.getEntryMatchingFailure(HAS_PRIMITIVE_PARAMETERS));
		assertTrue(f.getTransformations().isEmpty());
	}

	/**
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing,
	 */
	@Test
	public void testRetracing2() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasTensorParameter());
		assertFalse(f.getHasPrimitiveParameter());
		assertFalse(f.isHybrid());
		assertEquals(CONVERT_EAGER_FUNCTION_TO_HYBRID, f.getRefactoring());
		assertNotNull(f.getPassingPrecondition());
		assertEquals(P1, f.getPassingPrecondition());
		assertNull(f.getEntryMatchingFailure(HAS_PRIMITIVE_PARAMETERS));
		assertFalse(f.getTransformations().isEmpty());
		assertEquals(Collections.singleton(CONVERT_TO_HYBRID), f.getTransformations());
	}

	/**
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing,
	 */
	@Test
	public void testRetracing3() throws Exception {
		Function f = getFunction("f");

		assertTrue(f.isHybrid()); // hyb
		assertTrue(f.getHasTensorParameter()); // T
		assertTrue(f.getHasPrimitiveParameter()); // T
		assertFalse(f.getHasPythonSideEffects()); // F

		assertEquals(OPTIMIZE_HYBRID_FUNCTION, f.getRefactoring());
		assertNotNull(f.getPassingPrecondition());
		assertEquals(P3, f.getPassingPrecondition());
		assertFalse(f.getStatus().hasError());
		assertNull(f.getEntryMatchingFailure(HAS_PRIMITIVE_PARAMETERS));
		assertFalse(f.getTransformations().isEmpty());
		assertEquals(singleton(CONVERT_TO_EAGER), f.getTransformations());
	}

	/**
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing,
	 */
	@Test
	public void testRetracing4() throws Exception {
		Function f = getFunction("f");

		assertTrue(f.isHybrid()); // hyb
		assertTrue(f.getHasTensorParameter()); // T
		assertTrue(f.getHasPrimitiveParameter()); // T
		assertTrue(f.getHasPythonSideEffects()); // T

		assertEquals(OPTIMIZE_HYBRID_FUNCTION, f.getRefactoring());
		assertNull(f.getPassingPrecondition());
		assertTrue(f.getStatus().hasError());
		assertNull(f.getEntryMatchingFailure(HAS_PRIMITIVE_PARAMETERS));
		assertNotNull(f.getEntryMatchingFailure(HAS_PYTHON_SIDE_EFFECTS));
		assertTrue(f.getTransformations().isEmpty());
	}

	/**
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing,
	 */
	@Test
	public void testRetracing5() throws Exception {
		Function f = getFunction("f");

		assertTrue(f.isHybrid()); // hyb
		assertTrue(f.getHasTensorParameter()); // T
		assertFalse(f.getHasPrimitiveParameter()); // F
		assertFalse(f.getHasPythonSideEffects()); // F

		assertEquals(OPTIMIZE_HYBRID_FUNCTION, f.getRefactoring());
		assertNull(f.getPassingPrecondition());
		assertTrue(f.getStatus().hasError());
		assertNotNull(f.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
		assertNull(f.getEntryMatchingFailure(PreconditionFailure.HAS_TENSOR_PARAMETERS));
		assertTrue(f.getTransformations().isEmpty());
	}

	@Test
	public void testTensorFlowGanTutorial() throws Exception {
		Function f = getFunction("train_step");
		assertFalse(f.isHybrid());
		assertTrue("The tensor parameter comes from the dataset interprocedurally.", f.getHasTensorParameter());
		assertFalse("This function doesn't have a primitve parameter.", f.getHasPrimitiveParameter());
	}

	@Test
	public void testTensorFlowGanTutorial2() throws Exception {
		Function f = getFunction("train_step");
		assertTrue(f.isHybrid());
		assertTrue("The tensor parameter comes from the dataset interprocedurally.", f.getHasTensorParameter());
		assertFalse("This function doesn't have a primitve parameter.", f.getHasPrimitiveParameter());
	}

	@Test
	public void testTensorFlowEagerExecution() throws Exception {
		Function f = getFunction("MyModel.call");
		assertFalse(f.isHybrid());
		assertTrue(f.getHasTensorParameter());
		assertFalse(f.getHasPythonSideEffects());
		assertFalse(f.isRecursive());
		assertFalse(f.getHasPrimitiveParameter());
		assertEquals(P1, f.getPassingPrecondition());
		assertEquals(CONVERT_EAGER_FUNCTION_TO_HYBRID, f.getRefactoring());
		assertTrue(f.getErrors().isEmpty());
		assertFalse(f.getStatus().hasError());
		assertEquals(singleton(CONVERT_TO_HYBRID), f.getTransformations());

		f = getFunction("train_step");
		assertTrue(f.getHasTensorParameter());

		f = getFunction("test_step");
		assertTrue(f.getHasTensorParameter());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	@Ignore("Workaround https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/374.")
	public void testClassInDifferentFile() throws Exception {
		Set<Function> functions = getFunctions("B");
		Set<Function> set = functions.stream().filter(f -> f.getIdentifier().equals("Padding2D.call")).collect(Collectors.toSet());
		assertEquals(1, set.size());
		Function f = set.iterator().next();
		assertTrue("This function is called from A.py.", f.getHasTensorParameter());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	public void testClassInDifferentFile2() throws Exception {
		Set<Function> functions = getFunctions("B");
		Set<Function> set = functions.stream().filter(f -> f.getIdentifier().equals("Padding2D.call")).collect(Collectors.toSet());
		assertEquals(1, set.size());
		Function f = set.iterator().next();
		assertTrue("This function is called from A.py.", f.getHasTensorParameter());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	public void testClassInDifferentFile3() throws Exception {
		Set<Function> functions = getFunctions("B");
		Set<Function> set = functions.stream().filter(f -> f.getIdentifier().equals("Padding2D.call")).collect(Collectors.toSet());
		assertEquals(1, set.size());
		Function f = set.iterator().next();
		assertTrue("This function is called from A.py.", f.getHasTensorParameter());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	@Ignore("Workaround https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/374.")
	public void testClassInDifferentFile4() throws Exception {
		Set<Function> functions = getFunctions("B");
		Set<Function> set = functions.stream().filter(f -> f.getIdentifier().equals("Padding2D.call")).collect(Collectors.toSet());
		assertEquals(1, set.size());
		Function f = set.iterator().next();
		assertTrue("This function is called from A.py.", f.getHasTensorParameter());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	public void testClassInDifferentFile5() throws Exception {
		Set<Function> functions = getFunctions("B");
		Set<Function> set = functions.stream().filter(f -> f.getIdentifier().equals("SequentialModel.__call__"))
				.collect(Collectors.toSet());
		assertEquals(1, set.size());
		Function f = set.iterator().next();
		assertTrue("This function is called from A.py.", f.getHasTensorParameter());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/311.
	 */
	@Test
	public void testClassInDifferentFile6() throws Exception {
		Set<Function> functions = getFunctions("B");
		Set<Function> set = functions.stream().filter(f -> f.getIdentifier().equals("SequentialModel.__call__"))
				.collect(Collectors.toSet());
		assertEquals(1, set.size());
		Function f = set.iterator().next();
		assertTrue("This function is called from A.py.", f.getHasTensorParameter());
	}

	/**
	 * Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/308.
	 */
	@Test
	public void testTensorFlowKerasCustomLayer() throws Exception {
		Function function = getFunction("MyConvolution2D.call");
		assertNotNull(function);
		assertFalse(function.getHasPrimitiveParameter());
		assertTrue(function.getHasTensorParameter());
	}

	private static void testFunction(Function function, Boolean expectedHybrid, Boolean expectedTensorParameter,
			Boolean expectedPrimitiveParameter, Boolean expectedPythonSideEffects, Boolean expectedRecursive,
			Refactoring expectedRefactoring, PreconditionSuccess expectedPassingPrecondition, Set<Transformation> expectedTransformations,
			int expectedStatusSeverity) {
		assertEquals(expectedHybrid, function.isHybrid());
		assertEquals(expectedTensorParameter, function.getHasTensorParameter());
		assertEquals(expectedPrimitiveParameter, function.getHasPrimitiveParameter());
		assertEquals(expectedPythonSideEffects, function.getHasPythonSideEffects());
		assertEquals(expectedRecursive, function.isRecursive());
		assertEquals(expectedRefactoring, function.getRefactoring());
		assertEquals(expectedPassingPrecondition, function.getPassingPrecondition());
		assertEquals(expectedTransformations, function.getTransformations());
		assertEquals(expectedStatusSeverity, function.getStatus().getSeverity());
	}

	@Test
	public void testNeuralNetwork() throws Exception {
		Set<Function> functions = getFunctions();

		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "run_optimization":
			case "accuracy":
			case "cross_entropy_loss":
			case "NeuralNet.call":
				testFunction(function, false, true, false, false, false, CONVERT_EAGER_FUNCTION_TO_HYBRID, P1, singleton(CONVERT_TO_HYBRID),
						RefactoringStatus.INFO);
				break;
			case "NeuralNet.__init__":
				assertFalse(function.getStatus().isOK());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function.getIdentifier() + ".");
			}
		}
	}

	@Test
	public void testAutoEncoder() throws Exception {
		Set<Function> functions = getFunctions();

		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "encoder":
			case "mean_square":
			case "run_optimization":
			case "decoder":
				testFunction(function, false, true, false, false, false, CONVERT_EAGER_FUNCTION_TO_HYBRID, P1, singleton(CONVERT_TO_HYBRID),
						RefactoringStatus.INFO);
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function.getIdentifier() + ".");
			}
		}
	}

	@Test
	public void testDatasetGenerator() throws Exception {
		Function function = getFunction("add");
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testDatasetEnumeration() throws Exception {
		Function function = getFunction("summarize_weights");
		assertFalse(function.getHasTensorParameter());
	}

	@Test
	public void testDatasetIteration() throws Exception {
		Function function = getFunction("add");
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
	}

	@Test
	public void testDatasetIteration2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());
		assertTrue(functions.stream().filter(f -> f.getIdentifier().equals("add") || f.getIdentifier().equals("f"))
				.allMatch(f -> f.getHasTensorParameter() && !f.getHasPrimitiveParameter()));
	}

	@Test
	public void testDatasetIteration3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(3, functions.size());
		assertTrue(functions.stream().filter(f -> f.getIdentifier().equals("add") || f.getIdentifier().equals("f"))
				.allMatch(f -> f.getHasTensorParameter() && !f.getHasPrimitiveParameter()));
		assertTrue(functions.stream().filter(f -> f.getIdentifier().equals("g"))
				.allMatch(f -> !f.getHasTensorParameter() && !f.getHasPrimitiveParameter()));
	}

	@Test
	public void testDatasetIteration4() throws Exception {
		Function function = this.getSingleFunction();
		assertEquals("distributed_train_step", function.getIdentifier());
		assertTrue(function.isHybrid());
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
	}

	@Test
	public void testDataset() throws Exception {
		Function function = this.getSingleFunction();
		assertEquals("f", function.getIdentifier());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testDataset2() throws Exception {
		Function function = this.getSingleFunction();
		assertEquals("f", function.getIdentifier());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testDataset3() throws Exception {
		Function function = this.getSingleFunction();
		assertEquals("f", function.getIdentifier());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testDataset4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(3, functions.size());

		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
			case "g":
				assertTrue(function.getHasTensorParameter());
				break;
			case "filter_fn":
				assertNull(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Unknown function: " + function + ".");
			}
		}
	}

	@Test
	public void testDataset5() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());

		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
			case "g":
				assertTrue(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Unknown function: " + function + ".");
			}
		}
	}

	@Test
	public void testDataset6() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(3, functions.size());

		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
			case "g":
			case "h":
				assertTrue(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Unknown function: " + function + ".");
			}
		}
	}

	@Test
	public void testDataset7() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(5, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testDataset8() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(4, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testDataset9() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testDataset10() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testDataset11() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(14, functions.size());
		assertTrue(functions.stream().filter(f -> !f.getIdentifier().equals("n")).allMatch(Function::getHasTensorParameter));
		assertFalse(functions.stream().filter(f -> f.getIdentifier().equals("n")).allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testDataset12() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testDataset13() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testDataset14() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testDataset15() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(3, functions.size());

		Set<Function> addFunctions = functions.stream().filter(f -> f.getSimpleName().equals("add")).collect(toSet());
		assertEquals(1, addFunctions.size());

		Function function = addFunctions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testDataset16() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());

		Set<Function> addFunctions = functions.stream().filter(f -> f.getSimpleName().equals("add")).collect(toSet());
		assertEquals(1, addFunctions.size());

		Function function = addFunctions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testTFRange() throws Exception {
		Function function = getFunction("f");
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
	}

	@Test
	public void testTFRange2() throws Exception {
		Set<Function> functions = this.getFunctions("test_A");
		assertEquals(2, functions.size());
		long count = functions.stream().filter(f -> f.getIdentifier().equals("f")).filter(Function::getHasTensorParameter).count();
		assertEquals(1, count);

		count = functions.stream().filter(f -> f.getIdentifier().equals("f")).filter(Function::getHasPrimitiveParameter).count();
		assertEquals(0, count);
	}

	@Test
	public void testTFRange3() throws Exception {
		Function function = getFunction("f");
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
	}

	@Test
	public void testTFRange4() throws Exception {
		Function function = getFunction("f");
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
	}

	@Test
	public void testTFRange5() throws Exception {
		Function function = getFunction("f");
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
	}

	@Test
	public void testTFRange6() throws Exception {
		Function function = getFunction("f");
		assertFalse(function.getHasTensorParameter());
		assertTrue(function.getHasPrimitiveParameter());
	}

	@Test
	public void testTFRange7() throws Exception {
		Function function = getFunction("f");
		assertFalse(function.getHasTensorParameter());
		assertTrue(function.getHasPrimitiveParameter());
	}

	@Test
	public void testTFRange8() throws Exception {
		Function function = getFunction("f");
		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
	}

	@Test
	public void testPytest() throws Exception {
		Set<Function> functions = this.getFunctions("test_sample");
		assertEquals(2, functions.size());
		long count = functions.stream().filter(f -> f.getIdentifier().equals("func")).filter(Function::getHasPrimitiveParameter).count();
		assertEquals(1, count);
	}

	@Test
	public void testPytest2() throws Exception {
		Set<Function> functions = this.getFunctions("test_tf_range");
		assertEquals(2, functions.size());
		long count = functions.stream().filter(f -> f.getIdentifier().equals("f")).filter(Function::getHasTensorParameter).count();
		assertEquals(1, count);
	}

	private static void testGPModelHelper(Set<Function> functions) throws Exception {
		assertEquals(2, functions.size());

		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "test_compile_monitor":
				break;
			case "test_compile_monitor.tf_func":
				assertTrue(function.getHasTensorParameter());
				assertFalse(function.getHasPrimitiveParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testGPModel() throws Exception {
		Set<Function> functions = this.getFunctions("test_A");
		testGPModelHelper(functions);
	}

	@Test
	public void testGPModel2() throws Exception {
		Set<Function> functions = this.getFunctions();
		testGPModelHelper(functions);
	}

	@Test
	public void testGPModel3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(3, functions.size());

		Function[] array = functions.stream().filter(f -> f.getIdentifier().equals("func")).toArray(Function[]::new);
		assertEquals(1, array.length);

		Function function = array[0];
		assertNotNull(function);

		assertTrue(function.getHasTensorParameter());
		assertFalse(function.getHasPrimitiveParameter());
	}

	@Test
	public void testWildcardImport() throws Exception {
		Function function = this.getSingleFunction();
		assertEquals("f", function.getIdentifier());
		assertEquals(1, function.getNumberOfParameters());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testWildcardImport2() throws Exception {
		Function function = this.getSingleFunction();
		assertEquals("f", function.getIdentifier());
		assertEquals(1, function.getNumberOfParameters());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testWildcardImport3() throws Exception {
		Function function = this.getSingleFunction();
		assertEquals("f", function.getIdentifier());
		assertEquals(1, function.getNumberOfParameters());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testWildcardImport4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());
		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
				assertEquals(1, function.getNumberOfParameters());
				assertTrue(function.getHasTensorParameter());
				break;
			case "g":
				assertEquals(1, function.getNumberOfParameters());
				assertFalse(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testWildcardImport5() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());
		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
				assertEquals(1, function.getNumberOfParameters());
				assertTrue(function.getHasTensorParameter());
				break;
			case "g":
				assertEquals(1, function.getNumberOfParameters());
				assertFalse(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testWildcardImport6() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());
		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
			case "g":
				assertEquals(1, function.getNumberOfParameters());
				assertTrue(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testWildcardImport7() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());
		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
				assertEquals(1, function.getNumberOfParameters());
				assertFalse(function.getHasTensorParameter());
				break;
			case "g":
				assertEquals(1, function.getNumberOfParameters());
				assertTrue(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testWildcardImport8() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());
		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
			case "g":
				assertEquals(1, function.getNumberOfParameters());
				assertTrue(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testWildcardImport9() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());
		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
			case "g":
				assertEquals(1, function.getNumberOfParameters());
				assertTrue(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testWildcardImport10() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());
		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "f":
			case "g":
				assertEquals(1, function.getNumberOfParameters());
				assertTrue(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testModule() throws Exception {
		Set<Function> functions = this.getFunctions("B");
		assertEquals(1, functions.size());

		for (Function function : functions) {
			assertEquals("f", function.getIdentifier());
			assertEquals(1, function.getNumberOfParameters());
			assertTrue(function.getHasTensorParameter());
		}
	}

	@Test
	public void testModule2() throws Exception {
		Set<Function> functions = this.getFunctions("src/B");
		assertEquals(1, functions.size());

		for (Function function : functions) {
			assertEquals("f", function.getIdentifier());
			assertEquals(1, function.getNumberOfParameters());
			assertTrue(function.getHasTensorParameter());
		}
	}

	@Test
	public void testModule3() throws Exception {
		Set<Function> functions = this.getFunctions("src/B");
		assertEquals(1, functions.size());

		for (Function function : functions) {
			assertEquals("C.f", function.getIdentifier());
			assertEquals(1, function.getNumberOfParameters());
			assertTrue(function.getHasTensorParameter());
		}
	}

	@Test
	public void testModule4() throws Exception {
		Set<Function> functions = this.getFunctions("src/C/B");
		assertEquals(1, functions.size());

		for (Function function : functions) {
			assertEquals("f", function.getIdentifier());
			assertEquals(1, function.getNumberOfParameters());
			assertTrue(function.getHasTensorParameter());
		}
	}

	@Test
	public void testModule5() throws Exception {
		Set<Function> functions = this.getFunctions("C/B");
		assertEquals(1, functions.size());

		for (Function function : functions) {
			assertEquals("f", function.getIdentifier());
			assertEquals(1, function.getNumberOfParameters());
			assertTrue(function.getHasTensorParameter());
		}
	}

	@Test
	@Ignore("Workaround https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/374.")
	public void testModule6() throws Exception {
		Set<Function> functions = this.getFunctions("B");
		assertEquals(1, functions.size());

		for (Function function : functions) {
			assertEquals("f", function.getIdentifier());
			assertEquals(1, function.getNumberOfParameters());
			assertTrue(function.getHasTensorParameter());
		}
	}

	@Test
	public void testModule7() throws Exception {
		Set<Function> functions = this.getFunctions("C/B");
		assertEquals(1, functions.size());

		for (Function function : functions) {
			assertEquals("f", function.getIdentifier());
			assertEquals(1, function.getNumberOfParameters());
			assertTrue(function.getHasTensorParameter());
		}
	}

	@Test
	public void testModule8() throws Exception {
		Set<Function> functions = this.getFunctions("B");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("C.f", function.getIdentifier());
		assertEquals(2, function.getNumberOfParameters());
		assertTrue(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule9() throws Exception {
		Set<Function> functions = this.getFunctions("src/B");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("C.f", function.getIdentifier());
		assertEquals(2, function.getNumberOfParameters());
		assertTrue(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule10() throws Exception {
		Set<Function> functions = this.getFunctions("src/B");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("C.f", function.getIdentifier());
		assertEquals(2, function.getNumberOfParameters());
		assertTrue(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule11() throws Exception {
		Set<Function> functions = this.getFunctions("src/B");
		assertEquals(2, functions.size());

		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "C.f":
			case "D.g":
				assertEquals(2, function.getNumberOfParameters());
				assertTrue(function.isMethod());
				assertTrue(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testModule12() throws Exception {
		Set<Function> functions = this.getFunctions("B");
		assertEquals(2, functions.size());

		for (Function function : functions) {
			switch (function.getIdentifier()) {
			case "C.f":
				break;
			case "D.f":
				assertEquals(2, function.getNumberOfParameters());
				assertTrue(function.isMethod());
				assertTrue(function.getHasTensorParameter());
				break;
			default:
				throw new IllegalStateException("Not expecting: " + function + ".");
			}
		}
	}

	@Test
	public void testModule13() throws Exception {
		Set<Function> functions = this.getFunctions("C");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("D.f", function.getIdentifier());
		assertEquals(2, function.getNumberOfParameters());
		assertTrue(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule14() throws Exception {
		Set<Function> functions = this.getFunctions("src/C");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("D.f", function.getIdentifier());
		assertEquals(2, function.getNumberOfParameters());
		assertTrue(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule15() throws Exception {
		Set<Function> functions = this.getFunctions("src/C");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("D.f", function.getIdentifier());
		assertEquals(2, function.getNumberOfParameters());
		assertTrue(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule16() throws Exception {
		Set<Function> functions = this.getFunctions("src/C");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("D.g", function.getIdentifier());
		assertEquals(2, function.getNumberOfParameters());
		assertTrue(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule17() throws Exception {
		Set<Function> functions = this.getFunctions("src/B");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("f", function.getIdentifier());
		assertEquals(1, function.getNumberOfParameters());
		assertFalse(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule18() throws Exception {
		Set<Function> functions = this.getFunctions("src/B");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("f", function.getIdentifier());
		assertEquals(1, function.getNumberOfParameters());
		assertFalse(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule19() throws Exception {
		Set<Function> functions = this.getFunctions("src/B");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("C.f", function.getIdentifier());
		assertEquals(2, function.getNumberOfParameters());
		assertTrue(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testModule20() throws Exception {
		Set<Function> functions = this.getFunctions("src/B");
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("C.f", function.getIdentifier());
		assertEquals(2, function.getNumberOfParameters());
		assertTrue(function.isMethod());
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testBooleanMask() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testOnesLike() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testStack() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());

		for (Function function : functions)
			assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testCustomGradient() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());

		for (Function function : functions)
			if (function.getIdentifier().equals("log1pexp"))
				assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testExtractPatches() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testIdentity() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testMeshGrid() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(3, functions.size());

		for (Function function : functions)
			assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testRank() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testReshape() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testReshape2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testCond() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(3, functions.size());

		for (Function function : functions)
			if (function.getIdentifier().equals("f"))
				assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testWhileLoop() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testWhileLoop2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testWhileLoop3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testWhileLoop4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testWhileLoop5() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testWhileLoop6() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testDynamicPartition() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testDynamicPartition2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testDynamicStitch() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testDynamicStitch2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testEigenDecomposition() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testEigenDecomposition2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testMapFn() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testAbs() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testAbs2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testAddN() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testAddN2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testAddN3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testAddN4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testEqual() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testEqual2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testEqual3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testExp() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testExp2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testLinSpace() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testMatMul() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testNotEqual() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testNotEqual2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testNotEqual3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue(function.getHasTensorParameter());
	}

	@Test
	public void testReduceAll() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(3, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testShuffle() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testSort() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testSort2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testEinSum() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testClipByGlobalNorm() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(3, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	@Test
	public void testTopK() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());
		assertTrue(functions.stream().allMatch(Function::getHasTensorParameter));
	}

	/**
	 * Don't de-hybridize functions whose name has certain keywords.
	 */
	@Test
	public void testSpeculativeAnalysis() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream()
				.allMatch(f -> f.isHybrid() && f.getSimpleName().matches("train.*_step") && !Objects.equals(f.getPassingPrecondition(), P2)
						&& !Objects.equals(f.getPassingPrecondition(), P3) && !f.getTransformations().contains(CONVERT_TO_EAGER)
						&& f.getStatus().hasInfo() && Arrays.stream(f.getStatus().getEntries()).filter(e -> e.getSeverity() == INFO)
								.filter(e -> e.getCode() == SPECULATIVE_ANALYSIS.getCode()).count() == 1));
	}

	/**
	 * Don't de-hybridize functions whose name has certain keywords.
	 */
	@Test
	public void testSpeculativeAnalysis2() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		assertTrue(functions.stream()
				.allMatch(f -> f.isHybrid() && f.getSimpleName().matches("train.*_step") && Objects.equals(f.getPassingPrecondition(), P2)
						&& f.getTransformations().contains(CONVERT_TO_EAGER) && f.getStatus().hasInfo()
						&& Arrays.stream(f.getStatus().getEntries()).filter(e -> e.getSeverity() == INFO)
								.filter(e -> e.getCode() == SPECULATIVE_ANALYSIS.getCode()).count() == 0));
	}

	/**
	 * Don't de-hybridize functions whose name has certain keywords.
	 */
	@Test
	public void testSpeculativeAnalysis3() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();

		assertTrue(f.isHybrid());
		assertTrue(f.getSimpleName().matches("test.*_step"));
		assertNotEquals(P2, f.getPassingPrecondition());
		assertNotEquals(P3, f.getPassingPrecondition());
		assertFalse(f.getTransformations().contains(CONVERT_TO_EAGER));
		assertTrue(f.getStatus().hasInfo());
		assertEquals(1, Arrays.stream(f.getStatus().getEntries()).filter(e -> e.getSeverity() == INFO)
				.filter(e -> e.getCode() == SPECULATIVE_ANALYSIS.getCode()).count());
	}

	/**
	 * Don't de-hybridize functions whose name has certain keywords.
	 */
	@Test
	public void testSpeculativeAnalysis4() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();

		assertFalse(f.isHybrid());
		String name = f.getIdentifier();
		assertTrue(name.matches("[A-Z].*\\.call"));
		assertEquals(P1, f.getPassingPrecondition());
		assertTrue(f.getTransformations().contains(CONVERT_TO_HYBRID));

		assertEquals(1, Arrays.stream(f.getStatus().getEntries()).filter(e -> e.getSeverity() == INFO)
				.filter(e -> e.getCode() == SPECULATIVE_ANALYSIS.getCode()).count());
	}
}
