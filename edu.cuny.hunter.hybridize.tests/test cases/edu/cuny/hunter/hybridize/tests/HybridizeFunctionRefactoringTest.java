package edu.cuny.hunter.hybridize.tests;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static edu.cuny.hunter.hybridize.core.analysis.Function.PLUGIN_ID;
import static edu.cuny.hunter.hybridize.core.analysis.Information.INPUT_SIGNATURE_INFERENCE;
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
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P4;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P5;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P6;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.OPTIMIZE_HYBRID_FUNCTION;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.CONVERT_TO_EAGER;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.CONVERT_TO_HYBRID;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.RECONFIGURE;
import static java.lang.Boolean.TRUE;
import static java.lang.Integer.MAX_VALUE;
import static java.lang.System.getProperty;
import static java.nio.file.FileVisitOption.FOLLOW_LINKS;
import static java.util.Collections.singleton;
import static java.util.stream.Collectors.toSet;
import static org.eclipse.core.resources.ResourceAttributes.fromFile;
import static org.eclipse.core.runtime.Path.fromOSString;
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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.stream.Collectors;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.ResourceAttributes;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.ILogListener;
import org.eclipse.core.runtime.IPath;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.jface.text.Document;
import org.eclipse.jface.text.IDocument;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.RefactoringCore;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.RefactoringStatusEntry;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.eclipse.text.edits.TextEdit;
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
import org.python.pydev.parser.jython.ast.FunctionDef;
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

import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.python.pydev.analysis.additionalinfo.AbstractAdditionalDependencyInfo;
import com.python.pydev.analysis.additionalinfo.AdditionalProjectInterpreterInfo;

import edu.cuny.citytech.refactoring.common.tests.RefactoringTest;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.analysis.FunctionExtractor;
import edu.cuny.hunter.hybridize.core.analysis.InferenceResult;
import edu.cuny.hunter.hybridize.core.analysis.InputSignature;
import edu.cuny.hunter.hybridize.core.analysis.Parameter;
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

	private static final String RUN_OUTPUT_TEST_FILE_KEY = "edu.cuny.hunter.hybridize.tests.runOutput";

	private static final String COMPARE_OUTPUT_TEST_FILE_KEY = "edu.cuny.hunter.hybridize.tests.compareOutput";

	/**
	 * Add a module to the given {@link IPythonNature}.
	 *
	 * @param ast the ast that defines the module
	 * @param modName the module name
	 * @param natureToAdd the nature where the module should be added
	 * @param f the file backing the module on disk
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
	 *
	 * @throws MisconfigurationException If the project's interpreter or modules manager is misconfigured.
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
	 * @throws IOException If launching the {@code pip} subprocess or reading its output fails.
	 * @throws InterruptedException If the current thread is interrupted while waiting for {@code pip} to complete.
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
			try (BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
				errorOutput = errorReader.lines().collect(Collectors.joining("\n"));
			}
		}

		assertEquals("Error code should be 0. Error was:\n" + errorOutput + ".", 0, exitCode);
		return exitCode;
	}

	/**
	 * Runs python on the file presented by the given {@link Path}.
	 *
	 * @param path The {@link Path} of the file to interpret.
	 * @return The exit code.
	 * @throws IOException If launching the python subprocess or reading its output fails.
	 * @throws InterruptedException If the current thread is interrupted while waiting for python to complete.
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
					PythonPathHelper.parsePythonPathFromStr(path, new ArrayList<>()));

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
		InterpreterGeneralPreferences.FORCE_USE_TYPESHED = TRUE;
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
			return TRUE;
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
	 * True iff the output test Python file should be executed. Verifies the refactoring tool's "valid Python in, valid Python out" contract
	 * by running the expected {@code out/A.py} fixture.
	 */
	protected boolean runOutputTestFile = Boolean.getBoolean(RUN_OUTPUT_TEST_FILE_KEY);

	/**
	 * True iff the output test Python file should be compared.
	 */
	protected boolean compareOutputTestFile = Boolean.getBoolean(COMPARE_OUTPUT_TEST_FILE_KEY);

	/**
	 * True iff inferred input signatures should be emitted into the refactored source during {@code transform()}. Off by default so the
	 * suite's compare-output fixtures are unaffected by emission; the input-signature emission tests opt in through
	 * {@link #helperAssertInputSignatureEmission()}. Production gating is tracked at #481.
	 */
	protected boolean inferInputSignatures;

	/**
	 * The targeted k-CFA depth the harness forwards to the analysis engine, defaulting to the refactoring's own
	 * {@link HybridizeFunctionRefactoringProcessor#DEFAULT_TARGETED_CFA_DEPTH} so the harness does not override it. A test sets it via
	 * {@link #setTargetedCfaDepth(int)} to analyze a fixture at a chosen depth (#600).
	 */
	private int targetedCfaDepth = HybridizeFunctionRefactoringProcessor.DEFAULT_TARGETED_CFA_DEPTH;

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
	 * Returns whether the input test Python file should be executed under {@code python3.10} before analysis.
	 *
	 * @return True iff the input test Python file should be executed.
	 */
	public boolean getRunInputTestFile() {
		return this.runInputTestFile;
	}

	/**
	 * Returns whether the output test Python file should be executed under {@code python3.10} after analysis. Verifies the refactoring
	 * tool's "valid Python in, valid Python out" contract on the expected {@code out/A.py} fixture.
	 *
	 * @return True iff the output test Python file should be executed.
	 */
	public boolean getRunOutputTestFile() {
		return this.runOutputTestFile;
	}

	/**
	 * Returns whether the output test Python file should be compared against the expected output after analysis.
	 *
	 * @return True iff the output test Python file should be compared.
	 */
	public boolean getCompareOutputTestFile() {
		return this.compareOutputTestFile;
	}

	/**
	 * Returns whether inferred input signatures should be emitted into the refactored source. Scoped per test (off by default) rather than
	 * enabled suite-wide, so only the input-signature emission tests exercise emission.
	 *
	 * @return True iff inferred input signatures should be emitted.
	 */
	public boolean getInferInputSignatures() {
		return this.inferInputSignatures;
	}

	/**
	 * Sets whether inferred input signatures should be emitted into the refactored source. The input-signature emission tests call this to
	 * opt in (the flag is off by default suite-wide, #580).
	 *
	 * @param inferInputSignatures Whether inferred input signatures should be emitted.
	 */
	public void setInferInputSignatures(boolean inferInputSignatures) {
		this.inferInputSignatures = inferInputSignatures;
	}

	/**
	 * Sets the targeted k-CFA depth the harness forwards to the analysis engine (#600).
	 *
	 * @param targetedCfaDepth The targeted k-CFA depth.
	 */
	public void setTargetedCfaDepth(int targetedCfaDepth) {
		this.targetedCfaDepth = targetedCfaDepth;
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
			Set<Path> pythonFilesInTestFileDirectory;
			try (var stream = Files.find(inputTestFileDirectoryAbsolutePath, MAX_VALUE,
					(path, attr) -> path.toFile().getName().endsWith(".py"), FOLLOW_LINKS)) {
				pythonFilesInTestFileDirectory = stream.collect(toSet());
			}

			for (Path path : pythonFilesInTestFileDirectory) {
				boolean validSourceFile = PythonPathHelper.isValidSourceFile(path.toString());
				assertTrue("Source file must be valid.", validSourceFile);

				// Run the Python test file.
				rc = runPython(path);
				LOG.info("Running the test file was " + (rc == 0 ? "successful." : "unsuccessful."));
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
				// NOTE: This is incorrect when implementing https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/359.
				return fromOSString(inputTestFile.getAbsolutePath());
			}

			@Override
			public String getCharset(boolean checkImplicit) throws CoreException {
				return getProperty("file.encoding");
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
				ALWAYS_FOLLOW_TYPE_HINTS, USE_SPECULATIVE_ANALYSIS, this.getInferInputSignatures());
		processor.setTargetedCfaDepth(this.targetedCfaDepth);

		ProcessorBasedRefactoring refactoring = new ProcessorBasedRefactoring(processor);

		RefactoringStatus status = refactoring.checkAllConditions(new NullProgressMonitor());

		if (processor.getFunctions().stream().map(Function::getStatus).allMatch(RefactoringStatus::isOK))
			assertTrue(status.isOK());
		else
			assertFalse(status.isOK());

		// NOTE: `compareOutputTestFile` is the existing infra for asserting refactored output against `out/A.py`. The first layer of
		// #359 (missing `initializeValidationData(...)` before `performChange(...)`) is fixed here; the URI-resolution layer
		// (`ResourceStub`-backed `IFile`s) remains open at
		// https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/359.
		if (this.getCompareOutputTestFile()) {
			// check if there's an expected output.
			File outputTestFile = this.getOutputTestFile(fileNameWithoutExtension);

			if (outputTestFile.exists()) {
				Change change = refactoring.createChange(new NullProgressMonitor());
				change.initializeValidationData(new NullProgressMonitor());
				this.performChange(change);

				String expected = this.getFileContents(this.getOutputTestFileName(fileNameWithoutExtension));
				String actual = document.get();
				assertEqualLines(expected, actual);
			}
		}

		// Symmetric to the `runInputTestFile` block in `genericbefore`: when on, execute the expected `out/A.py` under `python3.10` to
		// verify the fixture represents a valid Python program. Catches a class of regressions where the expected output diverges from
		// runnable syntax (e.g., a hand-authored decorator with a typo). The stronger "run the actually-produced source" property
		// requires the `compareOutputTestFile` path's URI-resolution layer (#359); see
		// https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/576.
		if (this.getRunOutputTestFile()) {
			File outputTestFile = this.getOutputTestFile(fileNameWithoutExtension);

			if (outputTestFile.exists()) {
				LOG.info("Running output test file(s).");

				Path outputTestFileAbsolutePath = getAbsolutePath(this.getOutputTestFileName(fileNameWithoutExtension));
				Path outputTestFileDirectoryAbsolutePath = outputTestFileAbsolutePath.getParent();
				// Requirements live alongside the input fixture (`in/requirements.txt`); the output directory shares the same Python
				// environment.
				Path inputTestFileDirectoryAbsolutePath = getAbsolutePath(this.getInputTestFileName(fileNameWithoutExtension)).getParent();

				int rc = installRequirements(inputTestFileDirectoryAbsolutePath);
				LOG.info("Installing requirements was " + (rc == 0 ? "successful." : "unsuccessful."));

				int filesRun = 0;
				Set<Path> pythonFilesInOutputDirectory;
				try (var stream = Files.find(outputTestFileDirectoryAbsolutePath, MAX_VALUE,
						(path, attr) -> path.toFile().getName().endsWith(".py"), FOLLOW_LINKS)) {
					pythonFilesInOutputDirectory = stream.collect(toSet());
				}

				for (Path path : pythonFilesInOutputDirectory) {
					boolean validSourceFile = PythonPathHelper.isValidSourceFile(path.toString());
					assertTrue("Source file must be valid.", validSourceFile);

					rc = runPython(path);
					LOG.info("Running the output test file was " + (rc == 0 ? "successful." : "unsuccessful."));
					++filesRun;
				}

				assertTrue("Must have executed at least A.py.", filesRun > 0);
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

		assertTrue(!args.hasFuncParam() && args.hasInputSignatureParam() && !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
	}

	/**
	 * Returns the {@code Function.HybridizationParameters} of the single hybrid {@link Function} in the test file A.py.
	 *
	 * @return The hybridization parameters of the sole analyzed {@link Function}.
	 */
	private Function.HybridizationParameters getSingleHybridizationParameters() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue(function.isHybrid());
		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);
		return args;
	}

	/**
	 * Test for #557. A concrete, fully modeled keyword-form {@code input_signature=[tf.TensorSpec([2, 2], tf.float32)]} parses into a
	 * single rank-2 {@code float32} {@link TensorType}. The shape and dtype are supplied positionally <em>within</em> the
	 * {@code TensorSpec} call.
	 */
	@Test
	public void testSuppliedInputSignatureConcrete() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasInputSignatureParam());

		Optional<InputSignature> signature = args.getSuppliedInputSignature();
		assertTrue(signature.isPresent());
		assertEquals(List.of(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)))), signature.get().parameterTypes());
	}

	/**
	 * Test for #573. The positional form {@code @tf.function(None, [tf.TensorSpec(...)])} binds the signature to position 1; its content is
	 * parsed the same as the keyword form. {@code hasFuncParam} is true (position 0 is occupied by {@code None}) alongside
	 * {@code hasInputSignatureParam}.
	 */
	@Test
	public void testSuppliedInputSignaturePositional() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasFuncParam());
		assertTrue(args.hasInputSignatureParam());

		Optional<InputSignature> signature = args.getSuppliedInputSignature();
		assertTrue(signature.isPresent());
		assertEquals(List.of(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)))), signature.get().parameterTypes());
	}

	/**
	 * Test for #557. The keyword form {@code tf.TensorSpec(shape=(5,), dtype=tf.int32)} binds shape and dtype by name rather than by
	 * position; both forms must resolve to the same {@link TensorType}.
	 */
	@Test
	public void testSuppliedInputSignatureKeywordForm() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasInputSignatureParam());

		Optional<InputSignature> signature = args.getSuppliedInputSignature();
		assertTrue(signature.isPresent());
		assertEquals(List.of(new TensorType(INT32, List.of(new NumericDim(5)))), signature.get().parameterTypes());
	}

	/**
	 * Test for #557. A two-element {@code input_signature} list with differing dtypes parses into two {@link TensorType}s in declaration
	 * order.
	 */
	@Test
	public void testSuppliedInputSignatureMultipleParameters() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasInputSignatureParam());

		Optional<InputSignature> signature = args.getSuppliedInputSignature();
		assertTrue(signature.isPresent());
		assertEquals(List.of(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))),
				new TensorType(INT32, List.of(new NumericDim(5)))), signature.get().parameterTypes());
	}

	/**
	 * Test for #557. A well-formed empty {@code input_signature=[]} (a no-arg function) is itself fully modeled: it parses to a present,
	 * empty {@link InputSignature}, distinct from the presence-true/parse-empty "unmodelable" state.
	 */
	@Test
	public void testSuppliedInputSignatureEmptyList() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasInputSignatureParam());

		Optional<InputSignature> signature = args.getSuppliedInputSignature();
		assertTrue(signature.isPresent());
		assertEquals(List.of(), signature.get().parameterTypes());
	}

	/**
	 * Test for #557. A {@code None} entry in a {@code TensorSpec} shape parses to {@link DynamicDim#INSTANCE}, mixed with a concrete
	 * {@link NumericDim}.
	 */
	@Test
	public void testSuppliedInputSignatureDynamicDim() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasInputSignatureParam());

		Optional<InputSignature> signature = args.getSuppliedInputSignature();
		assertTrue(signature.isPresent());
		assertEquals(List.of(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32)))), signature.get().parameterTypes());
	}

	/**
	 * Test for #557. A scalar (rank-0) {@code TensorSpec} with an empty shape tuple parses to a {@link TensorType} with an empty dimension
	 * list.
	 */
	@Test
	public void testSuppliedInputSignatureScalar() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasInputSignatureParam());

		Optional<InputSignature> signature = args.getSuppliedInputSignature();
		assertTrue(signature.isPresent());
		assertEquals(List.of(new TensorType(INT32, List.of())), signature.get().parameterTypes());
	}

	/**
	 * Test for #557. A bare {@code shape=None} (unknown rank) parses to a {@link TensorType} with {@code null} dimensions, the shape-⊤
	 * encoding used elsewhere in the tool.
	 */
	@Test
	public void testSuppliedInputSignatureShapeNone() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasInputSignatureParam());

		Optional<InputSignature> signature = args.getSuppliedInputSignature();
		assertTrue(signature.isPresent());
		assertEquals(1, signature.get().parameterTypes().size());
		TensorType tensorType = signature.get().parameterTypes().get(0);
		assertEquals(FLOAT32, tensorType.getDType());
		assertNull(tensorType.getDims());
	}

	/**
	 * Test for #557. A {@code RaggedTensorSpec} element is supplied (so {@code hasInputSignatureParam()} is true), but the current
	 * {@link InputSignature} model cannot represent raggedness, so the parse drops to {@link Optional#empty}. This pins the
	 * presence-true/parse-empty state of the contract: a supplied-but-unmodelable signature must be left as-is, not overwritten. Ragged
	 * emission is tracked separately at #524.
	 */
	@Test
	public void testSuppliedInputSignatureRaggedSpec() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasInputSignatureParam());
		assertFalse(args.getSuppliedInputSignature().isPresent());
	}

	/**
	 * Test for #557. A valid TensorFlow dtype the tool does not model ({@code tf.bfloat16}) is supplied; the parse drops the whole
	 * signature to {@link Optional#empty} while {@code hasInputSignatureParam()} stays true (the presence-true/parse-empty contract state).
	 * Uses {@code bfloat16} rather than {@code complex64} so the dtype stays unmodeled after Ariadne added complex support (wala/ML#472).
	 */
	@Test
	public void testSuppliedInputSignatureUnmodeledDType() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertTrue(args.hasInputSignatureParam());
		assertFalse(args.getSuppliedInputSignature().isPresent());
	}

	/**
	 * Test for #557. A bare {@code @tf.function} with no {@code input_signature} argument: presence is false and the parsed signature is
	 * {@link Optional#empty}. This pins the presence-false state of the contract, distinguishing "no signature supplied" (safe to infer and
	 * write) from "supplied but unmodelable" (leave as-is).
	 */
	@Test
	public void testSuppliedInputSignatureAbsent() throws Exception {
		Function.HybridizationParameters args = this.getSingleHybridizationParameters();

		assertFalse(args.hasInputSignatureParam());
		assertFalse(args.getSuppliedInputSignature().isPresent());
	}

	/**
	 * Test for #108. Positional `input_signature` argument: `@tf.function(None, (tf.TensorSpec(...),))` passes `func=None` and
	 * `input_signature=(...)` positionally. We expect both `hasFuncParam` and `hasInputSignatureParam` to be true (the position-0 `func`
	 * slot is occupied even though its value is `None`; presence of the positional slot is what's detected, not its value).
	 */
	@Test
	public void testPositionalParameterInputSignature() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(args.hasFuncParam());
		assertTrue(args.hasInputSignatureParam());
		assertTrue(!args.hasAutoGraphParam() && !args.hasJitCompileParam() && !args.hasReduceRetracingParam()
				&& !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
	}

	/**
	 * Test for #108. Positional `autograph` argument at position 2: `@tf.function(None, None, False)`. We expect `hasFuncParam`,
	 * `hasInputSignatureParam`, and `hasAutoGraphParam` to all be true.
	 */
	@Test
	public void testPositionalParameterAutograph() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(args.hasFuncParam());
		assertTrue(args.hasInputSignatureParam());
		assertTrue(args.hasAutoGraphParam());
		assertTrue(!args.hasJitCompileParam() && !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam()
				&& !args.hasExperimentalAutographOptionsParam() && !args.hasExperimentalFollowTypeHintsParam());
	}

	/**
	 * Test for #108. Mixed positional + keyword in the same decorator: `@tf.function(None, autograph=False)`. The position-0 `func` slot is
	 * occupied positionally; `autograph` is bound by keyword. We expect both `hasFuncParam` and `hasAutoGraphParam` true, and notably
	 * `hasInputSignatureParam` false (no slot occupied at position 1, no `input_signature` keyword).
	 */
	@Test
	public void testMixedPositionalAndKeyword() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);

		assertTrue(function.isHybrid());

		Function.HybridizationParameters args = function.getHybridizationParameters();
		assertNotNull(args);

		assertTrue(args.hasFuncParam());
		assertTrue(args.hasAutoGraphParam());
		assertTrue(!args.hasInputSignatureParam() && !args.hasJitCompileParam() && !args.hasReduceRetracingParam()
				&& !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
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

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && args.hasExperimentalAutographOptionsParam()
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

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
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

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
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

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && !args.hasAutoGraphParam() && args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
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

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
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

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
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

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && !args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
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

		assertTrue(!args.hasFuncParam() && args.hasInputSignatureParam() && args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
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

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && args.hasAutoGraphParam() && !args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
				&& !args.hasExperimentalFollowTypeHintsParam());

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

		assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && !args.hasAutoGraphParam() && args.hasJitCompileParam()
				&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam() && !args.hasExperimentalAutographOptionsParam()
				&& !args.hasExperimentalFollowTypeHintsParam());
	}

	/**
	 * Test that an unrecognized {@code @tf.function} keyword argument is parsed without error and that none of the recognized
	 * {@code *Param} flags are set. Exercises the {@code default} branch of {@code markParam}, which logs a {@code WARNING} and otherwise
	 * leaves state untouched. The fixture uses {@code experimental_attributes}, a real TF kwarg added in TF 2.13 that Hybridize does not
	 * yet model; the per-fixture {@code requirements.txt} pins {@code tensorflow==2.13.1} so the fixture is Python-runnable. NOTE: If
	 * Hybridize promotes {@code experimental_attributes} to a first-class recognized parameter (its own case in {@code markParam}'s switch
	 * and a {@code hasExperimentalAttributesParam} accessor), this test will fail twice over—the new flag would be set by the fixture
	 * (failing the conjunction below), and the WARN log assertion would no longer fire. Re-pin the fixture to a different
	 * real-but-unrecognized kwarg at that point to keep exercising the {@code default} WARN branch.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/204">Issue 204</a>
	 */
	@Test
	public void testComputeParameters13() throws Exception {
		ILog functionLog = getLog(Function.class);
		List<IStatus> captured = new ArrayList<>();
		ILogListener listener = (status, plugin) -> captured.add(status);
		functionLog.addLogListener(listener);
		try {
			Set<Function> functions = this.getFunctions();
			assertNotNull(functions);
			assertEquals(1, functions.size());
			Function function = functions.iterator().next();
			assertNotNull(function);

			assertTrue(function.isHybrid());

			Function.HybridizationParameters args = function.getHybridizationParameters();
			assertNotNull(args);

			assertTrue(!args.hasFuncParam() && !args.hasInputSignatureParam() && !args.hasAutoGraphParam() && !args.hasJitCompileParam()
					&& !args.hasReduceRetracingParam() && !args.hasExperimentalImplementsParam()
					&& !args.hasExperimentalAutographOptionsParam() && !args.hasExperimentalFollowTypeHintsParam());
		} finally {
			functionLog.removeLogListener(listener);
		}

		boolean warnFired = captured.stream().anyMatch(s -> s.getSeverity() == IStatus.WARNING
				&& "Unknown @tf.function argument: experimental_attributes on func().".equals(s.getMessage()));
		assertTrue("Expected WARN log for unknown kwarg `experimental_attributes`.", warnFired);
	}

	/**
	 * Test that {@code convertToHybrid} emits an inferred {@code input_signature=[tf.TensorSpec(...)]} keyword into the generated
	 * {@code @tf.function(...)} decorator. Phase 2 of #563: the formatter from #564 plus the wired-through flag on {@link Function} and
	 * {@code HybridizeFunctionRefactoringProcessor}. The emission tests opt into the flag through
	 * {@link #helperAssertInputSignatureEmission()} (off by default suite-wide, #580); the user-facing/eval-facing gating in production
	 * wiring is tracked at #481.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	@Test
	public void testInferInputSignatureEmission() throws Exception {
		helperAssertInputSignatureEmission();
	}

	/**
	 * Wildcard-import variant of {@link #testInferInputSignatureEmission()}. With {@code from tensorflow import *}, both {@code function}
	 * and {@code TensorSpec} (plus the dtype constants) are reachable unqualified. The source-write should distinguish this empty-prefix
	 * shape from the {@code from tensorflow import function} shape and emit an unqualified
	 * {@code @function(input_signature=[TensorSpec(shape=(), dtype=float32)])} rather than skipping emission.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/565">PR 565</a>
	 */
	@Test
	public void testInferInputSignatureEmissionWildcardImport() throws Exception {
		helperAssertInputSignatureEmission();
	}

	/**
	 * Named-import variant of {@link #testInferInputSignatureEmission()}. With {@code from tensorflow import function, constant},
	 * {@code function} is reachable but {@code TensorSpec} is not. Even though analysis classifies {@code x} as a tensor and
	 * {@code inferInputSignature} would produce a signature, the source-write must skip the {@code input_signature=...} argument because
	 * {@code TensorSpec} is not in scope under this import shape, emitting a bare {@code @function}. Exercises the
	 * {@code tensorSpecReachable == false} gate that distinguishes the named-import shape from the wildcard shape.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/565">PR 565</a>
	 */
	@Test
	public void testInferInputSignatureEmissionNamedImport() throws Exception {
		helperAssertInputSignatureEmission();
	}

	/**
	 * Fully-qualified-import variant of {@link #testInferInputSignatureEmission()}. With bare {@code import tensorflow} (no {@code as}
	 * alias), the source-write must qualify every emitted name with the {@code tensorflow.} prefix, producing
	 * {@code @tensorflow.function(input_signature=[tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32)])}. Exercises the
	 * {@code import tensorflow} branch of the import-context detection and the non-empty-prefix path of
	 * {@link edu.cuny.hunter.hybridize.core.analysis.InputSignature#toTensorSpecList(String)}. The emitted decorator line exceeds black's
	 * 88-column limit, so the expected {@code out/A.py} is intentionally a single unwrapped line matching the tool's output; the pre-commit
	 * black hook is configured to skip {@code out/} fixtures.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/565">PR 565</a>
	 */
	@Test
	public void testInferInputSignatureEmissionFullyQualifiedImport() throws Exception {
		helperAssertInputSignatureEmission();
	}

	/**
	 * Bundled-import variant (#578, shape A). {@code from tensorflow import function, TensorSpec, float32, constant} brings the decorator,
	 * the spec type, and the dtype constant all into scope unqualified. The first imported name ({@code function}) must not short-circuit
	 * {@code getImportContext} and skip emission; the source-write emits an unqualified
	 * {@code @function(input_signature=[TensorSpec(shape=(), dtype=float32)])}.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/578">Issue 578</a>
	 */
	@Test
	public void testInferInputSignatureEmissionBundledImport() throws Exception {
		helperAssertInputSignatureEmission();
	}

	/**
	 * Mixed-import variant (#578, shape B). {@code from tensorflow import function} plus a later {@code import tensorflow as tf}: the
	 * qualified {@code tf.} import makes the whole signature reachable, so {@code getImportContext} must scan past the bare
	 * {@code from}-import rather than returning early, and emit a {@code tf.}-qualified
	 * {@code @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])}.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/578">Issue 578</a>
	 */
	@Test
	public void testInferInputSignatureEmissionMixedImport() throws Exception {
		helperAssertInputSignatureEmission();
	}

	/**
	 * Named-import variant where {@code TensorSpec} is in scope but the dtype constant is not (#585). With
	 * {@code from tensorflow import function, TensorSpec, constant}, both {@code function} and {@code TensorSpec} are reachable, so the
	 * {@code tensorSpecReachable} gate passes—but {@code inferInputSignature} produces {@code [TensorSpec(shape=(), dtype=float32)]} and
	 * {@code float32} is not in scope. The source-write must skip emission rather than emit an unqualified {@code dtype=float32} that would
	 * raise {@code NameError} at runtime, producing a bare {@code @function}. Exercises the dtype-reachability gate in
	 * {@code computeInputSignatureKeyword}, distinct from the {@code TensorSpec}-not-reachable gate of
	 * {@link #testInferInputSignatureEmissionNamedImport()}.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/585">Issue 585</a>
	 */
	@Test
	public void testInferInputSignatureEmissionNamedImportMissingDType() throws Exception {
		helperAssertInputSignatureEmission();
	}

	/**
	 * Auto-inject variant (#574). The fixture has no TensorFlow import at all, but {@code x} is tensor-typed via {@code np.ones}. The
	 * source-write must inject a {@code from tensorflow import ...} line carrying {@code function}, {@code TensorSpec}, and the inferred
	 * signature's dtype constant, then emit the unqualified {@code input_signature}—rather than injecting a bare
	 * {@code from tensorflow import function} and skipping emission for lack of {@code TensorSpec}/dtype scope.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/574">Issue 574</a>
	 */
	@Test
	public void testInferInputSignatureEmissionAutoInjectImport() throws Exception {
		helperAssertInputSignatureEmission();
	}

	/**
	 * Auto-inject union variant (#588). An import-less file with two hybridizable functions whose inferred signatures need divergent
	 * dtypes. The single auto-injected {@code from tensorflow import ...} line must carry the union of both functions' dtypes so each emits
	 * its {@code input_signature} unqualified, rather than carrying only the first-processed function's dtype and gating the other off
	 * emission. Exercises {@link Function#planAutoInjectedImports}, mirroring the pre-pass the processor runs in {@code createChange}.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/588">Issue 588</a>
	 */
	@Test
	public void testInferInputSignatureEmissionAutoInjectImportUnion() throws Exception {
		this.setInferInputSignatures(true);

		Set<Function> functions = this.getFunctions();
		assertEquals(2, functions.size());

		for (Function function : functions) {
			assertFalse("Fixture functions should be eager pre-refactoring.", function.isHybrid());
			assertTrue("Fixture functions should select `CONVERT_TO_HYBRID` after analysis.",
					function.getTransformations().contains(Transformation.CONVERT_TO_HYBRID));
		}

		// Mirror the processor's pre-pass so the auto-injected import line carries every function's dtypes, not just the first's (#588).
		Function.planAutoInjectedImports(functions);

		// All functions share the file's document; apply every function's edits to it. Highest-offset edits first so the low-offset
		// injected import does not shift later anchors (mirrors `helperAssertInputSignatureEmission`).
		IDocument doc = functions.iterator().next().getContainingDocument();

		List<TextEdit> edits = new ArrayList<>();
		for (Function function : functions)
			edits.addAll(function.transform());

		edits.sort(Comparator.comparingInt(TextEdit::getOffset).reversed());

		for (TextEdit edit : edits)
			edit.apply(doc);

		String expected = this.getFileContents(this.getOutputTestFileName("A"));
		assertEqualLines(expected, doc.get());
	}

	/**
	 * Runs the refactoring on the current test's fixture and asserts the produced source matches the expected {@code out/A.py}. The single
	 * fixture function must be eager pre-refactoring, select {@link Transformation#CONVERT_TO_HYBRID}, and carry the harness-enabled
	 * {@code inferInputSignatures} flag. Shared by the input-signature emission tests, which differ only in their import-shape fixture.
	 */
	private void helperAssertInputSignatureEmission() throws Exception {
		// Emission is opt-in per test (off by default suite-wide; see #580). The input-signature emission tests enable it here.
		this.setInferInputSignatures(true);

		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertFalse("Fixture function `f` should be eager pre-refactoring.", f.isHybrid());
		assertTrue("Fixture function `f` should select `CONVERT_TO_HYBRID` after analysis.",
				f.getTransformations().contains(Transformation.CONVERT_TO_HYBRID));
		assertTrue("Harness `inferInputSignatures` flag should propagate to the analyzed function's flag.", f.getInferInputSignatures());

		// The inferred signature is computed during analysis (not deferred to the change), so it is observable before `transform()` runs.
		// This pins the analysis-time computation the evaluator's inferred-signature column reads via `getInferredInputSignature()`.
		assertTrue("Inferred signature should be available after analysis for an eager->hybrid candidate.",
				f.getInferredInputSignature().isPresent());

		// Apply the `TextEdit`s directly to the function's in-memory document. The shared `compareOutputTestFile` path would do the
		// same comparison via the existing infrastructure, but the test's `ResourceStub`-backed `IFile` can't be resolved to a URI by
		// `TextFileBufferManager`. Tracked at #359. When that lands, these tests can collapse to setting `compareOutputTestFile`.
		IDocument doc = f.getContainingDocument();

		// Apply highest-offset edits first so an inserted import (low offset) does not shift the unapplied decorator edit's anchor. In
		// production the edits compose in one LTK change tree, which coordinates offsets; this loop applies them as independent trees.
		List<TextEdit> edits = new ArrayList<>(f.transform());
		edits.sort(Comparator.comparingInt(TextEdit::getOffset).reversed());

		for (TextEdit edit : edits)
			edit.apply(doc);

		String expected = this.getFileContents(this.getOutputTestFileName("A"));
		assertEqualLines(expected, doc.get());
	}

	/**
	 * Test that de-hybridizing a function whose {@code @tf.function} decorator carries arguments removes the ENTIRE decorator, not just its
	 * name. Regression for #681: {@code Function.convertToEager()} computed the delete span from the decorator name alone, orphaning a
	 * parameterized decorator's argument list as an invalid bare {@code (...)}.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/681">Issue 681</a>
	 */
	@Test
	public void testConvertToEagerParameterizedDecorator() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertTrue("Fixture function should be hybrid pre-refactoring.", f.isHybrid());
		assertTrue("Fixture function should select CONVERT_TO_EAGER.", f.getTransformations().contains(CONVERT_TO_EAGER));

		IDocument doc = f.getContainingDocument();

		List<TextEdit> edits = new ArrayList<>(f.transform());
		edits.sort(Comparator.comparingInt(TextEdit::getOffset).reversed());

		for (TextEdit edit : edits)
			edit.apply(doc);

		assertEqualLines(this.getFileContents(this.getOutputTestFileName("A")), doc.get());
	}

	/**
	 * Test that {@code RECONFIGURE} adds an inferred {@code input_signature=[tf.TensorSpec(...)]} to an already-hybrid function whose bare
	 * {@code @tf.function} decorator (no parentheses) carries no signature. The remaining half of #563: the eager case is
	 * {@code CONVERT_TO_HYBRID} (#565); this is the already-hybrid case. The source-write appends a parenthesized argument list right after
	 * the decorator name.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	@Test
	public void testReconfigureBareDecorator() throws Exception {
		helperAssertReconfigure();
	}

	/**
	 * Empty-parentheses variant of {@link #testReconfigureBareDecorator()}. The existing decorator is {@code @tf.function()}; the
	 * source-write inserts {@code input_signature=[...]} between the parentheses (front-of-arg-list insertion with no trailing arguments).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	@Test
	public void testReconfigureEmptyParens() throws Exception {
		helperAssertReconfigure();
	}

	/**
	 * Non-empty argument-list variant of {@link #testReconfigureBareDecorator()}. The existing decorator already carries a
	 * non-{@code input_signature} keyword ({@code @tf.function(reduce_retracing=True)}); the source-write appends
	 * {@code , input_signature=[...]} at the end of the argument list, after the existing keyword.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	@Test
	public void testReconfigureExistingArgs() throws Exception {
		helperAssertReconfigure();
	}

	/**
	 * Positional-argument variant of {@link #testReconfigureBareDecorator()}. The existing decorator passes {@code func} positionally
	 * ({@code @tf.function(None)}); the source-write appends {@code , input_signature=[...]} at the end of the argument list. Front
	 * insertion would place the keyword argument before the {@code None} positional argument, producing a Python syntax error; appending at
	 * the end keeps the result valid (#595 review).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	@Test
	public void testReconfigurePositionalFunc() throws Exception {
		helperAssertReconfigure();
	}

	/**
	 * Fully-qualified-import variant of {@link #testReconfigureBareDecorator()}. With bare {@code import tensorflow} (no {@code as} alias),
	 * the source-write must qualify every emitted name with the {@code tensorflow.} prefix when reconfiguring the existing
	 * {@code @tensorflow.function} decorator.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	@Test
	public void testReconfigureFullyQualifiedImport() throws Exception {
		helperAssertReconfigure();
	}

	/**
	 * A function that already supplies an {@code input_signature} must not be reconfigured. Per the presence/parse three-state contract
	 * (#557), a supplied signature must not be clobbered; validate-then-overwrite is future work. Even with inference enabled,
	 * {@code RECONFIGURE} is not selected, no passing precondition is set, and the function keeps its {@code HAS_NO_PRIMITIVE_PARAMETERS}
	 * failure. Pins the deferral of the supplied-signature case.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	@Test
	public void testReconfigureExistingSignatureDeferred() throws Exception {
		this.setInferInputSignatures(true);

		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertTrue("Fixture function `f` should be hybrid.", f.isHybrid());
		assertTrue("A supplied `input_signature` must be detected.", f.getHybridizationParameters().hasInputSignatureParam());
		assertFalse("A function that already supplies an `input_signature` must not be reconfigured.",
				f.getTransformations().contains(RECONFIGURE));
		assertNull("The supplied-signature case is deferred, so no passing precondition is set.", f.getPassingPrecondition());
		assertNotNull("The deferred function keeps its HAS_NO_PRIMITIVE_PARAMETERS failure.",
				f.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * A hybrid function under a named import ({@code from tensorflow import function}) that brings the decorator into scope but not
	 * {@code TensorSpec}: the inferred signature cannot be emitted unqualified, so {@code RECONFIGURE} must not be selected (it would be a
	 * no-op). The function keeps its {@code HAS_NO_PRIMITIVE_PARAMETERS} status. Pins the emittability gate on {@code RECONFIGURE}
	 * selection, ensuring a passing precondition is never reported for a transformation that would produce no edit.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	@Test
	public void testReconfigureNamedImportMissingTensorSpec() throws Exception {
		this.setInferInputSignatures(true);

		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertTrue("Fixture function `f` should be hybrid.", f.isHybrid());
		assertFalse("Emission is impossible under this import shape, so RECONFIGURE must not be selected.",
				f.getTransformations().contains(RECONFIGURE));
		assertNull("No passing precondition when the signature is not emittable.", f.getPassingPrecondition());
		assertNotNull("The function keeps its HAS_NO_PRIMITIVE_PARAMETERS failure.",
				f.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * With input-signature inference disabled (the suite default), the precondition matrix must be unchanged: a good-hybrid function still
	 * hits the {@code HAS_NO_PRIMITIVE_PARAMETERS} failure and selects no transformation. Proves {@code RECONFIGURE} is gated behind the
	 * flag and existing behavior is preserved for the whole suite.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	@Test
	public void testReconfigureFlagOff() throws Exception {
		// The `inferInputSignatures` flag defaults off; intentionally do not enable it.
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertTrue("Fixture function `f` should be hybrid.", f.isHybrid());
		assertFalse("With inference disabled, RECONFIGURE must not be selected.", f.getTransformations().contains(RECONFIGURE));
		assertTrue("The default precondition matrix must be unchanged: no transformation.", f.getTransformations().isEmpty());
		assertNull("No passing precondition with the flag off.", f.getPassingPrecondition());
		assertNotNull("The good-hybrid function must still hit the HAS_NO_PRIMITIVE_PARAMETERS failure when the flag is off.",
				f.getEntryMatchingFailure(HAS_NO_PRIMITIVE_PARAMETERS));
	}

	/**
	 * Runs the refactoring on the current test's fixture and asserts the produced source matches the expected {@code out/A.py}. The single
	 * fixture function must be hybrid pre-refactoring, carry no {@code input_signature}, select {@link Transformation#RECONFIGURE} with
	 * passing precondition {@link PreconditionSuccess#P4}, and carry the harness-enabled {@code inferInputSignatures} flag. Shared by the
	 * reconfigure tests, which differ only in their decorator/import-shape fixture.
	 */
	private void helperAssertReconfigure() throws Exception {
		// Emission is opt-in per test (off by default suite-wide; see #580). The reconfigure tests enable it here.
		this.setInferInputSignatures(true);

		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertTrue("Fixture function `f` should be hybrid pre-refactoring.", f.isHybrid());
		assertEquals("Fixture function `f` should select `RECONFIGURE` after analysis.", singleton(RECONFIGURE), f.getTransformations());
		assertEquals("`RECONFIGURE` selection should set the P4 passing precondition.", P4, f.getPassingPrecondition());

		// Apply the `TextEdit`s directly to the function's in-memory document, mirroring `helperAssertInputSignatureEmission` (the
		// `ResourceStub`-backed `IFile` can't be resolved to a URI by `TextFileBufferManager`; tracked at #359).
		IDocument doc = f.getContainingDocument();

		List<TextEdit> edits = new ArrayList<>(f.transform());
		edits.sort(Comparator.comparingInt(TextEdit::getOffset).reversed());

		for (TextEdit edit : edits)
			edit.apply(doc);

		String expected = this.getFileContents(this.getOutputTestFileName("A"));
		assertEqualLines(expected, doc.get());
	}

	/**
	 * Modify path (#596), supplied-tighter: the existing {@code input_signature} is more specific than the call-site evidence (a concrete
	 * rank-1 shape against call sites of differing rank, which infer an unknown-rank shape), so it is overwritten with the inferred one and
	 * reported informationally.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/596">Issue 596</a>
	 */
	@Test
	public void testReconfigureOverwriteTighter() throws Exception {
		helperAssertReconfigureOverwrite(false, "narrower than its call sites require");
	}

	/**
	 * Modify path (#596), incomparable: the existing {@code input_signature} is incomparable with the inferred one (a float32 dtype against
	 * an int32 call site), so it is overwritten with a warning.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/596">Issue 596</a>
	 */
	@Test
	public void testReconfigureOverwriteIncomparable() throws Exception {
		helperAssertReconfigureOverwrite(true, "disagrees with its call sites");
	}

	/**
	 * Shared assertion for the modify-path overwrite tests: the single hybrid fixture function selects {@link Transformation#RECONFIGURE}
	 * with passing precondition {@link PreconditionSuccess#P5}, emits a status of the expected severity (warning for an incomparable
	 * signature, informational otherwise) containing {@code messageFragment}, and rewrites the decorator to match {@code out/A.py}.
	 *
	 * @param expectWarning True iff the per-category status should be a warning (the incomparable case).
	 * @param messageFragment A fragment the per-category status message must contain.
	 */
	private void helperAssertReconfigureOverwrite(boolean expectWarning, String messageFragment) throws Exception {
		this.setInferInputSignatures(true);

		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertTrue("Fixture function `f` should be hybrid.", f.isHybrid());
		assertEquals("A modify-path overwrite should select `RECONFIGURE`.", singleton(RECONFIGURE), f.getTransformations());
		assertEquals("A modify-path overwrite should set the P5 passing precondition.", P5, f.getPassingPrecondition());

		boolean found = Arrays.stream(f.getStatus().getEntries())
				.anyMatch(e -> (expectWarning ? e.isWarning() : e.isInfo()) && e.getMessage().contains(messageFragment));
		assertTrue("Expected a " + (expectWarning ? "warning" : "informational") + " status containing: " + messageFragment, found);

		IDocument doc = f.getContainingDocument();
		List<TextEdit> edits = new ArrayList<>(f.transform());
		edits.sort(Comparator.comparingInt(TextEdit::getOffset).reversed());

		for (TextEdit edit : edits)
			edit.apply(doc);

		assertEqualLines(this.getFileContents(this.getOutputTestFileName("A")), doc.get());
	}

	/**
	 * Modify path (#596), supplied-broader: the existing {@code input_signature} (an unknown-rank shape) is broader than the call sites
	 * require, so it is preserved (not overwritten) and the divergence is reported informationally.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/596">Issue 596</a>
	 */
	@Test
	public void testReconfigurePreserveBroader() throws Exception {
		this.setInferInputSignatures(true);

		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertTrue("Fixture function `f` should be hybrid.", f.isHybrid());
		assertFalse("A broader supplied signature must not be overwritten.", f.getTransformations().contains(RECONFIGURE));
		assertTrue("Expected an informational status about preserving the broader signature.", Arrays.stream(f.getStatus().getEntries())
				.anyMatch(e -> e.isInfo() && e.getMessage().contains("broader than its call sites require")));
	}

	/**
	 * Modify path (#596), agreement: the supplied {@code input_signature} matches the inferred one, so the refactoring is a no-op (no
	 * transformation selected).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/596">Issue 596</a>
	 */
	@Test
	public void testReconfigureAgreement() throws Exception {
		this.setInferInputSignatures(true);

		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertTrue("Fixture function `f` should be hybrid.", f.isHybrid());
		assertFalse("An agreeing signature must not be reconfigured.", f.getTransformations().contains(RECONFIGURE));
	}

	/**
	 * Modify path (#596), flag-off guard: with input-signature inference disabled (the suite default), an existing signature is never
	 * compared or overwritten, so the default precondition matrix is unchanged.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/596">Issue 596</a>
	 */
	@Test
	public void testReconfigureModifyFlagOff() throws Exception {
		// The `inferInputSignatures` flag defaults off; intentionally do not enable it.
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function f = functions.iterator().next();
		assertTrue("Fixture function `f` should be hybrid.", f.isHybrid());
		assertFalse("With inference disabled, an existing signature must not be overwritten.",
				f.getTransformations().contains(RECONFIGURE));
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

		List<Parameter> params = function.getParameters();

		// no params.
		assertEquals(0, params.size());

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

		List<Parameter> params = function.getParameters();

		// one param.
		assertEquals(1, params.size());

		Parameter actualParameter = params.get(0);
		assertNotNull(actualParameter);

		String paramName = actualParameter.getName();
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

		List<Parameter> params = function.getParameters();

		// one param.
		assertEquals(1, params.size());

		Parameter actualParameter = params.get(0);
		assertNotNull(actualParameter);

		String paramName = actualParameter.getName();
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

		List<Parameter> params = function.getParameters();

		// one param.
		assertEquals(1, params.size());

		Parameter actualParameter = params.get(0);
		assertNotNull(actualParameter);

		String paramName = actualParameter.getName();
		assertEquals("x", paramName);

		assertFalse(function.getHasTensorParameter());
	}

	/**
	 * Exercises {@link Parameter#getTensorTypes()} on the shape-divergence/same-dtype scenario ported from wala/ML's
	 * {@code tf2_test_function8.py}: parameter {@code t} is reached by {@code tf.constant(l)} where {@code l} is one of two literal lists
	 * of different rank. Ariadne therefore associates two {@link TensorType}s with {@code t}, both {@code float32}, with shapes
	 * {@code (2, 1)} and {@code (2,)} respectively.
	 */
	@Test
	public void testInferredTensorTypes() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());

		List<Parameter> parameters = function.getParameters();
		assertNotNull(parameters);
		assertEquals("Function `func` has exactly one parameter `t`.", 1, parameters.size());

		Parameter t = parameters.get(0);
		assertEquals("t", t.getName());
		assertEquals(0, t.getIndex());

		Set<TensorType> inferred = t.getTensorTypes();
		assertNotNull(inferred);

		Set<TensorType> expected = Set.of(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(1))),
				new TensorType(FLOAT32, List.of(new NumericDim(2))));
		assertEquals(expected, inferred);

		// Multi-context input with rank disagreement (rank 2 vs rank 1): dtype consensus passes (both float32), shape axis degrades
		// to ⊥ (null dims). `inferInputSignature` emits a coarse `TensorType(FLOAT32, null)` signature rather than dropping the
		// parameter.
		Optional<InputSignature> signature = function.inferInputSignature().signature();
		assertTrue("Multi-context rank-disagreement emits a coarse `TensorType(FLOAT32, null)` signature.", signature.isPresent());
		assertEquals("Expected a single-parameter signature.", 1, signature.get().parameterTypes().size());
		TensorType spec = signature.get().parameterTypes().get(0);
		assertEquals("Spec dtype must be FLOAT32.", FLOAT32, spec.getDType());
		assertNull("Spec dims must be null (shape-⊤ from rank disagreement).", spec.getDims());

		// Wrapper identity contract: equals/hashCode/toString.
		assertEquals(t, t);
		assertEquals(t.hashCode(), t.hashCode());
		assertNotNull(t.toString());
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

		List<Parameter> params = function.getParameters();

		// no params.
		assertEquals(0, params.size());

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
		assertTrue(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());

		List<Parameter> params = function.getParameters();

		// no params.
		assertEquals(0, params.size());

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
		assertTrue(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());
		// TODO: And the value is true (#111).

		List<Parameter> params = function.getParameters();

		// one param.
		assertEquals(1, params.size());

		Parameter actualParameter = params.get(0);
		assertNotNull(actualParameter);

		String paramName = actualParameter.getName();
		assertEquals("x", paramName);

		// get the type hint.
		String typeHintName = actualParameter.getTypeHintName();

		// no type hint.
		assertNull(typeHintName);

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
		assertFalse(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());

		List<Parameter> params = function.getParameters();

		// one param.
		assertEquals(1, params.size());

		Parameter actualParameter = params.get(0);
		assertNotNull(actualParameter);

		String paramName = actualParameter.getName();
		assertEquals("x", paramName);

		// get the type hint.
		String typeHintName = actualParameter.getTypeHintName();

		// Tensor type hint.
		assertEquals("tf.Tensor", typeHintName);

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
		assertTrue(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());

		List<Parameter> params = function.getParameters();

		// one param.
		assertEquals(1, params.size());

		Parameter actualParameter = params.get(0);
		assertNotNull(actualParameter);

		String paramName = actualParameter.getName();
		assertEquals("x", paramName);

		// get the type hint.
		String typeHintName = actualParameter.getTypeHintName();

		assertEquals("tf.Tensor", typeHintName);

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
		assertTrue(function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());

		// But, it's set to False.
		// TODO: assert that the experimental type hints param is set to false (#111).

		List<Parameter> params = function.getParameters();

		// one param.
		assertEquals(1, params.size());

		Parameter actualParameter = params.get(0);
		assertNotNull(actualParameter);

		String paramName = actualParameter.getName();
		assertEquals("x", paramName);

		String typeHintName = actualParameter.getTypeHintName();
		assertEquals("tf.Tensor", typeHintName);

		// NOTE: Set to assertFalse() when #111 is fixed.
		assertTrue(function.getHasTensorParameter());
	}

	/**
	 * Test for #2. From https://tensorflow.org/guide/function#usage. The .py calls {@code add(tf.ones([1, 2]), tf.ones([2, 2]))} once, so
	 * {@code a} binds to FLOAT32 shape {@code (1, 2)} and {@code b} to FLOAT32 shape {@code (2, 2)}—asymmetric per-parameter expectations.
	 * The inferred input signature combines the per-parameter ground truths into a two-parameter signature pinning each to its concrete
	 * shape and dtype.
	 */
	@Test
	public void testHasLikelyTensorParameter11() throws Exception {
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
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

		List<Parameter> params = function.getParameters();

		// two params.
		assertEquals(2, params.size());

		Parameter actualParameter = params.get(0);
		assertNotNull(actualParameter);

		String paramName = actualParameter.getName();
		assertEquals("a", paramName);

		actualParameter = params.get(1);
		assertNotNull(actualParameter);

		paramName = actualParameter.getName();
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

			List<Parameter> params = function.getParameters();

			List<String> expectedParameters = fut.getParameters();
			assertEquals(expectedParameters.size(), params.size());

			for (int i = 0; i < params.size(); i++) {
				Parameter actualParameter = params.get(i);
				assertNotNull(actualParameter);

				String paramName = actualParameter.getName();
				assertEquals(expectedParameters.get(i), paramName);
			}

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.getHasTensorParameter());
		}

		// Precision audit. Top-level call: `dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))`. Inside the body,
		// `add(tf.matmul(x, w), b)` calls `add` with the matmul result and `b`. Reading the semantics, every tensor here is FLOAT32
		// with concrete shape: dense_layer's `x` is (3, 2), `w` is (2, 2), `b` is (2,); add's `a` is the matmul output (3, 2), and add's
		// `b` threads through from dense_layer's `b` so it is also (2,).
		Function denseLayerFunc = nameToFunctions.get("dense_layer").iterator().next();
		List<Parameter> dlParams = denseLayerFunc.getParameters();
		assertEquals(Set.of(new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(2)))), dlParams.get(0).getTensorTypes());
		assertEquals(Set.of(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)))), dlParams.get(1).getTensorTypes());
		assertEquals(Set.of(new TensorType(FLOAT32, List.of(new NumericDim(2)))), dlParams.get(2).getTensorTypes());
		Optional<InputSignature> dlSig = denseLayerFunc.inferInputSignature().signature();
		assertTrue(dlSig.isPresent());
		assertEquals(List.of(new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(2))),
				new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2))),
				new TensorType(FLOAT32, List.of(new NumericDim(2)))), dlSig.get().parameterTypes());

		Function addFunc = nameToFunctions.get("add").iterator().next();
		List<Parameter> addParams = addFunc.getParameters();
		assertEquals(Set.of(new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(2)))), addParams.get(0).getTensorTypes());
		assertEquals(Set.of(new TensorType(FLOAT32, List.of(new NumericDim(2)))), addParams.get(1).getTensorTypes());
		Optional<InputSignature> addSig = addFunc.inferInputSignature().signature();
		assertTrue(addSig.isPresent());
		assertEquals(List.of(new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(2))),
				new TensorType(FLOAT32, List.of(new NumericDim(2)))), addSig.get().parameterTypes());
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

			List<Parameter> params = function.getParameters();

			List<String> expectedParameters = fut.getParameters();
			assertEquals(expectedParameters.size(), params.size());

			for (int i = 0; i < params.size(); i++) {
				Parameter actualParameter = params.get(i);
				assertNotNull(actualParameter);

				String paramName = actualParameter.getName();
				assertEquals(expectedParameters.get(i), paramName);
			}

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.getHasTensorParameter());
		}

		// Precision audit. `image = tf.zeros([1, 200, 200, 100])` followed by `conv_fn(image)`. Single call, single tensor parameter,
		// FLOAT32 dtype, shape (1, 200, 200, 100).
		Function convFn = nameToFunctions.get("conv_fn").iterator().next();
		Parameter image = convFn.getParameters().get(0);
		TensorType expectedImage = new TensorType(FLOAT32,
				List.of(new NumericDim(1), new NumericDim(200), new NumericDim(200), new NumericDim(100)));
		assertEquals(Set.of(expectedImage), image.getTensorTypes());
		Optional<InputSignature> convFnSig = convFn.inferInputSignature().signature();
		assertTrue(convFnSig.isPresent());
		assertEquals(List.of(expectedImage), convFnSig.get().parameterTypes());
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

			List<Parameter> params = function.getParameters();

			List<String> expectedParameters = fut.getParameters();
			assertEquals(expectedParameters.size(), params.size());

			for (int i = 0; i < params.size(); i++) {
				Parameter actualParameter = params.get(i);
				assertNotNull(actualParameter);

				String paramName = actualParameter.getName();
				assertEquals(expectedParameters.get(i), paramName);
			}

			assertTrue("Expecting " + function + " to likely have a tensor-like parameter.", function.getHasTensorParameter());
		}

		// Precision audit. `double` is called with `tf.constant(1)` (INT32), `tf.constant(1.1)` (FLOAT32), and `tf.constant("a")` /
		// `tf.constant("b")` (STRING). Multi-context with three distinct scalar dtypes — algorithm drops signature on dtype
		// disagreement (|D| != 1).
		Function dbl = nameToFunctions.get("double").iterator().next();
		Parameter a = dbl.getParameters().get(0);
		assertEquals(Set.of(new TensorType(INT32, List.of()), new TensorType(FLOAT32, List.of()), new TensorType(STRING, List.of())),
				a.getTensorTypes());
		Optional<InputSignature> dblSig = dbl.inferInputSignature().signature();
		assertFalse("Expected signature drop due to dtype disagreement across call sites.", dblSig.isPresent());

		// See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/510: the `inferSpec`-side drop must surface a
		// per-parameter INFO naming the reason, not collapse silently.
		RefactoringStatusEntry dblEntry = dbl.getStatus().getEntryMatchingCode(Function.PLUGIN_ID, INPUT_SIGNATURE_INFERENCE.getCode());
		assertNotNull("Expected an INPUT_SIGNATURE_INFERENCE INFO for the `inferSpec` heterogeneous-dtype drop (#510).", dblEntry);
		assertEquals("Status entry must be INFO severity.", INFO, dblEntry.getSeverity());
		assertTrue("Status message must cite parameter `a`.", dblEntry.getMessage().contains("`a`"));
		assertTrue("Status message must name the dtype-disagreement reason.", dblEntry.getMessage().contains("conflicting dtypes"));
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

		// Precision audit. `f(tf.random.uniform([5]))` — FLOAT32 rank-1 with shape (5,).
		Function f = nameToFunctions.get("f").iterator().next();
		Parameter x = f.getParameters().get(0);
		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(5)));
		assertEquals(Set.of(expected), x.getTensorTypes());
		Optional<InputSignature> sig = f.inferInputSignature().signature();
		assertTrue(sig.isPresent());
		assertEquals(List.of(expected), sig.get().parameterTypes());
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

		List<Parameter> params = function.getParameters();
		assertEquals(3, params.size());

		Parameter z = params.get(0);
		assertEquals("z", z.getName());
		Parameter a = params.get(1);
		assertEquals("a", a.getName());
		Parameter b = params.get(2);
		assertEquals("b", b.getName());

		assertTrue("Expecting function with likely tensor parameter.", function.getHasTensorParameter());

		// Precision audit. `add(5, tf.ones([1, 2]), tf.ones([2, 2]))` — `z` gets primitive int 5 (not a tensor); `a` and `b` get
		// FLOAT32 tensors of shape (1, 2) and (2, 2).
		assertTrue("Expected z (primitive int) to carry no tensor types.", z.getTensorTypes().isEmpty());
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		assertEquals(Set.of(expectedA), a.getTensorTypes());
		assertEquals(Set.of(expectedB), b.getTensorTypes());
	}

	/**
	 * Test for #2 for TF API `tf.zeros`.
	 */
	@Test
	public void testHasLikelyTensorParameter20() throws Exception {
		// Precision audit. `add(tf.zeros([1, 2]), tf.zeros([2, 2]))` — FLOAT32 with concrete shapes (1, 2) and (2, 2).
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `tf.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter21() throws Exception {
		// Precision audit. `add(tf.constant([1, 2]), tf.constant([2, 2]))` — Python int literal lists give INT32 dtype; shape (2,) for
		// both.
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `tf.Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter22() throws Exception {
		// Precision audit. `add(tf.Variable([1.0, 2.0]), tf.Variable([2.0, 2.0]))` — verified at runtime: FLOAT32, shape (2,) for both.
		// `tf.Variable` is treated as tensor-equivalent for hybridization purposes (per testHasLikelyTensorParameter12 docstring).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `tf.random.uniform`.
	 */
	@Test
	public void testHasLikelyTensorParameter23() throws Exception {
		// Precision audit. `add(tf.random.uniform([1, 2]), tf.random.uniform([2, 2]))` — FLOAT32 default; shapes (1, 2) and (2, 2).
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `tf.SparseTensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter24() throws Exception {
		// Precision audit. `add(tf.SparseTensor([[0,0],[1,2]], [1,2], [3,4]), ...)` — verified at runtime: INT32 dtype, dense shape (3, 4).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), new NumericDim(4))).asSparse());
	}

	/**
	 * Test for #2 for TF API `tf.Tensor`. Direct `tf.Tensor(op, value_index, dtype)` construction in a `tf.Graph()` context — TF1-style
	 * code that exercises the low-level constructor. Verified at runtime via {@code python3.10}: FLOAT32, shape {@code ()} (rank-0 scalar).
	 */
	@Test
	public void testHasLikelyTensorParameter25() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of()));
	}

	/**
	 * Test for #2 for TF API `tf.fill`.
	 */
	@Test
	public void testHasLikelyTensorParameter26() throws Exception {
		// Precision audit. `tf.fill([1, 2], 2)` and `tf.fill([2, 2], 1)` — fill-value is int, so INT32 dtype; shapes (1, 2) and (2, 2).
		TensorType expectedA = new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(INT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `tf.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter27() throws Exception {
		// Precision audit. `tf.eye(2, 3)` × 2 — identity matrix; FLOAT32 default dtype; shape (2, 3).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))));
	}

	/**
	 * Test for #2 for TF API `tf.zero_likes`.
	 */
	@Test
	public void testHasLikelyTensorParameter28() throws Exception {
		// Precision audit. `tf.zeros_like([1, 2])` × 2 — input is a Python int list of length 2; output is INT32 shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `tf.one_hot`.
	 */
	@Test
	public void testHasLikelyTensorParameter29() throws Exception {
		// Precision audit. `tf.one_hot([0, 1, 2], 3)` × 2 — input length 3 + depth 3; default FLOAT32 dtype; output shape (3, 3).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(3))));
	}

	/**
	 * Test for #2 for TF API `tf.convert_to_tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter30() throws Exception {
		// Precision audit. `tf.convert_to_tensor(1)` / `tf.convert_to_tensor(2)` — Python ints → INT32 scalar (rank-0).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of()));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter31() throws Exception {
		// Precision audit. `tf.range(3, 18, 3)` / `tf.range(5)` — INT32 (range with int args); both produce shape (5,).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `ones`.
	 */
	@Test
	public void testHasLikelyTensorParameter32() throws Exception {
		// Precision audit. `tensorflow.ones(...)` (fully-qualified) — FLOAT32 shapes (1, 2) and (2, 2).
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `ones`.
	 */
	@Test
	public void testHasLikelyTensorParameter33() throws Exception {
		// Precision audit. `tf.ones(...)` + `ones(...)` from `tensorflow.python.ops.array_ops` — same FLOAT32 shapes.
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `random.uniform`.
	 */
	@Test
	public void testHasLikelyTensorParameter34() throws Exception {
		// Precision audit. `tf.random.uniform(...)` + `random.uniform(...)` from `tensorflow` — same FLOAT32 shapes.
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `random.uniform`.
	 */
	@Test
	public void testHasLikelyTensorParameter35() throws Exception {
		// Precision audit. `uniform(...)` directly + `random.uniform(...)` — same FLOAT32 shapes.
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `random.uniform`.
	 */
	@Test
	public void testHasLikelyTensorParameter36() throws Exception {
		// Precision audit. `uniform(...)` + `random_uniform(...)` from `tensorflow.python.ops.random_ops` — same FLOAT32 shapes.
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter37() throws Exception {
		// Precision audit. `tf.Variable([1.0, 2.0])` + `Variable([2.0, 2.0])` from `tensorflow.python.ops.variables` — FLOAT32 shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter38() throws Exception {
		// Precision audit. `tensorflow.Variable(...)` (fully-qualified) — FLOAT32 shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter39() throws Exception {
		// Precision audit. `tensors.Variable(...)` (aliased import) + `Variable(...)` directly — FLOAT32 shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter40() throws Exception {
		// Precision audit. `variables.Variable(...)` from `tensorflow.python.ops` — FLOAT32 shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter41() throws Exception {
		// Precision audit. Two FLOAT32 tensors with shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `Variable`.
	 */
	@Test
	public void testHasLikelyTensorParameter42() throws Exception {
		// Precision audit. Two FLOAT32 tensors with shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter43() throws Exception {
		// Precision audit. Two INT32 tensors with shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter44() throws Exception {
		// Precision audit. Two INT32 tensors with shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `zeros`.
	 */
	@Test
	public void testHasLikelyTensorParameter45() throws Exception {
		// Precision audit. Two FLOAT32 tensors with shapes (1, 2) and (2, 2).
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `SparseTensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter46() throws Exception {
		// Precision audit. Two INT32 SparseTensors with dense shape (3, 4).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), new NumericDim(4))).asSparse());
	}

	/**
	 * Test for #2 for TF API `fill`.
	 */
	@Test
	public void testHasLikelyTensorParameter47() throws Exception {
		// Precision audit. Two INT32 tensors with shapes (1, 2) and (2, 2).
		TensorType expectedA = new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(INT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * Test for #2 for TF API `zeros_like`.
	 */
	@Test
	public void testHasLikelyTensorParameter48() throws Exception {
		// Precision audit. Two INT32 tensors with shape (2,).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `one_hot`.
	 */
	@Test
	public void testHasLikelyTensorParameter49() throws Exception {
		// Precision audit. Two FLOAT32 one-hot tensors with shape (3, 3).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(3))));
	}

	/**
	 * Test for #2 for TF API `convert_tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter50() throws Exception {
		// Precision audit. Two INT32 scalar tensors (rank-0).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of()));
	}

	/**
	 * Test for #2 for TF API `range`.
	 */
	@Test
	public void testHasLikelyTensorParameter51() throws Exception {
		// Precision audit. `tensorflow.range(3, 18, 3)` + `range(5)` from `tensorflow.python.ops.math_ops` — INT32 shape (5,).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `Tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter52() throws Exception {
		// Precision audit. `t = tf.Tensor(c.op, 0, tf.float32)` where `c = tf.matmul(a, b)` and `a`, `b` are FLOAT32 (2, 2) constants.
		// Runtime: `t` is a FLOAT32 tensor wrapping the matmul output of shape (2, 2).
		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelperMultiFunction("func2", false, expected, expected);
	}

	/**
	 * Test for #2 for TF API `Tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter53() throws Exception {
		// Precision audit. `t = Tensor(c.op, 0, tensorflow.float32)` (raw import) wrapping `c = tensorflow.matmul(a, b)`.
		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelperMultiFunction("func2", false, expected, expected);
	}

	/**
	 * Test for #2 for TF API `Tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter54() throws Exception {
		// Precision audit. `t = Tensor(c.op, 0, tf.float32)` from `tensorflow.python.framework.ops` wrapping `c = tf.matmul(a, b)`.
		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelperMultiFunction("func2", false, expected, expected);
	}

	/**
	 * Test for #2 for TF API `eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter55() throws Exception {
		// Precision audit. `tensorflow.eye(2, 3)` + `eye(2, 3)` from `tensorflow.python.ops.linalg_ops` — FLOAT32 shape (2, 3).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))));
	}

	/**
	 * Test for #2 for TF API `Tensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter56() throws Exception {
		// Precision audit. Raw `Tensor(op, value_index, dtype)` from `tensorflow.python.framework.ops` — FLOAT32 scalar (rank-0).
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of()));
	}

	/**
	 * Test for #2 for TF API `SparseTensor`.
	 */
	@Test
	public void testHasLikelyTensorParameter57() throws Exception {
		// Precision audit. `SparseTensor(...)` from `tensorflow` — INT32 dense shape (3, 4).
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), new NumericDim(4))).asSparse());
	}

	/**
	 * Test for #2 for TF API `ones`.
	 */
	@Test
	public void testHasLikelyTensorParameter58() throws Exception {
		// Precision audit. `ones(...)` direct import — FLOAT32 shapes (1, 2) and (2, 2).
		TensorType expectedA = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedB = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedA), Set.of(expectedB), List.of(expectedA, expectedB));
	}

	/**
	 * General precision-audit helper for the canonical two-parameter fixture shape with an expected non-empty input signature. Verifies
	 * that the file under test contains exactly one function with exactly two parameters named {@code a} and {@code b}, asserts
	 * {@link Function#isHybrid()} and {@link Function#getHasTensorParameter()}, asserts each parameter's {@link Parameter#getTensorTypes()}
	 * set, and asserts the {@link Function#inferInputSignature()} value equals {@code expectedSignature}.
	 *
	 * @param expectingHybridFunction The expected value of {@link Function#isHybrid()} for the loaded function.
	 * @param expectingTensorParameter The expected value of {@link Function#getHasTensorParameter()} for the loaded function.
	 * @param aTensorTypes The expected {@link Parameter#getTensorTypes()} set for parameter {@code a}.
	 * @param bTensorTypes The expected {@link Parameter#getTensorTypes()} set for parameter {@code b}.
	 * @param expectedSignature The expected per-parameter {@link TensorType} list in {@code (a, b)} order; must be non-{@code null}. For
	 *        fixtures expecting the inferred signature to be dropped, use
	 *        {@link #testHasLikelyTensorParameterHelperExpectingDrop(boolean, boolean, Set, Set)} instead.
	 * @throws Exception If the underlying analysis fails.
	 */
	private void testHasLikelyTensorParameterHelper(boolean expectingHybridFunction, boolean expectingTensorParameter,
			Set<TensorType> aTensorTypes, Set<TensorType> bTensorTypes, List<TensorType> expectedSignature) throws Exception {
		assertNotNull("Helper contract: expectedSignature must be non-null;"
				+ " for an expected dropped signature, call testHasLikelyTensorParameterHelperExpectingDrop.", expectedSignature);
		assertNotNull("Helper contract: aTensorTypes must be non-null.", aTensorTypes);
		assertNotNull("Helper contract: bTensorTypes must be non-null.", bTensorTypes);

		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertEquals(expectingHybridFunction, function.isHybrid());

		List<Parameter> params = function.getParameters();
		assertEquals(2, params.size());
		Parameter a = params.get(0);
		Parameter b = params.get(1);
		assertNotNull(a);
		assertNotNull(b);
		assertEquals("a", a.getName());
		assertEquals("b", b.getName());
		assertEquals(expectingTensorParameter, function.getHasTensorParameter());
		assertEquals(aTensorTypes, a.getTensorTypes());
		assertEquals(bTensorTypes, b.getTensorTypes());
		assertEquals(Optional.of(expectedSignature), function.inferInputSignature().signature().map(InputSignature::parameterTypes));
	}

	/**
	 * Precision-audit helper for the canonical two-parameter fixture shape where the inferred input signature is expected to be dropped.
	 * The signature drop typically reflects a {@code inferSpec} short-circuit (e.g., no tensor parameter, dtype disagreement) where the
	 * per-parameter Ariadne data is still asserted but the overall signature cannot be inferred. Structural and per-parameter assertions
	 * match {@link #testHasLikelyTensorParameterHelper(boolean, boolean, Set, Set, List)}; only the final signature comparison differs.
	 *
	 * @param expectingHybridFunction The expected value of {@link Function#isHybrid()} for the loaded function.
	 * @param expectingTensorParameter The expected value of {@link Function#getHasTensorParameter()} for the loaded function.
	 * @param aTensorTypes The expected {@link Parameter#getTensorTypes()} set for parameter {@code a}.
	 * @param bTensorTypes The expected {@link Parameter#getTensorTypes()} set for parameter {@code b}.
	 * @throws Exception If the underlying analysis fails.
	 */
	private void testHasLikelyTensorParameterHelperExpectingDrop(boolean expectingHybridFunction, boolean expectingTensorParameter,
			Set<TensorType> aTensorTypes, Set<TensorType> bTensorTypes) throws Exception {
		assertNotNull("Helper contract: aTensorTypes must be non-null.", aTensorTypes);
		assertNotNull("Helper contract: bTensorTypes must be non-null.", bTensorTypes);

		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertEquals(expectingHybridFunction, function.isHybrid());

		List<Parameter> params = function.getParameters();
		assertEquals(2, params.size());
		Parameter a = params.get(0);
		Parameter b = params.get(1);
		assertNotNull(a);
		assertNotNull(b);
		assertEquals("a", a.getName());
		assertEquals("b", b.getName());
		assertEquals(expectingTensorParameter, function.getHasTensorParameter());
		assertEquals(aTensorTypes, a.getTensorTypes());
		assertEquals(bTensorTypes, b.getTensorTypes());
		assertEquals(Optional.empty(), function.inferInputSignature().signature().map(InputSignature::parameterTypes));
	}

	/**
	 * Precision-audit overload for the canonical two-parameter fixture shape where both parameters {@code a} and {@code b} are expected to
	 * share the same per-parameter type at each of Layer 1 (Ariadne) and Layer 2 (Hybridize). The two-layer form is needed when Ariadne's
	 * emitted {@link TensorType} differs from the {@link TensorType} the inference algorithm produces—an upstream representation gap or an
	 * algorithm-side collapse. The canonical case is ragged tensors; see {@link #testHasLikelyTensorParameter59()} for the TODO-anchored
	 * fixture and the upstream/downstream flip targets https://github.com/wala/ML/issues/544 and
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/524.
	 *
	 * @param expectedParameterTensorType The {@link TensorType} expected from Ariadne via {@link Parameter#getTensorTypes()}; applied
	 *        identically to {@code a} and {@code b}.
	 * @param expectedSignatureTensorType The {@link TensorType} expected in the inferred input signature via
	 *        {@link Function#inferInputSignature()}; applied identically to {@code a} and {@code b}.
	 * @throws Exception If the underlying analysis fails.
	 */
	private void testHasLikelyTensorParameterHelper(TensorType expectedParameterTensorType, TensorType expectedSignatureTensorType)
			throws Exception {
		testHasLikelyTensorParameterHelper(false, true, Set.of(expectedParameterTensorType), Set.of(expectedParameterTensorType),
				List.of(expectedSignatureTensorType, expectedSignatureTensorType));
	}

	/**
	 * Precision-audit overload for the canonical two-parameter fixture shape where Layer 1 (Ariadne) and Layer 2 (Hybridize) agree: both
	 * parameters {@code a} and {@code b} carry {@code expected} from Ariadne and the inferred input signature is
	 * {@code [expected, expected]}. Suitable for IDEAL fixtures (dense tensors with concrete shape and dtype) and for sparse-tensor
	 * fixtures whose numerical assertion matches the dense form even though their runtime spec emission is semantically distinct (see
	 * #533).
	 *
	 * @param expected The {@link TensorType} expected at Layer 1 (Ariadne's {@link Parameter#getTensorTypes()}) and at Layer 2 (Hybridize's
	 *        {@link Function#inferInputSignature()}), applied identically to both parameters {@code a} and {@code b}.
	 * @throws Exception If the underlying analysis fails.
	 */
	private void testHasLikelyTensorParameterHelper(TensorType expected) throws Exception {
		testHasLikelyTensorParameterHelper(expected, expected);
	}

	/**
	 * Precision-audit overload for the canonical two-parameter fixture shape where Layer 1 (Ariadne) and Layer 2 (Hybridize) agree and the
	 * expected hybrid-function status must be supplied. Equivalent to {@link #testHasLikelyTensorParameterHelper(TensorType)} but
	 * parameterized on {@code expectingHybridFunction}; needed for fixtures exercising {@code @tf.function}-decorated functions where the
	 * structural check expects {@code isHybrid() == true}.
	 *
	 * @param expectingHybridFunction The expected value of {@link Function#isHybrid()} for the loaded function.
	 * @param expected The {@link TensorType} expected at Layer 1 and Layer 2, applied identically to both parameters {@code a} and
	 *        {@code b}.
	 * @throws Exception If the underlying analysis fails.
	 */
	private void testHasLikelyTensorParameterHelper(boolean expectingHybridFunction, TensorType expected) throws Exception {
		testHasLikelyTensorParameterHelper(expectingHybridFunction, true, Set.of(expected), Set.of(expected), List.of(expected, expected));
	}

	/**
	 * Precision-audit overload for the no-tensor-parameter case: neither parameter is classified as a tensor. Asserts both
	 * {@link Parameter#getTensorTypes()} are empty and the inferred signature is {@link Optional#empty}—the signature drops at the
	 * per-parameter classification step before reaching {@code inferSpec}, distinct from in-{@code inferSpec} drops (e.g., dtype
	 * disagreement) where the per-parameter classification succeeded.
	 *
	 * @param expectingHybridFunction The expected value of {@link Function#isHybrid()} for the loaded function.
	 * @throws Exception If the underlying analysis fails.
	 */
	private void testHasLikelyTensorParameterHelperNoTensor(boolean expectingHybridFunction) throws Exception {
		testHasLikelyTensorParameterHelperExpectingDrop(expectingHybridFunction, false, Set.of(), Set.of());
	}

	/**
	 * Precision-audit overload for multi-context fixtures where each parameter is observed with multiple distinct {@link TensorType}s
	 * across call sites and the inference algorithm narrows to a single inferred type via per-dim consensus. Applies
	 * {@code expectedParameterTensorTypes} identically to both {@code a} and {@code b} and asserts the signature is
	 * {@code [expectedSignatureTensorType, expectedSignatureTensorType]}.
	 *
	 * @param expectedParameterTensorTypes The {@link TensorType} multi-context Set expected from Ariadne via
	 *        {@link Parameter#getTensorTypes()} for each parameter.
	 * @param expectedSignatureTensorType The single {@link TensorType} expected in the inferred input signature after per-dim consensus
	 *        narrowing.
	 * @throws Exception If the underlying analysis fails.
	 */
	private void testHasLikelyTensorParameterHelperMultiContext(Set<TensorType> expectedParameterTensorTypes,
			TensorType expectedSignatureTensorType) throws Exception {
		testHasLikelyTensorParameterHelper(false, true, expectedParameterTensorTypes, expectedParameterTensorTypes,
				List.of(expectedSignatureTensorType, expectedSignatureTensorType));
	}

	/**
	 * Precision-audit helper for multi-function fixtures with a single-parameter target function named {@code t}. Loads exactly two
	 * functions in the test fixture, picks the one with {@code targetFunctionSimpleName} (asserting exactly one match), and asserts the
	 * structural shape (single parameter named {@code t}), {@link Function#isHybrid()}, that the target has a tensor parameter, the
	 * per-parameter {@link Parameter#getTensorTypes()}, and {@link Function#inferInputSignature()}. This helper covers the
	 * tensor-parameter-present case only; if a future fixture needs to assert a signature drop or absent tensor parameter, add a sibling
	 * helper (analogous to {@link #testHasLikelyTensorParameterHelperExpectingDrop(boolean, boolean, Set, Set)}).
	 *
	 * @param targetFunctionSimpleName The simple name of the function under test (one of the two functions in the fixture).
	 * @param expectingHybridFunction The expected value of {@link Function#isHybrid()} for the target function.
	 * @param expectedParameterTensorType The {@link TensorType} expected from Ariadne via {@link Parameter#getTensorTypes()} for the single
	 *        parameter {@code t}.
	 * @param expectedSignatureTensorType The {@link TensorType} expected in the inferred input signature.
	 * @throws Exception If the underlying analysis fails.
	 */
	private void testHasLikelyTensorParameterHelperMultiFunction(String targetFunctionSimpleName, boolean expectingHybridFunction,
			TensorType expectedParameterTensorType, TensorType expectedSignatureTensorType) throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(2, functions.size());

		List<Function> matches = functions.stream().filter(f -> Objects.equals(f.getSimpleName(), targetFunctionSimpleName)).toList();
		assertEquals("Expected exactly one function named " + targetFunctionSimpleName + " in fixture.", 1, matches.size());
		Function target = matches.get(0);
		assertEquals(expectingHybridFunction, target.isHybrid());

		List<Parameter> params = target.getParameters();
		assertEquals(1, params.size());
		Parameter t = params.get(0);
		assertNotNull(t);
		assertEquals("t", t.getName());
		assertTrue(target.getHasTensorParameter());
		assertEquals(Set.of(expectedParameterTensorType), t.getTensorTypes());
		assertEquals(Optional.of(List.of(expectedSignatureTensorType)),
				target.inferInputSignature().signature().map(InputSignature::parameterTypes));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`. Canonical fixture for the ragged-tensor precision gap. Runtime:
	 * {@code tf.RaggedTensor.from_nested_row_splits(values, splits)} dtype=int32, shape=(3, None, None). Ariadne emits an INT32
	 * {@link TensorType} with three dims: a known constant 3 followed by two {@link RaggedDim} raggedness markers (per wala/ML#544 /
	 * ponder-lab/ML#320, shipped in Ariadne 0.45.0). The inference algorithm collapses ragged markers to {@code SymbolicDim("?")} wildcards
	 * in the inferred signature, producing a {@code TensorSpec}-shaped signature that admits the runtime call but loses the raggedness
	 * signal. TODO(https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/524): flip from a {@code TensorSpec} with wildcards
	 * to a {@code RaggedTensorSpec} once the consumer-side emission lands.
	 */
	@Test
	public void testHasLikelyTensorParameter59() throws Exception {
		TensorType expectedParameterTensorType = new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE));
		TensorType expectedSignatureTensorType = new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE));
		testHasLikelyTensorParameterHelper(expectedParameterTensorType, expectedSignatureTensorType);
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter60() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter61() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter62() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter63() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter64() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter65() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_splits`.
	 */
	@Test
	public void testHasLikelyTensorParameter66() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter67() throws Exception {
		testHasLikelyTensorParameterHelper(
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE)),
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, new SymbolicDim("?"))));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter68() throws Exception {
		testHasLikelyTensorParameterHelper(
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE)),
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, new SymbolicDim("?"))));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter69() throws Exception {
		testHasLikelyTensorParameterHelper(
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE)),
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, new SymbolicDim("?"))));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter70() throws Exception {
		testHasLikelyTensorParameterHelper(
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE)),
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, new SymbolicDim("?"))));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter71() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter72() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter73() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_limits`.
	 */
	@Test
	public void testHasLikelyTensorParameter74() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_limits`.
	 */
	@Test
	public void testHasLikelyTensorParameter75() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_limits`.
	 */
	@Test
	public void testHasLikelyTensorParameter76() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_limits`.
	 */
	@Test
	public void testHasLikelyTensorParameter77() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_starts`.
	 */
	@Test
	public void testHasLikelyTensorParameter78() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_row_starts`.
	 */
	@Test
	public void testHasLikelyTensorParameter79() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(5), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter80() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(4), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(4), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter81() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(4), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(4), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_value_rowids`.
	 */
	@Test
	public void testHasLikelyTensorParameter82() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(4), RaggedDim.INSTANCE)),
				new TensorType(INT32, List.of(new NumericDim(4), RaggedDim.INSTANCE)));
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter83() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(10), new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter84() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(10), new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter85() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(10), new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `random.gamma`.
	 */
	@Test
	public void testHasLikelyTensorParameter86() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(10), new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `random.normal`.
	 */
	@Test
	public void testHasLikelyTensorParameter87() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(4))));
	}

	/**
	 * Test for #2 for TF API `random.normal`.
	 */
	@Test
	public void testHasLikelyTensorParameter88() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(4))));
	}

	/**
	 * Test for #2 for TF API `random.normal`.
	 */
	@Test
	public void testHasLikelyTensorParameter89() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(4))));
	}

	/**
	 * Test for #2 for TF API `random.poisson`.
	 */
	@Test
	public void testHasLikelyTensorParameter90() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(10), new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `random.poisson`.
	 */
	@Test
	public void testHasLikelyTensorParameter91() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(10), new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `random.truncated_normal`.
	 */
	@Test
	public void testHasLikelyTensorParameter92() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `random.truncated_normal`.
	 */
	@Test
	public void testHasLikelyTensorParameter93() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `sparse.eye`. The parameter is inferred as a sparse {@code TensorType}, so the signature emits a
	 * {@code SparseTensorSpec} that admits the {@code SparseTensor} arguments (#533).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/533">Hybridize#533</a>
	 */
	@Test
	public void testHasLikelyTensorParameter94() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))).asSparse());
	}

	/**
	 * Test for #2 for TF API `sparse.eye`. The parameter is inferred as a sparse {@code TensorType}, so the signature emits a
	 * {@code SparseTensorSpec} that admits the {@code SparseTensor} arguments (#533).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/533">Hybridize#533</a>
	 */
	@Test
	public void testHasLikelyTensorParameter95() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))).asSparse());
	}

	/**
	 * Test for #2 for TF API `sparse.eye`. The parameter is inferred as a sparse {@code TensorType}, so the signature emits a
	 * {@code SparseTensorSpec} that admits the {@code SparseTensor} arguments (#533).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/533">Hybridize#533</a>
	 */
	@Test
	public void testHasLikelyTensorParameter96() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(3))).asSparse());
	}

	/**
	 * Test for #2 for TF API `linalg.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter97() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `linalg.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter98() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `linalg.eye`.
	 */
	@Test
	public void testHasLikelyTensorParameter99() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter100() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32))),
				new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(32))));
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter101() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32))),
				new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(32))));
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter102() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32))),
				new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(32))));
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter103() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32))),
				new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(32))));
	}

	/**
	 * Test for #2 for TF API `keras.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter104() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32))),
				new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(32))));
	}

	/**
	 * Test for #2 for TF API `ragged.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter105() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `ragged.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter106() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `ragged.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter107() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `ragged.constant`.
	 */
	@Test
	public void testHasLikelyTensorParameter108() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(2))));
	}

	/**
	 * Test for #2 for TF API `sparse.SparseTensor`. The parameter is inferred as a sparse {@code TensorType}, so the signature emits a
	 * {@code SparseTensorSpec} that admits the {@code SparseTensor} arguments (#533).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/533">Hybridize#533</a>
	 */
	@Test
	public void testHasLikelyTensorParameter109() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), new NumericDim(4))).asSparse());
	}

	/**
	 * Test for #2 for TF API `sparse.SparseTensor`. The parameter is inferred as a sparse {@code TensorType}, so the signature emits a
	 * {@code SparseTensorSpec} that admits the {@code SparseTensor} arguments (#533).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/533">Hybridize#533</a>
	 */
	@Test
	public void testHasLikelyTensorParameter110() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), new NumericDim(4))).asSparse());
	}

	/**
	 * Test for #2 for TF API `sparse.SparseTensor`. The parameter is inferred as a sparse {@code TensorType}, so the signature emits a
	 * {@code SparseTensorSpec} that admits the {@code SparseTensor} arguments (#533).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/533">Hybridize#533</a>
	 */
	@Test
	public void testHasLikelyTensorParameter111() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(3), new NumericDim(4))).asSparse());
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter112() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))),
				new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter113() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))),
				new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter114() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))),
				new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter115() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))),
				new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter116() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))),
				new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `ragged.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter117() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))),
				new TensorType(INT32, List.of(new NumericDim(1), new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter118() throws Exception {
		testHasLikelyTensorParameterHelper(
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE)),
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, new SymbolicDim("?"))));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter119() throws Exception {
		testHasLikelyTensorParameterHelper(
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE)),
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, new SymbolicDim("?"))));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter120() throws Exception {
		testHasLikelyTensorParameterHelper(
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE)),
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, new SymbolicDim("?"))));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter121() throws Exception {
		testHasLikelyTensorParameterHelper(
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE)),
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, new SymbolicDim("?"))));
	}

	/**
	 * Test for #2 for TF API `RaggedTensor.from_nested_row_lengths`.
	 */
	@Test
	public void testHasLikelyTensorParameter122() throws Exception {
		testHasLikelyTensorParameterHelper(
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE)),
				new TensorType(STRING, List.of(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, new SymbolicDim("?"))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter123() throws Exception {
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.keras.layers.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter124() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32))),
				new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(32))));
	}

	/**
	 * Test for #2 for TF API `tf.keras.layers.Input`.
	 */
	@Test
	public void testHasLikelyTensorParameter125() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of(DynamicDim.INSTANCE, new NumericDim(32))),
				new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(32))));
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter126() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of()));
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter127() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of()));
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter128() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of()));
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter129() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(FLOAT32, List.of()));
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter130() throws Exception {
		// Precision audit. `t = tf.experimental.numpy.ndarray(c.op, 0, tf.float32)` (fully-qualified) wrapping `c = tf.matmul(a, b)`.
		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelperMultiFunction("func2", false, expected, expected);
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter131() throws Exception {
		// Precision audit. `t = experimental.numpy.ndarray(c.op, 0, tf.float32)` (`from tensorflow import experimental`) wrapping
		// `c = tf.matmul(a, b)`.
		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelperMultiFunction("func2", false, expected, expected);
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter132() throws Exception {
		// Precision audit. `t = numpy.ndarray(c.op, 0, tf.float32)` (`from tensorflow.experimental import numpy`) wrapping
		// `c = tf.matmul(a, b)`.
		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelperMultiFunction("func2", false, expected, expected);
	}

	/**
	 * Test for #2 for TF API `tf.experimental.numpy.ndarray`.
	 */
	@Test
	public void testHasLikelyTensorParameter133() throws Exception {
		// Precision audit. `t = ndarray(c.op, 0, tf.float32)` (`from tensorflow.experimental.numpy import ndarray`) wrapping
		// `c = tf.matmul(a, b)`.
		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		testHasLikelyTensorParameterHelperMultiFunction("func2", false, expected, expected);
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter134() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter135() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but the tf.function has parentheses.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter136() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter137() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter138() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter139() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter140() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter141() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter142() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for #2 for TF API `tf.range`.
	 */
	@Test
	public void testHasLikelyTensorParameter143() throws Exception {
		// Same test as testHasLikelyTensorParameter123 but with a tf.function parameter.
		testHasLikelyTensorParameterHelper(true, new TensorType(INT32, List.of(new NumericDim(5))));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/265.
	 */
	@Test
	public void testHasLikelyTensorParameter144() throws Exception {
		testHasLikelyTensorParameterHelper(new TensorType(INT32, List.of()));
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/265. {@code add(element, element)} inside
	 * {@code for element in [tf.ones([1, 2]), tf.ones([2, 2])]}—multi-context across loop iterations; FLOAT32 dtype agrees, rank-2 agrees,
	 * but position-0 dim disagrees (1 vs 2) and position-1 agrees (2 vs 2). The inference algorithm narrows position-0 to a wildcard and
	 * keeps position-1.
	 */
	@Test
	public void testHasLikelyTensorParameter145() throws Exception {
		TensorType expectedParameterTensorTypeContext1 = new TensorType(FLOAT32, List.of(new NumericDim(1), new NumericDim(2)));
		TensorType expectedParameterTensorTypeContext2 = new TensorType(FLOAT32, List.of(new NumericDim(2), new NumericDim(2)));
		TensorType expectedSignatureTensorType = new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(2)));
		testHasLikelyTensorParameterHelperMultiContext(Set.of(expectedParameterTensorTypeContext1, expectedParameterTensorTypeContext2),
				expectedSignatureTensorType);
	}

	/**
	 * Test lists. {@code add(list, list)} where {@code list = [tf.ones([1, 2]), tf.ones([2, 2])]}—both parameters bind to a Python list
	 * literal containing tensors. Phase-3 container detection classifies each parameter as tensor-bearing (so
	 * {@link Function#getHasTensorParameter()} is {@code true}), but the parameter binds to the list itself, not to a tensor; per-parameter
	 * {@link Parameter#getTensorTypes()} is empty. Without a concrete per-parameter tensor type, {@code inferSpec} cannot produce a
	 * {@code TensorSpec}, so the inferred signature drops.
	 */
	@Test
	public void testHasLikelyTensorParameter146() throws Exception {
		testHasLikelyTensorParameterHelperExpectingDrop(false, true, Set.of(), Set.of());
	}

	/**
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/265. {@code add(element, element)} inside
	 * {@code for element in list} where {@code list = list()} then {@code .append(tf.ones(...))}—the dynamic-list construction prevents
	 * Ariadne from tracking the appended tensors into {@code element}, so neither parameter is classified as a tensor and the inferred
	 * signature drops at the per-parameter classification step.
	 */
	@Test
	public void testHasLikelyTensorParameter147() throws Exception {
		testHasLikelyTensorParameterHelperNoTensor(false);
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
	 * Test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/265. Dataset iteration with {@code .batch(2)}: source
	 * is {@code [1, 2, 3]} (length 3), so batching by 2 produces one full batch of shape (2,) and a residual batch of shape (1,). Both
	 * batches are INT32. Ariadne tracks both shapes as a multi-context per-parameter Set; the inference algorithm narrows the disagreeing
	 * dim-0 to a wildcard via per-dim consensus, yielding INT32 (?,) in the signature.
	 */
	@Test
	public void testHasLikelyTensorParameter157() throws Exception {
		TensorType expectedParameterTensorTypeBatch2 = new TensorType(INT32, List.of(new NumericDim(2)));
		TensorType expectedParameterTensorTypeBatch1 = new TensorType(INT32, List.of(new NumericDim(1)));
		TensorType expectedSignatureTensorType = new TensorType(INT32, List.of(new SymbolicDim("?")));
		testHasLikelyTensorParameterHelperMultiContext(Set.of(expectedParameterTensorTypeBatch2, expectedParameterTensorTypeBatch1),
				expectedSignatureTensorType);
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
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing.
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
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing.
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
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing.
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
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing.
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
	 * Test https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing.
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

	/**
	 * Pins the soundness gain of forwarding {@link PythonTensorAnalysisEngine#MODEL_FORWARD_CFA_DEPTH} to the analysis engine (#600).
	 * {@code accuracy}'s {@code y_pred} is bound to a model-forward output ({@code pred = neural_net(...)}) reached from two call sites
	 * that carry different shapes ({@code (256, 10)} from training, {@code (10000, 10)} from test). At the shallow
	 * {@link PythonTensorAnalysisEngine#DEFAULT_TARGETED_CFA_DEPTH} the two contexts collapse and the training shape leaks into the test
	 * call site, so the inferred {@link TensorType} set is a single, unsound concrete shape. At {@code MODEL_FORWARD_CFA_DEPTH} the
	 * contexts separate and the test context falls to a wildcard ({@code null}-dims) over-approximation, so the set strictly grows. A
	 * consumer emitting {@code input_signature} from the shallow result would unsoundly assert {@code y_pred} is always {@code (256, 10)}
	 * when the test path can pass {@code (10000, 10)}. Parameters bound directly to call-site arguments (e.g. {@code NeuralNet.call}'s
	 * {@code x}) are depth-invariant; this output-bound parameter is the one whose type set is depth-sensitive (wala/ML#587).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/600">Issue 600</a>
	 */
	@Test
	public void testTargetedCfaDepthSoundness() throws Exception {
		Set<TensorType> shallow = this.analyzeAccuracyYPredAtDepth(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH);
		Set<TensorType> deep = this.analyzeAccuracyYPredAtDepth(PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH);

		// The shallow targeted depth collapses the two call-site contexts into a single, concrete (unsound) shape.
		assertEquals("Shallow targeted depth should collapse `y_pred` to a single shape.", 1, shallow.size());
		assertTrue("The collapsed shape should be concrete.", shallow.stream().allMatch(t -> t.getDims() != null));

		// The deeper targeted depth separates the contexts: it keeps the concrete shape and adds the wildcard over-approximation.
		assertTrue("Deeper targeted depth must retain every shallow shape.", deep.containsAll(shallow));
		assertEquals("Deeper targeted depth should add exactly the wildcard over-approximation.", shallow.size() + 1, deep.size());

		Set<TensorType> added = new HashSet<>(deep);
		added.removeAll(shallow);
		assertTrue("The added type should be a wildcard (unconstrained-shape) over-approximation.",
				added.stream().allMatch(t -> t.getDims() == null));
	}

	/**
	 * Analyzes the soundness fixture at the given targeted k-CFA depth and returns the inferred {@link TensorType} set for
	 * {@code accuracy}'s {@code y_pred} parameter. Clears the analysis caches first so each depth gets a fresh call graph.
	 *
	 * @param targetedCfaDepth The targeted k-CFA depth to analyze at.
	 * @return The {@link TensorType}s inferred for {@code accuracy}'s {@code y_pred} parameter.
	 */
	private Set<TensorType> analyzeAccuracyYPredAtDepth(int targetedCfaDepth) throws Exception {
		Function.clearCaches();
		this.setTargetedCfaDepth(targetedCfaDepth);
		Function accuracy = this.getFunctions().stream().filter(f -> f.getIdentifier().equals("accuracy")).findFirst()
				.orElseThrow(() -> new AssertionError("Expected an `accuracy` function."));
		Parameter yPred = accuracy.getParameters().stream().filter(p -> "y_pred".equals(p.getName())).findFirst()
				.orElseThrow(() -> new AssertionError("Expected a `y_pred` parameter."));
		return yPred.getTensorTypes();
	}

	/**
	 * A keyword-only parameter (declared after a bare {@code *}) is wrapped as a {@link Parameter} and, when it carries a tensor-typed type
	 * hint, classifies as a tensor. PyDev's by-name type-hint resolver scans only the positional argument array, so {@link Parameter} reads
	 * the keyword-only annotation array directly.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/607">Issue 607</a>
	 */
	@Test
	public void testKwonlyTensorParameter() throws Exception {
		Function f = getFunction("f");

		Parameter y = f.getParameters().stream().filter(p -> "y".equals(p.getName())).findFirst()
				.orElseThrow(() -> new AssertionError("Expected the keyword-only parameter `y` to be wrapped."));

		assertEquals("tf.Tensor", y.getTypeHintName());
		assertTrue("Keyword-only parameter `y` should have a tensor type hint.", y.hasTensorTypeHint(new NullProgressMonitor()));
		assertTrue("Keyword-only parameter `y` should classify as a tensor via its type hint.", y.isTensor());
		assertTrue(f.getHasTensorParameter());
	}

	/**
	 * A keyword-only parameter whose tensor value arrives only through a keyword argument at the call site (no type hint) classifies as a
	 * tensor. WALA models keyword-only parameters as formal parameters (wala/ML#596, Ariadne 0.51.1), so the call-site keyword argument
	 * binds to the parameter and Ariadne types it via the call-site (Phase 2) path.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/607">Issue 607</a>
	 * @see <a href="https://github.com/wala/ML/issues/596">WALA ML issue 596</a>
	 */
	@Test
	public void testKwonlyTensorParameterCallSite() throws Exception {
		Function f = getFunction("f");

		Parameter y = f.getParameters().stream().filter(p -> "y".equals(p.getName())).findFirst()
				.orElseThrow(() -> new AssertionError("Expected the keyword-only parameter `y` to be wrapped."));

		assertTrue("Keyword-only parameter `y` should classify as a tensor from its call-site keyword argument.", y.isTensor());
		assertFalse("Keyword-only parameter `y` should carry a concrete tensor type from the call site.", y.getTensorTypes().isEmpty());
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
	@Ignore("Env-dependent flake; suspected https://github.com/wala/ML/issues/168 (unconfirmed).")
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
	@Ignore("Env-dependent flake; suspected https://github.com/wala/ML/issues/168 (unconfirmed).")
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
	@Ignore("Env-dependent flake; suspected https://github.com/wala/ML/issues/168 (unconfirmed).")
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
	@Ignore("Env-dependent flake; suspected https://github.com/wala/ML/issues/168 (unconfirmed).")
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

	/**
	 * Regression test for https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/324: a method whose only parameter is
	 * {@code self} (e.g., {@code def f(self):}) was previously misclassified as not-a-method because {@link Function#isMethod} required
	 * {@code parameters.size() > 1}. After the fix it requires {@code >= 1}.
	 */
	@Test
	public void testSelfOnlyMethod() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("C.f", function.getIdentifier());
		assertEquals(1, function.getNumberOfParameters());
		assertTrue("A method whose only parameter is `self` should be classified as a method.", function.isMethod());
	}

	/**
	 * Input-signature inference for a function called once with a single tensor of concrete dtype and shape. The expected signature is a
	 * singleton containing that single tensor type.
	 */
	@Test
	public void testInputSignatureScenario1() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter t = parameters.get(0);

		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2)));

		Set<TensorType> ariadne = t.getTensorTypes();
		assertEquals(Set.of(expected), ariadne);

		Optional<InputSignature> signature = function.inferInputSignature().signature();
		assertTrue(signature.isPresent());
		assertEquals(List.of(expected), signature.get().parameterTypes());
	}

	/**
	 * Input-signature inference when the function body uses a tensor that is not a parameter (a module-level tensor closed over by the
	 * function). The inferred signature should still contain only the parameter's TensorType, not the closure-captured tensor. The closure
	 * tensor's shape is intentionally distinct from the parameter's to make a leak observable: an implementation that collected all tensors
	 * and deduplicated by `TensorType` would surface both shapes and fail the singleton assertion.
	 */
	@Test
	public void testInputSignatureNonParameterTensor() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());

		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2)));

		Optional<InputSignature> signature = function.inferInputSignature().signature();
		assertTrue(signature.isPresent());
		assertEquals(List.of(expected), signature.get().parameterTypes());
	}

	/**
	 * Input-signature inference when the singleton {@link TensorType} carries a concrete dtype but shape-⊤ (null dims), as produced by
	 * `tf.keras.Input(shape=json.loads(...))` where Ariadne cannot trace `json.loads`. The shape and dtype axes degrade independently: the
	 * dtype carries through (concrete `float32`) while the shape axis emits ⊤; the result is a valid coarse
	 * {@link TensorType}{@code (FLOAT32, null)} signature rather than a dropped parameter.
	 */
	@Test
	public void testInputSignatureShapeUnknown() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter t = parameters.get(0);

		Set<TensorType> ariadne = t.getTensorTypes();
		assertEquals("Expected exactly one TensorType so we exercise the singleton branch, not the multi-context one.", 1, ariadne.size());
		TensorType single = ariadne.iterator().next();
		assertNull("Expected a shape-⊤ marker (TensorType with null dims).", single.getDims());

		Optional<InputSignature> signature = function.inferInputSignature().signature();
		assertTrue("Singleton with null dims emits a coarse `TensorType(FLOAT32, null)` signature.", signature.isPresent());
		assertEquals("Expected a single-parameter signature.", 1, signature.get().parameterTypes().size());
		TensorType spec = signature.get().parameterTypes().get(0);
		assertEquals("Spec dtype must be FLOAT32.", FLOAT32, spec.getDType());
		assertNull("Spec dims must be null (shape-⊤).", spec.getDims());
	}

	/**
	 * Regression test for #494 / #507 per-dim wildcard emission. The parameter `t` is reached by two same-rank (rank 1) tensors of
	 * differing size at position 0: shape (2,) and shape (3,). The per-dim consensus check at position 0 fails (|D_0| = 2), so `inferSpec`
	 * emits a `SymbolicDim("?")` wildcard at that position. Dtype consensus passes (both float32), so the emitted signature is
	 * `TensorType(FLOAT32, [SymbolicDim("?")])`.
	 */
	@Test
	public void testInputSignaturePerDimWildcard() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter t = parameters.get(0);

		Set<TensorType> ariadne = t.getTensorTypes();
		Set<TensorType> expectedAriadne = Set.of(new TensorType(FLOAT32, List.of(new NumericDim(2))),
				new TensorType(FLOAT32, List.of(new NumericDim(3))));
		assertEquals("Expected two TensorTypes of disagreeing size at position 0.", expectedAriadne, ariadne);

		Optional<InputSignature> signature = function.inferInputSignature().signature();
		assertTrue("Per-dim disagreement emits a wildcard-shape signature.", signature.isPresent());
		assertEquals("Expected a single-parameter signature.", 1, signature.get().parameterTypes().size());
		TensorType spec = signature.get().parameterTypes().get(0);
		assertEquals("Spec dtype must be FLOAT32.", FLOAT32, spec.getDType());
		assertNotNull("Spec dims must be non-null (rank consensus).", spec.getDims());
		assertEquals("Spec must be rank 1.", 1, spec.getDims().size());
		assertTrue("Position 0 must be a wildcard SymbolicDim.", spec.getDims().get(0) instanceof SymbolicDim);
	}

	/**
	 * Pins the concrete typing that wala/ML#539's fix (Ariadne 0.47.0) provides: {@code tf.constant(np.ones(..., dtype=...))} now
	 * propagates the numpy array's shape and dtype, so {@code consume}'s parameter is inferred as a concrete
	 * {@code TensorType(FLOAT32, (2, 3))} rather than full-⊤, and {@code inferInputSignature} produces a signature with no inferSpec-side
	 * drop. Replaces the earlier dtype-⊤ pinning test, which the bump to 0.47.0 inverted (the fix removed the only fixture trigger for the
	 * dtype-⊤ branch).
	 *
	 * @see <a href="https://github.com/wala/ML/issues/539">wala/ML#539</a>
	 */
	@Test
	public void testInputSignatureNumpyConstant() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter x = parameters.get(0);
		assertEquals("x", x.getName());

		// Post wala/ML#539 (Ariadne 0.47.0): the numpy shape/dtype propagate, so `x` types as a concrete FLOAT32 (2, 3) tensor.
		Set<TensorType> inferred = x.getTensorTypes();
		assertEquals("Expected exactly one TensorType for `x`.", 1, inferred.size());
		TensorType only = inferred.iterator().next();
		assertEquals("dtype must be concrete FLOAT32.", FLOAT32, only.getDType());
		assertNotNull("shape must be concrete, not ⊤.", only.getDims());
		assertEquals("Expected a rank-2 shape.", 2, only.getDims().size());

		Optional<InputSignature> signature = function.inferInputSignature().signature();
		assertTrue("A concrete tensor type yields an input signature.", signature.isPresent());
		assertEquals("[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)]", signature.get().toTensorSpecList("tf."));

		// No inferSpec-side drop, so no INPUT_SIGNATURE_INFERENCE drop INFO.
		assertNull("No drop INFO when the signature is inferred.",
				function.getStatus().getEntryMatchingCode(Function.PLUGIN_ID, INPUT_SIGNATURE_INFERENCE.getCode()));
	}

	/**
	 * Regression test for #508 category (b) (Phase-3 container path). A parameter classified as tensor-typed via Phase 3
	 * (`hasTensorContainer`) but with no Phase 2 (Ariadne call-site) shape/dtype evidence: `xs.isTensor()` is TRUE while
	 * `xs.getTensorTypes()` is empty. Per #508, `inferInputSignature` drops the signature and emits a per-parameter INFO referencing #509
	 * (the tool-side recovery: extend the `Parameter` API to surface container constituents).
	 */
	@Test
	public void testInputSignatureContainerParameter() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter xs = parameters.get(0);
		assertEquals("xs", xs.getName());

		assertTrue("Parameter `xs` should be classified as tensor-typed via Phase 3 (container).", xs.isTensor());
		assertEquals("Phase 3 must populate the `isTensorContainer` cache to TRUE.", TRUE, xs.isTensorContainer());
		assertTrue("Parameter `xs` must have an empty `getTensorTypes()` cache (no Phase 2 evidence for the container itself).",
				xs.getTensorTypes().isEmpty());

		Optional<InputSignature> signature = function.inferInputSignature().signature();
		assertFalse("Container-classified parameter without Phase 2 data must yield `Optional.empty`.", signature.isPresent());

		RefactoringStatusEntry entry = function.getStatus().getEntryMatchingCode(PLUGIN_ID, INPUT_SIGNATURE_INFERENCE.getCode());
		assertNotNull("Expected an INPUT_SIGNATURE_INFERENCE INFO status for category (b).", entry);
		assertEquals("Status entry must be INFO severity.", INFO, entry.getSeverity());
		assertTrue("Status message must cite parameter `xs`.", entry.getMessage().contains("`xs`"));
		assertTrue("Status message must reference the tool-side recovery tracker (#509).", entry.getMessage().contains("#509"));
	}

	/**
	 * Regression test for #508 category (b) (Phase-1 type-hint path). A parameter classified as tensor-typed via Phase 1
	 * (`hasTensorTypeHint`) but with no Phase 2 (Ariadne call-site) shape/dtype evidence: the parameter `x` has a `tf.Tensor` type-hint
	 * annotation, classifying it as tensor-typed, while the call site supplies a non-tensor (`int`), so Ariadne's per-parameter cache stays
	 * empty. Per #508, `inferInputSignature` drops the signature and emits a per-parameter INFO referencing #509.
	 */
	@Test
	public void testInputSignatureTypeHintOnly() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter x = parameters.get(0);
		assertEquals("x", x.getName());

		assertTrue("Parameter `x` should be classified as tensor-typed via Phase 1 (type hint).", x.isTensor());
		assertTrue("Parameter `x` must have an empty `getTensorTypes()` cache (call site supplies a non-tensor).",
				x.getTensorTypes().isEmpty());

		Optional<InputSignature> signature = function.inferInputSignature().signature();
		assertFalse("Type-hint-classified parameter without Phase 2 data must yield `Optional.empty`.", signature.isPresent());

		RefactoringStatusEntry entry = function.getStatus().getEntryMatchingCode(PLUGIN_ID, INPUT_SIGNATURE_INFERENCE.getCode());
		assertNotNull("Expected an INPUT_SIGNATURE_INFERENCE INFO status for category (b).", entry);
		assertEquals("Status entry must be INFO severity.", INFO, entry.getSeverity());
		assertTrue("Status message must cite parameter `x`.", entry.getMessage().contains("`x`"));
		assertTrue("Status message must reference the tool-side recovery tracker (#509).", entry.getMessage().contains("#509"));
	}

	/**
	 * Regression test for #508 category (a). A function with a mixed (tensor + non-tensor) parameter list: `t` is a tensor (Phase-2 hit via
	 * the `tf.constant(...)` call site) and `n` is a non-tensor (`int` literal at the call site). The non-tensor parameter `n` blocks
	 * input-signature inference: `inferInputSignature()` returns `Optional.empty` and the function emits an `INPUT_SIGNATURE_INFERENCE`
	 * INFO status with the source-side recovery suggestion (annotate `n` as `tf.Tensor`, wrap call sites with `tf.constant(...)`, rerun).
	 */
	@Test
	public void testInputSignatureNonTensorParameter() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(2, parameters.size());
		Parameter t = parameters.get(0);
		Parameter n = parameters.get(1);
		assertEquals("t", t.getName());
		assertEquals("n", n.getName());

		assertTrue("Parameter `t` should be classified as tensor-typed (Phase 2 hit).", t.isTensor());
		assertFalse("Parameter `n` should not be classified as tensor-typed.", n.isTensor());

		Optional<InputSignature> signature = function.inferInputSignature().signature();
		assertFalse("Mixed (tensor + non-tensor) parameter list must yield Optional.empty.", signature.isPresent());

		RefactoringStatusEntry entry = function.getStatus().getEntryMatchingCode(PLUGIN_ID, INPUT_SIGNATURE_INFERENCE.getCode());
		assertNotNull("Expected an INPUT_SIGNATURE_INFERENCE INFO status when a non-tensor parameter blocks inference.", entry);
		assertEquals("Status entry must be INFO severity.", INFO, entry.getSeverity());
		assertTrue("Status message must cite parameter `n`.", entry.getMessage().contains("`n`"));
		assertTrue("Status message must suggest the source-side recovery (annotate as `tf.Tensor`).",
				entry.getMessage().contains("tf.Tensor"));
	}

	/**
	 * Regression test for #508 category (a) accumulation. A function with multiple non-tensor parameters must emit one
	 * `INPUT_SIGNATURE_INFERENCE` INFO per blocking parameter in a single `inferInputSignature` call, not just the first one. Pins the
	 * accumulate-then-return semantics so developers see all source-side recovery suggestions at once rather than discovering them
	 * one-at-a-time across refactoring reruns.
	 */
	@Test
	public void testInputSignatureMultipleNonTensorParameters() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(2, parameters.size());
		Parameter m = parameters.get(0);
		Parameter n = parameters.get(1);
		assertEquals("m", m.getName());
		assertEquals("n", n.getName());

		assertFalse("Parameter `m` should not be classified as tensor-typed.", m.isTensor());
		assertFalse("Parameter `n` should not be classified as tensor-typed.", n.isTensor());

		InferenceResult result = function.inferInputSignature();
		assertFalse("All-non-tensor parameter list must not yield a signature.", result.signature().isPresent());
		assertEquals("All-non-tensor parameter list must report the non-tensor-parameter absence reason.",
				Optional.of(InferenceResult.AbsenceReason.NON_TENSOR_PARAMETER), result.absenceReason());

		List<RefactoringStatusEntry> infoEntries = Arrays.stream(function.getStatus().getEntries()).filter(e -> e.getSeverity() == INFO)
				.filter(e -> e.getCode() == INPUT_SIGNATURE_INFERENCE.getCode()).collect(Collectors.toList());

		assertEquals("Expected one INPUT_SIGNATURE_INFERENCE INFO per blocking parameter.", 2, infoEntries.size());
		assertTrue("Expected an INFO citing parameter `m`.", infoEntries.stream().anyMatch(e -> e.getMessage().contains("`m`")));
		assertTrue("Expected an INFO citing parameter `n`.", infoEntries.stream().anyMatch(e -> e.getMessage().contains("`n`")));

		// Per-parameter attribution (#654): both blocking parameters are recorded with their reason, in declaration order, where the
		// function-level `absenceReason()` collapses to only the first.
		Map<Parameter, InferenceResult.AbsenceReason> blocking = function.getBlockingParameterReasons();
		assertEquals("Both non-tensor parameters must be recorded as blocking.", 2, blocking.size());
		assertEquals("Parameter `m` blocks for the non-tensor-parameter reason.", InferenceResult.AbsenceReason.NON_TENSOR_PARAMETER,
				blocking.get(m));
		assertEquals("Parameter `n` blocks for the non-tensor-parameter reason.", InferenceResult.AbsenceReason.NON_TENSOR_PARAMETER,
				blocking.get(n));
		assertEquals("Blocking parameters must be in declaration order.", List.of(m, n), new ArrayList<>(blocking.keySet()));
	}

	/**
	 * Regression test for #497: a tensor-container parameter (reached via a list-of-tensors call site) classifies as tensor-typed via
	 * {@link Parameter#classifyAsTensor}'s Phase 3 but does not populate the per-Parameter {@link Set} of {@link TensorType}s (the
	 * container itself is not a tensor in Ariadne's analysis). Pins three relationships:
	 * <ul>
	 * <li>{@link Function#getHasTensorParameter} reflects the parameter-level Phase 3 result.
	 * <li>The {@link Parameter#isTensorContainer} cache returns {@code TRUE} for this parameter.
	 * <li>{@link Parameter#getTensorTypes} stays empty—the asymmetry between the boolean classifier and the type-set cache.
	 * </ul>
	 */
	@Test
	public void testTensorContainerParameterCache() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertTrue("Function with a tensor-container parameter classifies as having a tensor parameter.", function.getHasTensorParameter());

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter a = parameters.get(0);
		assertEquals("a", a.getName());

		assertTrue("Phase 3 classification must populate the `isTensorContainer` cache to TRUE.", a.isTensorContainer());

		// Asymmetry pin: `getTensorTypes()` stays empty because Ariadne does not emit a single TensorType for the container itself;
		// Phase 2's cache-population path runs but finds nothing for this parameter.
		Set<TensorType> tensorTypes = a.getTensorTypes();
		assertTrue("Container parameter must not surface a direct TensorType through `getTensorTypes()`.",
				tensorTypes == null || tensorTypes.isEmpty());
	}

	/**
	 * Pins the cross-method invariant from #501: a tensor-container parameter keeps the {@link Parameter#isTensorContainer} and
	 * {@link Parameter#isTensor} caches in sync, so {@code isTensorContainer() == TRUE} implies {@code isTensor() == TRUE}. Reuses
	 * {@link #testTensorContainerParameterCache}'s fixture shape ({@code f([tf.constant([1.0, 2.0])])}): Phase 3 container detection sets
	 * both caches, while {@link Parameter#getTensorTypes} stays empty (Ariadne emits no single {@link TensorType} for the container
	 * itself). This invariant could only be tested once both #499 ({@code isTensorContainer} cache) and #500 ({@code isTensor} cache) had
	 * landed.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/501">Issue 501</a>
	 */
	@Test
	public void testTensorContainerImpliesTensorClassification() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter a = parameters.get(0);
		assertEquals("a", a.getName());

		assertTrue("Phase 3 container detection must set `isTensorContainer()` to TRUE.", a.isTensorContainer());
		assertTrue("`isTensorContainer() == TRUE` must imply `isTensor() == TRUE` (the cross-method invariant).", a.isTensor());

		Set<TensorType> tensorTypes = a.getTensorTypes();
		assertTrue("The container parameter must not surface a direct TensorType through `getTensorTypes()`.",
				tensorTypes == null || tensorTypes.isEmpty());

		assertTrue("Function-level reflection: a tensor-container parameter implies `getHasTensorParameter() == TRUE`.",
				function.getHasTensorParameter());
	}

	/**
	 * Regression test for #498: pins the classifier→query contract on `Parameter`. After {@link Parameter#classifyAsTensor} runs
	 * (transitively via {@link Function#inferTensorParameters}), {@link Parameter#isTensor} returns the cached classification: {@code TRUE}
	 * for a tensor parameter, {@code FALSE} for a non-tensor parameter. Also pins the function-level reflection:
	 * {@link Function#getHasTensorParameter} is {@code TRUE} iff at least one non-{@code self} parameter has
	 * {@code Parameter.isTensor() == TRUE} (modulo the speculative-context override, which this fixture doesn't trigger).
	 */
	@Test
	public void testParameterClassifyAsTensorContract() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(2, parameters.size());
		Parameter t = parameters.get(0);
		Parameter n = parameters.get(1);
		assertEquals("t", t.getName());
		assertEquals("n", n.getName());

		assertTrue("Parameter `t` (tensor call site) classifies as tensor-typed.", t.isTensor());
		assertFalse("Parameter `n` (non-tensor call site) classifies as non-tensor.", n.isTensor());

		assertTrue("Function has at least one tensor parameter ⇒ `getHasTensorParameter()` is TRUE.", function.getHasTensorParameter());

		// Tighter cache→classifier invariant: non-empty `getTensorTypes()` ⇒ `isTensor() == TRUE`.
		assertFalse("Tensor parameter must have a non-empty `getTensorTypes()` cache (Phase 2 fired).", t.getTensorTypes().isEmpty());

		// The call site is `tf.constant([1.0, 2.0])`: shape (2,), dtype float32.
		Set<Map.Entry<DType, List<Integer>>> tensorShapesDtypes = t.getTensorTypes().stream()
				.map(tt -> Map.entry(tt.getDType(),
						tt.getDims().stream()
								.map(d -> d instanceof TensorType.NumericDim ? ((TensorType.NumericDim) d).value() : Integer.valueOf(-1))
								.collect(Collectors.toList())))
				.collect(toSet());
		assertEquals("Tensor parameter `t` from `tf.constant([1.0, 2.0])` has dtype FLOAT32 and shape (2,).",
				Set.of(Map.entry(FLOAT32, List.of(2))), tensorShapesDtypes);

		// Non-tensor parameter must not surface a TensorType.
		assertTrue("Non-tensor parameter `n` must have an empty `getTensorTypes()` cache.", n.getTensorTypes().isEmpty());
	}

	/**
	 * Regression test for #498 (reverse direction): if {@link Function#getHasTensorParameter} is {@code FALSE}, then no non-{@code self}
	 * parameter has {@code Parameter.isTensor() == TRUE}. The fixture uses a function named `f` (outside the speculative-context regex)
	 * with two non-tensor int arguments, so the speculative-context fallback does not fire.
	 */
	@Test
	public void testParameterClassifyAsTensorContractReverse() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertFalse("Function with no tensor parameters and no speculative-context match: `getHasTensorParameter()` is FALSE.",
				function.getHasTensorParameter());

		List<Parameter> parameters = function.getParameters();
		assertEquals(2, parameters.size());
		for (Parameter param : parameters)
			if (!param.isSelf())
				assertFalse("`getHasTensorParameter()` is FALSE ⇒ no non-self parameter classifies as tensor-typed.", param.isTensor());
	}

	/**
	 * Regression test for #498: a `self` parameter is skipped by `Function.inferTensorParameters` (the classifier never runs on it), so its
	 * cached `isTensor()` stays {@code null}. The owning function has only `self`, so `Function.getHasTensorParameter() == FALSE`: the
	 * speculative-context fallback is gated on `!onlySelfParam`, and the function name `f` doesn't match the regex anyway.
	 */
	@Test
	public void testSelfParameterClassification() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertFalse("Self-only method has no tensor parameter.", function.getHasTensorParameter());

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter self = parameters.get(0);
		assertEquals("self", self.getName());

		assertTrue("Parameter `self` must classify as self.", self.isSelf());
		assertNotEquals("Self parameter is not classified as tensor-typed by `Function.inferTensorParameters` (skipped).", TRUE,
				self.isTensor());
	}

	/**
	 * Regression test for #498: pins the asymmetry that {@link Function#getHasTensorParameter} {@code == TRUE} does NOT imply
	 * {@code ∃ non-self param p : p.isTensor() == TRUE}. The speculative-context fallback (in {@link Function#inferTensorParameters}) can
	 * set `hasTensorParameter = TRUE` based on function name + class lineage, with zero parameter-level tensor evidence.
	 */
	@Test
	public void testSpeculativeContextOverridesParameterEvidence() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals("__call__", function.getSimpleName());

		// Function-level signal: speculative-context fired.
		assertTrue("Speculative-context override sets `getHasTensorParameter()` to TRUE.", function.getHasTensorParameter());

		// Parameter-level signal: no individual parameter classifies as tensor-typed (no type hint, no Ariadne call-site classification,
		// no container).
		List<Parameter> parameters = function.getParameters();
		assertEquals(2, parameters.size());
		Parameter self = parameters.get(0);
		Parameter x = parameters.get(1);
		assertEquals("self", self.getName());
		assertEquals("x", x.getName());

		assertNotEquals("Self parameter is skipped by classifier; not classified as tensor.", TRUE, self.isTensor());
		assertFalse("Asymmetry: function-level TRUE does not imply parameter-level TRUE under speculative-context override.", x.isTensor());
	}

	/**
	 * Regression test for #498: a function with no parameters trivially has no tensor parameter. Pins the empty-param-list edge case:
	 * {@code Function.getHasTensorParameter() == FALSE} (the classification loop iterates over an empty list).
	 */
	@Test
	public void testZeroParameterFunctionHasNoTensorParameter() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertEquals(0, function.getParameters().size());
		assertFalse("Zero-parameter function has no tensor parameter.", function.getHasTensorParameter());
	}

	/**
	 * Regression test for #498: pins that ALL non-self tensor parameters classify independently. Two-tensor-parameter fixture; both
	 * parameters individually classify as tensor-typed, and the function reflects the OR.
	 */
	@Test
	public void testMultipleTensorParameters() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(2, parameters.size());
		Parameter a = parameters.get(0);
		Parameter b = parameters.get(1);
		assertEquals("a", a.getName());
		assertEquals("b", b.getName());

		assertTrue("First tensor parameter classifies as tensor-typed.", a.isTensor());
		assertTrue("Second tensor parameter classifies as tensor-typed independently of the first.", b.isTensor());
		assertTrue("Function with multiple tensor parameters has `getHasTensorParameter() == TRUE`.", function.getHasTensorParameter());
	}

	/**
	 * Regression test for #498: under the test harness's global `ALWAYS_FOLLOW_TYPE_HINTS=true`, a tensor-typed type hint classifies the
	 * parameter even when the per-decorator `experimental_follow_type_hints` flag is absent. Pins that the `followTypeHints` predicate is
	 * the OR of (global flag, per-decorator flag) and that either alone suffices.
	 */
	@Test
	public void testTensorTypeHintHonoredViaGlobalFlag() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		assertTrue("Function is decorated with `@tf.function`.", function.isHybrid());
		assertFalse("`experimental_follow_type_hints` is NOT supplied on the decorator.",
				function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter x = parameters.get(0);
		assertEquals("x", x.getName());

		assertTrue("Type hint honored via the test harness's global `ALWAYS_FOLLOW_TYPE_HINTS=true`.", x.isTensor());
		assertTrue("Type-hint-classified parameter ⇒ `getHasTensorParameter()` is TRUE.", function.getHasTensorParameter());
	}

	/**
	 * Regression test for #498: pins classification across a method-style call (self + tensor parameter in the same function). Self is
	 * skipped by `Function.inferTensorParameters` (cache stays at default); `t` is classified by Ariadne from the tensor call site. The
	 * function reflects the parameter-level tensor signal.
	 */
	@Test
	public void testMethodWithSelfAndTensorParameter() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(2, parameters.size());
		Parameter self = parameters.get(0);
		Parameter t = parameters.get(1);
		assertEquals("self", self.getName());
		assertEquals("t", t.getName());

		assertTrue("Parameter `self` classifies as self.", self.isSelf());
		assertNotEquals("Self parameter is skipped by classifier; not classified as tensor.", TRUE, self.isTensor());
		assertTrue("Tensor parameter `t` classifies as tensor-typed.", t.isTensor());
		assertTrue("Method with a tensor parameter ⇒ `getHasTensorParameter() == TRUE`.", function.getHasTensorParameter());

		// The call site is `C().m(tf.constant([1.0, 2.0]))`: shape (2,), dtype float32.
		Set<Map.Entry<DType, List<Integer>>> tensorShapesDtypes = t.getTensorTypes().stream()
				.map(tt -> Map.entry(tt.getDType(),
						tt.getDims().stream()
								.map(d -> d instanceof TensorType.NumericDim ? ((TensorType.NumericDim) d).value() : Integer.valueOf(-1))
								.collect(Collectors.toList())))
				.collect(toSet());
		assertEquals("Tensor parameter `t` from `tf.constant([1.0, 2.0])` has dtype FLOAT32 and shape (2,).",
				Set.of(Map.entry(FLOAT32, List.of(2))), tensorShapesDtypes);
	}

	/**
	 * Regression test for #498: pins that Phase 1's `hasTensorTypeHint` is specific to TF tensor types. An `int` type hint does NOT
	 * classify the parameter as tensor-typed, even under the test harness's global `ALWAYS_FOLLOW_TYPE_HINTS=true`.
	 */
	@Test
	public void testNonTensorTypeHintNotClassified() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter x = parameters.get(0);
		assertEquals("x", x.getName());

		assertFalse("Non-tensor type hint (`int`) must NOT classify the parameter as tensor-typed.", x.isTensor());
		assertFalse("Function with no tensor classification: `getHasTensorParameter()` is FALSE.", function.getHasTensorParameter());
	}

	/**
	 * Test for #434. A {@code tf.RaggedTensor} type hint classifies the parameter as tensor-typed via the Phase 1 type-hint path: a
	 * {@code RaggedTensor} is a tensor for refactoring purposes, so its FQN is among those recognized by {@code hasTensorTypeHint}. The
	 * argument is a non-tensor, isolating the type-hint signal.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/434">Issue 434</a>
	 */
	@Test
	public void testHasLikelyTensorParameterRagged() throws Exception {
		assertTensorTypeHintClassifies();
	}

	/**
	 * Test for #434. A {@code tf.SparseTensor} type hint classifies the parameter as tensor-typed via the Phase 1 type-hint path.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/434">Issue 434</a>
	 */
	@Test
	public void testHasLikelyTensorParameterSparse() throws Exception {
		assertTensorTypeHintClassifies();
	}

	/**
	 * Test for #434. A {@code tf.Variable} type hint classifies the parameter as tensor-typed via the Phase 1 type-hint path; a
	 * {@code Variable} is treated as tensor-equivalent for hybridization purposes.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/434">Issue 434</a>
	 */
	@Test
	public void testHasLikelyTensorParameterVariable() throws Exception {
		assertTensorTypeHintClassifies();
	}

	/**
	 * Negative test for #434. A {@code tf.RaggedTensorSpec} type hint must NOT classify the parameter as tensor-typed: a {@code *Spec}
	 * descriptor describes a tensor but is not itself one, so its FQN is excluded from the recognized set. With the type hint rejected and
	 * a non-tensor argument, the parameter has no classifying signal.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/434">Issue 434</a>
	 */
	@Test
	public void testHasLikelyTensorParameterRaggedSpec() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue("Fixture function is decorated with `@tf.function`.", function.isHybrid());

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter t = parameters.get(0);
		assertEquals("t", t.getName());

		assertNotEquals("A `*Spec` type hint must NOT classify the parameter as tensor-typed.", TRUE, t.isTensor());
		assertFalse("With no tensor classification, `getHasTensorParameter()` is FALSE.", function.getHasTensorParameter());
	}

	/**
	 * A parameter receiving a {@code tf.RaggedTensor} (here via {@code tf.ragged.constant}) is typed by Ariadne with a {@code RaggedDim} at
	 * the ragged position. Inference preserves that marker rather than collapsing it to a symbolic wildcard, and the emission produces a
	 * {@code tf.RaggedTensorSpec} rather than a dense {@code tf.TensorSpec} (#524).
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/524">Issue 524</a>
	 */
	@Test
	public void testInferRaggedTensorSpec() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue("Parameter receiving a `tf.RaggedTensor` should be classified as tensor-typed.", function.getHasTensorParameter());

		/*
		 * Regression pin for the raw Ariadne type (the probe finding underpinning #524): Ariadne types the `tf.RaggedTensor` parameter with
		 * a `RaggedDim` at the ragged axis. If a future Ariadne version stops emitting `RaggedDim` here, `RaggedTensorSpec` emission would
		 * silently regress to a dense `TensorSpec`.
		 */
		Parameter t = function.getParameters().get(0);
		assertEquals("Ariadne should type the `tf.RaggedTensor` parameter as `(INT32, [NumericDim(2), RaggedDim])`.",
				Set.of(new TensorType(INT32, List.of(new NumericDim(2), RaggedDim.INSTANCE))), t.getTensorTypes());

		InputSignature signature = function.inferInputSignature().signature()
				.orElseThrow(() -> new AssertionError("Expected an inferred signature for the ragged parameter."));

		// The ragged marker survives reduction (not collapsed to a symbolic wildcard).
		List<TensorType.Dimension<?>> dims = signature.parameterTypes().get(0).getDims();
		assertTrue("Inference should preserve the `RaggedDim` marker.", dims.get(dims.size() - 1) instanceof TensorType.RaggedDim);

		// ...and the emission is a `RaggedTensorSpec`, with the ragged position rendered as `None`.
		assertEquals("[tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)]", signature.toTensorSpecList("tf."));
	}

	/**
	 * Shared assertion for the positive #434 type-hint tests: the current fixture's single {@code @tf.function} parameter {@code t} is
	 * classified as tensor-typed via the Phase 1 type-hint path (honored by the harness's global {@code ALWAYS_FOLLOW_TYPE_HINTS}), and the
	 * function reflects that at {@link Function#getHasTensorParameter()}.
	 */
	private void assertTensorTypeHintClassifies() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertTrue("Fixture function is decorated with `@tf.function`.", function.isHybrid());

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter t = parameters.get(0);
		assertEquals("t", t.getName());

		assertTrue("A TF tensor subtype type hint must classify the parameter as tensor-typed (#434).", t.isTensor());
		assertTrue("A type-hint-classified parameter implies `getHasTensorParameter() == TRUE`.", function.getHasTensorParameter());
	}

	/**
	 * Regression test for #486 (shape-⊤). A tensor-typed parameter whose shape Ariadne cannot resolve must surface the shape-⊤ marker (a
	 * {@link TensorType} with {@code null} {@linkplain TensorType#getDims() dims}) so downstream code can distinguish it from a concrete
	 * shape. The marker is visible at the {@link Parameter#getTensorTypes()} level because the {@code TensorTypeAnalysis} iterator emits
	 * the underlying {@link TensorType} unchanged.
	 */
	@Test
	public void testInferredTensorTypesUnknownShapeTop() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());
		assertTrue(function.getHasTensorParameter());

		List<Parameter> parameters = function.getParameters();
		assertNotNull(parameters);
		assertEquals(1, parameters.size());

		Parameter t = parameters.get(0);
		assertEquals("t", t.getName());

		Set<TensorType> inferred = t.getTensorTypes();
		assertNotNull(inferred);
		// NOTE: This assertion is fragile. The fixture relies on Ariadne NOT seeing through `json.loads("[32]")` to defeat shape inference.
		// If Ariadne ever learns to model `json.loads` for compile-time-constant string inputs (tracked at wala/ML#536), this fixture stops
		// producing a shape-⊤ marker and the assertions below flip. A durable shape-⊤ source (immune to wala/ML#536) is now available via
		// the `tf.constant(np.array(...))` construction in testInferredTensorTypesDtypeTop (#491), which pins full-⊤; this fixture is
		// retained as the shape-⊤-with-concrete-dtype lattice point (`float32` from `tf.keras.Input`'s default). Tight assertions (exact
		// size + null dims) ensure that future Ariadne changes—either dropping the marker or emitting additional TensorTypes alongside
		// it—are caught cleanly rather than silently masked.
		assertEquals("Expected exactly one TensorType for parameter `t`.", 1, inferred.size());
		TensorType only = inferred.iterator().next();
		assertNull("Expected shape-⊤ marker (null dims).", only.getDims());
	}

	/**
	 * Regression test for #491 (full-⊤). Pins the full-⊤ marker {@code TensorType(UNKNOWN, null)}—both axes simultaneously unknown—on a
	 * decorated parameter, end-to-end through Ariadne. The source is {@code tf.constant(numpy.array([...]))} with no {@code dtype=}
	 * argument and a list-literal first argument:
	 * <ul>
	 * <li><em>dtype-⊤.</em> {@code NpArray.getDefaultDTypes} returns {@code EnumSet.of(DType.UNKNOWN)} whenever the {@code dtype} argument
	 * is absent (numpy infers the dtype from the data at runtime, which a static points-to analysis does not model), per the wala/ML
	 * lattice contract; {@code tf.constant} propagates it (wala/ML#539).</li>
	 * <li><em>shape-⊤.</em> {@code NpArray.getDefaultShapes} returns the shape of arg 0, and a bare Python list literal's shape is not
	 * modeled, so it falls through to {@code null}.</li>
	 * </ul>
	 * The {@code tf.constant} wrap is load-bearing: a bare {@code numpy.array(...)} does not classify the parameter as tensor-typed (an
	 * un-wrapped ndarray's {@code TensorType} does not propagate to the callee parameter, wala/ML#598), so the {@code tf.constant}
	 * TensorFlow tensor is what carries the ⊤ type to {@code t}. This is a durable full-⊤ source—both axes are unknown by construction
	 * rather than by defeating a specific Ariadne model—unlike the {@code json.loads} shape-⊤ source in
	 * {@link #testInferredTensorTypesUnknownShapeTop()}, which is fragile against wala/ML#536.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/491">Issue 491</a>
	 */
	@Test
	public void testInferredTensorTypesDtypeTop() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());
		assertTrue("The `tf.constant(np.array(...))` source should classify parameter `t` as tensor-typed.",
				function.getHasTensorParameter());

		List<Parameter> parameters = function.getParameters();
		assertNotNull(parameters);
		assertEquals(1, parameters.size());

		Parameter t = parameters.get(0);
		assertEquals("t", t.getName());

		Set<TensorType> inferred = t.getTensorTypes();
		assertNotNull(inferred);
		assertEquals("Expected exactly one TensorType for parameter `t`.", 1, inferred.size());
		TensorType only = inferred.iterator().next();
		assertEquals("Expected dtype-⊤ marker (UNKNOWN). Got: " + only, DType.UNKNOWN, only.getDType());
		assertNull("Expected shape-⊤ marker (null dims) for full-⊤. Got: " + only, only.getDims());
	}

	/**
	 * Characterizes that Ariadne types a parameter from its call-site argument (forward), not from its in-body uses (backward). The
	 * parameter `adj` is used as sparse in one arm of an `isinstance(adj, tf.SparseTensor)` guard and dense in the other, but the single
	 * call site passes a <em>dense</em> argument; `adj` carries only a dense type, with no sparse type leaking from the sparse use. This is
	 * the property that makes the call-site-precise recovery contemplated in #653 unnecessary: because the parameter's inferred type
	 * already reflects its actual arguments, a mixed sparse/dense union is genuine polymorphism (correctly abandoned, #642) rather than a
	 * spurious artifact to recover from. A regression here (a sparse type appearing on `adj`) would mean Ariadne had switched to use-based
	 * typing, reopening #653.
	 */
	@Test
	public void testParameterTypedFromArgumentNotUse() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();

		List<Parameter> parameters = function.getParameters();
		assertEquals(1, parameters.size());
		Parameter adj = parameters.get(0);
		assertEquals("adj", adj.getName());

		Set<TensorType> inferred = adj.getTensorTypes();
		assertFalse("Parameter `adj` should be tensor-typed from its dense argument.", inferred.isEmpty());
		assertEquals(
				"No sparse type should leak onto `adj` from the in-body sparse use; it is typed from its dense argument. Got: " + inferred,
				0, inferred.stream().filter(TensorType::isSparse).count());
	}

	/**
	 * Coverage: a method parameter that receives a direct tensor argument at the call site is tensor-typed. Companion to
	 * {@link #testParameterTypedFromArgumentNotUse} (a module function with a direct tensor argument), adding the instance-method
	 * {@code self} offset to exercise the parameter-position handling.
	 */
	@Test
	public void testTensorParamMethodDirectArg() throws Exception {
		Parameter x = findParameter(this.getFunctions(), "x");
		assertFalse("Method with a direct tensor argument: `x` should be tensor-typed.", x.getTensorTypes().isEmpty());
	}

	/**
	 * Coverage: a module-function parameter that receives a <em>derived</em> tensor argument (another function's result, not a direct
	 * {@code tf.constant}) is tensor-typed, exercising tensor-type propagation across the producing call.
	 */
	@Test
	public void testTensorParamFunctionDerivedArg() throws Exception {
		Parameter x = findParameter(this.getFunctions(), "x");
		assertFalse("Function with a derived tensor argument: `x` should be tensor-typed.", x.getTensorTypes().isEmpty());
	}

	/**
	 * Coverage: an instance method whose parameters receive derived tensor arguments (a method-call result and a threaded parameter) are
	 * tensor-typed, the interprocedural multi-parameter case.
	 */
	@Test
	public void testTensorParamMethodDerivedArg() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertFalse("Method with a derived argument: `real` should be tensor-typed.",
				findParameter(functions, "real").getTensorTypes().isEmpty());
		assertFalse("Method with a derived argument: `pred` should be tensor-typed.",
				findParameter(functions, "pred").getTensorTypes().isEmpty());
	}

	/**
	 * Coverage: a parameter fed a value tuple-unpacked from a Keras {@code __call__} result ({@code predictions, _ = self(inputs)}) is
	 * tensor-typed, exercising tuple-element propagation through the implicit Keras dispatch. The full-program counterpart of this shape is
	 * the under-approximation tracked in wala/ML#618.
	 */
	@Test
	public void testTensorParamTupleUnpackKeras() throws Exception {
		Parameter pred = findParameter(this.getFunctions(), "pred");
		assertFalse("Tuple-unpacked Keras __call__ result: `pred` should be tensor-typed.", pred.getTensorTypes().isEmpty());
	}

	/**
	 * Coverage: the same tuple unpack as {@link #testTensorParamTupleUnpackKeras}, but from a plain method call rather than a Keras
	 * {@code __call__}, isolating tuple-element propagation from the Keras dispatch.
	 */
	@Test
	public void testTensorParamTupleUnpackDirect() throws Exception {
		Parameter pred = findParameter(this.getFunctions(), "pred");
		assertFalse("Tuple-unpacked method result: `pred` should be tensor-typed.", pred.getTensorTypes().isEmpty());
	}

	/**
	 * Finds the analyzed {@link Parameter} with the given name across all the given {@link Function}s.
	 *
	 * @param functions The analyzed functions.
	 * @param name The parameter name to find.
	 * @return The first {@link Parameter} so named.
	 */
	private static Parameter findParameter(Set<Function> functions, String name) {
		return functions.stream().flatMap(f -> f.getParameters().stream()).filter(p -> name.equals(p.getName())).findFirst()
				.orElseThrow(() -> new AssertionError("No parameter named `" + name + "` among analyzed functions."));
	}

	private static Function findFunction(Set<Function> functions, String identifier) {
		return functions.stream().filter(f -> identifier.equals(f.getIdentifier())).findFirst()
				.orElseThrow(() -> new AssertionError("No function with identifier `" + identifier + "` among analyzed functions."));
	}

	/**
	 * Consumer-side guard for the distributed training reach (wala/ML#461). A {@code get_loss(targets, predictions)} call is reached
	 * through {@code distributed_train_step -> strategy.run(_train_step, args=(inputs, targets))}; both parameters resolve to the
	 * {@code (2, 2)} {@code int32} element type, the {@code real} parameter via the dataset target threaded through {@code _train_step} and
	 * the {@code pred} parameter via the keras call result. Pins that the distributed {@code strategy.run} argument forwarding types the
	 * callee parameters from the consumer's perspective.
	 */
	@Test
	public void testGetLossDistributedReach() throws Exception {
		Set<Function> fns = this.getFunctions();
		Set<TensorType> expected = Set.of(new TensorType(INT32, List.of(new NumericDim(2), new NumericDim(2))));
		assertEquals("`get_loss`'s `real` types through the distributed `strategy.run` reach (wala/ML#461).", expected,
				findParameter(fns, "real").getTensorTypes());
		assertEquals("`get_loss`'s `pred` types from the keras call result.", expected, findParameter(fns, "pred").getTensorTypes());
	}

	/**
	 * Guards on the vendored {@code akanyaani/gpt-2-tensorflow2.0} subject. The full subject is vendored verbatim (the model body, its
	 * {@code layers}/{@code utils} packages, and the {@code input_fn} dataset pipeline) and driven through
	 * {@code fit -> train_step -> _train_step -> get_loss(targets, predictions)}.
	 * <p>
	 * (a) {@code get_loss} call-site-to-callee typing (wala/ML#618): as of Ariadne 0.52.8, {@code real} receives the dataset element type
	 * ({@code int32} with a dynamic dimension), so the call-site-to-callee gap is fixed for it; {@code pred}, which flows from the keras
	 * call result rather than the dataset, still receives no tensor type, the residual tracked in wala/ML#618.
	 * <p>
	 * (b) Barren-eager benefit precondition (#709/#712): {@code OutputLayer.call} performs tensor operations ({@code tf.matmul},
	 * {@code tf.reshape}, {@code tf.shape}), but the {@code tf} module global has an empty points-to set in this whole-program context, so
	 * the tensor-op detector must recognize the op via the import-alias fallback ({@link edu.cuny.hunter.hybridize.core.analysis.Util})
	 * rather than misreport the function as performing no tensor computation and block its hybridization.
	 */
	@Test
	public void testGpt2GetLossVendored() throws Exception {
		Set<Function> fns = this.getFunctions();
		assertEquals("`get_loss`'s `real` types as the dataset element type (wala/ML#618 fixed for `real` in Ariadne 0.52.8).",
				Set.of(new TensorType(INT32, List.of(DynamicDim.INSTANCE))), findParameter(fns, "real").getTensorTypes());
		// TODO(wala/ML#618): `pred` should also carry the element type once the residual keras-call-result reach is fixed.
		assertEquals("`get_loss`'s `pred` does not yet type in the full subject (residual wala/ML#618).", Set.of(),
				findParameter(fns, "pred").getTensorTypes());
		assertEquals("`OutputLayer.call` performs a tensor computation (`tf.matmul`), recognized via the import-alias fallback (#712).",
				Boolean.TRUE, findFunction(fns, "OutputLayer.call").getHasTensorComputation());
	}

	/**
	 * Regression guard against Python list-repetition tensor over-typing (fixed by https://github.com/wala/ML/issues/653, released in
	 * Ariadne 0.52.9). {@code rep}'s {@code value} receives {@code [0] * 3}: a Python list repetition, not a tensor, so the parameter must
	 * not be typed as a tensor. Two controls isolate the scenario to a list operand of {@code *}, not the list or {@code *} alone:
	 * {@code lit}'s {@code value} (the list literal {@code [1, 2, 3]}, no {@code *}) and {@code mul}'s {@code value} (scalar {@code 2 * 3})
	 * are likewise not typed as tensors. Distilled from {@code voc_ap} ({@code tp = [0] * nd}) in
	 * {@code YunYang1994/TensorFlow2.0-Examples} (mAP), which the 2024 evaluation typed non-tensor.
	 */
	@Test
	public void testListRepetitionTensorOverTyping() throws Exception {
		assertFalse("`rep`'s `value` receives a list repetition (`[0] * 3`), a Python list and not a tensor.",
				getFunction("rep").getHasTensorParameter());
		assertFalse("`lit`'s `value` receives a list literal (`[1, 2, 3]`); correctly not a tensor.",
				getFunction("lit").getHasTensorParameter());
		assertFalse("`mul`'s `value` receives scalar `2 * 3`; correctly not a tensor (only a list operand of `*` was over-typed).",
				getFunction("mul").getHasTensorParameter());
	}

	/**
	 * Regression guard against typing a parameter fed a subscript-slice of an opaque (argparse) value as a tensor (fixed by
	 * https://github.com/wala/ML/issues/656, released in Ariadne 0.52.9): no tensor reaches {@code check}'s {@code value}, so it must not
	 * be typed as a tensor.
	 */
	@Test
	public void testSubscriptSliceOpaqueOverTyping() throws Exception {
		assertFalse("`check`'s `value` (slice of an opaque argparse attribute) is not a tensor.",
				getFunction("check").getHasTensorParameter());
		// Control: the same opaque attribute without a slice.
		assertFalse("`plain`'s `value` (the opaque argparse attribute, no slice); not a tensor.",
				getFunction("plain").getHasTensorParameter());
	}

	/**
	 * Pins the hybrid-to-eager benefit precondition (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/709): a hybrid
	 * function with a tensor parameter that performs no tensor computation and has no Python side-effects is de-hybridized (P6,
	 * {@link Transformation#CONVERT_TO_EAGER}), while a computing hybrid function is not.
	 */
	@Test
	public void testBarrenHybridDehybridizes() throws Exception {
		Function barren = getFunction("barren");
		assertTrue("`barren` is hybrid.", barren.isHybrid());
		assertTrue("`barren` has a tensor parameter.", barren.getHasTensorParameter());
		assertFalse("`barren` performs no tensor computation.", barren.getHasTensorComputation());
		assertEquals("A barren hybrid function de-hybridizes (P6).", P6, barren.getPassingPrecondition());
		assertTrue("`barren` selects CONVERT_TO_EAGER.", barren.getTransformations().contains(CONVERT_TO_EAGER));

		Function compute = getFunction("compute");
		assertTrue("`compute` is hybrid.", compute.isHybrid());
		assertTrue("`compute` performs a tensor computation.", compute.getHasTensorComputation());
		assertFalse("A computing hybrid function is not de-hybridized as barren.", compute.getTransformations().contains(CONVERT_TO_EAGER));
	}

	/**
	 * Pins the eager-to-hybrid benefit precondition (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/709): an eager
	 * function with a tensor parameter but no tensor computation must not hybridize (it fails with
	 * {@link PreconditionFailure#NO_TENSOR_COMPUTATION}), while one that performs a tensor op still passes P1.
	 */
	@Test
	public void testNoTensorComputationBlocksHybridization() throws Exception {
		Function barren = getFunction("barren");
		assertTrue("`barren` has a tensor parameter.", barren.getHasTensorParameter());
		assertFalse("`barren` performs no tensor computation.", barren.getHasTensorComputation());
		assertNull("`barren` must not pass a precondition; it performs no tensor computation.", barren.getPassingPrecondition());
		assertNotNull("`barren` fails with NO_TENSOR_COMPUTATION.",
				barren.getStatus().getEntryMatchingCode(Function.PLUGIN_ID, PreconditionFailure.NO_TENSOR_COMPUTATION.getCode()));

		Function compute = getFunction("compute");
		assertTrue("`compute` performs a tensor computation.", compute.getHasTensorComputation());
		assertEquals("`compute` performs a tensor op, so it still hybridizes (P1).", P1, compute.getPassingPrecondition());
	}

	/**
	 * Barren counterpart of {@link #testSpeculativeAnalysis4}: the same keras {@code Model} with the original empty {@code call} body. The
	 * parameter still types speculatively, but the function performs no tensor computation, so it fails with
	 * {@link PreconditionFailure#NO_TENSOR_COMPUTATION} instead of hybridizing (issue 709).
	 */
	@Test
	public void testSpeculativeAnalysis4Barren() throws Exception {
		Function f = this.getFunctions().iterator().next();
		assertFalse(f.isHybrid());
		assertFalse("The empty `call` body performs no tensor computation.", f.getHasTensorComputation());
		assertNull(f.getPassingPrecondition());
		assertNotNull(f.getEntryMatchingFailure(PreconditionFailure.NO_TENSOR_COMPUTATION));
	}

	/**
	 * Barren counterpart of {@link #testPreconditionChecking2}: {@code func} has a tensor parameter but an empty body, so it fails with
	 * {@link PreconditionFailure#NO_TENSOR_COMPUTATION} instead of hybridizing (issue 709).
	 */
	@Test
	public void testPreconditionChecking2Barren() throws Exception {
		Function f = getFunction("func");
		assertTrue(f.getHasTensorParameter());
		assertFalse("The empty body performs no tensor computation.", f.getHasTensorComputation());
		assertNull(f.getPassingPrecondition());
		assertNotNull(f.getEntryMatchingFailure(PreconditionFailure.NO_TENSOR_COMPUTATION));
	}

	/**
	 * Barren counterpart of {@link #testRetracing2}: {@code f} returns its tensor parameter unchanged, so it performs no tensor computation
	 * and fails with {@link PreconditionFailure#NO_TENSOR_COMPUTATION} instead of hybridizing (issue 709).
	 */
	@Test
	public void testRetracing2Barren() throws Exception {
		Function f = getFunction("f");
		assertTrue(f.getHasTensorParameter());
		assertFalse("Returning the parameter unchanged performs no tensor computation.", f.getHasTensorComputation());
		assertNull(f.getPassingPrecondition());
		assertNotNull(f.getEntryMatchingFailure(PreconditionFailure.NO_TENSOR_COMPUTATION));
	}

	/**
	 * Barren counterpart of {@link #testRecursion2}: {@code not_recursive_fn} computes {@code abs(n - 1)}. The {@code n - 1} subtraction
	 * types as tensor computation in isolation, but wrapping it in the opaque Python builtin {@code abs()} suppresses the analysis's typing
	 * of the intermediate, so no tensor computation is detected and the function fails with
	 * {@link PreconditionFailure#NO_TENSOR_COMPUTATION} instead of hybridizing (issue 709). This is an analysis-modeling limitation;
	 * blocking the eager-to-hybrid conversion here is incompleteness-safe (we decline to hybridize but never violate semantics).
	 */
	@Test
	public void testRecursion2Barren() throws Exception {
		Function f = getFunction("not_recursive_fn");
		assertTrue(f.getHasTensorParameter());
		assertFalse(f.isRecursive());
		assertFalse("The abs() wrapper suppresses typing of the n - 1 intermediate, so no tensor computation is detected.",
				f.getHasTensorComputation());
		assertNull(f.getPassingPrecondition());
		assertNotNull(f.getEntryMatchingFailure(PreconditionFailure.NO_TENSOR_COMPUTATION));
	}

	/**
	 * Regression guard for #429. The argument {@code tf.zeros([2 * 14])} has a literal-arithmetic shape that only folds to a numeric
	 * dimension (28) when Jython's interpreter is healthy under Tycho-OSGi, i.e. when the {@code edu.cuny.hunter.hybridize.jython.frozen}
	 * fragment puts {@code _frozen_importlib.class} on the wrapped Ariadne bundle's classloader. On a degraded interpreter the
	 * constant-folding pass falls back to a {@link SymbolicDim}, so this assertion fails if the fragment regresses.
	 */
	@Test
	public void testFrozenImportlibConstantFolding() throws Exception {
		Parameter t = findParameter(this.getFunctions(), "t");
		assertEquals("`t` receives tf.zeros([2 * 14]); its shape must fold to the numeric dimension 28, not a symbolic one.",
				Set.of(new TensorType(FLOAT32, List.of(new NumericDim(28)))), t.getTensorTypes());
	}

	/**
	 * Regression test for #486 (no-iterator-entry case). A non-tensor parameter produces an empty {@link Set} at the
	 * {@link Parameter#getTensorTypes()} level. The wala/ML lattice is defined per-shape and per-dtype inside individual {@link TensorType}
	 * objects (`getDims() == null` for shape-⊤, `getDType() == UNKNOWN` for dtype-⊤); the absence of any {@link TensorType} for this
	 * variable corresponds to Ariadne's ⊥ classification (provably not a tensor) when generators are contract-compliant—they emit a
	 * placeholder {@code TensorType(UNKNOWN, null)} for "tensor with unknown info" cases, so an empty {@code state} means no generator
	 * classified the variable as a tensor. The iterator filter (`state != null && !state.isEmpty()`) collapses "variable not analyzed" with
	 * this not-a-tensor case at the API surface; both behave identically for downstream consumers.
	 */
	@Test
	public void testInferredTensorTypesBottomNotTensor() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertFalse(function.isHybrid());
		assertFalse(function.getHasTensorParameter());

		List<Parameter> parameters = function.getParameters();
		assertNotNull(parameters);
		assertEquals(1, parameters.size());

		Parameter x = parameters.get(0);
		assertEquals("x", x.getName());

		Set<TensorType> inferred = x.getTensorTypes();
		assertNotNull(inferred);
		assertTrue("Non-tensor parameter yields no iterator entry, so the inferred set is empty.", inferred.isEmpty());
	}

	/**
	 * Regression test for #495: under `experimental_follow_type_hints=True` with a `tf.Tensor` type hint, the per-Parameter tensor-types
	 * cache must be populated even though `Parameter.classifyAsTensor`'s Phase 1 (type hints) returns true before Phase 2 (Ariadne query)
	 * runs. Without the hoist landed in #496, `param.getTensorTypes()` returned the empty default for type-hint-classified parameters even
	 * when Ariadne had a concrete `TensorType` from the call site.
	 */
	@Test
	public void testInferredTensorTypesUnderFollowTypeHints() throws Exception {
		Set<Function> functions = this.getFunctions();
		assertNotNull(functions);
		assertEquals(1, functions.size());
		Function function = functions.iterator().next();
		assertNotNull(function);
		assertTrue("Function is decorated with `@tf.function(experimental_follow_type_hints=True)`.", function.isHybrid());
		assertTrue("The `experimental_follow_type_hints` parameter is supplied on the `tf.function` decorator.",
				function.getHybridizationParameters().hasExperimentalFollowTypeHintsParam());
		assertTrue("The function has a tensor parameter (via the type hint).", function.getHasTensorParameter());

		List<Parameter> parameters = function.getParameters();
		assertNotNull(parameters);
		assertEquals(1, parameters.size());
		Parameter t = parameters.get(0);
		assertEquals("t", t.getName());

		// The load-bearing assertion: the cache is populated from the call site's `tf.constant([1.0, 2.0])`, even though Phase 1's
		// type-hint hit causes `classifyAsTensor` to return true before Phase 2 reads the cache. Without the hoist, this assertion fails
		// (the cache stays at the empty default).
		TensorType expected = new TensorType(FLOAT32, List.of(new NumericDim(2)));
		assertEquals("Cache must be populated from Ariadne's call-site classification under followTypeHints.", Set.of(expected),
				t.getTensorTypes());
	}
}
