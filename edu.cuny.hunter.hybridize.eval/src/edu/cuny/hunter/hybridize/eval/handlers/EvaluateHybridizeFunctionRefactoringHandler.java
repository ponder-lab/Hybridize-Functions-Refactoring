package edu.cuny.hunter.hybridize.eval.handlers;

import static edu.cuny.hunter.hybridize.core.analysis.Util.getFullyQualifiedName;
import static edu.cuny.hunter.hybridize.core.analysis.Util.getSelection;
import static edu.cuny.hunter.hybridize.core.utils.Util.createHybridizeFunctionRefactoring;
import static edu.cuny.hunter.hybridize.core.utils.Util.getDocument;
import static edu.cuny.hunter.hybridize.core.utils.Util.getFile;
import static edu.cuny.hunter.hybridize.core.utils.Util.getModuleName;
import static edu.cuny.hunter.hybridize.core.utils.Util.getPythonNature;
import static org.eclipse.core.runtime.IProgressMonitor.UNKNOWN;
import static org.eclipse.core.runtime.Platform.getLog;
import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVPrinter;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectNature;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.resources.IncrementalProjectBuilder;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IPath;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.core.runtime.Status;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.core.runtime.jobs.Job;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.RefactoringStatusEntry;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.eclipse.ui.ISources;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.navigator.elements.PythonNode;
import org.python.pydev.navigator.elements.PythonSourceFolder;
import org.python.pydev.outline.ParsedItem;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.VisitorBase;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.parser.visitors.scope.ASTEntryWithChildren;

import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;

import edu.cuny.citytech.refactoring.common.core.TimeCollector;
import edu.cuny.citytech.refactoring.common.eval.handlers.EvaluateRefactoringHandler;
import edu.cuny.hunter.hybridize.core.analysis.AmbiguousDeclaringModuleException;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.Function.HybridizationParameters;
import edu.cuny.hunter.hybridize.core.analysis.InputSignature;
import edu.cuny.hunter.hybridize.core.analysis.NoDeclaringModuleException;
import edu.cuny.hunter.hybridize.core.analysis.NoTextSelectionException;
import edu.cuny.hunter.hybridize.core.analysis.Parameter;
import edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess;
import edu.cuny.hunter.hybridize.core.analysis.Refactoring;
import edu.cuny.hunter.hybridize.core.analysis.Transformation;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;
import edu.cuny.hunter.hybridize.core.utils.Util;
import edu.cuny.hunter.hybridize.eval.config.EvaluationOption;

public class EvaluateHybridizeFunctionRefactoringHandler extends EvaluateRefactoringHandler {

	private static final boolean BUILD_WORKSPACE = false;

	private static final ILog LOG = getLog(EvaluateHybridizeFunctionRefactoringHandler.class);

	private static final String RESULTS_CSV_FILENAME = "results.csv";

	private static final String FUNCTIONS_CSV_FILENAME = "functions.csv";

	private static final String CANDIDATES_CSV_FILENAME = "candidate_functions.csv";

	private static final String TRANSFORMATIONS_CSV_FILENAME = "transformations.csv";

	private static final String OPTMIZABLE_CSV_FILENAME = "optimizable.csv";

	private static final String NONOPTMIZABLE_CSV_FILENAME = "nonoptimizable.csv";

	private static final String FAILED_PRECONDITIONS_CSV_FILENAME = "failed_preconditions.csv";

	private static final String STATUS_CSV_FILENAME = "statuses.csv";

	private static final String DECORATOR_CSV_FILENAME = "decorators.csv";

	private static final String CALL_CSV_FILENAME = "calls.csv";

	private static final String[] CALLS_HEADER = { "subject", "callee", "expr" };

	private static final String BLOCKED_PARAMETERS_CSV_FILENAME = "blocked_parameters.csv";

	private static final String INPUT_SIGNATURES_CSV_FILENAME = "input_signatures.csv";

	/** The {@code eval.properties} key for the targeted k-CFA depth; also the suffix of its system-property key. */
	private static final String TARGETED_CFA_DEPTH_PROPERTY_KEY = "targetedCfaDepth";

	/**
	 * The targeted k-CFA depth system-property key: {@link EvaluationOption#PREFIX} prepended to {@link #TARGETED_CFA_DEPTH_PROPERTY_KEY}.
	 */
	private static final String TARGETED_CFA_DEPTH_KEY = EvaluationOption.PREFIX + TARGETED_CFA_DEPTH_PROPERTY_KEY;

	private static String[] buildAttributeColumnNames(String... additionalColumnNames) {
		String[] primaryColumns = new String[] { "subject", "function", "module", "relative path" };
		List<String> ret = new ArrayList<>(Arrays.asList(primaryColumns));
		ret.addAll(Arrays.asList(additionalColumnNames));
		return ret.toArray(String[]::new);
	}

	private static Object[] buildAttributeColumnValues(Function function, Object... additionalColumnValues) {
		IProject project = function.getProject();
		Path relativePath = project.getLocation().toFile().toPath().relativize(function.getContainingFile().toPath());
		Object[] primaryColumns = new Object[] { project.getName(), function.getIdentifier(), function.getContainingModuleName(),
				relativePath };
		List<Object> ret = new ArrayList<>(Arrays.asList(primaryColumns));
		ret.addAll(Arrays.asList(additionalColumnValues));
		return ret.toArray(Object[]::new);
	}

	private boolean alwaysCheckPythonSideEffects = Boolean.getBoolean(EvaluationOption.ALWAYS_CHECK_PYTHON_SIDE_EFFECTS.key());

	private boolean alwaysCheckRecursion = Boolean.getBoolean(EvaluationOption.ALWAYS_CHECK_RECURSION.key());

	private boolean processFunctionsInParallel = Boolean.getBoolean(EvaluationOption.PROCESS_FUNCTIONS_IN_PARALLEL.key());

	private boolean useTestEntrypoints = Boolean.getBoolean(EvaluationOption.USE_TEST_ENTRYPOINTS.key());

	private boolean alwaysFollowTypeHints = Boolean.getBoolean(EvaluationOption.ALWAYS_FOLLOW_TYPE_HINTS.key());

	private boolean useSpeculativeAnalysis = Boolean.getBoolean(EvaluationOption.USE_SPECULATIVE_ANALYSIS.key());

	/**
	 * True iff the refactoring should emit an inferred {@code input_signature=...} keyword into the generated {@code @tf.function}
	 * decorator. Off by default; set via the {@code edu.cuny.hunter.hybridize.eval.inferInputSignatures} system property.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/481">Issue 481</a>
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	private boolean inferInputSignatures = Boolean.getBoolean(EvaluationOption.INFER_INPUT_SIGNATURES.key());

	/**
	 * The global targeted k-CFA depth, set via the {@code edu.cuny.hunter.hybridize.eval.targetedCfaDepth} system property and defaulting
	 * to {@link HybridizeFunctionRefactoringProcessor#DEFAULT_TARGETED_CFA_DEPTH}. A per-project {@code eval.properties}
	 * {@code targetedCfaDepth} entry overrides this for that project; see {@link #getTargetedCfaDepth(IProject)}.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/600">Issue 600</a>
	 */
	private int targetedCfaDepth = Integer.getInteger(TARGETED_CFA_DEPTH_KEY,
			HybridizeFunctionRefactoringProcessor.DEFAULT_TARGETED_CFA_DEPTH);

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		Job.create("Evaluating Hybridize Functions refactoring...", monitor -> {
			try {
				return evaluate(getSelectedPythonProjectsFromEvent(event), monitor);
			} catch (CoreException | ExecutionException e) {
				return Status.error("Could not determine the projects to evaluate.", e);
			}
		}).schedule();

		return null;
	}

	/**
	 * Runs the evaluation over the given projects, writing the result CSVs. This is the UI-independent core of {@link #execute}: the
	 * handler supplies the workbench selection, but a headless entry point can supply an args-derived project set. Configuration is read
	 * from the {@code edu.cuny.hunter.hybridize.eval.*} system properties, so it needs no UI.
	 *
	 * @param pythonProjects The Python projects to evaluate.
	 * @param monitor The progress monitor.
	 * @return An {@link IStatus} describing the outcome.
	 */
	public IStatus evaluate(IProject[] pythonProjects, IProgressMonitor monitor) {
		List<String> resultsHeader = new ArrayList<>(
				Arrays.asList("subject", "functions", "optimization available functions", "optimizable functions", "failed preconditions"));

		for (Refactoring refactoring : Refactoring.values())
			resultsHeader.add(refactoring.toString());

		for (PreconditionSuccess preconditionSuccess : PreconditionSuccess.values())
			resultsHeader.add(preconditionSuccess.toString());

		for (Transformation transformation : Transformation.values())
			resultsHeader.add(transformation.toString());

		String[] experimentalSettingsHeader = new String[] { "side-effects", "recursion", "type hints", "parallel", "speculative",
				"test entrypoints", "infer input signatures", "targeted CFA depth" };
		resultsHeader.addAll(Arrays.asList(experimentalSettingsHeader));

		resultsHeader.add("time (s)");

		HybridizeFunctionRefactoringProcessor processor = null;
		List<String> failedProjects = new ArrayList<>();

		try (CSVPrinter resultsPrinter = createCSVPrinter(RESULTS_CSV_FILENAME, resultsHeader.toArray(String[]::new));
				CSVPrinter functionsPrinter = createCSVPrinter(FUNCTIONS_CSV_FILENAME, buildFunctionAttributeColumnNames());
				CSVPrinter candidatesPrinter = createCSVPrinter(CANDIDATES_CSV_FILENAME, buildAttributeColumnNames());
				CSVPrinter transformationsPrinter = createCSVPrinter(TRANSFORMATIONS_CSV_FILENAME,
						buildAttributeColumnNames("transformation"));
				CSVPrinter optimizableFunctionPrinter = createCSVPrinter(OPTMIZABLE_CSV_FILENAME, buildAttributeColumnNames());
				CSVPrinter nonOptimizableFunctionPrinter = createCSVPrinter(NONOPTMIZABLE_CSV_FILENAME, buildAttributeColumnNames());
				CSVPrinter errorPrinter = createCSVPrinter(FAILED_PRECONDITIONS_CSV_FILENAME,
						buildAttributeColumnNames("refactoring", "severity", "code", "message"));
				CSVPrinter statusPrinter = createCSVPrinter(STATUS_CSV_FILENAME,
						buildAttributeColumnNames("refactoring", "severity", "code", "message"));
				CSVPrinter decoratorPrinter = createCSVPrinter(DECORATOR_CSV_FILENAME, buildAttributeColumnNames("decorator"));
				CSVPrinter callPrinter = createCSVPrinter(CALL_CSV_FILENAME, CALLS_HEADER);
				CSVPrinter blockedParametersPrinter = createCSVPrinter(BLOCKED_PARAMETERS_CSV_FILENAME,
						buildAttributeColumnNames("param index", "param name", "absence reason"));
				CSVPrinter inputSignaturesPrinter = createCSVPrinter(INPUT_SIGNATURES_CSV_FILENAME,
						buildAttributeColumnNames("param index", "source", "absence reason", "dtype", "shape"));) {
			if (BUILD_WORKSPACE) {
				// build the workspace.
				monitor.beginTask("Building workspace ...", IProgressMonitor.UNKNOWN);
				ResourcesPlugin.getWorkspace().build(IncrementalProjectBuilder.FULL_BUILD, monitor.slice(IProgressMonitor.UNKNOWN));
			}

			monitor.beginTask("Analyzing projects...", pythonProjects.length);

			for (IProject project : pythonProjects) {
				// Per-project isolation: record and skip a single project's failure rather than aborting the whole run.
				List<Object> resultsRecord = new ArrayList<>();

				try {
					if (!(project.isOpen() && project.exists() && project.hasNature(PYTHON_NATURE_ID)
							&& project.isNatureEnabled(PYTHON_NATURE_ID)))
						throw new IllegalStateException("Python project: " + project.getName() + " must be open and must exist.");

					// subject.
					resultsRecord.add(project.getName());

					// calls.
					if (Boolean.getBoolean(EvaluationOption.OUTPUT_CALLS.key()))
						printCalls(callPrinter, project, monitor.slice(IProgressMonitor.UNKNOWN));

					// set up analysis for single project.
					int targetedCfaDepth = getTargetedCfaDepth(project);
					TimeCollector resultsTimeCollector = new TimeCollector();

					resultsTimeCollector.start();
					processor = createHybridizeFunctionRefactoring(new IProject[] { project }, this.getAlwaysCheckPythonSideEffects(),
							this.getProcessFunctionsInParallel(), this.getAlwaysCheckRecusion(), this.getUseTestEntrypoints(),
							this.getAlwaysFollowTypeHints(), this.getUseSpeculativeAnalysis(), this.getInferInputSignatures());
					processor.setTargetedCfaDepth(targetedCfaDepth);
					resultsTimeCollector.stop();

					// run the precondition checking.
					RefactoringStatus status = null;
					ProcessorBasedRefactoring refactoring = new ProcessorBasedRefactoring(processor);

					if (shouldPerformAnalysis()) {
						resultsTimeCollector.start();
						status = refactoring.checkAllConditions(new NullProgressMonitor());
						resultsTimeCollector.stop();

						LOG.info("Preconditions " + (status.isOK() ? "passed" : "failed") + ".");
					} else
						status = new RefactoringStatus();

					// functions.
					Set<Function> functions = processor.getFunctions();
					resultsRecord.add(functions.size());

					for (Function func : functions) {
						printFunction(functionsPrinter, func);
						printBlockedParameters(blockedParametersPrinter, func);
						printInputSignatures(inputSignaturesPrinter, func);
					}

					// optimization available functions. These are the "filtered" functions. We consider functions to be candidates iff they
					// have a tensor-like parameter or are currently hybrid.
					Set<Function> candidates = functions.stream().filter(Function::isHybridizationAvailable).filter(
							f -> f.isHybrid() != null && f.isHybrid() || f.getHasTensorParameter() != null && f.getHasTensorParameter())
							.collect(Collectors.toSet());
					resultsRecord.add(candidates.size()); // number.

					// candidate functions.
					for (Function function : candidates) {
						candidatesPrinter.printRecord(buildAttributeColumnValues(function));

						// transformations.
						for (Transformation transformation : function.getTransformations())
							transformationsPrinter.printRecord(buildAttributeColumnValues(function, transformation));
					}

					// optimizable candidate functions.
					Set<Function> optimizableFunctions = Sets.intersection(candidates, processor.getOptimizableFunctions());
					resultsRecord.add(optimizableFunctions.size()); // number.

					for (Function function : optimizableFunctions)
						optimizableFunctionPrinter.printRecord(buildAttributeColumnValues(function));

					// failed functions.
					SetView<Function> failures = Sets.difference(candidates, optimizableFunctions);

					for (Function function : failures)
						nonOptimizableFunctionPrinter.printRecord(buildAttributeColumnValues(function));

					// failed preconditions.
					Collection<RefactoringStatusEntry> errorEntries = getRefactoringStatusEntries(failures,
							RefactoringStatusEntry::isError);

					resultsRecord.add(errorEntries.size()); // number.

					printStatuses(errorPrinter, errorEntries);

					// general refactoring statuses.
					Set<RefactoringStatusEntry> generalEntries = getRefactoringStatusEntries(functions, x -> true);
					printStatuses(statusPrinter, generalEntries);

					// decorators.
					printDecorators(decoratorPrinter, functions, monitor.slice(IProgressMonitor.UNKNOWN));

					// refactoring type counts.
					for (Refactoring refactoringKind : Refactoring.values())
						resultsRecord.add(candidates.parallelStream().map(Function::getRefactoring)
								.filter(r -> Objects.equals(r, refactoringKind)).count());

					// precondition success counts.
					for (PreconditionSuccess preconditionSuccess : PreconditionSuccess.values())
						resultsRecord.add(candidates.parallelStream().map(Function::getPassingPrecondition)
								.filter(pp -> Objects.equals(pp, preconditionSuccess)).count());

					// transformation counts.
					for (Transformation transformation : Transformation.values())
						resultsRecord.add(candidates.parallelStream().map(Function::getTransformations).filter(Objects::nonNull)
								.flatMap(as -> as.parallelStream()).filter(a -> Objects.equals(a, transformation)).count());

					// side-effects.
					resultsRecord.add(this.getAlwaysCheckPythonSideEffects());

					// recursion.
					resultsRecord.add(this.getAlwaysCheckRecusion());

					// type hints.
					resultsRecord.add(this.getAlwaysFollowTypeHints());

					// parallel.
					resultsRecord.add(this.getProcessFunctionsInParallel());

					// speculative.
					resultsRecord.add(this.getUseSpeculativeAnalysis());

					// test entrypoints.
					resultsRecord.add(this.getUseTestEntrypoints());

					// infer input signatures.
					resultsRecord.add(this.getInferInputSignatures());

					// targeted CFA depth.
					resultsRecord.add(targetedCfaDepth);

					// actually perform the refactoring if there are no fatal
					// errors.
					if (shouldPerformChange() && !status.hasFatalError()) {
						resultsTimeCollector.start();
						Change change = refactoring.createChange(monitor.slice(IProgressMonitor.UNKNOWN));
						change.perform(monitor.slice(IProgressMonitor.UNKNOWN));
						resultsTimeCollector.stop();
					}

					// overall results time.
					resultsRecord.add(
							(resultsTimeCollector.getCollectedTime() - processor.getExcludedTimeCollector().getCollectedTime()) / 1000.0);

					// end the record.
					resultsPrinter.printRecord(resultsRecord);
				} catch (Throwable e) {
					// Per-project isolation (#689): a subject whose analysis throws, including a recoverable java.lang.Error such
					// as wala's UnimplementedError (wala/ML#616), is recorded and skipped rather than aborting the whole run. A
					// VirtualMachineError (OutOfMemoryError, StackOverflowError, ...) is rethrown: the JVM is then unrecoverable, so
					// isolating one subject and continuing would be meaningless.
					if (e instanceof VirtualMachineError vmError)
						throw vmError;

					LOG.error("Evaluation failed for project " + project.getName() + "; recording and skipping it.", e);
					failedProjects.add(project.getName() + " (" + e.getClass().getSimpleName()
							+ (e.getMessage() != null ? ": " + e.getMessage() : "") + ")");
				} finally {
					if (processor != null)
						processor.clearCaches();
				}

				monitor.worked(1);
			}
		} catch (Exception e) {
			return Status.error("Encountered error with evaluation.", e);
		} finally {
			// clear cache.
			if (processor != null)
				processor.clearCaches();

			SubMonitor.done(monitor);
		}

		if (!failedProjects.isEmpty())
			return Status.warning("Evaluation completed; " + failedProjects.size() + " of " + pythonProjects.length
					+ " project(s) failed and were skipped: " + String.join(", ", failedProjects));

		return Status.info("Evaluation was successful.");
	}

	private static boolean shouldPerformChange() {
		return Boolean.getBoolean(EvaluationOption.PERFORM_CHANGE.key());
	}

	private static boolean shouldPerformAnalysis() {
		return Boolean.getBoolean(EvaluationOption.PERFORM_ANALYSIS.key());
	}

	private static Set<RefactoringStatusEntry> getRefactoringStatusEntries(Set<Function> functionSet,
			Predicate<? super RefactoringStatusEntry> predicate) {
		return functionSet.parallelStream().map(Function::getStatus).flatMap(s -> Arrays.stream(s.getEntries())).filter(predicate)
				.collect(Collectors.toSet());
	}

	private static void printStatuses(CSVPrinter printer, Collection<RefactoringStatusEntry> entries) throws IOException {
		for (RefactoringStatusEntry entry : entries) {
			if (!entry.isFatalError()) {
				Object correspondingElement = entry.getData();

				if (!(correspondingElement instanceof Function))
					throw new IllegalStateException("The element: " + correspondingElement + " is not a Function. Instead, it is a: "
							+ correspondingElement.getClass());

				Function function = (Function) correspondingElement;

				printer.printRecord(buildAttributeColumnValues(function, function.getRefactoring(), entry.getSeverity(), entry.getCode(),
						entry.getMessage()));
			}
		}
	}

	private static void printDecorators(CSVPrinter printer, Set<Function> functions, IProgressMonitor monitor) throws IOException {
		SubMonitor progress = SubMonitor.convert(monitor, "Printing decorators", functions.size());

		for (Function function : functions) {
			Set<String> decoratorNames = function.getDecoratorNames(progress.split(1));
			for (String name : decoratorNames)
				printer.printRecord(buildAttributeColumnValues(function, name));
		}
	}

	private static void printCalls(CSVPrinter printer, IProject project, IProgressMonitor monitor)
			throws IOException, ExecutionException, CoreException {
		SubMonitor progress = SubMonitor.convert(monitor, "Printing calls.", UNKNOWN);
		Set<PythonNode> pythonNodes = Util.getPythonNodes(project);
		progress.setWorkRemaining(pythonNodes.size());
		Map<String, Set<String>> fqnToExprs = new HashMap<>();

		for (PythonNode pythonNode : pythonNodes) {
			String moduleName = getModuleName(pythonNode);
			File file = getFile(pythonNode);
			IDocument document = getDocument(pythonNode);
			IPythonNature nature = getPythonNature(pythonNode);

			ParsedItem entry = pythonNode.entry;
			ASTEntryWithChildren ast = entry.getAstThis();
			SimpleNode node = ast.node;

			try {
				node.accept(new VisitorBase() {

					@Override
					public void traverse(SimpleNode node) throws Exception {
						node.traverse(this);
					}

					@Override
					protected Object unhandled_node(SimpleNode node) throws Exception {
						return null;
					}

					@Override
					public Object visitCall(Call call) throws Exception {
						String fqn = null;
						PySelection selection = null;

						try {
							selection = getSelection(call.func, document);
							fqn = getFullyQualifiedName(call, moduleName, file, selection, nature, progress.split(UNKNOWN));
						} catch (AmbiguousDeclaringModuleException | NoDeclaringModuleException | NoTextSelectionException
								| BadLocationException e) {
							String selectedText = null;

							try {
								if (selection != null)
									selectedText = selection.getSelectedText();
							} catch (BadLocationException e1) {
								LOG.info("Can't get selected text for selection: " + selection, e1);
							}

							LOG.info(String.format(
									"Can't determine FQN of function call expression: %s in selection: %s, module: %s, file: %s, and project: %s.",
									NodeUtils.getFullRepresentationString(call), selectedText, moduleName, file, project), e);
						}

						if (fqn != null)
							fqnToExprs.merge(fqn, Sets.newHashSet(NodeUtils.getFullRepresentationString(call.func)), (s1, s2) -> {
								s1.addAll(s2);
								return s1;
							});

						return super.visitCall(call);
					}
				});
			} catch (Exception e) {
				LOG.error("Failed to collect function calls..", e);
				throw new ExecutionException("Failed to collect function calls.", e);
			}

			progress.worked(1);
		}

		for (String fqn : fqnToExprs.keySet()) {
			Set<String> exprs = fqnToExprs.get(fqn);

			for (String expr : exprs)
				printer.printRecord(project.getName(), fqn, expr);
		}
	}

	private static String[] buildFunctionAttributeColumnNames() {
		return buildAttributeColumnNames("method reference", "type reference", "method", "parameters", "tensor parameter",
				"primitive parameter", "hybrid", "side-effects", "recursive", "autograph", "experimental_autograph_options",
				"experimental_follow_type_hints", "experimental_implements", "func", "input_signature", "supplied input_signature",
				"jit_compile", "reduce_retracing", "inferred input_signature", "input_signature relation", "input_signature absence reason",
				"refactoring", "passing precondition", "status");
	}

	/**
	 * Emits one row per parameter that blocked input-signature inference for the given function, naming the parameter and its
	 * {@link edu.cuny.hunter.hybridize.core.analysis.InferenceResult.AbsenceReason}. Where the {@code functions.csv} {@code input_signature
	 * absence reason} column reports only the function's first blocking reason, this surfaces the per-parameter attribution (#654). Reads
	 * the memoized inference result without triggering inference, so it leaves the function's status untouched.
	 *
	 * @param printer The {@code blocked_parameters.csv} printer.
	 * @param function The function whose blocking parameters to emit.
	 * @throws IOException If a record cannot be written.
	 */
	private static void printBlockedParameters(CSVPrinter printer, Function function) throws IOException {
		for (var entry : function.getBlockingParameterReasons().entrySet())
			printer.printRecord(
					buildAttributeColumnValues(function, entry.getKey().getIndex(), entry.getKey().getName(), entry.getValue()));
	}

	/**
	 * Emits the function's input signatures at per-parameter (per-{@code TensorSpec}) granularity into {@code input_signatures.csv}, so
	 * downstream analysis is a group-by rather than a parse of the joined {@code input_signature} string in {@code functions.csv}. There is
	 * one row per non-{@code self} parameter of the inferred signature (source {@code "inferred"}) and of the developer-supplied signature
	 * (source {@code "supplied"}); when inference was blocked, one row per blocking parameter (source {@code "absent"}) carrying its
	 * {@link edu.cuny.hunter.hybridize.core.analysis.InferenceResult.AbsenceReason}. The {@code dtype} and {@code shape} columns hold the
	 * raw per-parameter values, with rank and wildcard counts left to derive downstream. Reads the memoized inference result without
	 * recomputing, so it leaves the function's status untouched.
	 *
	 * @param printer The {@code input_signatures.csv} printer.
	 * @param function The function whose per-parameter signatures to emit.
	 * @throws IOException If a record cannot be written.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/665">Issue 665</a>
	 */
	private static void printInputSignatures(CSVPrinter printer, Function function) throws IOException {
		// The non-self parameters in declaration order, aligned position-wise with InputSignature.parameterSpecs().
		List<Parameter> parameters = function.getParameters().stream().filter(p -> !p.isSelf()).toList();

		if (function.getInferredInputSignature().isPresent())
			printSignatureRows(printer, function, parameters, function.getInferredInputSignature().get(), "inferred");
		else
			for (var entry : function.getBlockingParameterReasons().entrySet())
				printer.printRecord(
						buildAttributeColumnValues(function, entry.getKey().getIndex(), "absent", entry.getValue(), null, null));

		HybridizationParameters hybridizationParameters = function.getHybridizationParameters();
		if (hybridizationParameters != null && hybridizationParameters.getSuppliedInputSignature().isPresent())
			printSignatureRows(printer, function, parameters, hybridizationParameters.getSuppliedInputSignature().get(), "supplied");
	}

	/**
	 * Emits one {@code input_signatures.csv} row per parameter of the given signature, tagging each with {@code source} and pairing it
	 * position-wise with the function's non-{@code self} parameters for the {@code param index} join key. A signature entry beyond the
	 * declared non-{@code self} parameter count (a parameter-count mismatch) gets a {@code null} index rather than failing the row.
	 *
	 * @param printer The {@code input_signatures.csv} printer.
	 * @param function The function the signature belongs to.
	 * @param parameters The function's non-{@code self} parameters in declaration order.
	 * @param signature The signature whose per-parameter rows to emit.
	 * @param source The source tag ({@code "inferred"} or {@code "supplied"}).
	 * @throws IOException If a record cannot be written.
	 */
	private static void printSignatureRows(CSVPrinter printer, Function function, List<Parameter> parameters, InputSignature signature,
			String source) throws IOException {
		var specs = signature.parameterSpecs();

		for (int i = 0; i < specs.size(); i++) {
			Integer index = i < parameters.size() ? parameters.get(i).getIndex() : null;
			printer.printRecord(buildAttributeColumnValues(function, index, source, null, specs.get(i).dtype(), specs.get(i).shape()));
		}
	}

	private static void printFunction(CSVPrinter printer, Function function) throws IOException, CoreException {
		Object[] initialColumnValues = buildAttributeColumnValues(function, function.getMethodReference(), function.getDeclaringClass(),
				function.isMethod(), function.getNumberOfParameters(), function.getHasTensorParameter(),
				function.getHasPrimitiveParameter(), function.isHybrid(), function.getHasPythonSideEffects(), function.isRecursive());

		for (Object columnValue : initialColumnValues)
			printer.print(columnValue);

		HybridizationParameters hybridizationParameters = function.getHybridizationParameters();

		printer.print(hybridizationParameters == null ? null : hybridizationParameters.hasAutoGraphParam());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.hasExperimentalAutographOptionsParam());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.hasExperimentalFollowTypeHintsParam());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.hasExperimentalImplementsParam());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.hasFuncParam());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.hasInputSignatureParam());
		printer.print(hybridizationParameters == null ? null
				: hybridizationParameters.getSuppliedInputSignature().map(s -> s.toTensorSpecList("tf.")).orElse(null));
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.hasJitCompileParam());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.hasReduceRetracingParam());

		/*
		 * The inferred signature the refactoring computed for this function, how it relates to an explicitly supplied one, and—when no
		 * signature was inferred—why. All read the memoized inference result without recomputing, so they leave the status untouched. When
		 * the refactoring did not run inference on a function, all three are blank. When it did, exactly one of the inferred-content and
		 * absence-reason columns is populated: the content when a signature was inferred, the reason when inference was blocked. (The
		 * inferred-content column is thus blank both when inference never ran and when it ran but was blocked.)
		 */
		printer.print(function.getInferredInputSignature().map(s -> s.toTensorSpecList("tf.")).orElse(null));
		printer.print(hybridizationParameters == null ? null
				: hybridizationParameters.getSuppliedInputSignature()
						.flatMap(supplied -> function.getInferredInputSignature().map(supplied::relate)).orElse(null));
		printer.print(function.getInferredInputSignatureAbsenceReason().orElse(null));

		printer.print(function.getRefactoring());
		printer.print(function.getPassingPrecondition());
		printer.print(function.getStatus().isOK() ? 0 : function.getStatus().getEntryWithHighestSeverity().getSeverity());

		// end the record.
		printer.println();
	}

	private static IProject[] getSelectedPythonProjectsFromEvent(ExecutionEvent event) throws CoreException, ExecutionException {
		Set<IProject> ret = new LinkedHashSet<>();
		IStructuredSelection selection = getCurrentStructuredSelectionChecked(event);

		for (Object obj : selection) {
			IProject project = getProject(obj);

			if (project != null) {
				IProjectNature nature = project.getNature(PYTHON_NATURE_ID);

				if (nature != null)
					// We have a Python project.
					ret.add(project);
			}
		}

		return ret.toArray(IProject[]::new);
	}

	/**
	 * Return the current structured selection.
	 *
	 * @param event The execution event that contains the application context.
	 * @return The current IStructuredSelection.Will not return <code>null</code>.
	 * @throws ExecutionException If the current selection variable is not found.
	 */
	private static IStructuredSelection getCurrentStructuredSelectionChecked(ExecutionEvent event) throws ExecutionException {
		ISelection selection = HandlerUtil.getCurrentSelectionChecked(event);

		if (selection instanceof IStructuredSelection)
			return (IStructuredSelection) selection;

		throw new ExecutionException("Incorrect type for " //$NON-NLS-1$
				+ ISources.ACTIVE_CURRENT_SELECTION_NAME + " found while executing " //$NON-NLS-1$
				+ event.getCommand().getId() + ", expected " + IStructuredSelection.class.getName() //$NON-NLS-1$
				+ " found " + selection.getClass()); //$NON-NLS-1$
	}

	private static IProject getProject(Object obj) {
		if (obj instanceof PythonSourceFolder) {
			PythonSourceFolder folder = (PythonSourceFolder) obj;
			IResource actualObject = folder.getActualObject();
			return actualObject.getProject();
		}

		if (obj instanceof IProject) {
			IProject project = (IProject) obj;
			return project;
		}

		return null;
	}

	public boolean getAlwaysCheckPythonSideEffects() {
		return alwaysCheckPythonSideEffects;
	}

	public boolean getAlwaysCheckRecusion() {
		return alwaysCheckRecursion;
	}

	public boolean getProcessFunctionsInParallel() {
		return this.processFunctionsInParallel;
	}

	public boolean getUseTestEntrypoints() {
		return this.useTestEntrypoints;
	}

	public boolean getAlwaysFollowTypeHints() {
		return alwaysFollowTypeHints;
	}

	public boolean getUseSpeculativeAnalysis() {
		return this.useSpeculativeAnalysis;
	}

	public boolean getInferInputSignatures() {
		return this.inferInputSignatures;
	}

	/**
	 * Returns the targeted k-CFA depth for the given project: the {@code targetedCfaDepth} entry of the nearest {@code eval.properties}
	 * file (searched from the project's location upward) when present and a positive integer, otherwise the global
	 * {@link #targetedCfaDepth} (the {@code edu.cuny.hunter.hybridize.eval.targetedCfaDepth} system property, or
	 * {@link HybridizeFunctionRefactoringProcessor#DEFAULT_TARGETED_CFA_DEPTH}). A missing file or entry, a non-positive or malformed
	 * value, or a read failure falls back to the global depth. Mirrors how the Java 8 stream-refactoring evaluator reads its per-project
	 * analysis depth from {@code eval.properties}.
	 *
	 * @param project The project being evaluated.
	 * @return The targeted k-CFA depth for the project.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/600">Issue 600</a>
	 */
	private int getTargetedCfaDepth(IProject project) {
		IPath location = project.getLocation();
		File file = location == null ? null : this.findEvaluationPropertiesFile(location.toFile());

		if (file != null && file.exists())
			try (Reader reader = new FileReader(file, StandardCharsets.UTF_8)) {
				Properties properties = new Properties();
				properties.load(reader);
				String value = properties.getProperty(TARGETED_CFA_DEPTH_PROPERTY_KEY);

				if (value != null)
					try {
						int depth = Integer.parseInt(value.trim());

						if (depth >= 1) {
							LOG.info("Using eval.properties targeted CFA depth: " + depth + ".");
							return depth;
						}

						LOG.warn("Ignoring non-positive eval.properties targeted CFA depth: " + depth + ". Using " + this.targetedCfaDepth
								+ ".");
					} catch (NumberFormatException e) {
						LOG.warn("Ignoring malformed eval.properties targeted CFA depth: " + value.trim() + ". Using "
								+ this.targetedCfaDepth + ".", e);
					}
			} catch (IOException e) {
				LOG.warn("Could not read " + file + " for the targeted CFA depth. Using " + this.targetedCfaDepth + ".", e);
			}

		return this.targetedCfaDepth;
	}
}
