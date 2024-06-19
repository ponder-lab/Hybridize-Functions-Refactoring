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
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
import org.eclipse.core.runtime.IProgressMonitor;
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
import edu.cuny.hunter.hybridize.core.analysis.NoDeclaringModuleException;
import edu.cuny.hunter.hybridize.core.analysis.NoTextSelectionException;
import edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess;
import edu.cuny.hunter.hybridize.core.analysis.Refactoring;
import edu.cuny.hunter.hybridize.core.analysis.Transformation;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;
import edu.cuny.hunter.hybridize.core.utils.Util;

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

	private static final String PERFORM_ANALYSIS_PROPERTY_KEY = "edu.cuny.hunter.hybridize.eval.performAnalysis";

	private static final String PERFORM_CHANGE_PROPERTY_KEY = "edu.cuny.hunter.hybridize.eval.performChange";

	private static final String ALWAYS_CHECK_PYTHON_SIDE_EFFECTS_PROPERTY_KEY = "edu.cuny.hunter.hybridize.eval.alwaysCheckPythonSideEffects";

	private static final String ALWAYS_CHECK_RECURSION_PROPERTY_KEY = "edu.cuny.hunter.hybridize.eval.alwaysCheckRecursion";

	private static final String PROCESS_FUNCTIONS_IN_PARALLEL_PROPERTY_KEY = "edu.cuny.hunter.hybridize.eval.processFunctionsInParallel";

	private static final String USE_TEST_ENTRYPOINTS_KEY = "edu.cuny.hunter.hybridize.eval.useTestEntrypoints";

	private static final String ALWAYS_FOLLOW_TYPE_HINTS_KEY = "edu.cuny.hunter.hybridize.eval.alwaysFollowTypeHints";

	private static final String USE_SPECULATIVE_ANALYSIS_KEY = "edu.cuny.hunter.hybridize.eval.useSpeculativeAnalysis";

	private static final String OUTPUT_CALLS_KEY = "edu.cuny.hunter.hybridize.eval.outputCalls";

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

	private boolean alwaysCheckPythonSideEffects = Boolean.getBoolean(ALWAYS_CHECK_PYTHON_SIDE_EFFECTS_PROPERTY_KEY);

	private boolean alwaysCheckRecursion = Boolean.getBoolean(ALWAYS_CHECK_RECURSION_PROPERTY_KEY);

	private boolean processFunctionsInParallel = Boolean.getBoolean(PROCESS_FUNCTIONS_IN_PARALLEL_PROPERTY_KEY);

	private boolean useTestEntrypoints = Boolean.getBoolean(USE_TEST_ENTRYPOINTS_KEY);

	private boolean alwaysFollowTypeHints = Boolean.getBoolean(ALWAYS_FOLLOW_TYPE_HINTS_KEY);

	private boolean useSpeculativeAnalysis = Boolean.getBoolean(USE_SPECULATIVE_ANALYSIS_KEY);

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		Job.create("Evaluating Hybridize Functions refactoring...", monitor -> {
			List<String> resultsHeader = new ArrayList<>(Arrays.asList("subject", "functions", "optimization available functions",
					"optimizable functions", "failed preconditions"));

			for (Refactoring refactoring : Refactoring.values())
				resultsHeader.add(refactoring.toString());

			for (PreconditionSuccess preconditionSuccess : PreconditionSuccess.values())
				resultsHeader.add(preconditionSuccess.toString());

			for (Transformation transformation : Transformation.values())
				resultsHeader.add(transformation.toString());

			resultsHeader.add("time (s)");

			HybridizeFunctionRefactoringProcessor processor = null;

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
					CSVPrinter callPrinter = createCSVPrinter(CALL_CSV_FILENAME, CALLS_HEADER);) {
				if (BUILD_WORKSPACE) {
					// build the workspace.
					monitor.beginTask("Building workspace ...", IProgressMonitor.UNKNOWN);
					ResourcesPlugin.getWorkspace().build(IncrementalProjectBuilder.FULL_BUILD, monitor.slice(IProgressMonitor.UNKNOWN));
				}

				IProject[] pythonProjects = getSelectedPythonProjectsFromEvent(event);

				monitor.beginTask("Analyzing projects...", pythonProjects.length);

				for (IProject project : pythonProjects) {
					if (!(project.isOpen() && project.exists() && project.hasNature(PYTHON_NATURE_ID)
							&& project.isNatureEnabled(PYTHON_NATURE_ID)))
						throw new IllegalStateException("Python project: " + project.getName() + " must be open and must exist.");

					// subject.
					resultsPrinter.print(project.getName());

					// calls.
					if (Boolean.getBoolean(OUTPUT_CALLS_KEY))
						printCalls(callPrinter, project, monitor.slice(IProgressMonitor.UNKNOWN));

					// set up analysis for single project.
					TimeCollector resultsTimeCollector = new TimeCollector();

					resultsTimeCollector.start();
					processor = createHybridizeFunctionRefactoring(new IProject[] { project }, this.getAlwaysCheckPythonSideEffects(),
							this.getProcessFunctionsInParallel(), this.getAlwaysCheckRecusion(), this.getUseTestEntrypoints(),
							this.getAlwaysFollowTypeHints(), this.getUseSpeculativeAnalysis());
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
					resultsPrinter.print(functions.size());

					for (Function func : functions)
						printFunction(functionsPrinter, func);

					// optimization available functions. These are the "filtered" functions. We consider functions to be candidates iff they
					// have a tensor-like parameter or are currently hybrid.
					Set<Function> candidates = functions.stream().filter(Function::isHybridizationAvailable)
							.filter(f -> f.isHybrid() != null && f.isHybrid() || f.hasTensorParameter() != null && f.hasTensorParameter())
							.collect(Collectors.toSet());
					resultsPrinter.print(candidates.size()); // number.

					// candidate functions.
					for (Function function : candidates) {
						candidatesPrinter.printRecord(buildAttributeColumnValues(function));

						// transformations.
						for (Transformation transformation : function.getTransformations())
							transformationsPrinter.printRecord(buildAttributeColumnValues(function, transformation));
					}

					// optimizable candidate functions.
					Set<Function> optimizableFunctions = Sets.intersection(candidates, processor.getOptimizableFunctions());
					resultsPrinter.print(optimizableFunctions.size()); // number.

					for (Function function : optimizableFunctions)
						optimizableFunctionPrinter.printRecord(buildAttributeColumnValues(function));

					// failed functions.
					SetView<Function> failures = Sets.difference(candidates, optimizableFunctions);

					for (Function function : failures)
						nonOptimizableFunctionPrinter.printRecord(buildAttributeColumnValues(function));

					// failed preconditions.
					Collection<RefactoringStatusEntry> errorEntries = getRefactoringStatusEntries(failures,
							RefactoringStatusEntry::isError);

					resultsPrinter.print(errorEntries.size()); // number.

					printStatuses(errorPrinter, errorEntries);

					// general refactoring statuses.
					Set<RefactoringStatusEntry> generalEntries = getRefactoringStatusEntries(functions, x -> true);
					printStatuses(statusPrinter, generalEntries);

					// decorators.
					printDecorators(decoratorPrinter, functions, monitor.slice(IProgressMonitor.UNKNOWN));

					// refactoring type counts.
					for (Refactoring refactoringKind : Refactoring.values())
						resultsPrinter.print(candidates.parallelStream().map(Function::getRefactoring)
								.filter(r -> Objects.equals(r, refactoringKind)).count());

					// precondition success counts.
					for (PreconditionSuccess preconditionSuccess : PreconditionSuccess.values())
						resultsPrinter.print(candidates.parallelStream().map(Function::getPassingPrecondition)
								.filter(pp -> Objects.equals(pp, preconditionSuccess)).count());

					// transformation counts.
					for (Transformation transformation : Transformation.values())
						resultsPrinter.print(candidates.parallelStream().map(Function::getTransformations).filter(Objects::nonNull)
								.flatMap(as -> as.parallelStream()).filter(a -> Objects.equals(a, transformation)).count());

					// actually perform the refactoring if there are no fatal
					// errors.
					if (shouldPerformChange() && !status.hasFatalError()) {
						resultsTimeCollector.start();
						Change change = refactoring.createChange(monitor.slice(IProgressMonitor.UNKNOWN));
						change.perform(monitor.slice(IProgressMonitor.UNKNOWN));
						resultsTimeCollector.stop();
					}

					// overall results time.
					resultsPrinter.print(
							(resultsTimeCollector.getCollectedTime() - processor.getExcludedTimeCollector().getCollectedTime()) / 1000.0);

					// end the record.
					resultsPrinter.println();

					// clear the cache.
					processor.clearCaches();

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

			return Status.info("Evaluation was successful.");
		}).schedule();

		return null;
	}

	private static boolean shouldPerformChange() {
		return Boolean.getBoolean(PERFORM_CHANGE_PROPERTY_KEY);
	}

	private static boolean shouldPerformAnalysis() {
		return Boolean.getBoolean(PERFORM_ANALYSIS_PROPERTY_KEY);
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
									NodeUtils.getFullRepresentationString(call), selectedText, moduleName, file, project, e));
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
				"experimental_follow_type_hints", "experimental_implements", "func", "input_signature", "jit_compile", "reduce_retracing",
				"refactoring", "passing precondition", "status");
	}

	private static void printFunction(CSVPrinter printer, Function function) throws IOException, CoreException {
		Object[] initialColumnValues = buildAttributeColumnValues(function, function.getMethodReference(), function.getDeclaringClass(),
				function.isMethod(), function.getNumberOfParameters(), function.hasTensorParameter(), function.hasPrimitiveParameters(),
				function.isHybrid(), function.hasPythonSideEffects(), function.isRecursive());

		for (Object columnValue : initialColumnValues)
			printer.print(columnValue);

		HybridizationParameters hybridizationParameters = function.getHybridizationParameters();

		printer.print(hybridizationParameters == null ? null : hybridizationParameters.isAutoGraphParamExists());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.isExperimentalAutographOptParamExists());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.isExperimentalFollowTypeHintsParamExists());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.isExperimentalImplementsParamExists());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.isFuncParamExists());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.isInputSignatureParamExists());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.isJitCompileParamExists());
		printer.print(hybridizationParameters == null ? null : hybridizationParameters.isReduceRetracingParamExists());

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

	private boolean getAlwaysCheckRecusion() {
		return alwaysCheckRecursion;
	}

	private boolean getProcessFunctionsInParallel() {
		return this.processFunctionsInParallel;
	}

	private boolean getUseTestEntrypoints() {
		return this.useTestEntrypoints;
	}

	public boolean getAlwaysFollowTypeHints() {
		return alwaysFollowTypeHints;
	}

	public boolean getUseSpeculativeAnalysis() {
		return this.useSpeculativeAnalysis;
	}
}
