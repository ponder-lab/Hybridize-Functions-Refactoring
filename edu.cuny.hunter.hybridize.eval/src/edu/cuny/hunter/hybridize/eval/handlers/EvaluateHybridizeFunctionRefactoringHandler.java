package edu.cuny.hunter.hybridize.eval.handlers;

import static edu.cuny.hunter.hybridize.core.utils.Util.createHybridizeFunctionRefactoring;
import static org.eclipse.core.runtime.Platform.getLog;
import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVPrinter;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectNature;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.Status;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.core.runtime.jobs.Job;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.RefactoringStatusEntry;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.eclipse.ui.ISources;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.navigator.elements.PythonSourceFolder;

import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;

import edu.cuny.citytech.refactoring.common.core.TimeCollector;
import edu.cuny.citytech.refactoring.common.eval.handlers.EvaluateRefactoringHandler;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess;
import edu.cuny.hunter.hybridize.core.analysis.Refactoring;
import edu.cuny.hunter.hybridize.core.analysis.Transformation;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;

public class EvaluateHybridizeFunctionRefactoringHandler extends EvaluateRefactoringHandler {

	private static final ILog LOG = getLog(EvaluateHybridizeFunctionRefactoringHandler.class);

	private static final String RESULTS_CSV_FILENAME = "results.csv";

	private static final String CANDIDATE_CSV_FILENAME = "candidate_functions.csv";

	private static final String TRANSFORMATIONS_CSV_FILENAME = "transformations.csv";

	private static final String PERFORM_CHANGE_PROPERTY_KEY = "edu.cuny.hunter.hybridize.eval.performChange";

	private static String[] buildAttributeColumnNames(String... additionalColumnNames) {
		String[] primaryColumns = new String[] { "subject", "function", "module", "relative path" };
		List<String> ret = new ArrayList<>(Arrays.asList(primaryColumns));
		ret.addAll(Arrays.asList(additionalColumnNames));
		return ret.toArray(String[]::new);
	}

	private static Object[] buildAttributeColumnValues(Function function, Object... additioanlColumnValues) {
		IProject project = function.getProject();
		Path relativePath = project.getLocation().toFile().toPath().relativize(function.getContainingFile().toPath());
		String[] primaryColumns = new String[] { project.getName(), function.getIdentifer(), function.getContainingModuleName(),
				relativePath.toString() };
		List<Object> ret = new ArrayList<>(Arrays.asList(primaryColumns));
		ret.addAll(Arrays.asList(additioanlColumnValues));
		return ret.toArray(Object[]::new);
	}

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

			try (CSVPrinter resultsPrinter = createCSVPrinter(RESULTS_CSV_FILENAME, resultsHeader.toArray(String[]::new));
					CSVPrinter candidatePrinter = createCSVPrinter(CANDIDATE_CSV_FILENAME,
							buildAttributeColumnNames("parameters", "tensor parameter", "hybrid", "refactoring", "passing precondition",
									"status"));
					CSVPrinter transformationsPrinter = createCSVPrinter(TRANSFORMATIONS_CSV_FILENAME,
							buildAttributeColumnNames("transformation"))) {
				IProject[] pythonProjectsFromEvent = getSelectedPythonProjectsFromEvent(event);

				monitor.beginTask("Analyzing projects...", pythonProjectsFromEvent.length);

				for (IProject project : pythonProjectsFromEvent) {
					// subject.
					resultsPrinter.print(project.getName());

					// set up analysis for single project.
					TimeCollector resultsTimeCollector = new TimeCollector();

					resultsTimeCollector.start();
					HybridizeFunctionRefactoringProcessor processor = createHybridizeFunctionRefactoring(new IProject[] { project });
					resultsTimeCollector.stop();

					// run the precondition checking.
					ProcessorBasedRefactoring refactoring = new ProcessorBasedRefactoring(processor);
					resultsTimeCollector.start();
					RefactoringStatus status = refactoring.checkAllConditions(monitor);
					resultsTimeCollector.stop();

					LOG.info("Preconditions " + (status.isOK() ? "passed" : "failed") + ".");

					// functions.
					Set<Function> functions = processor.getFunctions();
					resultsPrinter.print(functions.size());

					// optimization available functions. These are the "filtered" functions.
					Set<Function> candidates = functions.stream().filter(f -> f.isHybridizationAvailable()).collect(Collectors.toSet());
					resultsPrinter.print(candidates.size()); // number.

					// candidate functions.
					for (Function function : candidates) {
						RefactoringStatus functionStatus = function.getStatus();

						candidatePrinter.printRecord(buildAttributeColumnValues(function, function.getNumberOfParameters(),
								function.getLikelyHasTensorParameter(), function.isHybrid(), function.getRefactoring(),
								function.getPassingPrecondition(),
								functionStatus.isOK() ? 0 : functionStatus.getEntryWithHighestSeverity().getSeverity()));

						// transformations.
						for (Transformation transformation : function.getTransformations())
							transformationsPrinter.printRecord(buildAttributeColumnValues(function, transformation));
					}

					// optimizable functions.
					Set<Function> optimizableFunctions = processor.getOptimizableFunctions();
					resultsPrinter.print(optimizableFunctions.size()); // number.

					// failed functions.
					SetView<Function> failures = Sets.difference(candidates, optimizableFunctions);

					// failed preconditions.
					Collection<RefactoringStatusEntry> errorEntries = failures.parallelStream().map(Function::getStatus)
							.flatMap(s -> Arrays.stream(s.getEntries())).filter(RefactoringStatusEntry::isError)
							.collect(Collectors.toSet());

					resultsPrinter.print(errorEntries.size()); // number.

					// refactoring type counts.
					for (Refactoring refactoringKind : Refactoring.values())
						resultsPrinter.print(functions.parallelStream().map(Function::getRefactoring)
								.filter(r -> Objects.equals(r, refactoringKind)).count());

					// precondition success counts.
					for (PreconditionSuccess preconditionSuccess : PreconditionSuccess.values())
						resultsPrinter.print(functions.parallelStream().map(Function::getPassingPrecondition)
								.filter(pp -> Objects.equals(pp, preconditionSuccess)).count());

					// transformation counts.
					for (Transformation transformation : Transformation.values())
						resultsPrinter.print(functions.parallelStream().map(Function::getTransformations).filter(Objects::nonNull)
								.flatMap(as -> as.parallelStream()).filter(a -> Objects.equals(a, transformation)).count());

					if (Boolean.getBoolean(PERFORM_CHANGE_PROPERTY_KEY) && !status.hasFatalError()) {
						resultsTimeCollector.start();
						Change change = refactoring.createChange(monitor.slice(IProgressMonitor.UNKNOWN));
						change.perform(monitor.slice(IProgressMonitor.UNKNOWN));
						resultsTimeCollector.stop();
					}

					// overall results time.
					resultsPrinter.print(
							// TODO:
							// (
							resultsTimeCollector.getCollectedTime()
									// - processor.getExcludedTimeCollector().getCollectedTime())
									/ 1000.0);

					// end the record.
					resultsPrinter.println();

					monitor.worked(1);
				}
			} catch (IOException | ExecutionException e) {
				throw new CoreException(Status.error("Encountered error with evaluation.", e));
			} finally {
				SubMonitor.done(monitor);
			}
		}).schedule();

		return null;
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
}
