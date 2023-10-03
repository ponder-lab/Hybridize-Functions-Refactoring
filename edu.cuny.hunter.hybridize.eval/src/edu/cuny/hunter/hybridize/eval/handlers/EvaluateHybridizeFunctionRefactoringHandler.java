package edu.cuny.hunter.hybridize.eval.handlers;

import static edu.cuny.hunter.hybridize.core.utils.Util.createHybridizeFunctionRefactoring;
import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.csv.CSVPrinter;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectNature;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.Status;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.core.runtime.jobs.Job;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.eclipse.ui.ISources;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.navigator.elements.PythonSourceFolder;

import edu.cuny.citytech.refactoring.common.core.TimeCollector;
import edu.cuny.citytech.refactoring.common.eval.handlers.EvaluateRefactoringHandler;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess;
import edu.cuny.hunter.hybridize.core.analysis.Refactoring;
import edu.cuny.hunter.hybridize.core.analysis.Transformation;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;

public class EvaluateHybridizeFunctionRefactoringHandler extends EvaluateRefactoringHandler {

	private static final String RESULTS_CSV_FILENAME = "results.csv";

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

			try (CSVPrinter resultsPrinter = createCSVPrinter(RESULTS_CSV_FILENAME, resultsHeader.toArray(String[]::new))) {
				IProject[] pythonProjectsFromEvent = getSelectedPythonProjectsFromEvent(event);

				monitor.beginTask("Analyzing projects...", pythonProjectsFromEvent.length);

				for (IProject project : pythonProjectsFromEvent) {
					// subject.
					resultsPrinter.print(project.getName());

					// set up analysis for single project.
					TimeCollector resultsTimeCollector = new TimeCollector();

					resultsTimeCollector.start();
					HybridizeFunctionRefactoringProcessor processor = createHybridizeFunctionRefactoring(new IProject[] { project },
							monitor);
					resultsTimeCollector.stop();

					// run the precondition checking.
					resultsTimeCollector.start();
					RefactoringStatus status = new ProcessorBasedRefactoring(processor).checkAllConditions(monitor);
					resultsTimeCollector.stop();

					// functions.
					Set<Function> functions = processor.getFunctions();
					resultsPrinter.print(functions.size());

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
