package edu.cuny.hunter.hybridize.eval.handlers;

import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.io.IOException;
import java.util.LinkedHashSet;
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
import org.eclipse.ui.ISources;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.navigator.elements.PythonSourceFolder;

import edu.cuny.citytech.refactoring.common.core.TimeCollector;
import edu.cuny.citytech.refactoring.common.eval.handlers.EvaluateRefactoringHandler;

public class EvaluateHybridizeFunctionRefactoringHandler extends EvaluateRefactoringHandler {

	private static final String RESULTS_CSV_FILENAME = "results.csv";

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		Job.create("Evaluating Hybridize Functions refactoring...", monitor -> {
			try (CSVPrinter resultsPrinter = createCSVPrinter(RESULTS_CSV_FILENAME, new String[] { "subject", "time (s)" })) {
				IProject[] pythonProjectsFromEvent = getSelectedPythonProjectsFromEvent(event);

				for (IProject project : pythonProjectsFromEvent) {
					// subject.
					resultsPrinter.print(project.getName());

					// set up analysis for single project.
					TimeCollector resultsTimeCollector = new TimeCollector();

					resultsTimeCollector.start();
					// TODO
					resultsTimeCollector.stop();

					// overall results time.
					resultsPrinter.print(
					// TODO:
//							(
							resultsTimeCollector.getCollectedTime()
//							- processor.getExcludedTimeCollector().getCollectedTime())
									/ 1000.0);

					// end the record.
					resultsPrinter.println();
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
