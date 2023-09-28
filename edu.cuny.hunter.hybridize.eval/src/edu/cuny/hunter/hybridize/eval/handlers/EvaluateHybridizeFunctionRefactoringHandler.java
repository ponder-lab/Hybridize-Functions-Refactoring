package edu.cuny.hunter.hybridize.eval.handlers;

import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.io.IOException;
import java.util.LinkedHashSet;

import org.apache.commons.csv.CSVPrinter;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectNature;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.Status;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.core.runtime.jobs.Job;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.navigator.elements.PythonSourceFolder;

import edu.cuny.citytech.refactoring.common.eval.handlers.EvaluateRefactoringHandler;

public class EvaluateHybridizeFunctionRefactoringHandler extends EvaluateRefactoringHandler {

	private static final String RESULTS_CSV_FILENAME = "results.csv";

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		Job.create("Evaluating Hybridize Functions refactoring...", monitor -> {
			IProject[] pythonProjectsFromEvent = getSelectedPythonProjectsFromEvent(event);

			try (CSVPrinter resultsPrinter = createCSVPrinter(RESULTS_CSV_FILENAME, new String[] { "subject", "time (s)" })) {
				for (IProject project : pythonProjectsFromEvent) {
					// subject.
					resultsPrinter.print(project.getName());

					// TODO: overall results time.
					/*
					resultsPrinter.print((resultsTimeCollector.getCollectedTime()
							- processor.getExcludedTimeCollector().getCollectedTime()) / 1000.0);
					*/
					resultsPrinter.print(0);

					// end the record.
					resultsPrinter.println();
				}
			} catch (IOException e) {
				IStatus status = Status.error("Encountered error with evaluation.", e);
				throw new CoreException(status);
			} finally {
				SubMonitor.done(monitor);
			}
		}).schedule();

		return null;
	}

	private static IProject[] getSelectedPythonProjectsFromEvent(ExecutionEvent event) throws CoreException {
		IStructuredSelection selection = HandlerUtil.getCurrentStructuredSelection(event);
		java.util.Set<IProject> ret = new LinkedHashSet<>();

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
