package edu.cuny.hunter.hybridize.eval.handlers;

import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.util.LinkedHashSet;
import java.util.stream.Stream;

import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectNature;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.jobs.Job;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.navigator.elements.PythonSourceFolder;

import edu.cuny.citytech.refactoring.common.eval.handlers.EvaluateRefactoringHandler;

public class EvaluateHybridizeFunctionRefactoringHandler extends EvaluateRefactoringHandler {

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		Job.create("Evaluating Hybridize Functions refactoring...", monitor -> {
			IProject[] pythonProjectsFromEvent = getSelectedPythonProjectsFromEvent(event);

		}).schedule();

		return null;
	}

	private IProject[] getSelectedPythonProjectsFromEvent(ExecutionEvent event) throws CoreException {
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
