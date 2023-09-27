package edu.cuny.hunter.hybridize.eval.handlers;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.navigator.elements.PythonSourceFolder;

public class EvaluateHybridizeFunctionRefactoringHandler extends AbstractHandler {

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		IStructuredSelection selection = HandlerUtil.getCurrentStructuredSelection(event);

		for (Object obj : selection) {
			IProject project = getProject(obj);
			System.out.println(project.getName());
		}

		return null;
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

		throw new IllegalArgumentException("Unknown entity type: " + obj.getClass() + " for argument: " + obj + ".");
	}
}
