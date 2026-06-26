package edu.cuny.hunter.hybridize.eval.application;

import static org.eclipse.core.runtime.Platform.getLog;
import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.equinox.app.IApplication;
import org.eclipse.equinox.app.IApplicationContext;

import edu.cuny.hunter.hybridize.eval.handlers.EvaluateHybridizeFunctionRefactoringHandler;

/**
 * Headless (command-line) entry point for the evaluator. Runs
 * {@link EvaluateHybridizeFunctionRefactoringHandler#evaluate(IProject[], org.eclipse.core.runtime.IProgressMonitor)} over the open Python
 * projects already in the workspace, without the IDE.
 * <p>
 * The workspace must be pre-populated with the subjects as PyDev projects (e.g., imported once via the UI); this application enumerates the
 * open Python projects and evaluates them. Configuration comes from the same {@code edu.cuny.hunter.hybridize.eval.*} system properties as
 * the IDE launch. By default all open Python projects are evaluated; set {@code edu.cuny.hunter.hybridize.eval.projects} to a
 * comma-separated list of project names to evaluate only a subset. Launch with
 * {@code eclipse -application edu.cuny.hunter.hybridize.eval.evaluate -data <workspace> ...}. See
 * <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/657">issue 657</a>; programmatic project import (so a
 * workspace need not be pre-populated) is the follow-up
 * <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/658">issue 658</a>.
 */
public class EvaluateHybridizeFunctionRefactoringApplication implements IApplication {

	private static final ILog LOG = getLog(EvaluateHybridizeFunctionRefactoringApplication.class);

	/** Exit code returned when the workspace contains no open Python project to evaluate. */
	private static final Integer EXIT_NO_PROJECTS = Integer.valueOf(2);

	/** Exit code returned when the evaluation completes but its status is not OK. */
	private static final Integer EXIT_EVALUATION_FAILED = Integer.valueOf(1);

	/**
	 * System property naming a comma-separated subset of project names to evaluate; when unset or blank, all open Python projects in the
	 * workspace are evaluated.
	 */
	private static final String PROJECTS_PROPERTY_KEY = "edu.cuny.hunter.hybridize.eval.projects";

	@Override
	public Object start(IApplicationContext context) throws Exception {
		// Refresh so the workspace reflects the subjects' current on-disk state (e.g., after a git checkout of the evaluated branch).
		ResourcesPlugin.getWorkspace().getRoot().refreshLocal(IResource.DEPTH_INFINITE, new NullProgressMonitor());

		IProject[] projects = getOpenPythonProjects();

		if (projects.length == 0) {
			LOG.warn("No open Python projects in the workspace to evaluate. Import the subjects as PyDev projects first.");
			return EXIT_NO_PROJECTS;
		}

		LOG.info("Evaluating " + projects.length + " Python project(s) headlessly.");
		IStatus status = new EvaluateHybridizeFunctionRefactoringHandler().evaluate(projects, new NullProgressMonitor());

		if (!status.isOK()) {
			LOG.log(status);
			return EXIT_EVALUATION_FAILED;
		}

		return IApplication.EXIT_OK;
	}

	@Override
	public void stop() {
		// Nothing to clean up; evaluate(...) runs synchronously within start(...).
	}

	/**
	 * Returns the open, Python-natured projects in the workspace, restricted to the names listed in the {@value #PROJECTS_PROPERTY_KEY}
	 * system property when it is set.
	 *
	 * @return The open Python projects to evaluate.
	 * @throws org.eclipse.core.runtime.CoreException If a project's nature cannot be read.
	 */
	private static IProject[] getOpenPythonProjects() throws Exception {
		String filter = System.getProperty(PROJECTS_PROPERTY_KEY);
		Set<String> wanted = null;

		if (filter != null && !filter.isBlank()) {
			wanted = new HashSet<>();

			for (String name : filter.split(","))
				if (!name.isBlank())
					wanted.add(name.strip());
		}

		List<IProject> pythonProjects = new ArrayList<>();

		for (IProject project : ResourcesPlugin.getWorkspace().getRoot().getProjects())
			if (project.isOpen() && project.hasNature(PYTHON_NATURE_ID) && (wanted == null || wanted.contains(project.getName())))
				pythonProjects.add(project);

		return pythonProjects.toArray(IProject[]::new);
	}
}
