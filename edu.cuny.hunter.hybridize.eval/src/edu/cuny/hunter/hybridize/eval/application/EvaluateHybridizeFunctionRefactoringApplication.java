package edu.cuny.hunter.hybridize.eval.application;

import static org.eclipse.core.runtime.Platform.getLog;
import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.equinox.app.IApplication;
import org.eclipse.equinox.app.IApplicationContext;

import edu.cuny.hunter.hybridize.eval.config.EvaluationOption;
import edu.cuny.hunter.hybridize.eval.handlers.EvaluateHybridizeFunctionRefactoringHandler;

/**
 * Headless (command-line) entry point for the evaluator. Runs
 * {@link EvaluateHybridizeFunctionRefactoringHandler#evaluate(IProject[], org.eclipse.core.runtime.IProgressMonitor)} over the open Python
 * projects already in the workspace, without the IDE.
 * <p>
 * The workspace must be pre-populated with the subjects as PyDev projects (e.g., imported once via the UI); this application enumerates the
 * open Python projects and evaluates them. Configuration may be given either as {@code --kebab-case} program arguments (e.g.
 * {@code --perform-change --projects=A,B}) or, equivalently, as the {@code edu.cuny.hunter.hybridize.eval.*} system properties used by the
 * IDE launch; a bare flag means {@code true}, and any name not given on the command line falls back to its system property. By default all
 * open Python projects are evaluated; {@code --projects} (or {@code edu.cuny.hunter.hybridize.eval.projects}) restricts to a
 * comma-separated subset. Launch with {@code eclipse -application edu.cuny.hunter.hybridize.eval.evaluate -data <workspace> ...}. See
 * <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/657">issue 657</a> and
 * <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/664">issue 664</a>; programmatic project import (so a
 * workspace need not be pre-populated) is the follow-up
 * <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/658">issue 658</a>.
 */
public class EvaluateHybridizeFunctionRefactoringApplication implements IApplication {

	private static final ILog LOG = getLog(EvaluateHybridizeFunctionRefactoringApplication.class);

	/** Exit code returned when the workspace contains no open Python project to evaluate. */
	private static final int EXIT_NO_PROJECTS = 2;

	/** Exit code returned when the evaluation completes but its status is not OK. */
	private static final int EXIT_EVALUATION_FAILED = 1;

	/** Exit code returned when a command-line argument is unrecognized. */
	private static final int EXIT_BAD_ARGUMENTS = 3;

	@Override
	public Object start(IApplicationContext context) throws Exception {
		if (!applyArguments(context))
			return EXIT_BAD_ARGUMENTS;

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
	 * Translates {@code --kebab-name[=value]} program arguments into their backing {@code edu.cuny.hunter.hybridize.eval.}<i>name</i>
	 * system properties, so the headless CLI need not pass evaluator configuration as {@code -vmargs}. A bare flag sets {@code "true"}. Any
	 * configuration not named on the command line still falls back to its system property, preserving the existing {@code -D} surface.
	 *
	 * @param context The application context carrying the program arguments.
	 * @return True iff every argument was a recognized option.
	 */
	private static boolean applyArguments(IApplicationContext context) {
		Object rawArguments = context.getArguments().get(IApplicationContext.APPLICATION_ARGS);
		String[] arguments = rawArguments instanceof String[] strings ? strings : new String[0];

		for (String argument : arguments) {
			if (!argument.startsWith("--")) {
				LOG.warn("Ignoring unexpected non-option argument: " + argument);
				continue;
			}

			String body = argument.substring("--".length());
			int separator = body.indexOf('=');
			String flag = separator < 0 ? body : body.substring(0, separator);
			String value = separator < 0 ? Boolean.TRUE.toString() : body.substring(separator + 1);
			String name = toCamelCase(flag);

			if (!EvaluationOption.propertyNames().contains(name)) {
				LOG.error("Unrecognized evaluator option --" + flag + ". Recognized configuration names (pass as --kebab-case): "
						+ new TreeSet<>(EvaluationOption.propertyNames()));
				return false;
			}

			System.setProperty(EvaluationOption.PREFIX + name, value);
		}

		return true;
	}

	/**
	 * Converts a kebab-case command-line flag to the camelCase configuration name backing its system property.
	 *
	 * @param flag The kebab-case flag, without the leading dashes.
	 * @return The camelCase configuration name.
	 */
	private static String toCamelCase(String flag) {
		String[] parts = flag.split("-");
		StringBuilder name = new StringBuilder(parts[0]);

		for (int part = 1; part < parts.length; part++)
			if (!parts[part].isEmpty())
				name.append(Character.toUpperCase(parts[part].charAt(0))).append(parts[part].substring(1));

		return name.toString();
	}

	/**
	 * Returns the open, Python-natured projects in the workspace, restricted to the names listed in the {@link EvaluationOption#PROJECTS
	 * projects} system property when it is set.
	 *
	 * @return The open Python projects to evaluate.
	 * @throws org.eclipse.core.runtime.CoreException If a project's nature cannot be read.
	 */
	private static IProject[] getOpenPythonProjects() throws Exception {
		String filter = System.getProperty(EvaluationOption.PROJECTS.key());
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
