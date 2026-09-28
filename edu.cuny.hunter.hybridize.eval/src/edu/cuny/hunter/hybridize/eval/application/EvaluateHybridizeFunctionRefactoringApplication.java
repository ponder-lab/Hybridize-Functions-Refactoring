package edu.cuny.hunter.hybridize.eval.application;

import static org.eclipse.core.runtime.Platform.getLog;
import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectDescription;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.resources.IWorkspace;
import org.eclipse.core.resources.IWorkspaceRoot;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.core.runtime.Path;
import org.eclipse.equinox.app.IApplication;
import org.eclipse.equinox.app.IApplicationContext;
import org.python.pydev.core.MisconfigurationException;
import org.python.pydev.core.PythonNatureWithoutProjectException;
import org.python.pydev.plugin.nature.PythonNature;

import edu.cuny.hunter.hybridize.eval.config.EvaluationOption;
import edu.cuny.hunter.hybridize.eval.handlers.EvaluateHybridizeFunctionRefactoringHandler;

/**
 * Headless (command-line) entry point for the evaluator. Runs
 * {@link EvaluateHybridizeFunctionRefactoringHandler#evaluate(IProject[], org.eclipse.core.runtime.IProgressMonitor)} over the open Python
 * projects already in the workspace, without the IDE.
 * <p>
 * The subjects must be PyDev projects in the workspace. They can be imported once through the UI, or on the command line with
 * {@code --import-projects=
 *
<dir>
 * ,
 *
<dir>
 * } (or {@code edu.cuny.hunter.hybridize.eval.importProjects}), which registers each directory's committed {@code .project} before the run;
 * see {@link #importProjects()}. This application then enumerates the open Python projects and evaluates them. Configuration may be given
 * either as {@code --kebab-case} program arguments (e.g. {@code --perform-change --projects=A,B}) or, equivalently, as the
 * {@code edu.cuny.hunter.hybridize.eval.*} system properties used by the IDE launch; a bare flag means {@code true}, and any name not given
 * on the command line falls back to its system property. By default all open Python projects are evaluated; {@code --projects} (or
 * {@code edu.cuny.hunter.hybridize.eval.projects}) restricts to a comma-separated subset. Launch with
 * {@code eclipse -application edu.cuny.hunter.hybridize.eval.evaluate -data <workspace> ...}. See
 * <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/657">issue 657</a> and
 * <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/664">issue 664</a>, and, for the import,
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

	/** Exit code returned when a project named by {@link EvaluationOption#IMPORT_PROJECTS} cannot be imported. */
	private static final int EXIT_IMPORT_FAILED = 4;

	/** Name of the file holding an Eclipse project's description. */
	private static final String PROJECT_DESCRIPTION_FILE = ".project";

	@Override
	public Object start(IApplicationContext context) throws Exception {
		if (!applyArguments(context))
			return EXIT_BAD_ARGUMENTS;

		if (!importProjects())
			return EXIT_IMPORT_FAILED;

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
	 * Imports the projects whose directories are listed, comma-separated, in the {@link EvaluationOption#IMPORT_PROJECTS importProjects}
	 * system property, so that a workspace need not be populated through the UI first. Each directory must be absolute and must hold a
	 * committed {@code .project}; its {@code .pydevproject}, if any, comes along unchanged. A project already in the workspace at the same
	 * location is opened if closed and otherwise left alone, so repeating an import is harmless. A project already in the workspace under
	 * the same name at a different location is an error rather than a silent substitution, since the run would then evaluate another tree.
	 * <p>
	 * Every imported project must also resolve its PyDev interpreter. A {@code .pydevproject} names its interpreter (usually
	 * {@code Default}) without defining it, and the definition lives in the workspace, so a fresh workspace imports the project but cannot
	 * analyze it. That is checked here, where it can fail loudly, rather than left to surface as an empty analysis.
	 *
	 * @return True iff every listed project is in the workspace, open, Python-natured, and has a resolvable interpreter. True when nothing
	 *         is listed.
	 * @throws Exception If the workspace cannot be read or saved.
	 */
	private static boolean importProjects() throws Exception {
		String list = System.getProperty(EvaluationOption.IMPORT_PROJECTS.key());

		if (list == null || list.isBlank())
			return true;

		IWorkspace workspace = ResourcesPlugin.getWorkspace();
		IWorkspaceRoot root = workspace.getRoot();
		File workspaceDirectory = root.getLocation().toFile().getCanonicalFile();
		boolean imported = true;

		for (String entry : list.split(",")) {
			if (entry.isBlank())
				continue;

			File directory = new File(entry.strip());

			if (!directory.isAbsolute()) {
				LOG.error("Cannot import " + directory
						+ ": the path must be absolute, since the evaluator's working directory is its output" + " directory.");
				imported = false;
				continue;
			}

			directory = directory.getCanonicalFile();
			File descriptionFile = new File(directory, PROJECT_DESCRIPTION_FILE);

			if (!descriptionFile.isFile()) {
				LOG.error("Cannot import " + directory + ": it has no " + PROJECT_DESCRIPTION_FILE
						+ ". Commit the subject's Eclipse project metadata first.");
				imported = false;
				continue;
			}

			IProjectDescription description = workspace.loadProjectDescription(new Path(descriptionFile.getPath()));
			IProject project = root.getProject(description.getName());

			if (project.exists()) {
				File existing = project.getLocation().toFile().getCanonicalFile();

				if (!existing.equals(directory)) {
					LOG.error("Cannot import " + directory + ": the workspace already has a project named " + project.getName() + " at "
							+ existing + ".");
					imported = false;
					continue;
				}
			} else {
				// A project inside the workspace directory must use the default location, which the workspace refuses to have set
				// explicitly.
				if (workspaceDirectory.equals(directory.getParentFile()))
					description.setLocation(null);

				project.create(description, new NullProgressMonitor());
				LOG.info("Imported project " + project.getName() + " from " + directory + ".");
			}

			if (!project.isOpen())
				project.open(new NullProgressMonitor());

			if (!project.hasNature(PYTHON_NATURE_ID)) {
				LOG.error("Imported project " + project.getName() + " is not a PyDev project: its " + PROJECT_DESCRIPTION_FILE
						+ " lacks the Python nature.");
				imported = false;
				continue;
			}

			try {
				PythonNature.getPythonNature(project).getProjectInterpreter();
			} catch (MisconfigurationException | PythonNatureWithoutProjectException e) {
				LOG.error("Imported project " + project.getName()
						+ " has no usable PyDev interpreter; configure it in the workspace before evaluating. PyDev reports: "
						+ e.getMessage());
				imported = false;
			}
		}

		// Persist the registration, so a later run against the same workspace sees these projects even if this one fails.
		workspace.save(true, new NullProgressMonitor());
		return imported;
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
