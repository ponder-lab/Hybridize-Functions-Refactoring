package edu.cuny.hunter.hybridize.eval.application;

import static org.eclipse.core.runtime.Platform.getLog;
import static org.python.pydev.plugin.nature.PythonNature.PYTHON_NATURE_ID;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectDescription;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.resources.IWorkspace;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IPath;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.core.runtime.Path;
import org.eclipse.equinox.app.IApplication;
import org.eclipse.equinox.app.IApplicationContext;
import org.python.pydev.ast.interpreter_managers.InterpreterManagersAPI;
import org.python.pydev.core.IInterpreterInfo;
import org.python.pydev.core.IInterpreterManager;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.MisconfigurationException;
import org.python.pydev.core.PythonNatureWithoutProjectException;
import org.python.pydev.plugin.nature.PythonNature;

import edu.cuny.hunter.hybridize.eval.config.EvaluationOption;
import edu.cuny.hunter.hybridize.eval.handlers.EvaluateHybridizeFunctionRefactoringHandler;

/**
 * Headless (command-line) entry point for the evaluator. Runs
 * {@link EvaluateHybridizeFunctionRefactoringHandler#evaluate(IProject[], org.eclipse.core.runtime.IProgressMonitor)} over the open Python
 * projects in the workspace, without the IDE.
 * <p>
 * The subjects must be PyDev projects in the workspace. They can be imported once through the UI, or on the command line with
 * {@code --import-projects=}<i>entry</i>{@code ,}<i>entry</i> (or {@code edu.cuny.hunter.hybridize.eval.importProjects}), which registers
 * each entry's directory before the run; see {@link #importProjects()}. A directory with committed Eclipse metadata is imported from it.
 * One without is created as a PyDev project from the source folders the entry names, as <i>directory</i>{@code =}<i>folder</i>{@code :}
 * <i>folder</i>, or, with {@code --source-folders-from-packaging}, from its packaging metadata. The source folders are never guessed: which
 * directories are import roots is the developer's call, as it is in the IDE's own project wizard. The workspace's PyDev interpreter, which
 * committed metadata names ({@code Default}) without defining, can be set with {@code --python-interpreter=}<i>executable</i>; see
 * {@link #configureInterpreter()}. This application then enumerates the open Python projects and evaluates them. Configuration may be given
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

	/** Exit code returned when the interpreter named by {@link EvaluationOption#PYTHON_INTERPRETER} cannot be configured. */
	private static final int EXIT_INTERPRETER_FAILED = 5;

	/** Name of the file holding an Eclipse project's description. */
	private static final String PROJECT_DESCRIPTION_FILE = ".project";

	/** Name of the file holding a PyDev project's configuration, including its source folders. */
	private static final String PYDEV_PROJECT_FILE = ".pydevproject";

	/** Separates an import entry's directory from its source folders. */
	private static final char SOURCE_FOLDERS_SEPARATOR = '=';

	/** Separates an import entry's source folders from one another. */
	private static final String SOURCE_FOLDER_SEPARATOR = ":";

	/** The variable PyDev expands to a project's directory in a {@code .pydevproject} source path, keeping the file relocatable. */
	private static final String PROJECT_DIR_VARIABLE = "/${PROJECT_DIR_NAME}";

	/** Separates the entries of a PyDev source path. */
	private static final String PYDEV_PATH_SEPARATOR = "|";

	/** The source path property of a {@code .pydevproject}, whose {@code path} elements are the project's source folders. */
	private static final Pattern PYDEV_SOURCE_PATH = Pattern
			.compile("name=\"org\\.python\\.pydev\\.PROJECT_SOURCE_PATH\">(.*?)</pydev_pathproperty>", Pattern.DOTALL);

	/** One source folder in a {@code .pydevproject} source path. */
	private static final Pattern PYDEV_PATH = Pattern.compile("<path>([^<]*)</path>");

	@Override
	public Object start(IApplicationContext context) throws Exception {
		if (!applyArguments(context))
			return EXIT_BAD_ARGUMENTS;

		if (!configureInterpreter())
			return EXIT_INTERPRETER_FAILED;

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
	 * Makes the executable named by the {@link EvaluationOption#PYTHON_INTERPRETER pythonInterpreter} system property the workspace's
	 * default PyDev interpreter, the one a project naming {@code Default} (as committed metadata usually does) resolves to. PyDev resolves
	 * {@code Default} to the first configured interpreter, so the executable is moved to the front of the list, and added to it first if
	 * the workspace does not have it yet; the interpreters already configured are kept. An executable already first is left alone, so
	 * repeating the option is harmless. PyDev reads the new interpreter's library paths by running it, choosing the paths it would select
	 * by default rather than asking. One exception is PyDev's own: a project with a {@code Pipfile} whose pipenv interpreter is configured
	 * resolves {@code Default} to that interpreter instead.
	 *
	 * @return True iff no interpreter is named, or the named one is now the workspace's default.
	 */
	private static boolean configureInterpreter() {
		String executable = System.getProperty(EvaluationOption.PYTHON_INTERPRETER.key());

		if (executable == null || executable.isBlank())
			return true;

		File file = new File(executable.strip());

		if (!file.isAbsolute() || !file.isFile() || !file.canExecute()) {
			LOG.error("Cannot configure the Python interpreter " + file + ": it must be the absolute path of an executable.");
			return false;
		}

		IInterpreterManager manager = InterpreterManagersAPI.getPythonInterpreterManager();
		List<IInterpreterInfo> infos = new ArrayList<>(Arrays.asList(manager.getInterpreterInfos()));
		Optional<IInterpreterInfo> existing = infos.stream().filter(info -> samePath(new File(info.getExecutableOrJar()), file))
				.findFirst();

		if (existing.isPresent() && infos.indexOf(existing.get()) == 0) {
			LOG.info(file + " is already the workspace's default Python interpreter.");
			return true;
		}

		IInterpreterInfo info;
		Set<String> toRestore;

		if (existing.isPresent()) {
			info = existing.get();
			infos.remove(info);
			toRestore = Set.of();
		} else {
			try {
				info = manager.createInterpreterInfo(file.getPath(), new NullProgressMonitor(), false);
			} catch (RuntimeException e) {
				LOG.error("Cannot configure the Python interpreter " + file + ": " + e.getMessage(), e);
				return false;
			}

			if (info == null) {
				LOG.error("Cannot configure the Python interpreter " + file + ": PyDev could not read its configuration.");
				return false;
			}

			// A new interpreter's library index is built on configuration, as the IDE does when one is added.
			toRestore = Set.of(info.getExecutableOrJar());
		}

		infos.add(0, info);
		manager.setInfos(infos.toArray(IInterpreterInfo[]::new), toRestore, new NullProgressMonitor());
		LOG.info("Configured " + file + " as the workspace's default Python interpreter.");
		return true;
	}

	/**
	 * Whether two paths name the same file without resolving symbolic links. An interpreter is compared this way, since a virtual
	 * environment's {@code python} is a link to its base interpreter, and resolving it would take two environments for one.
	 *
	 * @param a One path.
	 * @param b The other path.
	 * @return True iff they are the same absolute, normalized path.
	 */
	private static boolean samePath(File a, File b) {
		return a.toPath().toAbsolutePath().normalize().equals(b.toPath().toAbsolutePath().normalize());
	}

	/**
	 * Whether two files are the same, after resolving symbolic links where possible.
	 *
	 * @param a One file.
	 * @param b The other file.
	 * @return True iff they name the same file.
	 */
	private static boolean sameFile(File a, File b) {
		try {
			return a.getCanonicalFile().equals(b.getCanonicalFile());
		} catch (IOException _) {
			return a.getAbsoluteFile().equals(b.getAbsoluteFile());
		}
	}

	/**
	 * Imports the projects whose entries are listed, comma-separated, in the {@link EvaluationOption#IMPORT_PROJECTS importProjects} system
	 * property, so that a workspace need not be populated through the UI first. An entry is a project's absolute directory, optionally
	 * followed by {@code =} and its source folders, colon-separated and relative to the directory ({@code .} for the directory itself); for
	 * example, {@code /subjects/psiz=.:src}. Each entry is imported by {@link #importProject(IWorkspace, String, List)}; a failed entry is
	 * logged and the rest are still tried, so one run reports every problem.
	 *
	 * @return True iff every listed project is in the workspace, open, Python-natured, and has a resolvable interpreter. True when nothing
	 *         is listed.
	 * @throws CoreException If the workspace cannot be saved.
	 */
	private static boolean importProjects() throws CoreException {
		String list = System.getProperty(EvaluationOption.IMPORT_PROJECTS.key());

		if (list == null || list.isBlank())
			return true;

		IWorkspace workspace = ResourcesPlugin.getWorkspace();
		boolean imported = true;

		for (String entry : list.split(",")) {
			if (entry.isBlank())
				continue;

			int separator = entry.indexOf(SOURCE_FOLDERS_SEPARATOR);
			String path = (separator < 0 ? entry : entry.substring(0, separator)).strip();
			List<String> sourceFolders = separator < 0 ? null
					: Arrays.stream(entry.substring(separator + 1).split(SOURCE_FOLDER_SEPARATOR)).map(String::strip)
							.filter(folder -> !folder.isEmpty()).toList();

			if (!importProject(workspace, path, sourceFolders))
				imported = false;
		}

		// Persist the registration, so a later run against the same workspace sees these projects even if this one fails.
		workspace.save(true, new NullProgressMonitor());
		return imported;
	}

	/**
	 * Imports one project from the given directory, which must be absolute. A directory with a committed {@code .project} is imported from
	 * it, and its {@code .pydevproject}, if any, comes along unchanged; source folders may not be given for it, since that file already
	 * decides them. A directory without one is created as a PyDev project by {@link #createProject(IWorkspace, File, List)}. The project
	 * keeps the location its description loads with: its directory, or the default location when that directory is the workspace's own
	 * directory for the project's name. A directory elsewhere under the workspace root is refused by the workspace, which this reports as a
	 * failure rather than redirecting the project to an empty default location.
	 * <p>
	 * A project already in the workspace at the same location is opened if closed and otherwise left alone, so repeating an import is
	 * harmless. A project already in the workspace under the same name at a different location is an error rather than a silent
	 * substitution, since the run would then evaluate another tree.
	 * <p>
	 * The project must also resolve its PyDev interpreter. A {@code .pydevproject} names its interpreter (usually {@code Default}) without
	 * defining it, and the definition lives in the workspace, so a fresh workspace imports the project but cannot analyze it. That is
	 * checked here, where it can fail loudly, rather than left to surface as an empty analysis. A project this call created that fails the
	 * check is removed from the workspace again, so that a later run without the import does not evaluate it unchecked. An imported
	 * project's directory is left untouched, while a created one's loses the metadata this call wrote, so that a later run does not take it
	 * for committed metadata. Repeating a creation entry is harmless too: the directory then has the metadata the first run wrote, which is
	 * imported when its source folders are the ones the entry names.
	 *
	 * @param workspace The workspace to import into.
	 * @param path The project's directory.
	 * @param sourceFolders The source folders the entry names, relative to the directory, or {@code null} if it names none.
	 * @return True iff the project is in the workspace, open, Python-natured, and has a resolvable interpreter.
	 */
	private static boolean importProject(IWorkspace workspace, String path, List<String> sourceFolders) {
		File directory = new File(path);

		if (!directory.isAbsolute()) {
			LOG.error("Cannot import " + directory
					+ ": the path must be absolute, since the evaluator's working directory is its output directory.");
			return false;
		}

		IProject project = null;
		boolean created = false;
		boolean writesMetadata = false;

		try {
			directory = directory.getCanonicalFile();
			File descriptionFile = new File(directory, PROJECT_DESCRIPTION_FILE);

			if (!descriptionFile.isFile()) {
				if (new File(directory, PYDEV_PROJECT_FILE).isFile()) {
					LOG.error("Cannot import " + directory + ": it has a " + PYDEV_PROJECT_FILE + " but no " + PROJECT_DESCRIPTION_FILE
							+ ". Commit both, or neither, so that its source folders have one source.");
					return false;
				}

				List<String> folders = sourceFolders != null ? sourceFolders : sourceFoldersFromPackaging(directory);

				if (folders == null)
					return false;

				project = workspace.getRoot().getProject(directory.getName());

				if (project.exists()) {
					IPath location = project.getLocation();

					if (location == null || !sameFile(location.toFile(), directory)) {
						LOG.error("Cannot create a project for " + directory + ": the workspace already has a project named "
								+ project.getName() + " at " + project.getLocationURI() + ".");
						return false;
					}

					// Registered here before, and its metadata since removed (as by `git clean`): forget the registration and start over.
					LOG.info("Recreating project " + project.getName() + ", registered at " + directory + " without its "
							+ PROJECT_DESCRIPTION_FILE + ".");
					project.delete(IResource.NEVER_DELETE_PROJECT_CONTENT | IResource.FORCE, new NullProgressMonitor());
				}

				// Marked before the attempt, so a failure partway through still removes what was made.
				created = true;
				writesMetadata = true;

				if (createProject(workspace, directory, folders) && verifyPyDevProject(project, created))
					return true;

				unregister(project, created);
				return removeWrittenMetadata(directory);
			}

			if (sourceFolders != null) {
				// Metadata an earlier run created from this same entry is imported; any other decides its own source folders.
				List<String> requested = sourcePath(directory, sourceFolders);

				if (requested == null)
					return false;

				Set<String> existing = committedSourcePath(directory);

				if (!new HashSet<>(requested).equals(existing)) {
					LOG.error("Cannot import " + directory + " with the source folders " + sourceFolders + ": it has a committed "
							+ PROJECT_DESCRIPTION_FILE + ", whose " + PYDEV_PROJECT_FILE + " names the source path " + existing
							+ " instead. Drop the folders from the entry, or change the committed metadata.");
					return false;
				}
			}

			IProjectDescription description = workspace.loadProjectDescription(new Path(descriptionFile.getPath()));
			project = workspace.getRoot().getProject(description.getName());

			if (project.exists()) {
				IPath location = project.getLocation();
				File existing = location == null ? null : location.toFile().getCanonicalFile();

				if (!directory.equals(existing)) {
					LOG.error("Cannot import " + directory + ": the workspace already has a project named " + project.getName() + " at "
							+ (existing == null ? project.getLocationURI() : existing) + ".");
					return false;
				}
			} else {
				project.create(description, new NullProgressMonitor());
				created = true;
				LOG.info("Imported project " + project.getName() + " from " + directory + ".");
			}

			if (!project.isOpen())
				project.open(new NullProgressMonitor());

			return verifyPyDevProject(project, created);
		} catch (CoreException | IOException | RuntimeException e) {
			// A runtime failure inside PyDev or the workspace must still undo what this call made.
			LOG.error("Cannot import " + directory + ": " + e.getMessage(), e);
			unregister(project, created);
			return writesMetadata ? removeWrittenMetadata(directory) : false;
		}
	}

	/**
	 * Deletes the {@code .project} and {@code .pydevproject} a failed creation wrote into a directory that had neither, so that a later run
	 * does not take them for committed metadata.
	 *
	 * @param directory The project's directory.
	 * @return False, the failed import's result, so that a caller can return this call directly.
	 */
	private static boolean removeWrittenMetadata(File directory) {
		for (String name : new String[] { PROJECT_DESCRIPTION_FILE, PYDEV_PROJECT_FILE }) {
			File file = new File(directory, name);

			if (file.exists() && !file.delete())
				LOG.error("Could not remove " + file + ", written for a project that failed to be created; delete it before the next run.");
		}

		return false;
	}

	/**
	 * Returns the source folders a directory without Eclipse metadata is created with when its import entry names none. They are read from
	 * its packaging metadata when {@link EvaluationOption#SOURCE_FOLDERS_FROM_PACKAGING sourceFoldersFromPackaging} is set; otherwise, or
	 * when there is no packaging metadata, none are, and the import fails. The failure suggests what the packaging metadata declares, if
	 * anything, so the entry can be completed deliberately.
	 *
	 * @param directory The project's directory.
	 * @return The source folders, or {@code null} if there are none to use.
	 * @throws IOException If a packaging file cannot be read.
	 */
	private static List<String> sourceFoldersFromPackaging(File directory) throws IOException {
		Optional<List<String>> declared = PackagingSourceFolders.of(directory);

		if (Boolean.getBoolean(EvaluationOption.SOURCE_FOLDERS_FROM_PACKAGING.key())) {
			if (declared.isPresent()) {
				LOG.info("Using the source folders " + declared.get() + " that " + directory + "'s packaging metadata declares.");
				return declared.get();
			}

			LOG.error("Cannot create a project for " + directory + ": it has no " + PROJECT_DESCRIPTION_FILE
					+ " and no packaging metadata (setup.cfg, pyproject.toml, or setup.py) to read its source folders from. Name them in the "
					+ "entry, as " + directory + SOURCE_FOLDERS_SEPARATOR + "folder" + SOURCE_FOLDER_SEPARATOR + "folder.");
			return null;
		}

		LOG.error("Cannot import " + directory + ": it has no " + PROJECT_DESCRIPTION_FILE
				+ ". Commit its Eclipse project metadata, or name its source folders in the entry, as " + directory
				+ SOURCE_FOLDERS_SEPARATOR + "folder" + SOURCE_FOLDER_SEPARATOR + "folder"
				+ declared
						.map(folders -> ". Its packaging metadata declares " + directory + SOURCE_FOLDERS_SEPARATOR
								+ String.join(SOURCE_FOLDER_SEPARATOR, folders) + ", which --source-folders-from-packaging would use")
						.orElse("")
				+ ".");
		return null;
	}

	/**
	 * Creates a PyDev project for a directory without Eclipse metadata, named after the directory, with the given source folders and the
	 * workspace's default interpreter. Eclipse keeps a project's description in its directory, so this writes a {@code .project} and a
	 * {@code .pydevproject} there, as the IDE's project wizard does; the source path is written relative to the project
	 * ({@code ${PROJECT_DIR_NAME}}), so the files stay valid if the directory moves. Each source folder must be an existing directory
	 * within the project's.
	 *
	 * @param workspace The workspace to create the project in.
	 * @param directory The project's directory, canonical.
	 * @param sourceFolders The source folders, relative to the directory.
	 * @return True iff the project was created.
	 * @throws CoreException If the project cannot be created or given the Python nature.
	 * @throws IOException If a source folder cannot be resolved.
	 */
	private static boolean createProject(IWorkspace workspace, File directory, List<String> sourceFolders)
			throws CoreException, IOException {
		List<String> sourcePath = sourcePath(directory, sourceFolders);

		if (sourcePath == null)
			return false;

		IProject project = workspace.getRoot().getProject(directory.getName());
		IProjectDescription description = workspace.newProjectDescription(project.getName());
		File defaultLocation = workspace.getRoot().getLocation().append(project.getName()).toFile();

		// A directory at the project's default location must be left to the workspace; any other is named explicitly.
		if (!sameFile(directory, defaultLocation))
			description.setLocation(new Path(directory.getPath()));

		project.create(description, new NullProgressMonitor());
		project.open(new NullProgressMonitor());
		PythonNature.addNature(project, new NullProgressMonitor(), IPythonNature.PYTHON_VERSION_INTERPRETER,
				String.join(PYDEV_PATH_SEPARATOR, sourcePath), null, IPythonNature.DEFAULT_INTERPRETER, null);
		LOG.info("Created project " + project.getName() + " for " + directory + " with the source folders " + sourceFolders + ".");
		return true;
	}

	/**
	 * Resolves an entry's source folders to the PyDev source path a created project is given: each folder relative to the project's
	 * directory, written with {@code ${PROJECT_DIR_NAME}} so the file stays valid if the directory moves. Each folder must be an existing
	 * directory within the project's.
	 *
	 * @param directory The project's directory, canonical.
	 * @param sourceFolders The source folders, relative to the directory.
	 * @return The source path's entries, without duplicates, or {@code null} if a folder is invalid or none is given.
	 * @throws IOException If a source folder cannot be resolved.
	 */
	private static List<String> sourcePath(File directory, List<String> sourceFolders) throws IOException {
		if (sourceFolders.isEmpty()) {
			LOG.error("Cannot create a project for " + directory + ": its entry names no source folders.");
			return null;
		}

		List<String> ret = new ArrayList<>();

		for (String folder : sourceFolders) {
			File resolved = new File(folder).isAbsolute() ? null : new File(directory, folder).getCanonicalFile();

			if (resolved == null || !resolved.toPath().startsWith(directory.toPath()) || !resolved.isDirectory()) {
				LOG.error("Cannot create a project for " + directory + ": the source folder " + folder
						+ " is not an existing directory within it, relative to it.");
				return null;
			}

			String relative = directory.toPath().relativize(resolved.toPath()).toString().replace(File.separatorChar, '/');
			String entry = relative.isEmpty() ? PROJECT_DIR_VARIABLE : PROJECT_DIR_VARIABLE + "/" + relative;

			if (!ret.contains(entry))
				ret.add(entry);
		}

		return ret;
	}

	/**
	 * Reads the source path a directory's {@code .pydevproject} names.
	 *
	 * @param directory The project's directory.
	 * @return The source path's entries, empty if the file is absent or names none.
	 * @throws IOException If the file cannot be read.
	 */
	private static Set<String> committedSourcePath(File directory) throws IOException {
		File file = new File(directory, PYDEV_PROJECT_FILE);
		Set<String> ret = new HashSet<>();

		if (!file.isFile())
			return ret;

		Matcher property = PYDEV_SOURCE_PATH.matcher(new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8));

		if (property.find()) {
			Matcher path = PYDEV_PATH.matcher(property.group(1));

			while (path.find())
				ret.add(path.group(1).strip());
		}

		return ret;
	}

	/**
	 * Checks that an imported or created project is a usable PyDev project: it has the Python nature, PyDev can load it, and its
	 * interpreter resolves. A project this run created that fails is removed from the workspace again.
	 *
	 * @param project The project.
	 * @param created True iff this run created it.
	 * @return True iff the project is usable.
	 * @throws CoreException If the project's natures cannot be read.
	 */
	private static boolean verifyPyDevProject(IProject project, boolean created) throws CoreException {
		if (!project.hasNature(PYTHON_NATURE_ID)) {
			LOG.error("Project " + project.getName() + " is not a PyDev project: its " + PROJECT_DESCRIPTION_FILE
					+ " lacks the Python nature.");
			return unregister(project, created);
		}

		PythonNature nature = PythonNature.getPythonNature(project);

		if (nature == null) {
			LOG.error("Project " + project.getName() + " has the Python nature, but PyDev could not load it.");
			return unregister(project, created);
		}

		try {
			nature.getProjectInterpreter();
		} catch (MisconfigurationException | PythonNatureWithoutProjectException e) {
			LOG.error("Project " + project.getName()
					+ " has no usable PyDev interpreter; configure it in the workspace, or pass --python-interpreter, before evaluating. PyDev reports: "
					+ e.getMessage());
			return unregister(project, created);
		}

		return true;
	}

	/**
	 * Removes a project that failed its import checks from the workspace, if this run created it, without deleting its directory's
	 * contents. A project that was already in the workspace is left as it was.
	 *
	 * @param project The project that failed.
	 * @param created True iff this run created it.
	 * @return False, the failed import's result, so that a caller can return this call directly.
	 */
	private static boolean unregister(IProject project, boolean created) {
		if (created && project.exists())
			try {
				project.delete(IResource.NEVER_DELETE_PROJECT_CONTENT | IResource.FORCE, new NullProgressMonitor());
			} catch (CoreException e) {
				LOG.error("Could not remove the failed project " + project.getName() + " from the workspace.", e);
			}

		return false;
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
