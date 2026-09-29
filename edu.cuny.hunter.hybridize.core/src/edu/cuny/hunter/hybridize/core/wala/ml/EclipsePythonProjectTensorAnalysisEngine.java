package edu.cuny.hunter.hybridize.core.wala.ml;

import static edu.cuny.hunter.hybridize.core.utils.Util.getPath;
import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IPath;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.loader.PythonLoaderFactory;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.util.PythonInterpreter;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.classLoader.ModuleEntry;

public class EclipsePythonProjectTensorAnalysisEngine extends PythonTensorAnalysisEngine {

	private static final String PYTHON3_INTERPRETER_FQN = "com.ibm.wala.cast.python.util.Python3Interpreter";

	private static final String PYTHON3_LOADER_FACTORY_FQN = "com.ibm.wala.cast.python.loader.Python3LoaderFactory";

	private static final ILog LOG = getLog(EclipsePythonProjectTensorAnalysisEngine.class);

	private IProject project;

	static {
		try {
			@SuppressWarnings("unchecked")
			Class<? extends PythonLoaderFactory> j3 = (Class<? extends PythonLoaderFactory>) Class.forName(PYTHON3_LOADER_FACTORY_FQN);
			PythonAnalysisEngine.setLoaderFactory(j3);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException("Can't find: " + PYTHON3_LOADER_FACTORY_FQN + ".", e);
		}

		try {
			Class<?> i3 = Class.forName(PYTHON3_INTERPRETER_FQN);
			PythonInterpreter interpreter = (PythonInterpreter) i3.getDeclaredConstructor().newInstance();
			PythonInterpreter.setInterpreter(interpreter);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException("Can't find: " + PYTHON3_INTERPRETER_FQN + ".", e);
		} catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
			throw new RuntimeException("Can't instantiate: " + PYTHON3_INTERPRETER_FQN + ".", e);
		}
	}

	/**
	 * Constructs an engine that selects framework methods, {@code tf.keras.Model} subclasses, and user model-forward methods with the given
	 * targeted k-CFA depth, rather than the default {@link PythonTensorAnalysisEngine#DEFAULT_TARGETED_CFA_DEPTH}. A deeper depth recovers
	 * precise per-context tensor shapes for the model-forward archetype (chained-layer calls), at a cost confined to those targeted methods
	 * (#600). The given scripts are left out of the analysis: scripts under no PYTHONPATH entry that the caller chose to skip rather than
	 * fail on (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/990).
	 *
	 * @param project The project to analyze.
	 * @param pythonPath The Python path entries.
	 * @param targetedCfaDepth The targeted k-CFA depth to forward to the analysis engine.
	 * @param excludedScripts The scripts to leave out, relative to the project's directory.
	 */
	public EclipsePythonProjectTensorAnalysisEngine(IProject project, List<File> pythonPath, int targetedCfaDepth,
			Collection<IPath> excludedScripts) {
		super(pythonPath, TENSORFLOW, targetedCfaDepth);
		this.initialize(project, excludedScripts);
	}

	private void initialize(IProject project, Collection<IPath> excludedScripts) {
		assert this.project == null : "Engine is meant to be initialized only once.";

		this.project = project;
		IPath projectPath = getPath(project);

		// The module leaves out each file whose path, the project's joined with the file's relative path, is excluded.
		Module dirModule = new EclipsePythonSourceDirectoryTreeModule(projectPath, ".py",
				excludedScripts.stream().map(projectPath::append).collect(Collectors.toUnmodifiableSet()));
		LOG.info("Creating engine from: " + dirModule + ".");

		this.setModuleFiles(Collections.singleton(dirModule));

		for (Iterator<? extends ModuleEntry> entries = dirModule.getEntries(); entries.hasNext();) {
			ModuleEntry entry = entries.next();
			LOG.info("Found entry: " + entry + ".");
		}
	}

	public IProject getProject() {
		return project;
	}
}
