package edu.cuny.hunter.hybridize.core.wala.ml;

import static org.eclipse.core.runtime.Platform.getLog;

import java.lang.reflect.InvocationTargetException;
import java.util.Collections;
import java.util.Iterator;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IPath;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.loader.PythonLoaderFactory;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.util.PythonInterpreter;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.classLoader.ModuleEntry;
import com.ibm.wala.ide.classloader.EclipseSourceDirectoryTreeModule;

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

	public EclipsePythonProjectTensorAnalysisEngine(IProject project) {
		this.project = project;
		IPath projectPath = project.getFullPath();
		Module dirModule = new EclipseSourceDirectoryTreeModule(projectPath, null, ".py");
		LOG.info("Creating engine from: " + dirModule);

		this.setModuleFiles(Collections.singleton(dirModule));

		for (Iterator<? extends ModuleEntry> entries = dirModule.getEntries(); entries.hasNext();) {
			ModuleEntry entry = entries.next();
			LOG.info("Found entry: " + entry);
		}
	}

	public IProject getProject() {
		return project;
	}
}
