package edu.cuny.hunter.hybridize.core.wala.ml;

import static org.eclipse.core.runtime.Platform.getLog;

import java.util.Collections;
import java.util.Iterator;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IPath;

import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.classLoader.ModuleEntry;
import com.ibm.wala.ide.classloader.EclipseSourceDirectoryTreeModule;

public class EclipsePythonProjectTensorAnalysisEngine extends PythonTensorAnalysisEngine {

	private static final String ANALYSIS_ENGINE_FQN = "com.ibm.wala.cast.python.client.PythonAnalysisEngine";

	private static final ILog LOG = getLog(EclipsePythonProjectTensorAnalysisEngine.class);

	private IProject project;

	static {
		// Ensure that the following class is loaded to invoke the static initializer.
		try {
			Class.forName(ANALYSIS_ENGINE_FQN);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException("Can't load: " + ANALYSIS_ENGINE_FQN, e);
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
