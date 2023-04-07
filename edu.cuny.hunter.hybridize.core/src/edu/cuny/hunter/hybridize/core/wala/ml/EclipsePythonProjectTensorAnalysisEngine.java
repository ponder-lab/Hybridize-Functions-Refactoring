package edu.cuny.hunter.hybridize.core.wala.ml;

import java.util.Collections;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.IPath;

import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ide.classloader.EclipseSourceDirectoryTreeModule;

public class EclipsePythonProjectTensorAnalysisEngine extends PythonTensorAnalysisEngine {

	public EclipsePythonProjectTensorAnalysisEngine(IProject project) {
		IPath projectPath = project.getFullPath();
		EclipseSourceDirectoryTreeModule dirModule = new EclipseSourceDirectoryTreeModule(projectPath, null, ".py");
		this.setModuleFiles(Collections.singleton(dirModule));
	}
}
