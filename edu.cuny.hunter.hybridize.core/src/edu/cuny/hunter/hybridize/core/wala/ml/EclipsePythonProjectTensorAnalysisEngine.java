package edu.cuny.hunter.hybridize.core.wala.ml;

import java.util.Collections;
import java.util.Iterator;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.IPath;

import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.classLoader.ModuleEntry;
import com.ibm.wala.ide.classloader.EclipseSourceDirectoryTreeModule;

public class EclipsePythonProjectTensorAnalysisEngine extends PythonTensorAnalysisEngine {

	public EclipsePythonProjectTensorAnalysisEngine(IProject project) {
		IPath projectPath = project.getFullPath();
		Module dirModule = new EclipseSourceDirectoryTreeModule(projectPath, null, ".py");
		System.out.println(dirModule);

		this.setModuleFiles(Collections.singleton(dirModule));

		for (Iterator<? extends ModuleEntry> entries2 = dirModule.getEntries(); entries2.hasNext();) {
			ModuleEntry entry = entries2.next();
			System.out.println(entry);
		}

		// PythonSSAPropagationCallGraphBuilder builder = this.defaultCallGraphBuilder();
		// System.out.println(builder);
	}
}
