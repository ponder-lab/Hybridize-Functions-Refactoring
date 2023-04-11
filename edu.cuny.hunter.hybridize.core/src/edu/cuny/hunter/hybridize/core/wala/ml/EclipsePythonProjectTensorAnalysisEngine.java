package edu.cuny.hunter.hybridize.core.wala.ml;

import java.io.IOException;
import java.util.Collections;
import java.util.Iterator;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.IPath;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.classLoader.FileModule;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.ide.classloader.EclipseSourceDirectoryTreeModule;
import com.ibm.wala.ide.classloader.EclipseSourceFileModule;

public class EclipsePythonProjectTensorAnalysisEngine extends PythonTensorAnalysisEngine {

	public EclipsePythonProjectTensorAnalysisEngine(IProject project) throws IllegalArgumentException, IOException {
		IPath projectPath = project.getFullPath();
		Module dirModule = new EclipseSourceDirectoryTreeModule(projectPath, null, ".py");
		System.out.println(dirModule);
		this.setModuleFiles(Collections.singleton(dirModule));
		System.out.println(this);
//		for (Iterator<FileModule> entries = dirModule.getEntries(); entries.hasNext();) {
//			EclipseSourceFileModule fileModule = (EclipseSourceFileModule) entries.next();
//			System.out.println(fileModule);
//		}

		// TODO: Add more stuff here. Log?
		Module module = (Module) dirModule;
		System.out.println(module);
		
		
		PythonSSAPropagationCallGraphBuilder builder = this.defaultCallGraphBuilder();
		System.out.println(builder);
	}
}
