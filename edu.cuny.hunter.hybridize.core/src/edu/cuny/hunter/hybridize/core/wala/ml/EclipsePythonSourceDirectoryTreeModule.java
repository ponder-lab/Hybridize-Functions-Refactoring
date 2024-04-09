package edu.cuny.hunter.hybridize.core.wala.ml;

import java.io.File;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IWorkspace;
import org.eclipse.core.resources.IWorkspaceRoot;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.IPath;

import com.ibm.wala.classLoader.FileModule;
import com.ibm.wala.ide.classloader.EclipseSourceDirectoryTreeModule;

/**
 * A representation of a source directory tree module for an Eclipse Python (PyDev) project.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class EclipsePythonSourceDirectoryTreeModule extends EclipseSourceDirectoryTreeModule {

	protected IPath rootPath;

	public EclipsePythonSourceDirectoryTreeModule(IPath root, IPath[] excludePaths) {
		super(root, excludePaths);
		// We need a copy of this because `com.ibm.wala.ide.classloader.EclipseSourceDirectoryTreeModule.rootIPath` is private.
		this.rootPath = root;
	}

	public EclipsePythonSourceDirectoryTreeModule(IPath root, IPath[] excludePaths, String fileExt) {
		super(root, excludePaths, fileExt);
		this.rootPath = root;
	}

	@Override
	protected FileModule makeFile(File file) {
		IPath p = this.getRootPath().append(file.getPath().substring(root.getPath().length()));
		IWorkspace ws = ResourcesPlugin.getWorkspace();
		IWorkspaceRoot root = ws.getRoot();
		IFile ifile = root.getFile(p);
		assert ifile.getFullPath().toFile().exists();
		return new EclipsePythonSourceFileModule(ifile);
	}

	protected IPath getRootPath() {
		return rootPath;
	}

	@Override
	public String toString() {
		return this.getClass().getSimpleName() + ":" + this.getRootPath();
	}
}
