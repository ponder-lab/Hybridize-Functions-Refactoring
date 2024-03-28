package edu.cuny.hunter.hybridize.core.wala.ml;

import java.io.File;

import org.eclipse.core.resources.IFile;

import com.ibm.wala.classLoader.SourceFileModule;

/**
 * A module which is a wrapper around a .py file.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class EclipsePythonSourceFileModule extends SourceFileModule {

	protected final IFile f;

	public EclipsePythonSourceFileModule(IFile f) {
		super(f == null ? null : new File(f.getFullPath().toOSString()), f == null ? null : f.getName(), null);
		this.f = f;
	}

	public IFile getIFile() {
		return f;
	}

	@Override
	public String toString() {
		return this.getClass().getSimpleName() + ":" + getFile().toString();
	}

	/**
	 * @implNote We consider the name of the Python file as the "class name."
	 */
	@Override
	public String getClassName() {
		return this.getName();
	}

	/**
	 * @implNote We consider the full name of the Python file as its "name."
	 */
	@Override
	public String getName() {
		return this.getFile().getAbsolutePath();
	}
}
