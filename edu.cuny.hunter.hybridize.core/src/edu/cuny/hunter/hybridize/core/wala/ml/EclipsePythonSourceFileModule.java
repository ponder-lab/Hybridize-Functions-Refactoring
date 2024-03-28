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

	protected final IFile file;

	public EclipsePythonSourceFileModule(IFile file) {
		super(file == null ? null : new File(file.getFullPath().toOSString()), file == null ? null : file.getName(), null);
		this.file = file;
	}

	public IFile getIFile() {
		return this.file;
	}

	@Override
	public String toString() {
		return this.getClass().getSimpleName() + ":" + this.getFile().toString();
	}

	/**
	 * {@inheritDoc}
	 *
	 * @implNote We consider the name of the Python file as the "class name."
	 */
	@Override
	public String getClassName() {
		return this.getName();
	}

	/**
	 * {@inheritDoc}
	 *
	 * @implNote We consider the full name of the Python file as its "name."
	 */
	@Override
	public String getName() {
		return this.getFile().getAbsolutePath();
	}
}
