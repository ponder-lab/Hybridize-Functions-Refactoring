package edu.cuny.hunter.hybridize.core.wala.ml;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

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

	/**
	 * The files left out of the module, as the root joined with each file's path relative to it. Compared exactly, rather than through the
	 * superclass's exclusion patterns, which are regular expressions that escape only {@code .}: a path with {@code +}, {@code (} or
	 * {@code $} would mis-match, and one with <code>{{</code> would not compile (#990).
	 */
	private Set<IPath> excludedFiles = Collections.emptySet();

	public EclipsePythonSourceDirectoryTreeModule(IPath root, IPath[] excludePaths) {
		super(root, excludePaths);
		// We need a copy of this because `com.ibm.wala.ide.classloader.EclipseSourceDirectoryTreeModule.rootIPath` is private.
		this.rootPath = root;
	}

	public EclipsePythonSourceDirectoryTreeModule(IPath root, IPath[] excludePaths, String fileExt) {
		super(root, excludePaths, fileExt);
		this.rootPath = root;
	}

	/**
	 * Constructs a module over the files under the root with the given extension, leaving out the given files exactly (#990).
	 *
	 * @param root The root directory.
	 * @param fileExt The extension of the files to include.
	 * @param excludedFiles The files to leave out, as the root joined with each file's path relative to it.
	 */
	public EclipsePythonSourceDirectoryTreeModule(IPath root, String fileExt, Set<IPath> excludedFiles) {
		this(root, null, fileExt);
		this.excludedFiles = excludedFiles;
	}

	@Override
	protected boolean includeFile(File file) {
		return super.includeFile(file)
				&& !this.excludedFiles.contains(this.getRootPath().append(file.getPath().substring(root.getPath().length())));
	}

	/**
	 * Returns this module's files sorted by path. The superclass lists each directory in the order the file system returns it, which
	 * differs between machines for the same files (ext4, for one, orders a directory by a hash of the names seeded per file system). The
	 * analysis is sensitive to the order of its modules, so the same project would otherwise analyze differently depending on where it was
	 * checked out (see wala/ML#168). Every entry's path starts with this module's root, so sorting by the whole path sorts by the path
	 * relative to the root.
	 *
	 * @return An iterator over this module's files in path order.
	 */
	@Override
	public Iterator<FileModule> getEntries() {
		List<FileModule> entries = new ArrayList<>();
		super.getEntries().forEachRemaining(entries::add);
		entries.sort(Comparator.comparing(entry -> entry.getFile().getPath()));
		return entries.iterator();
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
