package edu.cuny.hunter.hybridize.tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.eclipse.core.resources.IFolder;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.CoreException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.ibm.wala.classLoader.FileModule;

import edu.cuny.hunter.hybridize.core.wala.ml.EclipsePythonSourceDirectoryTreeModule;

/**
 * Tests that {@link EclipsePythonSourceDirectoryTreeModule} hands the analysis its files in one order however the file system lists them.
 * The analysis is sensitive to the order of its modules, so an order taken from the directory listing makes the same project analyze
 * differently on two machines (see wala/ML#168).
 */
public class SourceDirectoryTreeModuleTest {

	/** The files of each tree, relative to its root, in the order the entries must come in. */
	private static final List<String> SORTED = List.of("a.py", "b/c.py", "b/d.py", "e.py", "f/g.py", "f/h.py", "i.py", "j.py", "k.py");

	private IProject project;

	@Before
	public void createProject() throws CoreException {
		this.project = ResourcesPlugin.getWorkspace().getRoot().getProject(this.getClass().getSimpleName());

		if (this.project.exists())
			this.project.delete(true, true, null);

		this.project.create(null);
		this.project.open(null);
	}

	@After
	public void deleteProject() throws CoreException {
		this.project.delete(true, true, null);
	}

	/**
	 * Files the file system lists out of path order still come in path order. They are created in reverse, so a file system that lists a
	 * directory in creation order lists them backwards; one that lists by a hash of the name, as ext4 does, lists them in no particular
	 * order. Either way the listing is not the sorted order, which the first assertion checks, so the second shows the sort.
	 */
	@Test
	public void testEntriesAreSortedByPath() throws CoreException {
		IFolder tree = this.createTree("tree", SORTED.reversed());

		assertNotEquals("The file system does not already list the tree in path order.", SORTED, list(tree));
		assertEquals("Entries come in path order.", SORTED, entries(tree));
	}

	/**
	 * Files a module leaves out are still left out once its entries are sorted (#990).
	 */
	@Test
	public void testExcludedFilesStayExcluded() throws CoreException {
		IFolder tree = this.createTree("excluded", SORTED.reversed());
		EclipsePythonSourceDirectoryTreeModule module = new EclipsePythonSourceDirectoryTreeModule(tree.getLocation(), ".py",
				Set.of(tree.getLocation().append("b/c.py")));

		assertEquals("The excluded file is not an entry.", List.of("a.py", "b/d.py", "e.py", "f/g.py", "f/h.py", "i.py", "j.py", "k.py"),
				entries(tree, module));
	}

	private IFolder createTree(String name, List<String> files) throws CoreException {
		IFolder root = this.project.getFolder(name);
		root.create(true, true, null);

		for (String path : files) {
			IFolder parent = root;

			if (path.contains("/")) {
				parent = root.getFolder(path.substring(0, path.lastIndexOf('/')));

				if (!parent.exists())
					parent.create(true, true, null);
			}

			parent.getFile(path.substring(path.lastIndexOf('/') + 1))
					.create(new ByteArrayInputStream("x = 1\n".getBytes(StandardCharsets.UTF_8)), true, null);
		}

		return root;
	}

	/**
	 * The file system's own listing of the tree, depth-first in the order each directory is listed.
	 */
	private static List<String> list(IFolder root) {
		List<String> ret = new ArrayList<>();
		list(root.getLocation().toFile(), "", ret);
		return ret;
	}

	private static void list(File directory, String prefix, List<String> out) {
		for (File file : directory.listFiles())
			if (file.isDirectory())
				list(file, prefix + file.getName() + "/", out);
			else
				out.add(prefix + file.getName());
	}

	private static List<String> entries(IFolder root) {
		return entries(root, new EclipsePythonSourceDirectoryTreeModule(root.getLocation(), ".py", Set.of()));
	}

	private static List<String> entries(IFolder root, EclipsePythonSourceDirectoryTreeModule module) {
		List<String> ret = new ArrayList<>();
		String prefix = root.getLocation().toString() + "/";

		for (Iterator<FileModule> it = module.getEntries(); it.hasNext();)
			ret.add(it.next().getFile().getPath().substring(prefix.length()));

		return ret;
	}
}
