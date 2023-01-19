package edu.cuny.hunter.hybridize.ui.handlers;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.jface.text.Document;
import org.eclipse.jface.text.IDocument;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.navigator.elements.PythonFile;
import org.python.pydev.navigator.elements.PythonNode;
import org.python.pydev.navigator.elements.PythonSourceFolder;
import org.python.pydev.plugin.nature.PythonNature;

public class Util {
	private static IFile getActualFile(PythonNode pythonNode) {
		PythonFile pythonFile = pythonNode.getPythonFile();
		return pythonFile.getActualObject();
	}

	public static IDocument getDocument(PythonNode pythonNode) throws CoreException, IOException {
		IFile file = getActualFile(pythonNode);

		try (InputStream contentStream = file.getContents()) {
			byte[] bytes = contentStream.readAllBytes();
			String content = new String(bytes, UTF_8);
			return new Document(content);
		}
	}

	public static File getFile(PythonNode pythonNode) {
		IFile file = getActualFile(pythonNode);
		URI uri = file.getRawLocationURI();
		return new File(uri);
	}

	static String getFileName(PythonNode pythonNode) {
		IFile file = getActualFile(pythonNode);
		return file.getName();
	}

	public static String getModuleName(PythonNode pythonNode) {
		String fileName = getFileName(pythonNode);
		int separatorPos = fileName.indexOf('.');
		return fileName.substring(0, separatorPos);
	}

	public static IPythonNature getPythonNature(PythonNode pythonNode) {
		IProject project = getProject(pythonNode);
		PythonNature pythonNature = PythonNature.getPythonNature(project);
		return pythonNature;
	}

	private static IProject getProject(PythonNode pythonNode) {
		PythonFile pythonFile = pythonNode.getPythonFile();
		PythonSourceFolder sourceFolder = pythonFile.getSourceFolder();
		IResource resource = sourceFolder.getActualObject();
		IProject project = resource.getProject();
		return project;
	}

	private Util() {
	}
}
