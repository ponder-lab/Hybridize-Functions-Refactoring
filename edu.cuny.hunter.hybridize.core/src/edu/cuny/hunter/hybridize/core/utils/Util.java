package edu.cuny.hunter.hybridize.core.utils;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IPath;
import org.eclipse.jface.text.Document;
import org.eclipse.jface.text.IDocument;
import org.eclipse.ltk.core.refactoring.Refactoring;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.navigator.PythonModelProvider;
import org.python.pydev.navigator.elements.PythonFile;
import org.python.pydev.navigator.elements.PythonNode;
import org.python.pydev.navigator.elements.PythonSourceFolder;
import org.python.pydev.outline.ParsedItem;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.visitors.scope.ASTEntryWithChildren;
import org.python.pydev.plugin.nature.PythonNature;

import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.analysis.FunctionExtractor;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;

public class Util {

	private static final ILog LOG = getLog(Util.class);

	private static final PythonModelProvider provider = new PythonModelProvider();

	public static Refactoring createRefactoring(Set<FunctionDefinition> functionDefinitions) throws TooManyMatchesException {
		return new ProcessorBasedRefactoring(new HybridizeFunctionRefactoringProcessor(functionDefinitions));
	}

	public static HybridizeFunctionRefactoringProcessor createHybridizeFunctionRefactoring(IProject[] projects,
			boolean alwaysCheckPythonSideEffects, boolean processFunctionsInParallel, boolean alwaysCheckRecusion,
			boolean useTestEntryPoints, boolean alwaysFollowTypeHints) throws ExecutionException, CoreException, IOException {
		Set<FunctionDefinition> functionDefinitions = getFunctionDefinitions(Arrays.asList(projects));
		return new HybridizeFunctionRefactoringProcessor(functionDefinitions, alwaysCheckPythonSideEffects, processFunctionsInParallel,
				alwaysCheckRecusion, useTestEntryPoints, alwaysFollowTypeHints);
	}

	public static Set<FunctionDefinition> getFunctionDefinitions(Iterable<?> iterable)
			throws ExecutionException, CoreException, IOException {
		Set<FunctionDefinition> ret = new HashSet<>();

		for (Object obj : iterable) {
			Set<PythonNode> nodeSet = getPythonNodes(obj);

			for (PythonNode node : nodeSet)
				ret.addAll(process(node));
		}

		return ret;
	}

	public static Set<PythonNode> getPythonNodes(Object obj) {
		Set<PythonNode> ret = new HashSet<>();

		if (obj instanceof PythonNode) {
			PythonNode pythonNode = (PythonNode) obj;
			ret.add(pythonNode);
		} else {
			Object[] children = provider.getChildren(obj);
			for (Object child : children)
				ret.addAll(getPythonNodes(child));
		}
		return ret;
	}

	public static Set<FunctionDefinition> process(PythonNode pythonNode) throws ExecutionException, CoreException, IOException {
		Set<FunctionDefinition> ret = new HashSet<>();

		String moduleName = getModuleName(pythonNode);
		File file = getFile(pythonNode);
		IDocument document = getDocument(pythonNode);
		IPythonNature nature = Util.getPythonNature(pythonNode);

		ParsedItem entry = pythonNode.entry;
		ASTEntryWithChildren ast = entry.getAstThis();
		SimpleNode simpleNode = ast.node;

		// extract function definitions.
		FunctionExtractor functionExtractor = new FunctionExtractor();
		try {
			simpleNode.accept(functionExtractor);
		} catch (Exception e) {
			LOG.error("Failed to start refactoring.", e);
			throw new ExecutionException("Failed to start refactoring.", e);
		}

		Collection<FunctionDef> definitions = functionExtractor.getDefinitions();

		for (FunctionDef def : definitions) {
			FunctionDefinition function = new FunctionDefinition(def, moduleName, file, document, nature);
			ret.add(function);
		}

		return ret;
	}

	private static IFile getActualFile(PythonNode pythonNode) {
		PythonFile pythonFile = pythonNode.getPythonFile();
		return pythonFile.getActualObject();
	}

	static IDocument getDocument(PythonNode pythonNode) throws CoreException, IOException {
		IFile file = getActualFile(pythonNode);

		try (InputStream contentStream = file.getContents()) {
			byte[] bytes = contentStream.readAllBytes();
			String content = new String(bytes, UTF_8);
			return new Document(content);
		}
	}

	static File getFile(PythonNode pythonNode) {
		IFile file = getActualFile(pythonNode);
		URI uri = file.getRawLocationURI();
		return new File(uri);
	}

	static String getFileName(PythonNode pythonNode) {
		IFile file = getActualFile(pythonNode);
		return file.getName();
	}

	static String getModuleName(PythonNode pythonNode) {
		String fileName = getFileName(pythonNode);
		int separatorPos = fileName.indexOf('.');
		return fileName.substring(0, separatorPos);
	}

	static IPythonNature getPythonNature(PythonNode pythonNode) {
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

	public static IPath getPath(IProject project) {
		IPath path = project.getFullPath();

		if (!path.toFile().exists())
			path = project.getLocation();

		return path;
	}
}
