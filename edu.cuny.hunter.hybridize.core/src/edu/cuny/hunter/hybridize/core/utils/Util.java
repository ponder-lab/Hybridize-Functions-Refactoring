package edu.cuny.hunter.hybridize.core.utils;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.toList;
import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IContainer;
import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectNature;
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
import org.python.pydev.navigator.elements.IWrappedResource;
import org.python.pydev.navigator.elements.PythonFile;
import org.python.pydev.navigator.elements.PythonNode;
import org.python.pydev.navigator.elements.PythonSourceFolder;
import org.python.pydev.outline.ParsedItem;
import org.python.pydev.parser.PyParser;
import org.python.pydev.parser.PyParser.ParserInfo;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.visitors.scope.ASTEntryWithChildren;
import org.python.pydev.plugin.nature.PythonNature;
import org.python.pydev.shared_core.parsing.BaseParser.ParseOutput;

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
			boolean useTestEntryPoints, boolean alwaysFollowTypeHints, boolean useSpeculativeAnalysis)
			throws ExecutionException, CoreException, IOException {
		return createHybridizeFunctionRefactoring(projects, alwaysCheckPythonSideEffects, processFunctionsInParallel, alwaysCheckRecusion,
				useTestEntryPoints, alwaysFollowTypeHints, useSpeculativeAnalysis, false);
	}

	public static HybridizeFunctionRefactoringProcessor createHybridizeFunctionRefactoring(IProject[] projects,
			boolean alwaysCheckPythonSideEffects, boolean processFunctionsInParallel, boolean alwaysCheckRecusion,
			boolean useTestEntryPoints, boolean alwaysFollowTypeHints, boolean useSpeculativeAnalysis, boolean inferInputSignatures)
			throws ExecutionException, CoreException, IOException {
		Set<FunctionDefinition> functionDefinitions = getFunctionDefinitions(Arrays.asList(projects));
		return new HybridizeFunctionRefactoringProcessor(functionDefinitions, alwaysCheckPythonSideEffects, processFunctionsInParallel,
				alwaysCheckRecusion, useTestEntryPoints, alwaysFollowTypeHints, useSpeculativeAnalysis, inferInputSignatures);
	}

	public static Set<FunctionDefinition> getFunctionDefinitions(Iterable<?> iterable)
			throws ExecutionException, CoreException, IOException {
		Set<FunctionDefinition> ret = new HashSet<>();

		for (Object obj : iterable) {
			IResource resource = toResource(obj);

			if (resource != null)
				ret.addAll(getFunctionDefinitions(resource));
		}

		return ret;
	}

	/**
	 * Returns the {@link FunctionDefinition}s under the given resource, resolved without the UI navigator. A file yields itself if it is
	 * Python; a folder yields its Python files; a project yields the Python files under its (project-local) source folders. Each file is
	 * parsed directly, so this works headlessly (e.g., the evaluator) as well as in the IDE.
	 *
	 * @param resource The resource (project, folder, or file) to search.
	 * @return The {@link FunctionDefinition}s found under the resource.
	 */
	public static Set<FunctionDefinition> getFunctionDefinitions(IResource resource) throws ExecutionException, CoreException, IOException {
		Set<FunctionDefinition> ret = new HashSet<>();

		for (IFile file : getPythonSourceFiles(resource))
			ret.addAll(extractFunctionDefinitions(file));

		return ret;
	}

	/**
	 * Unwraps the given selection element to its underlying {@link IResource}. Accepts raw resources, PyDev navigator elements (which wrap
	 * a resource), and a {@link PythonNode} (whose containing file is used).
	 *
	 * @param obj The selection element.
	 * @return The underlying resource, or {@code null} if the element does not correspond to one.
	 */
	private static IResource toResource(Object obj) {
		if (obj instanceof IResource resource)
			return resource;

		Object element = obj instanceof PythonNode pythonNode ? pythonNode.getPythonFile() : obj;

		if (element instanceof IWrappedResource wrapped && wrapped.getActualObject() instanceof IResource resource)
			return resource;

		return null;
	}

	/**
	 * Returns the Python source files reachable from the given resource. For a project, only the (project-local) source folders are walked,
	 * mirroring the navigator, which resolves source folders from the same nature source-path set.
	 *
	 * @param resource The resource (project, folder, or file) to search.
	 * @return The Python source files.
	 */
	private static Set<IFile> getPythonSourceFiles(IResource resource) throws CoreException {
		Set<IFile> ret = new LinkedHashSet<>();

		if (resource instanceof IFile file) {
			if (isPythonFile(file))
				ret.add(file);
		} else if (resource instanceof IProject project && project.isOpen()) {
			PythonNature nature = PythonNature.getPythonNature(project);

			if (nature != null)
				for (IResource sourceFolder : nature.getPythonPathNature().getProjectSourcePathFolderSet())
					collectPythonFiles(sourceFolder, ret);
		} else if (resource instanceof IContainer container)
			collectPythonFiles(container, ret);

		return ret;
	}

	private static void collectPythonFiles(IResource resource, Set<IFile> out) throws CoreException {
		if (resource instanceof IFile file) {
			if (isPythonFile(file))
				out.add(file);
		} else if (resource instanceof IContainer container && container.isAccessible())
			for (IResource member : container.members())
				collectPythonFiles(member, out);
	}

	private static boolean isPythonFile(IFile file) {
		return "py".equalsIgnoreCase(file.getFileExtension());
	}

	/**
	 * Parses the given Python file directly (no navigator) and returns its {@link FunctionDefinition}s.
	 *
	 * @param pythonFile The Python file to parse.
	 * @return The {@link FunctionDefinition}s in the file.
	 */
	private static Set<FunctionDefinition> extractFunctionDefinitions(IFile pythonFile)
			throws ExecutionException, CoreException, IOException {
		Set<FunctionDefinition> ret = new HashSet<>();

		IDocument document = getDocument(pythonFile);
		File file = getFile(pythonFile);
		String moduleName = getModuleName(pythonFile);
		PythonNature nature = getPythonNature(pythonFile);

		FunctionExtractor functionExtractor = new FunctionExtractor();

		try {
			ParserInfo parserInfo = new ParserInfo(document, nature, moduleName, file);
			ParseOutput parseOutput = PyParser.reparseDocument(parserInfo);

			if (!(parseOutput.ast instanceof SimpleNode node))
				return ret;

			node.accept(functionExtractor);
		} catch (Exception e) {
			LOG.error("Failed to extract functions from: " + pythonFile + ".", e);
			throw new ExecutionException("Failed to extract functions from: " + pythonFile + ".", e);
		}

		for (FunctionDef def : functionExtractor.getDefinitions())
			ret.add(new FunctionDefinition(def, moduleName, file, pythonFile, document, nature));

		return ret;
	}

	public static String getModuleName(IFile file) {
		String fileName = file.getName();
		int separatorPos = fileName.indexOf('.');
		return separatorPos < 0 ? fileName : fileName.substring(0, separatorPos);
	}

	public static PythonNature getPythonNature(IFile file) {
		return PythonNature.getPythonNature(file.getProject());
	}

	public static IDocument getDocument(IFile file) throws CoreException, IOException {
		try (InputStream contentStream = file.getContents()) {
			byte[] bytes = contentStream.readAllBytes();
			String content = new String(bytes, UTF_8);
			return new Document(content);
		}
	}

	public static File getFile(IFile file) {
		URI uri = file.getRawLocationURI();
		return new File(uri);
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
		IFile actualFile = getActualFile(pythonNode);
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
			FunctionDefinition function = new FunctionDefinition(def, moduleName, file, actualFile, document, nature);
			ret.add(function);
		}

		return ret;
	}

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

	public static IPath getPath(IProject project) {
		IPath path = project.getFullPath();

		if (!path.toFile().exists())
			path = project.getLocation();

		return path;
	}

	public static List<File> getPythonPath(IProject project) throws CoreException {
		IProjectNature projectNature = project.getNature(PythonNature.PYTHON_NATURE_ID);

		if (projectNature == null)
			throw new IllegalArgumentException("Can only work with PyDev projects.");

		IPythonNature pythonNature = (IPythonNature) projectNature;
		String[] pythonPath = pythonNature.getPythonPathNature().getOnlyProjectPythonPathStr(false).split("\\|");
		return Arrays.stream(pythonPath).map(File::new).collect(toList());
	}
}
