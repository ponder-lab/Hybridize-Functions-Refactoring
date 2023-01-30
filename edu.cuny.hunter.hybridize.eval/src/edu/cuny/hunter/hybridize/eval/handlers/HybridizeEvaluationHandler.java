package edu.cuny.hunter.hybridize.eval.handlers;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.navigator.PythonModelProvider;
import org.python.pydev.navigator.elements.PythonNode;
import org.python.pydev.navigator.elements.PythonProjectSourceFolder;
import org.python.pydev.outline.ParsedItem;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.visitors.scope.ASTEntryWithChildren;

import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.analysis.FunctionExtractor;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;
import edu.cuny.hunter.hybridize.ui.handlers.Util;

public class HybridizeEvaluationHandler extends AbstractHandler {

	private static final ILog LOG = getLog(HybridizeEvaluationHandler.class);

	private static final PythonModelProvider provider = new PythonModelProvider();

	private static CSVPrinter createCSVPrinter(String fileName, String[] header) throws IOException {
		return new CSVPrinter(new FileWriter(fileName, true), CSVFormat.EXCEL.withHeader(header));
	}

	private static String getProjectName(Object obj) {
		if (obj instanceof PythonProjectSourceFolder) {
			PythonProjectSourceFolder pyProject = (PythonProjectSourceFolder) obj;
			IResource actualObject = pyProject.getActualObject();
			IProject project = actualObject.getProject();

			return project.getName();

		}
		if (obj instanceof IProject) {
			IProject iProject = (IProject) obj;
			return iProject.getName();
		}
		return obj.toString();
	}

	private static Set<ClassDef> getClasses(Set<FunctionDefinition> functionDefinition) {
		Set<ClassDef> classesSet = new HashSet<>();

		for (FunctionDefinition funcDefinition : functionDefinition) {
			FunctionDef funcDef = funcDefinition.getFunctionDef();
			SimpleNode parentNode = funcDef.parent;

			while (parentNode instanceof ClassDef) {
				ClassDef classFunction = (ClassDef) parentNode;
				classesSet.add(classFunction);
				parentNode = parentNode.parent;
			}
		}

		return classesSet;
	}

	private static Set<PythonNode> getPythonNodes(Object obj) {
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

	private static Set<FunctionDefinition> process(PythonNode pythonNode) throws ExecutionException, CoreException, IOException {
		Set<FunctionDefinition> ret = new HashSet<>();

		String moduleName = Util.getModuleName(pythonNode);
		File file = Util.getFile(pythonNode);
		IDocument document = Util.getDocument(pythonNode);
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

	/**
	 * Gather all functions from the user's selection.
	 */
	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		CSVPrinter resultsPrinter = null;

		HybridizeFunctionRefactoringProcessor processor = null;

		NullProgressMonitor monitor = new NullProgressMonitor();

		try {
			ISelection currentSelection = HandlerUtil.getCurrentSelectionChecked(event);

			List<String> resultsHeader = new ArrayList<>(Arrays.asList("Subject", "SLOC", "classes", "functions",
					"hybrid functions (original)", "optimization available functions", "optimized functions",
					"convert eager Function to hybrid", "convert eager function to hybrid", "failed preconditions", "time (s)"));

			resultsPrinter = createCSVPrinter("results.csv", resultsHeader.toArray(new String[resultsHeader.size()]));

			if (currentSelection instanceof IStructuredSelection) {
				List<?> list = ((IStructuredSelection) currentSelection).toList();

				if (list != null)
					for (Object obj : list) {

						// get Subject Name
						String projectName = getProjectName(obj);

						// Project Name for results.csv
						resultsPrinter.print(projectName);

						Set<PythonNode> nodeSet = getPythonNodes(obj);

						// Map functions to PythoNode
						Map<Set<FunctionDefinition>, PythonNode> functionsToNodes = new HashMap<>();

						for (PythonNode node : nodeSet)
							try {
								functionsToNodes.put(process(node), node);
							} catch (ExecutionException | CoreException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}

						// Counters for aggregate information
						int countfunctionsPerNode = 0;
						int countHybridPerProject = 0;
						int slocTotal = 0;
						Set<ClassDef> classesTotal = new HashSet<>();

						// Getting information about the functions
						for (Map.Entry<Set<FunctionDefinition>, PythonNode> map : functionsToNodes.entrySet()) {

							// Getting functions
							processor = new HybridizeFunctionRefactoringProcessor(map.getKey(), monitor);

							classesTotal.addAll(getClasses(map.getKey()));

							Set<Function> functionsPerNode = processor.getFunctions();
							
							IDocument document = null;

							// Iterate over the functions per Python Node
							for (Function func : functionsPerNode) {
								document = func.getContainingDocument();
								if (func.isHybrid()) {
									countHybridPerProject++;
								}
								countfunctionsPerNode++;
							}
							if (document != null)
								slocTotal += document.getNumberOfLines();;
						}

						// Printing SLOC per project
						resultsPrinter.print(slocTotal);

						// Printing number of classes per project
						resultsPrinter.print(classesTotal.size());

						// Printing number of functions per project
						resultsPrinter.print(countfunctionsPerNode);

						// Printing number of hybrid functions per project
						resultsPrinter.print(countHybridPerProject);

						// Finish row of results.csv
						resultsPrinter.println();
					}
				// Close printer
				if (resultsPrinter != null) {
					resultsPrinter.close();
				}
			}
		} catch (IOException | TooManyMatchesException | BadLocationException e) {
			throw new ExecutionException("Error working with CSV file.", e);
		}

		return null;
	}
}