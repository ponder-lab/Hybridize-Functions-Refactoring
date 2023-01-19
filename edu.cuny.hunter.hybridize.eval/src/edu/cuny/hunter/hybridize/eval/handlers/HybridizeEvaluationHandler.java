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
import org.eclipse.core.runtime.IPath;
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
		CSVPrinter functionsPrinter = null;
		CSVPrinter decoratorsPrinter = null;

		HybridizeFunctionRefactoringProcessor processor = null;

		NullProgressMonitor monitor = new NullProgressMonitor();

		try {
			ISelection currentSelection = HandlerUtil.getCurrentSelectionChecked(event);

			List<String> resultsHeader = new ArrayList<>(Arrays.asList("project", "# functions", "# hybrid functions",
					"# decorators with hybrid parameters", "# func param", "# input_signature param", "# autograph param",
					"# jit_compile param", "# reduce_retracing param", "# experimental_implements param",
					"# experimental_autograph_options param", "# experimental_follow_type_hints param"));

			List<String> functionsHeader = new ArrayList<>(Arrays.asList("project", "file/path", "function id", "is_method",
					"is_embedded_function", "is_hybrid", "has_hybrid_params?"));

			List<String> decoratorsHeader = new ArrayList<>(Arrays.asList("project", "file/path", "function id", "func", "input_signature",
					"autograph", "jit_compile", "reduce_retracing", "experimental_implements", "experimental_autograph_options",
					"experimental_follow_type_hints"));

			resultsPrinter = createCSVPrinter("results.csv", resultsHeader.toArray(new String[resultsHeader.size()]));

			functionsPrinter = createCSVPrinter("functions.csv", functionsHeader.toArray(new String[functionsHeader.size()]));

			decoratorsPrinter = createCSVPrinter("decorators.csv", decoratorsHeader.toArray(new String[decoratorsHeader.size()]));

			if (currentSelection instanceof IStructuredSelection) {
				List<?> list = ((IStructuredSelection) currentSelection).toList();

				if (list != null)
					for (Object obj : list) {

						// get projectName
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
						int countFunctionsPerProject = 0;
						int countHybridPerProject = 0;

						// Counters for tf.function parameters
						int countFuncParamsPerProject = 0;
						int countInputSignatureParamsPerProject = 0;
						int countAutoGraphParamsPerProject = 0;
						int countJitCompileParamsPerProject = 0;
						int countReduceRetrainingParamsPerProject = 0;
						int countExpImplementsParamsPerProject = 0;
						int countExpAutographOptParamsPerProject = 0;
						int countExpTypeHintParamsPerProject = 0;
						int countDecoratorsWithHybridParams = 0;

						// Getting information about the functions
						for (Map.Entry<Set<FunctionDefinition>, PythonNode> map : functionsToNodes.entrySet()) {

							// Getting functions
							processor = new HybridizeFunctionRefactoringProcessor(map.getKey(), monitor);

							Set<Function> functionsPerNode = processor.getFunctions();

							// Iterate over the functions per Python Node
							for (Function func : functionsPerNode) {

								// Print each project for each function
								functionsPrinter.print(projectName);

								// Getting relative path
								IPath relativePath = map.getValue().pythonFile.getActualObject().getProjectRelativePath();

								// Printing relative path
								functionsPrinter.print(relativePath);

								// Getting function identifier
								String functionId = func.getIdentifer();

								// Printing function identifier
								functionsPrinter.print(functionId);

								functionsPrinter.print(func.getIsMethod());

								functionsPrinter.print(func.getIsEmbedded());

								// Getting tf.function parameter information
								// This already checks if the func is hybrid
								Function.HybridizationParameters args = func.getHybridizationParameters();
								boolean existenceOfTfParam = false;

								if (args != null) {

									// If args not null, then it means that it is hybrid
									functionsPrinter.print("True");

									// Counting the number of hybrid projects per project
									++countHybridPerProject;

									// Print each project for each function
									decoratorsPrinter.print(projectName);

									// Print each project for each function
									decoratorsPrinter.print(relativePath);

									// Printing function identifier
									decoratorsPrinter.print(functionId);

									if (args.hasFuncParam()) {
										decoratorsPrinter.print("True");
										existenceOfTfParam = true;
										++countFuncParamsPerProject;
									} else
										decoratorsPrinter.print("False");
									if (args.hasInputSignatureParam()) {
										decoratorsPrinter.print("True");
										existenceOfTfParam = true;
										++countInputSignatureParamsPerProject;
									} else
										decoratorsPrinter.print("False");
									if (args.hasAutoGraphParam()) {
										decoratorsPrinter.print("True");
										existenceOfTfParam = true;
										++countAutoGraphParamsPerProject;
									} else
										decoratorsPrinter.print("False");
									if (args.hasJitCompileParam()) {
										decoratorsPrinter.print("True");
										existenceOfTfParam = true;
										++countJitCompileParamsPerProject;
									} else
										decoratorsPrinter.print("False");
									if (args.hasReduceRetracingParam()) {
										decoratorsPrinter.print("True");
										existenceOfTfParam = true;
										++countReduceRetrainingParamsPerProject;
									} else
										decoratorsPrinter.print("False");
									if (args.hasExperimentalImplementsParam()) {
										decoratorsPrinter.print("True");
										existenceOfTfParam = true;
										++countExpImplementsParamsPerProject;
									} else
										decoratorsPrinter.print("False");
									if (args.hasExperimentalAutographOptParam()) {
										decoratorsPrinter.print("True");
										existenceOfTfParam = true;
										++countExpAutographOptParamsPerProject;
									} else
										decoratorsPrinter.print("False");
									if (args.hasExperimentalFollowTypeHintsParam()) {
										decoratorsPrinter.print("True");
										existenceOfTfParam = true;
										++countExpTypeHintParamsPerProject;
									} else
										decoratorsPrinter.print("False");
									decoratorsPrinter.println();
								} else
									functionsPrinter.print("False");

								if (existenceOfTfParam)
									++countDecoratorsWithHybridParams;

								functionsPrinter.print(existenceOfTfParam);

								functionsPrinter.println();
								++countFunctionsPerProject;
							}

						}

						// Printing number of functions per project
						resultsPrinter.print(countFunctionsPerProject);

						// Printing number of hybrid functions per project
						resultsPrinter.print(countHybridPerProject);

						// Printing number functions with hybrid parameters
						resultsPrinter.print(countDecoratorsWithHybridParams);

						// Printing number of tf.function parameters per project
						resultsPrinter.print(countFuncParamsPerProject);
						resultsPrinter.print(countInputSignatureParamsPerProject);
						resultsPrinter.print(countAutoGraphParamsPerProject);
						resultsPrinter.print(countJitCompileParamsPerProject);
						resultsPrinter.print(countReduceRetrainingParamsPerProject);
						resultsPrinter.print(countExpImplementsParamsPerProject);
						resultsPrinter.print(countExpAutographOptParamsPerProject);
						resultsPrinter.print(countExpTypeHintParamsPerProject);

						// Finish row of results.csv
						resultsPrinter.println();

					}
				// Close printer
				if (resultsPrinter != null) {
					resultsPrinter.close();
					functionsPrinter.close();
					decoratorsPrinter.close();
				}
			}
		} catch (IOException | TooManyMatchesException | BadLocationException e) {
			throw new ExecutionException("Error working with CSV file.", e);
		}

		return null;
	}
}