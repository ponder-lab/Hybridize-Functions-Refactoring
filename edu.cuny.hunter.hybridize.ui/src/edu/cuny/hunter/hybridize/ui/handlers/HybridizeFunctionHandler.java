package edu.cuny.hunter.hybridize.ui.handlers;

import static org.eclipse.core.runtime.Platform.getLog;
import static org.eclipse.ui.handlers.HandlerUtil.getActiveShellChecked;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.ILog;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.navigator.PythonModelProvider;
import org.python.pydev.navigator.elements.IWrappedResource;
import org.python.pydev.navigator.elements.PythonFile;
import org.python.pydev.navigator.elements.PythonFolder;
import org.python.pydev.navigator.elements.PythonNode;
import org.python.pydev.navigator.elements.PythonProjectSourceFolder;
import org.python.pydev.outline.ParsedItem;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.scope.ASTEntryWithChildren;

import edu.cuny.hunter.hybridize.core.analysis.FunctionExtractor;
import edu.cuny.hunter.hybridize.core.utils.RefactoringAvailabilityTester;
import edu.cuny.hunter.hybridize.ui.wizards.HybridizeFunctionRefactoringWizard;

public class HybridizeFunctionHandler extends AbstractHandler {

	private static final ILog LOG = getLog(HybridizeFunctionHandler.class);

	/**
	 * Gather all functions from the user's selection.
	 */
	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		ISelection currentSelection = HandlerUtil.getCurrentSelectionChecked(event);
		PythonModelProvider provider = new PythonModelProvider();
		Set<FunctionDef> functions = new HashSet<FunctionDef>();
		if (currentSelection instanceof IStructuredSelection) {
			List<?> list = ((IStructuredSelection) currentSelection).toList();

			if (list != null)
				for (Object obj : list) {
					if (obj instanceof PythonProjectSourceFolder) {
						// Drill down and extract function definitions.
						Map<IResource, IWrappedResource> projectChildren = ((PythonProjectSourceFolder) obj).children;
						functions.addAll(process(projectChildren, provider));
					} else if (obj instanceof PythonNode) {
						// Drill down and extract function definitions.
						PythonNode pythonNode = (PythonNode) obj;
						functions.addAll(process(pythonNode));
					} else if (obj instanceof PythonFolder) {
						// Drill down and extract function definitions.
						functions.addAll(process(obj, provider));
					} else if (obj instanceof PythonFile) {
						// Drill down and extract function definitions.
						functions.addAll(process(obj, provider));
					}
				}
		}

		// Refactoring on found functions

		LOG.info("Found " + functions.size() + " function definitions.");

		Set<FunctionDef> availableFunctions = functions.stream()
				.filter(RefactoringAvailabilityTester::isHybridizationAvailable).collect(Collectors.toSet());
		LOG.info("Found " + availableFunctions.size() + " available functions.");

		Shell shell = getActiveShellChecked(event);

		HybridizeFunctionRefactoringWizard
				.startRefactoring(availableFunctions.toArray(new FunctionDef[availableFunctions.size()]), shell);

		return null;
	}

	// Maintaining from previous code versions
	private void printStatements(SimpleNode simpleNode) {

		// ---------------------------------------------------------------------------------

		if (simpleNode instanceof FunctionDef) {
			FunctionDef function = (FunctionDef) simpleNode;
			System.out.println(function);

			argumentsType args = function.args;
			System.out.println(args);
			exprType[] annotation = args.annotation;

			for (exprType annot : annotation)
				if (annot != null)
					System.out.println(annot);

			exprType[] args2 = args.args;

			if (args2 != null)
				for (exprType argType : args2)
					System.out.println(argType);
		}

		// ---------------------------------------------------------------------------------
	}

	/**
	 * Process a Python Project Source Folder
	 * Params: Map of the Project Source
	 * Folder Children and the Python Model Provider that enables us to obtain
	 * the children if it is a Python File
	 * Return the function definitions
	 */
	private Set<FunctionDef> process(Map<IResource, IWrappedResource> projectChildren, PythonModelProvider provider)
			throws ExecutionException {
		
		Set<FunctionDef> functions = new HashSet<FunctionDef>();

		for (IWrappedResource child : projectChildren.values())
			// We receive all the children. (e.g. project/folder and project/folder/file)
			// Only interested on the files because we don't want to traverse redundant information
			if (child instanceof PythonFile) {
				Object file = child;
				functions.addAll(process(file, provider));
			}
		return functions;
	}

	/**
	 * Process a Folder or a File
	 * Params: Object that represents a folder or a
	 * file, we need it as type Object because the Python Model Provider
	 * getChildren() takes an Object which enables us to obtain the children for
	 * the folder and file
	 * Return the function definitions
	 */
	private Set<FunctionDef> process(Object folderOrFile, PythonModelProvider provider) throws ExecutionException {

		Object[] children = provider.getChildren(folderOrFile);
		Set<FunctionDef> functions = new HashSet<FunctionDef>();
		

		for (Object child : children) {
			// Object received was a File or Folder, its children could
			// be a File or Folder. We need to process again to obtain the nodes
			if ((child instanceof PythonFile) || (child instanceof PythonFolder))
				functions.addAll(process(child, provider));
			// Object received was a File, its children could be a Python Node
			if (child instanceof PythonNode) {
				PythonNode pythonNode = (PythonNode) child;
				functions.addAll(process(pythonNode));
			}
		}
		return functions;
	}

	/**
	 * Process a Python Node
	 * Params: PythonNode
	 * Return the function definitions
	 *
	 * @throws ExecutionException
	 */
	private Set<FunctionDef> process(PythonNode node) throws ExecutionException {
		ParsedItem entry = node.entry;
		ASTEntryWithChildren ast = entry.getAstThis();
		SimpleNode simpleNode = ast.node;

		printStatements(simpleNode);

		return process(simpleNode);

	}

	/**
	 * Process a Simple Node
	 * Params: SimpleNode
	 * Return the function definitions
	 */
	private Set<FunctionDef> process(SimpleNode simpleNode) throws ExecutionException {

		// extract function definitions.
		FunctionExtractor functionExtractor = new FunctionExtractor();
		try {
			simpleNode.accept(functionExtractor);
		} catch (Exception e) {
			LOG.error("Failed to start refactoring.", e);
			throw new ExecutionException("Failed to start refactoring.", e);
		}

		return functionExtractor.getDefinitions();

	}
}