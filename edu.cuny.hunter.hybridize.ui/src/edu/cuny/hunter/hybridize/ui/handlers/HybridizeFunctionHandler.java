package edu.cuny.hunter.hybridize.ui.handlers;

import static org.eclipse.ui.handlers.HandlerUtil.getActiveShellChecked;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.eclipse.core.runtime.Platform.getLog;
import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.runtime.ILog;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.core.log.Log;
import org.python.pydev.navigator.elements.IWrappedResource;
import org.python.pydev.navigator.elements.PythonFile;
import org.python.pydev.navigator.elements.PythonFolder;
import org.python.pydev.navigator.elements.PythonNode;
import org.python.pydev.navigator.elements.PythonProjectSourceFolder;
import org.python.pydev.outline.ParsedItem;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.scope.ASTEntryWithChildren;
import org.python.pydev.navigator.PythonModelProvider;
import org.python.pydev.plugin.PydevPlugin;

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
						processPythonProjectSourceFolder(obj, event, provider, functions);
					} else if (obj instanceof PythonNode) {
						// Drill down and extract function definitions.
						processPythonNode(obj, event, functions);

					} else if (obj instanceof PythonFolder) {
						// Drill down and extract function definitions.
						processPythonFolder(obj, event, provider, functions);

					} else if (obj instanceof PythonFile) {
						// Drill down and extract function definitions.
						processPythonFile(obj, event, provider, functions);
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

	private void processFunctionDefinitions(SimpleNode simpleNode, ExecutionEvent event, Set<FunctionDef> functions)
			throws ExecutionException {

		// extract function definitions.
		FunctionExtractor functionExtractor = new FunctionExtractor();
		try {
			simpleNode.accept(functionExtractor);
		} catch (Exception e) {
			Log.log(e); // TODO: Use our own logger.
			throw new ExecutionException("Failed to start refactoring.", e);
		}

		functions.addAll(functionExtractor.getDefinitions());

	}

	private void processPythonNode(Object obj, ExecutionEvent event, Set<FunctionDef> functions)
			throws ExecutionException {
		PythonNode pythonNode = (PythonNode) obj;
		ParsedItem entry = pythonNode.entry;
		ASTEntryWithChildren ast = entry.getAstThis();
		SimpleNode simpleNode = ast.node;

		processFunctionDefinitions(simpleNode, event, functions);

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

	private void processPythonFile(Object obj, ExecutionEvent event, PythonModelProvider provider,
			Set<FunctionDef> functions) throws ExecutionException {

		Object[] children = provider.getChildren(obj);

		for (Object child : children) {
			if (child instanceof PythonNode) {
				PythonNode pythonNode = (PythonNode) child;
				ParsedItem entry = pythonNode.entry;
				ASTEntryWithChildren ast = entry.getAstThis();
				SimpleNode simpleNode = ast.node;

				if (simpleNode instanceof FunctionDef || simpleNode instanceof ClassDef)
					processFunctionDefinitions(simpleNode, event, functions);
			}
		}
	}

	private void processPythonFolder(Object obj, ExecutionEvent event, PythonModelProvider provider,
			Set<FunctionDef> functions) throws ExecutionException {

		Object[] children = provider.getChildren(obj);

		for (Object child : children) {
			if (child instanceof PythonFile)
				processPythonFile(child, event, provider, functions);
			if (child instanceof PythonFolder)
				processPythonFolder(child, event, provider, functions);
		}
	}

	private void processPythonProjectSourceFolder(Object obj, ExecutionEvent event, PythonModelProvider provider,
			Set<FunctionDef> functions) throws ExecutionException {
		Map<IResource, IWrappedResource> children = ((PythonProjectSourceFolder) obj).children;

		for (IWrappedResource child : children.values()) {

			if (child instanceof PythonFile) {
				Object childValue = child;
				processPythonFile(childValue, event, provider, functions);
			}
		}
	}
}