package edu.cuny.hunter.hybridize.ui.handlers;

import static org.eclipse.core.runtime.Platform.getLog;
import static org.eclipse.ui.handlers.HandlerUtil.getActiveShellChecked;

import java.util.Collection;
import java.util.Collections;
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
	
	private static final PythonModelProvider provider = new PythonModelProvider();

	/**
	 * Gather all functions from the user's selection.
	 */
	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		Set<FunctionDef> functions = new HashSet<>();
		ISelection currentSelection = HandlerUtil.getCurrentSelectionChecked(event);

		if (currentSelection instanceof IStructuredSelection) {
			List<?> list = ((IStructuredSelection) currentSelection).toList();

			if (list != null)
				for (Object obj : list) {
					Set<PythonNode> nodeSet = getPythonNodes(obj);

					for (PythonNode node : nodeSet)
						functions.addAll(process(node));
				}
		}

		LOG.info("Found " + functions.size() + " function definitions.");

		Set<FunctionDef> availableFunctions = functions.stream()
				.filter(RefactoringAvailabilityTester::isHybridizationAvailable).collect(Collectors.toSet());
		LOG.info("Found " + availableFunctions.size() + " available functions.");

		Shell shell = getActiveShellChecked(event);

		HybridizeFunctionRefactoringWizard
				.startRefactoring(availableFunctions.toArray(new FunctionDef[availableFunctions.size()]), shell);

		return null;
	}

	private static Set<PythonNode> getPythonNodes(Object obj) {
		Set<PythonNode> ret = new HashSet<>();

		if (obj instanceof PythonNode) {
			PythonNode pythonNode = (PythonNode) obj;
			ret.add(pythonNode);
		} else {
			Object[] children = provider.getChildren(obj);
			for (Object child : children) {
				ret.addAll(getPythonNodes(child));
			}
		}
		return ret;
	}

	private static Set<FunctionDef> process(PythonNode pythonNode) throws ExecutionException {
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

		return functionExtractor.getDefinitions();
	}
}
