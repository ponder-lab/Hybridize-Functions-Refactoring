package edu.cuny.hunter.hybridize.ui.handlers;

import static org.eclipse.ui.handlers.HandlerUtil.getActiveShellChecked;

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

	// FIXME: Use our own logger.
	private static final ILog LOG = PydevPlugin.getDefault().getLog();

	/**
	 * Gather all functions from the user's selection.
	 */
	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		ISelection currentSelection = HandlerUtil.getCurrentSelectionChecked(event);

		if (currentSelection instanceof IStructuredSelection) {
			List<?> list = ((IStructuredSelection) currentSelection).toList();

			if (list != null)
				for (Object obj : list) {
					if (obj instanceof PythonProjectSourceFolder) {
						processPythonProjectSourceFolder(obj, event);
						
					} else if (obj instanceof PythonNode) {
						processPythonNode(obj, event);

					} else if (obj instanceof PythonFolder) {
						processPythonFolder(obj, event);
						
					} else if (obj instanceof PythonFile) {		
						processPythonFile(obj, event);
					}
				}
		}

		return null;
	}
	
	public void processFunctionDefinitions(SimpleNode simpleNode, ExecutionEvent event) throws ExecutionException {

		// extract function definitions.
		FunctionExtractor functionExtractor = new FunctionExtractor();
		try {
			simpleNode.accept(functionExtractor);
		} catch (Exception e) {
			Log.log(e); // TODO: Use our own logger.
			throw new ExecutionException("Failed to start refactoring.", e);
		}

		Set<FunctionDef> functions = functionExtractor.getDefinitions();
		LOG.info("Found " + functions.size() + " function definitions.");

		Set<FunctionDef> availableFunctions = functions.stream()
				.filter(RefactoringAvailabilityTester::isHybridizationAvailable)
				.collect(Collectors.toSet());
		LOG.info("Found " + availableFunctions.size() + " available functions.");

		Shell shell = getActiveShellChecked(event);

		HybridizeFunctionRefactoringWizard.startRefactoring(
				availableFunctions.toArray(new FunctionDef[availableFunctions.size()]), shell);
		
	}
	
	public void processPythonNode(Object obj, ExecutionEvent event) throws ExecutionException {
		PythonNode pythonNode = (PythonNode) obj;
		ParsedItem entry = pythonNode.entry;
		ASTEntryWithChildren ast = entry.getAstThis();
		SimpleNode simpleNode = ast.node;
		
		processFunctionDefinitions(simpleNode, event);
	}
	
	public void processPythonFile(Object obj, ExecutionEvent event) throws ExecutionException {
		PythonModelProvider provider = new PythonModelProvider();
		
		Object[] children = provider.getChildren(obj);
		
		for (Object child: children) {
			if (child instanceof PythonNode) {
				PythonNode pythonNode = (PythonNode) child;
				ParsedItem entry = pythonNode.entry;
				ASTEntryWithChildren ast = entry.getAstThis();
				SimpleNode simpleNode = ast.node;

				if(simpleNode instanceof FunctionDef || simpleNode instanceof ClassDef) 
					processFunctionDefinitions(simpleNode, event);
			}
		}
	}
	
	public void processPythonFolder(Object obj, ExecutionEvent event) throws ExecutionException {
		PythonModelProvider provider = new PythonModelProvider();
		
		Object[] children = provider.getChildren(obj);
		
		for (Object child: children) {
			if (child instanceof PythonFile)
				processPythonFile(child, event);
			if (child instanceof PythonFolder)
				processPythonFolder(child, event);
		}
	}
	
	public void processPythonProjectSourceFolder(Object obj, ExecutionEvent event) throws ExecutionException{
		PythonProjectSourceFolder folder = (PythonProjectSourceFolder) obj;
		Map<IResource, IWrappedResource> children = folder.children;
		
		for (Map.Entry<IResource, IWrappedResource> child: children.entrySet()) {
			
			if(child.getValue() instanceof PythonFile) {
				Object childValue = child.getValue();
				processPythonFile(childValue, event);
			}
			
			// Not necessary because the children given goes to the files for the folder already, so it would be duplicate
			if (child.getValue() instanceof PythonFolder)
				System.out.println("Folder");
		}
	}
}
