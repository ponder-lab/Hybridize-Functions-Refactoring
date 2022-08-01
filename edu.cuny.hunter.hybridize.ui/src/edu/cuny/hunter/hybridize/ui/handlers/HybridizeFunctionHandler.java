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
						System.out.println("Python Project Source Folder");
						PythonProjectSourceFolder folder = (PythonProjectSourceFolder) obj;
						Map<IResource, IWrappedResource> children = folder.children;
						for (Map.Entry<IResource, IWrappedResource> child: children.entrySet()) {
							if(child.getValue() instanceof PythonFile) {
								PythonModelProvider provider = new PythonModelProvider();
								Object childValue = child.getValue();
								
								Object[] childrenFile = provider.getChildren(childValue);
								
								for(Object childFile: childrenFile) {
									if(childFile instanceof PythonNode) {
										PythonNode pythonNode = (PythonNode) childFile;
										ParsedItem entry = pythonNode.entry;
										ASTEntryWithChildren ast = entry.getAstThis();
										SimpleNode simpleNode = ast.node;
										
										//children can be imports, etc. We are only interested in the classes and methods
										if(simpleNode instanceof FunctionDef || simpleNode instanceof ClassDef) {
											extractFunctionDefinitions(simpleNode, event);
										}
									}
								}
							}
							//Not necessary because the children given goes to the files for the folder already, so it would be duplicate
							if(child.getValue() instanceof PythonFolder) {
								System.out.println("Folder");
							}
						}
					} else if (obj instanceof PythonNode) {
						PythonNode pythonNode = (PythonNode) obj;
						ParsedItem entry = pythonNode.entry;
						ASTEntryWithChildren ast = entry.getAstThis();
						SimpleNode simpleNode = ast.node;
						
						extractFunctionDefinitions(simpleNode, event);

					} else if (obj instanceof PythonFolder) {
						PythonFolder folder = (PythonFolder) obj;
						System.out.println(folder);
						PythonModelProvider provider = new PythonModelProvider();
						
						Object[] children = provider.getChildren(obj);
						
						for(Object child: children) {
							if(child instanceof PythonFile) {
								Object[] childrenFile = provider.getChildren(child);
								for(Object childFile: childrenFile) {
									if(childFile instanceof PythonNode) {
										PythonNode pythonNode = (PythonNode) childFile;
										ParsedItem entry = pythonNode.entry;
										ASTEntryWithChildren ast = entry.getAstThis();
										SimpleNode simpleNode = ast.node;
										
										//children can be imports, etc. We are only interested in the classes and methods
										if(simpleNode instanceof FunctionDef || simpleNode instanceof ClassDef) {
											extractFunctionDefinitions(simpleNode, event);
										}
									}
								}
							}
						}
					} else if (obj instanceof PythonFile) {		
						PythonModelProvider provider = new PythonModelProvider();
						
						Object[] children = provider.getChildren(obj);
						
						for(Object child: children) {
							if(child instanceof PythonNode) {
								PythonNode pythonNode = (PythonNode) child;
								ParsedItem entry = pythonNode.entry;
								ASTEntryWithChildren ast = entry.getAstThis();
								SimpleNode simpleNode = ast.node;
								
								//children can be imports, etc. We are only interested in the classes and methods
								if(simpleNode instanceof FunctionDef || simpleNode instanceof ClassDef) {
									extractFunctionDefinitions(simpleNode, event);
								}
							}
						}
						
					}
				}
		}

		return null;
	}
	
	public void extractFunctionDefinitions(SimpleNode simpleNode, ExecutionEvent event) throws ExecutionException {

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
	
	
}
