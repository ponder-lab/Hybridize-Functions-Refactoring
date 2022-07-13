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
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.scope.ASTEntryWithChildren;
import org.python.pydev.plugin.PydevPlugin;

import edu.cuny.hunter.hybridize.core.analysis.FunctionDefintionExtractor;
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
						PythonProjectSourceFolder folder = (PythonProjectSourceFolder) obj;
						System.out.println(folder);
						Map<IResource, IWrappedResource> children = folder.children;
						System.out.println(children);
						// TODO: Drill down and extract function definitions.
					} else if (obj instanceof PythonNode) {
						PythonNode pythonNode = (PythonNode) obj;
						ParsedItem entry = pythonNode.entry;
						ASTEntryWithChildren ast = entry.getAstThis();
						SimpleNode simpleNode = ast.node;

						// extract function definitions.
						FunctionDefintionExtractor functionExtractor = new FunctionDefintionExtractor();
						try {
							simpleNode.accept(functionExtractor);
						} catch (Exception e) {
							Log.log(e);
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

						// ---------------------------------------------------------------------------------

						if (simpleNode instanceof FunctionDef) {
							FunctionDef function = (FunctionDef) simpleNode;
							System.out.println(function);

							decoratorsType[] decs = function.decs;

							if (decs != null)
								for (decoratorsType dt : decs)
									System.out.println(dt);

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
					} else if (obj instanceof PythonFolder) {
						// Could be something like a "package."
						System.out.println("Package?");
						PythonFolder folder = (PythonFolder) obj;
						System.out.println(folder);
						// TODO: Drill down here? Doesn't seem to be any constituent elements except for
						// going up to the parent.
					} else if (obj instanceof PythonFile) {
						PythonFile file = (PythonFile) obj;
						System.out.println(file);
						// TODO: Drill down and extract function definitions.
						// NOTE: Do not re-parse the elements if it all possible.
					}
				}
		}

		return null;
	}
}
