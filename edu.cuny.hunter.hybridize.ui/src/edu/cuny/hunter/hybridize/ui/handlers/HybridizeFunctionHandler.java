package edu.cuny.hunter.hybridize.ui.handlers;

import java.util.List;
import java.util.Map;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.ui.handlers.HandlerUtil;
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

public class HybridizeFunctionHandler extends AbstractHandler {

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
					} else if (obj instanceof PythonNode) {
						PythonNode node = (PythonNode) obj;
						System.out.println(node);
						ParsedItem entry = node.entry;
						ASTEntryWithChildren ast = entry.getAstThis();
						System.out.println(ast);
						SimpleNode node2 = ast.node;
						System.out.println(node2);

						if (node2 instanceof FunctionDef) {
							FunctionDef function = (FunctionDef) node2;
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
						// TODO: Drll down here? Doesn't seem to be any constituent elements except for
						// going up to the parent.
					} else if (obj instanceof PythonFile) {
						PythonFile file = (PythonFile) obj;
						System.out.println(file);
						// TODO: Drill down file.
					}
				}

		}

		return null;
	}

}
