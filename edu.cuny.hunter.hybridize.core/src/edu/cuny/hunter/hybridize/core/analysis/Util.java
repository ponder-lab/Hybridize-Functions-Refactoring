package edu.cuny.hunter.hybridize.core.analysis;

import java.io.File;
import java.util.List;

import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.python.pydev.ast.item_pointer.ItemPointer;
import org.python.pydev.ast.refactoring.AbstractPyRefactoring;
import org.python.pydev.ast.refactoring.IPyRefactoring;
import org.python.pydev.ast.refactoring.RefactoringRequest;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.plugin.nature.PythonNature;

import com.python.pydev.refactoring.actions.PyGoToDefinition;

public class Util {

	private Util() {
	}

	public static String getDeclaringModuleName(decoratorsType decorator, File file, IPythonNature nature, IProgressMonitor monitor) {
		// NOTE: __module__ gives us what we need. Either use dynamic analysis to get it or analyze imports?
		// Is there an import scope visitor? Module name getter?
		// Have a look at https://github.com/fabioz/Pydev/search?q=declared.
		// What module is thing declared in? __module__ is the name of the module the function was defined in,
		
		IPyRefactoring pyRefactoring = AbstractPyRefactoring.getPyRefactoring();
	
		// TODO: Need a PySelection.
		RefactoringRequest request = new RefactoringRequest(file, null, nature);
		request.acceptTypeshed = true;
		request.pushMonitor(monitor);
		
		try {
			ItemPointer[] pointers = pyRefactoring.findDefinition(request);
			System.out.println(pointers);
		} catch (TooManyMatchesException | BadLocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}

	public static String getFullyQualifiedName(decoratorsType decorator, File file, IPythonNature nature, IProgressMonitor monitor) {
		String declaringModuleName = getDeclaringModuleName(decorator, file, nature, monitor);

		exprType decoratorFunction = decorator.func;
		String decoratorfullRepresentationString = NodeUtils.getRepresentationString(decoratorFunction);

		return declaringModuleName + "." + decoratorfullRepresentationString;
	}

}
