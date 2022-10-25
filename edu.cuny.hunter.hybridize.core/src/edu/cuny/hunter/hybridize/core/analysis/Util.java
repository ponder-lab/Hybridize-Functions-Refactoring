package edu.cuny.hunter.hybridize.core.analysis;

import java.io.File;
import java.util.List;

import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.python.pydev.ast.codecompletion.revisited.ModulesManager;
import org.python.pydev.ast.codecompletion.revisited.modules.AbstractModule;
import org.python.pydev.ast.codecompletion.revisited.modules.SourceModule;
import org.python.pydev.ast.codecompletion.revisited.visitors.Definition;
import org.python.pydev.ast.item_pointer.ItemPointer;
import org.python.pydev.ast.refactoring.AbstractPyRefactoring;
import org.python.pydev.ast.refactoring.IPyRefactoring;
import org.python.pydev.ast.refactoring.RefactoringRequest;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IModule;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.MisconfigurationException;
import org.python.pydev.core.ModulesKey;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.plugin.nature.PythonNature;

import com.python.pydev.analysis.additionalinfo.AbstractAdditionalDependencyInfo;
import com.python.pydev.analysis.additionalinfo.AdditionalProjectInterpreterInfo;
import com.python.pydev.analysis.refactoring.refactorer.Refactorer;
import com.python.pydev.refactoring.actions.PyGoToDefinition;

public class Util {

	private Util() {
	}

	public static String getDeclaringModuleName(decoratorsType decorator, String modName, File file,
			PySelection selection, IPythonNature nature, IProgressMonitor monitor)
			throws TooManyMatchesException, BadLocationException {
		// NOTE: __module__ gives us what we need. Either use dynamic analysis to get it or analyze imports?
		// Is there an import scope visitor? Module name getter?
		// Have a look at https://github.com/fabioz/Pydev/search?q=declared.
		// What module is thing declared in? __module__ is the name of the module the function was defined in,

		IPyRefactoring pyRefactoring = AbstractPyRefactoring.getPyRefactoring();

		RefactoringRequest request = new RefactoringRequest(file, selection, nature);

		request.acceptTypeshed = true;
		request.moduleName = modName;
		request.pushMonitor(monitor);

		// FIXME: I don't think this belongs here. We should have the nature set at this point.
		// NOTE: I think this already done anyway.
//		SimpleNode ast = request.getAST();
//		addModuleToNature(ast, modName, nature, file);

		ItemPointer[] pointers = pyRefactoring.findDefinition(request);

		if (pointers.length == 0)
			throw new IllegalArgumentException("Can't find declaring module for " + decorator + ".");
		else if (pointers.length > 1)
			throw new TooManyMatchesException("Ambigious definitions found for " + decorator + ".", pointers.length);

		ItemPointer itemPointer = pointers[0];
		Definition definition = itemPointer.definition;
		IModule module = definition.module;
		return module.getName();
	}

	public static String getFullyQualifiedName(decoratorsType decorator, String modName, File file,
			PySelection selection, IPythonNature nature, IProgressMonitor monitor)
			throws TooManyMatchesException, BadLocationException {
		String declaringModuleName = getDeclaringModuleName(decorator, modName, file, selection, nature, monitor);

		exprType decoratorFunction = decorator.func;
		String decoratorfullRepresentationString = NodeUtils.getRepresentationString(decoratorFunction);

		return declaringModuleName + "." + decoratorfullRepresentationString;
	}

	/**
	 * FIXME: This probably belongs in the test code.
	 * 
	 * @param ast the ast that defines the module
	 * @param modName the module name
	 * @param natureToAdd the nature where the module should be added
	 */
	private static void addModuleToNature(final SimpleNode ast, String modName, IPythonNature natureToAdd, File f) {
		// this is to add the info from the module that we just created...
		AbstractAdditionalDependencyInfo additionalInfo;
		try {
			additionalInfo = AdditionalProjectInterpreterInfo.getAdditionalInfoForProject(natureToAdd);
		} catch (MisconfigurationException e) {
			throw new RuntimeException(e);
		}
		additionalInfo.addAstInfo(ast, new ModulesKey(modName, f), false);
		ModulesManager modulesManager = (ModulesManager) natureToAdd.getAstManager().getModulesManager();
		SourceModule mod = (SourceModule) AbstractModule.createModule(ast, f, modName, natureToAdd);
		modulesManager.doAddSingleModule(new ModulesKey(modName, f), mod);
	}
}
