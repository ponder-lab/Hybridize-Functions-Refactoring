package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;

import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.python.pydev.ast.codecompletion.revisited.visitors.Definition;
import org.python.pydev.ast.item_pointer.ItemPointer;
import org.python.pydev.ast.refactoring.AbstractPyRefactoring;
import org.python.pydev.ast.refactoring.IPyRefactoring;
import org.python.pydev.ast.refactoring.RefactoringRequest;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IModule;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;

public class Util {

	private static final ILog LOG = getLog(Util.class);

	/**
	 * Get the name of the module defining the entity described in the given {@link PySelection}.
	 *
	 * @param selection The {@link PySelection} in question.
	 * @param containingModName The name of the module containing the {@link PySelection}.
	 * @param containingFile The {@link File} containing the module.
	 * @param nature The {@link IPythonNature} to use.
	 * @param monitor The IProgressMonitor to use.
	 * @return The name of the module defining the given {@link PySelection}.
	 * @throws TooManyMatchesException On ambiguous definitions found.
	 * @throws BadLocationException On parsing error.
	 */
	public static String getDeclaringModuleName(PySelection selection, String containingModName, File containingFile, IPythonNature nature,
			IProgressMonitor monitor) throws TooManyMatchesException, BadLocationException {
		monitor.beginTask("Getting declaring module name.", 1);
		
		RefactoringRequest request = new RefactoringRequest(containingFile, selection, nature);

		request.acceptTypeshed = true;
		request.moduleName = containingModName;
		request.pushMonitor(monitor);

		IPyRefactoring pyRefactoring = AbstractPyRefactoring.getPyRefactoring();
		ItemPointer[] pointers = pyRefactoring.findDefinition(request);
		LOG.info("Found " + pointers.length + "\"pointer(s).\"");

		if (pointers.length == 0)
			throw new IllegalArgumentException("Can't find declaring module for " + selection + ".");

		if (pointers.length > 1)
			throw new TooManyMatchesException("Ambigious definitions found for " + selection + ".", pointers.length);

		ItemPointer itemPointer = pointers[0];
		Definition definition = itemPointer.definition;

		LOG.info("Found definition: " + definition + ".");

		IModule module = definition.module;

		LOG.info(String.format("Found module: %s.", module));

		monitor.done();
		return module.getName();
	}

	/**
	 * Get the FQN of the given decorator.
	 *
	 * @param decorator The {@link decoratorsType} in question.
	 * @param containingModName The name of the module where the decorator is used.
	 * @param containingFile The {@link File} where the containingModName is defined.
	 * @param containingSelection The {@link PySelection} containing the decorator.
	 * @param nature The {@link IPythonNature} to use.
	 * @param monitor The IProgressMonitor to use.
	 * @return The FQN of the given {@link decoratorsType}.
	 * @throws TooManyMatchesException If the definition of the decorator is ambiguous.
	 * @throws BadLocationException When the containing entities cannot be parsed.
	 */
	public static String getFullyQualifiedName(decoratorsType decorator, String containingModName, File containingFile,
			PySelection containingSelection, IPythonNature nature, IProgressMonitor monitor)
			throws TooManyMatchesException, BadLocationException {
		monitor.beginTask("Getting decorator FQN.", 3);

		monitor.subTask("Getting declaring module name.");
		String declaringModuleName = getDeclaringModuleName(containingSelection, containingModName, containingFile, nature, monitor);
		LOG.info(String.format("Found declaring module: %s.", declaringModuleName));
		monitor.worked(1);

		exprType decoratorFunction = decorator.func;
		String decoratorFullRepresentationString = NodeUtils.getRepresentationString(decoratorFunction);
		LOG.info(String.format("The \"full representation\" of %s is %s.", decoratorFunction, decoratorFullRepresentationString));
		monitor.worked(1);

		String fqn = declaringModuleName + "." + decoratorFullRepresentationString;
		LOG.info(String.format("FQN is: %s.", fqn));
		monitor.worked(1);

		monitor.done();
		return fqn;
	}

	private Util() {
	}
}
