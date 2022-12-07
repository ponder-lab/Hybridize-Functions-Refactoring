package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;

import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.python.pydev.ast.codecompletion.revisited.visitors.Definition;
import org.python.pydev.ast.item_pointer.ItemPointer;
import org.python.pydev.ast.refactoring.AbstractPyRefactoring;
import org.python.pydev.ast.refactoring.IPyRefactoring;
import org.python.pydev.ast.refactoring.RefactoringRequest;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IModule;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.shared_core.string.CoreTextSelection;

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
		LOG.info("Found " + pointers.length + " \"pointer(s).\"");

		if (pointers.length == 0)
			throw new IllegalArgumentException(
					"Can't find declaring module for " + selection.getLineWithoutCommentsOrLiterals().trim() + ".");

		if (pointers.length > 1)
			throw new TooManyMatchesException(
					"Ambigious definitions found for " + selection.getLineWithoutCommentsOrLiterals().trim() + ".", pointers.length);

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

		exprType decoratorFunction = decorator.func;
		String fqn = getFullyQualifiedName(decoratorFunction, containingModName, containingFile, containingSelection, nature, monitor);

		monitor.done();
		return fqn;
	}

	public static String getFullyQualifiedName(SimpleNode node, String containingModName, File containingFile,
			PySelection containingSelection, IPythonNature nature, IProgressMonitor monitor) throws BadLocationException {
		monitor.subTask("Getting declaring module name.");

		String declaringModuleName = getDeclaringModuleName(containingSelection, containingModName, containingFile, nature, monitor);
		LOG.info(String.format("Found declaring module: %s.", declaringModuleName));
		monitor.worked(1);

		String representationString = NodeUtils.getRepresentationString(node);
		LOG.info(String.format("\"Representation\" of %s: %s.", node, representationString));
		monitor.worked(1);

		String fqn = declaringModuleName + "." + representationString;
		LOG.info(String.format("FQN is: %s.", fqn));

		monitor.worked(1);
		return fqn;
	}

	private Util() {
	}

	/**
	 * Returns the qualified name corresponding to the given {@link FunctionDef}.
	 *
	 * @see <a href="https://peps.python.org/pep-3155">PEP 3155</a>
	 * @param functionDef The {@link FunctionDef} in question.
	 * @return The corresponding qualified name per PEP 3155.
	 */
	public static String getQualifiedName(FunctionDef functionDef) {
		String identifier = NodeUtils.getFullRepresentationString(functionDef);
		StringBuilder ret = new StringBuilder();
		SimpleNode parentNode = functionDef.parent;

		int count = 0;

		while (parentNode instanceof ClassDef || parentNode instanceof FunctionDef) {
			String identifierParent = NodeUtils.getFullRepresentationString(parentNode);

			if (count == 0) {
				ret.append(identifierParent);
				ret.append(".");
			} else {
				ret.insert(0, ".");
				ret.insert(0, identifierParent);
			}
			count++;

			parentNode = parentNode.parent;
		}

		ret.append(identifier);

		return ret.toString();
	}

	public static CoreTextSelection getCoreTextSelection(IDocument document, SimpleNode expression) {
		int offset = NodeUtils.getOffset(document, expression);
		String representationString = NodeUtils.getRepresentationString(expression);
		CoreTextSelection coreTextSelection = new CoreTextSelection(document, offset, representationString.length());
		return coreTextSelection;
	}
}
