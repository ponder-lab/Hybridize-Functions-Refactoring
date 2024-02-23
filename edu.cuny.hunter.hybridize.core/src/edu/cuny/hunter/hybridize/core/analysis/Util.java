package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

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
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.shared_core.string.CoreTextSelection;

import com.google.common.collect.Sets;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;

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
	 * @throws AmbiguousDeclaringModuleException On ambiguous definitions found.
	 * @throws BadLocationException On a parsing error.
	 */
	public static String getDeclaringModuleName(PySelection selection, String containingModName, File containingFile, IPythonNature nature,
			IProgressMonitor monitor) throws BadLocationException, AmbiguousDeclaringModuleException {
		monitor.beginTask("Getting declaring module name.", 1);

		LOG.info(String.format("Getting declaring module name for selection: %s in line: %s, module: %s, file: %s, and project: %s.",
				selection.getSelectedText(), selection.getLineWithoutCommentsOrLiterals().strip(), containingModName, containingFile,
				nature.getProject()));

		RefactoringRequest request = new RefactoringRequest(containingFile, selection, nature);

		request.acceptTypeshed = true;
		request.moduleName = containingModName;
		request.pushMonitor(monitor);

		IPyRefactoring pyRefactoring = AbstractPyRefactoring.getPyRefactoring();

		ItemPointer[] pointers;
		try {
			pointers = pyRefactoring.findDefinition(request);
		} catch (TooManyMatchesException e) {
			throw new AmbiguousDeclaringModuleException(selection, containingModName, containingFile, nature, e);
		}

		LOG.info("Found " + pointers.length + " \"pointer(s).\"");

		if (pointers.length == 0)
			throw new IllegalArgumentException(
					String.format("Can't find declaring module for selection: %s in line: %s, module: %s, file: %s, and project: %s.",
							selection.getSelectedText(), selection.getLineWithoutCommentsOrLiterals().strip(), containingModName,
							containingFile.getName(), nature.getProject()));

		// Collect the potential declaring module names.
		Set<String> potentialDeclaringModuleNames = new HashSet<>();

		// for each match.
		for (ItemPointer itemPointer : pointers) {
			Definition definition = itemPointer.definition;
			LOG.info("Found definition: " + definition + ".");

			IModule module = definition.module;
			LOG.info(String.format("Found module: %s.", module));

			String moduleName = module.getName();
			LOG.info(String.format("Found module name: %s.", moduleName));

			// add it to the set of found module names.
			potentialDeclaringModuleNames.add(moduleName);
		}

		// if we found a unique module name.
		if (potentialDeclaringModuleNames.size() == 1) {
			monitor.done();

			// return the first one.
			return potentialDeclaringModuleNames.iterator().next();
		}

		// otherwise, we have an ambiguous declaring module name.
		throw new AmbiguousDeclaringModuleException(selection, containingModName, containingFile, nature,
				potentialDeclaringModuleNames.size());
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
	 * @throws BadLocationException When the containing entities cannot be parsed.
	 * @throws AmbiguousDeclaringModuleException If the definition of the decorator is ambiguous.
	 */
	public static String getFullyQualifiedName(decoratorsType decorator, String containingModName, File containingFile,
			PySelection containingSelection, IPythonNature nature, IProgressMonitor monitor)
			throws BadLocationException, AmbiguousDeclaringModuleException {
		monitor.beginTask("Getting decorator FQN.", 3);

		exprType decoratorFunction = decorator.func;
		String fqn = getFullyQualifiedName(decoratorFunction, containingModName, containingFile, containingSelection, nature, monitor);

		monitor.done();
		return fqn;
	}

	public static String getFullyQualifiedName(SimpleNode node, String containingModName, File containingFile,
			PySelection containingSelection, IPythonNature nature, IProgressMonitor monitor)
			throws BadLocationException, AmbiguousDeclaringModuleException {
		monitor.subTask("Getting declaring module name.");
		LOG.info("Getting declaring module name for SimpleNode: " + node + ".");

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

	public static PySelection getSelection(decoratorsType decorator, IDocument document) {
		exprType expression = getExpressionFromFunction(decorator);
		LOG.info("Getting PySelection for exprType: " + expression + ".");
		return getSelection(expression, document);
	}

	public static PySelection getSelection(SimpleNode node, IDocument document) {
		CoreTextSelection coreTextSelection = getCoreTextSelection(document, node);
		return new PySelection(document, coreTextSelection);
	}

	public static CoreTextSelection getCoreTextSelection(IDocument document, SimpleNode expression) {
		int offset = NodeUtils.getOffset(document, expression);
		String representationString = NodeUtils.getRepresentationString(expression);
		CoreTextSelection coreTextSelection = new CoreTextSelection(document, offset, representationString.length());
		return coreTextSelection;
	}

	/**
	 * Returns the {@link exprType} associated with the given {@link decoratorsType}'s "function."
	 *
	 * @param decorator The {@link decoratorsType} for which to retrieve the associated {@link exprType} from its "function."
	 * @return The {@link exprType} associated with the given {@link decoratorsType}'s "function."
	 */
	public static exprType getExpressionFromFunction(decoratorsType decorator) {
		exprType func = decorator.func;
		return getInnerExpression(func);
	}

	private static exprType getInnerExpression(exprType expr) {
		if (expr instanceof Attribute || expr instanceof Name)
			return expr;

		if (expr instanceof Call) {
			Call call = (Call) expr;
			exprType func = call.func;
			return getInnerExpression(func);
		}

		throw new IllegalArgumentException("Can't find attribute of: " + expr + ".");
	}

	/**
	 * Returns true iff the given {@link decoratorsType} corresponds to a Python generated decorator (e.g., "setter" for properties).
	 *
	 * @param decorator The {@link decoratorsType} in question.
	 * @return True iff the given {@link decoratorsType} is generated by the run-time (e.g., a property).
	 */
	public static boolean isGenerated(decoratorsType decorator) {
		String decoratorRepresentation = NodeUtils.getRepresentationString(decorator.func);
		return decoratorRepresentation.equals("setter");
	}

	public static boolean isBuiltIn(decoratorsType decorator) {
		String decoratorRepresentation = NodeUtils.getRepresentationString(decorator.func);
		return decoratorRepresentation.equals("property");
	}

	public static boolean calls(CGNode node, MethodReference methodReference, CallGraph callGraph) {
		return calls(node, methodReference, callGraph, Sets.newHashSet());
	}

	private static boolean calls(CGNode node, MethodReference methodReference, CallGraph callGraph, Set<MethodReference> seen) {
		seen.add(node.getMethod().getReference());

		// check the callees.
		for (Iterator<CGNode> succNodes = callGraph.getSuccNodes(node); succNodes.hasNext();) {
			CGNode next = succNodes.next();
			MethodReference reference = next.getMethod().getReference();

			if (methodReference.equals(reference))
				return true;

			// otherwise, check its callees.
			if (!seen.contains(reference) && calls(next, methodReference, callGraph))
				return true;
		}

		return false;
	}

	public static boolean isContainerType(TypeReference reference) {
		return reference.equals(PythonTypes.dict) || reference.equals(PythonTypes.enumerate) || reference.equals(PythonTypes.list)
				|| reference.equals(PythonTypes.set) || reference.equals(PythonTypes.tuple);
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static void addEntryPoints(Collection target, Iterable source) {
		for (Object entryPoint : source)
			if (target.add(entryPoint))
				LOG.info("Adding entrypoint: " + entryPoint);
	}
}
