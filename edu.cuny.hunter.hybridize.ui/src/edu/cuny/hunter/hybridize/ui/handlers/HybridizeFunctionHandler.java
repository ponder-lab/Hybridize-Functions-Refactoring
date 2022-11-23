package edu.cuny.hunter.hybridize.ui.handlers;

import static edu.cuny.hunter.hybridize.ui.handlers.Util.getDocument;
import static edu.cuny.hunter.hybridize.ui.handlers.Util.getFile;
import static edu.cuny.hunter.hybridize.ui.handlers.Util.getModuleName;
import static org.eclipse.core.runtime.Platform.getLog;
import static org.eclipse.ui.handlers.HandlerUtil.getActiveShellChecked;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.navigator.PythonModelProvider;
import org.python.pydev.navigator.elements.PythonNode;
import org.python.pydev.outline.ParsedItem;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.visitors.scope.ASTEntryWithChildren;

import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.analysis.FunctionExtractor;
import edu.cuny.hunter.hybridize.core.utils.RefactoringAvailabilityTester;
import edu.cuny.hunter.hybridize.ui.wizards.HybridizeFunctionRefactoringWizard;

public class HybridizeFunctionHandler extends AbstractHandler {

	private static final ILog LOG = getLog(HybridizeFunctionHandler.class);

	private static final PythonModelProvider provider = new PythonModelProvider();

	private static Set<PythonNode> getPythonNodes(Object obj) {
		Set<PythonNode> ret = new HashSet<>();

		if (obj instanceof PythonNode) {
			PythonNode pythonNode = (PythonNode) obj;
			ret.add(pythonNode);
		} else {
			Object[] children = provider.getChildren(obj);
			for (Object child : children)
				ret.addAll(getPythonNodes(child));
		}
		return ret;
	}

	private static Set<FunctionDefinition> process(PythonNode pythonNode) throws ExecutionException, CoreException, IOException {
		Set<FunctionDefinition> ret = new HashSet<>();

		String moduleName = getModuleName(pythonNode);
		File file = getFile(pythonNode);
		IDocument document = getDocument(pythonNode);
		IPythonNature nature = Util.getPythonNature(pythonNode);

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

		Collection<FunctionDef> definitions = functionExtractor.getDefinitions();

		for (FunctionDef def : definitions) {
			FunctionDefinition function = new FunctionDefinition(def, moduleName, file, document, nature);
			ret.add(function);
		}

		return ret;
	}

	/**
	 * Gather all functions from the user's selection.
	 */
	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		Set<FunctionDefinition> functions = new HashSet<>();
		ISelection currentSelection = HandlerUtil.getCurrentSelectionChecked(event);

		if (currentSelection instanceof IStructuredSelection) {
			List<?> list = ((IStructuredSelection) currentSelection).toList();

			if (list != null)
				for (Object obj : list) {
					Set<PythonNode> nodeSet = getPythonNodes(obj);

					for (PythonNode node : nodeSet)
						try {
							functions.addAll(process(node));
						} catch (CoreException | IOException e) {
							throw new ExecutionException("Unable to process python node:" + node + ".", e);
						}
				}
		}

		LOG.info("Found " + functions.size() + " function definition(s).");

		Set<FunctionDefinition> availableFunctions = functions.stream()
				.filter(f -> RefactoringAvailabilityTester.isHybridizationAvailable(f.getFunctionDef())).collect(Collectors.toSet());
		LOG.info("Found " + availableFunctions.size() + " available functions.");

		Shell shell = getActiveShellChecked(event);

		try {
			HybridizeFunctionRefactoringWizard.startRefactoring(availableFunctions, shell, new NullProgressMonitor());
		} catch (TooManyMatchesException | BadLocationException e) {
			throw new ExecutionException("Unable to start refactoring.", e);
		}

		return null;
	}
}
