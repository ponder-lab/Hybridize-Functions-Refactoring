package edu.cuny.hunter.hybridize.ui.handlers;

import static edu.cuny.hunter.hybridize.core.utils.Util.getFunctionDefinitions;
import static org.eclipse.core.runtime.Platform.getLog;
import static org.eclipse.ui.handlers.HandlerUtil.getActiveShellChecked;

import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.ui.handlers.HandlerUtil;
import org.python.pydev.ast.refactoring.TooManyMatchesException;

import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.utils.RefactoringAvailabilityTester;
import edu.cuny.hunter.hybridize.ui.wizards.HybridizeFunctionRefactoringWizard;

public class HybridizeFunctionHandler extends AbstractHandler {

	private static final ILog LOG = getLog(HybridizeFunctionHandler.class);

	/**
	 * Gather all functions from the user's selection.
	 */
	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		Set<FunctionDefinition> functions = null;
		ISelection currentSelection = HandlerUtil.getCurrentSelectionChecked(event);

		if (currentSelection instanceof IStructuredSelection) {
			List<?> list = ((IStructuredSelection) currentSelection).toList();

			if (list != null)
				try {
					functions = getFunctionDefinitions(list);
				} catch (CoreException | IOException e) {
					throw new ExecutionException("Unable to get functions from selections.", e);
				}
		}

		LOG.info("Found " + functions.size() + " function definition(s).");

		Set<FunctionDefinition> availableFunctions = functions.stream()
				.filter(f -> RefactoringAvailabilityTester.isHybridizationAvailable(f.getFunctionDef())).collect(Collectors.toSet());
		LOG.info("Found " + availableFunctions.size() + " available functions.");

		Shell shell = getActiveShellChecked(event);

		try {
			HybridizeFunctionRefactoringWizard.startRefactoring(availableFunctions, shell);
		} catch (TooManyMatchesException e) {
			throw new ExecutionException("Unable to start refactoring.", e);
		}

		return null;
	}
}
