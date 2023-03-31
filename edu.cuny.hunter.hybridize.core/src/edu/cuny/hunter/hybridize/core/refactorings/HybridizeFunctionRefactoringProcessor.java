package edu.cuny.hunter.hybridize.core.refactorings;

import static org.eclipse.core.runtime.Platform.getLog;

import java.util.LinkedHashSet;
import java.util.Set;

import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.NullChange;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.participants.CheckConditionsContext;
import org.eclipse.ltk.core.refactoring.participants.RefactoringParticipant;
import org.eclipse.ltk.core.refactoring.participants.SharableParticipants;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.preferences.InterpreterGeneralPreferences;

import edu.cuny.citytech.refactoring.common.core.RefactoringProcessor;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.descriptors.HybridizeFunctionRefactoringDescriptor;
import edu.cuny.hunter.hybridize.core.messages.Messages;

public class HybridizeFunctionRefactoringProcessor extends RefactoringProcessor {

	private static final ILog LOG = getLog(HybridizeFunctionRefactoringProcessor.class);

	private static RefactoringStatus checkDecorators(Function func) {
		RefactoringStatus status = new RefactoringStatus();
		LOG.info("Checking decorators for: " + func + ".");
		// TODO: Is the function already decorated with tf.function? NOTE: May move to checkFinalConditions() as this
		// will be dependent on other things.
		return status;
	}

	private static RefactoringStatus checkParameters(Function func) {
		RefactoringStatus status = new RefactoringStatus();
		LOG.info("Checking parameters for: " + func + ".");
		// TODO: Does the function have a tensor parameter (#2)?
		// NOTE: Not sure if we will be checking everything individually here since we'll do the computation in the
		// Function class. Instead, we may just need to check everything in checkFinalConditions(), as it is likely that
		// the checking will depend on several things.
		return status;
	}

	private Set<Function> functions = new LinkedHashSet<>();

	public HybridizeFunctionRefactoringProcessor() {
	}

	public HybridizeFunctionRefactoringProcessor(Set<FunctionDefinition> functionDefinitionSet, IProgressMonitor monitor)
			throws TooManyMatchesException, BadLocationException {
		// Force the use of typeshed. It's an experimental feature of PyDev.
		InterpreterGeneralPreferences.FORCE_USE_TYPESHED = Boolean.TRUE;

		// Convert the FunctionDefs to Functions.
		if (functions != null) {
			Set<Function> functionSet = this.getFunctions();

			for (FunctionDefinition fd : functionDefinitionSet) {
				Function function = new Function(fd, monitor);

				// Add the Function to the Function set.
				functionSet.add(function);
			}
		}
	}

	@Override
	public RefactoringStatus checkFinalConditions(IProgressMonitor pm, CheckConditionsContext context)
			throws CoreException, OperationCanceledException {
		RefactoringStatus status = new RefactoringStatus();
		SubMonitor progress = SubMonitor.convert(pm, Messages.CheckingPreconditions, 100);
		// TODO: Adjust amount of work later.

		status.merge(this.checkFunctions(progress.split(1)));

		return status;
	}

	private RefactoringStatus checkFunctions(IProgressMonitor monitor) {
		RefactoringStatus status = new RefactoringStatus();
		Set<Function> functions = this.getFunctions();
		@SuppressWarnings("unused")
		SubMonitor progress = SubMonitor.convert(monitor, Messages.CheckingFunctions, functions.size());
		LOG.info("Checking " + functions.size() + " function" + (functions.size() > 1 ? "s" : "") + ".");

		for (Function func : functions) {
			LOG.info("Checking function: " + func + ".");

			// TODO: Whether a function has a tensor argument should probably be an initial
			// condition: functions w/o such arguments should not be candidates.
			// TODO: It might be time to now go back to the paper to see how we can
			// formulate the preconditions. Have a look at the stream refactoring paper.

			status.merge(checkParameters(func));
			status.merge(checkDecorators(func));
		}

		return status;
	}

	@Override
	public RefactoringStatus checkInitialConditions(IProgressMonitor pm) throws CoreException, OperationCanceledException {
		return super.checkInitialConditions(pm);
	}

	@Override
	protected void clearCaches() {
		// NOTE: Nothing right now.
	}

	@Override
	public Change createChange(IProgressMonitor pm) throws CoreException, OperationCanceledException {
		// TODO Auto-generated method stub
		return new NullChange();
	}

	@Override
	public Object[] getElements() {
		// TODO Auto-generated method stub
		return null;
	}

	public Set<Function> getFunctions() {
		return this.functions;
	}

	@Override
	public String getIdentifier() {
		return HybridizeFunctionRefactoringDescriptor.REFACTORING_ID;
	}

	@Override
	public String getProcessorName() {
		return Messages.Name;
	}

	@Override
	public boolean isApplicable() throws CoreException {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public RefactoringParticipant[] loadParticipants(RefactoringStatus status, SharableParticipants sharedParticipants)
			throws CoreException {
		// TODO Auto-generated method stub
		return new RefactoringParticipant[0];
	}
}
