package edu.cuny.hunter.hybridize.core.refactorings;

import static org.python.pydev.core.log.Log.logInfo;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.NullChange;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.participants.CheckConditionsContext;
import org.eclipse.ltk.core.refactoring.participants.RefactoringParticipant;
import org.eclipse.ltk.core.refactoring.participants.SharableParticipants;
import org.python.pydev.parser.jython.ast.FunctionDef;

import edu.cuny.citytech.refactoring.common.core.RefactoringProcessor;
import edu.cuny.hunter.hybridize.core.descriptors.HybridizeFunctionRefactoringDescriptor;
import edu.cuny.hunter.hybridize.core.messages.Messages;

// FIXME: Use our own logger.

public class HybridizeFunctionRefactoringProcessor extends RefactoringProcessor {

	private Set<FunctionDef> functions = new LinkedHashSet<>();

	// FIXME: Use our own logger.

	protected Set<FunctionDef> getFunctions() {
		return functions;
	}

	public HybridizeFunctionRefactoringProcessor() {
	}

	public HybridizeFunctionRefactoringProcessor(FunctionDef[] functions) {
		Collections.addAll(this.getFunctions(), functions);
	}

	@Override
	public Object[] getElements() {
		// TODO Auto-generated method stub
		return null;
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
	public RefactoringStatus checkInitialConditions(IProgressMonitor pm)
			throws CoreException, OperationCanceledException {
		return super.checkInitialConditions(pm);
	}

	@Override
	public RefactoringStatus checkFinalConditions(IProgressMonitor pm, CheckConditionsContext context)
			throws CoreException, OperationCanceledException {
		RefactoringStatus status = new RefactoringStatus();
		SubMonitor progress = SubMonitor.convert(pm, Messages.CheckingPreconditions, 100); // TODO: Adjust amount of
																							// work later.

		status.merge(checkFunctions(progress.split(1)));

		return status;
	}

	private RefactoringStatus checkFunctions(IProgressMonitor monitor) {
		RefactoringStatus status = new RefactoringStatus();

		Set<FunctionDef> functions = this.getFunctions();
		SubMonitor progress = SubMonitor.convert(monitor, Messages.CheckingFunctions, functions.size());
		logInfo("Checking " + functions.size() + " functions.");

		for (FunctionDef func : functions) {
			logInfo("Checking function: " + func + ".");

			// TODO: Can we create a Function class and record all the info?
			// TODO: Whether a function has a tensor argument should probably be an initial
			// condition: functions w/o such arguments should not be candidates.
			// TODO: It might be time to now go back to the paper to see how we can
			// formulate the precoditions. Have a look at the stream refactoring paper.

			status.merge(checkParameters(func));
			status.merge(checkDecorators(func));
		}

		return status;
	}

	private RefactoringStatus checkParameters(FunctionDef func) {
		RefactoringStatus status = new RefactoringStatus();
		// TODO: Does the function have a tensor parameter?
		return status;
	}

	private RefactoringStatus checkDecorators(FunctionDef func) {
		RefactoringStatus status = new RefactoringStatus();
		// TODO: Is the function already decorated with tf.function?
		return status;
	}

	@Override
	public Change createChange(IProgressMonitor pm) throws CoreException, OperationCanceledException {
		// TODO Auto-generated method stub
		return new NullChange();
	}

	@Override
	public RefactoringParticipant[] loadParticipants(RefactoringStatus status, SharableParticipants sharedParticipants)
			throws CoreException {
		// TODO Auto-generated method stub
		return new RefactoringParticipant[0];
	}

	@Override
	protected void clearCaches() {
		// NOTE: Nothing right now.
	}
}
