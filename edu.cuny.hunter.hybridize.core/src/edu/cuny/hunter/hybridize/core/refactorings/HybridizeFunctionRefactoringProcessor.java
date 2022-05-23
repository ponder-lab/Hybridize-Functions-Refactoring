package edu.cuny.hunter.hybridize.core.refactorings;

import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.NullChange;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.participants.CheckConditionsContext;
import org.eclipse.ltk.core.refactoring.participants.RefactoringParticipant;
import org.eclipse.ltk.core.refactoring.participants.RefactoringProcessor;
import org.eclipse.ltk.core.refactoring.participants.SharableParticipants;

import edu.cuny.hunter.hybridize.core.descriptors.HybridizeFunctionRefactoringDescriptor;
import edu.cuny.hunter.hybridize.core.messages.Messages;

public class HybridizeFunctionRefactoringProcessor extends RefactoringProcessor {

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
		// TODO Auto-generated method stub
		// TODO: Going to have to turn the common plug-in into a hierarchy.
		// From streams:
//		try {
//			this.clearCaches();
//			this.getExcludedTimeCollector().clear();
//
//			// if (this.getSourceMethods().isEmpty())
//			// return
//			// RefactoringStatus.createFatalErrorStatus(Messages.StreamsNotSpecified);
//			// else {
//			RefactoringStatus status = new RefactoringStatus();
//			pm.beginTask(Messages.CheckingPreconditions, 1);
//			return status;
//			// }
//		} catch (Exception e) {
//			JavaPlugin.log(e);
//			throw e;
//		} finally {
//			pm.done();
//		}
		return new RefactoringStatus();
	}

	@Override
	public RefactoringStatus checkFinalConditions(IProgressMonitor pm, CheckConditionsContext context)
			throws CoreException, OperationCanceledException {
		// TODO Auto-generated method stub
		return new RefactoringStatus();
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
}
