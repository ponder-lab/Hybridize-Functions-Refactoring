package edu.cuny.hunter.hybridize.core.descriptors;

import java.util.Map;

import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.participants.RefactoringProcessor;

import edu.cuny.citytech.refactoring.common.core.Refactoring;
import edu.cuny.citytech.refactoring.common.core.RefactoringDescriptor;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;

public class HybridizeFunctionRefactoringDescriptor extends RefactoringDescriptor {

	public static final String REFACTORING_ID = "edu.cuny.hunter.refactoring.hybridize.function"; // $NON-NLS-1$

	public HybridizeFunctionRefactoringDescriptor(String refactoringID, String project, String description,
			String comment, @SuppressWarnings("rawtypes") Map arguments) {
		super(refactoringID, project, description, comment, arguments);
	}

	public HybridizeFunctionRefactoringDescriptor(String project, @SuppressWarnings("rawtypes") Map arguments) {
		super(REFACTORING_ID, project, "Hybridize a TensforFlow client function.",
				"Improves efficiency in imperatives style TensorFlow client code.", arguments);
	}

	@Override
	protected Refactoring createRefactoring() {
		RefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor();
		return new Refactoring() {

			@Override
			public String getName() {
				return processor.getProcessorName();
			}

			@Override
			public Change createChange(IProgressMonitor pm) throws CoreException, OperationCanceledException {
				return processor.createChange(pm);
			}

			@Override
			public RefactoringStatus checkInitialConditions(IProgressMonitor pm)
					throws CoreException, OperationCanceledException {
				return processor.checkInitialConditions(pm);
			}

			@Override
			public RefactoringStatus checkFinalConditions(IProgressMonitor pm)
					throws CoreException, OperationCanceledException {
				return processor.checkFinalConditions(pm, null);
			}
		};
	}
}
