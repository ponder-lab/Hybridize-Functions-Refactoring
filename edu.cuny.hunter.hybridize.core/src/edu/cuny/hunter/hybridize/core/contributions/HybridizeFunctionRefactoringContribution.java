package edu.cuny.hunter.hybridize.core.contributions;

import java.util.Map;

import org.eclipse.ltk.core.refactoring.RefactoringDescriptor;

import edu.cuny.citytech.refactoring.common.core.RefactoringContribution;
import edu.cuny.hunter.hybridize.core.descriptors.HybridizeFunctionRefactoringDescriptor;

public class HybridizeFunctionRefactoringContribution extends RefactoringContribution {

	@Override
	public RefactoringDescriptor createDescriptor(String id, String project, String description, String comment,
			Map<String, String> arguments, int flags) throws IllegalArgumentException {
		return new HybridizeFunctionRefactoringDescriptor(id, project, description, comment, arguments);
	}
}
