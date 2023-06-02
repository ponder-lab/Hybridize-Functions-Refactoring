package edu.cuny.hunter.hybridize.core.utils;

import java.util.Set;

import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.ltk.core.refactoring.Refactoring;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.python.pydev.ast.refactoring.TooManyMatchesException;

import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;

public class Util {

	public static Refactoring createRefactoring(Set<FunctionDefinition> functionDefinitions) throws TooManyMatchesException {
		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(functionDefinitions);
		return new ProcessorBasedRefactoring(processor);
	}
}
