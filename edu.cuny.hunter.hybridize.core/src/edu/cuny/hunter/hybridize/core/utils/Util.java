package edu.cuny.hunter.hybridize.core.utils;

import org.eclipse.ltk.core.refactoring.Refactoring;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.python.pydev.parser.jython.ast.FunctionDef;

import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;

public class Util {

	public static Refactoring createRefactoring(FunctionDef[] functions) {
		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(functions);
		return new ProcessorBasedRefactoring(processor);
	}

}
