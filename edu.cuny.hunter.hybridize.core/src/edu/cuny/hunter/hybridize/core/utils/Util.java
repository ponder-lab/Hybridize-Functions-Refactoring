package edu.cuny.hunter.hybridize.core.utils;

import java.util.Optional;

import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.ltk.core.refactoring.Refactoring;
import org.eclipse.ltk.core.refactoring.participants.ProcessorBasedRefactoring;
import org.python.pydev.parser.jython.ast.FunctionDef;

import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;

public class Util {

	public static Refactoring createRefactoring(FunctionDef[] functions, Optional<IProgressMonitor> monitor) {
		HybridizeFunctionRefactoringProcessor processor = new HybridizeFunctionRefactoringProcessor(functions);
		return new ProcessorBasedRefactoring(processor);
	}

}
