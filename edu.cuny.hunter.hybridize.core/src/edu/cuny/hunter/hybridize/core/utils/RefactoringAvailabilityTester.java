package edu.cuny.hunter.hybridize.core.utils;

import static org.eclipse.core.runtime.Platform.getLog;

import org.eclipse.core.runtime.ILog;
import org.python.pydev.parser.jython.ast.FunctionDef;
import edu.cuny.hunter.hybridize.core.analysis.Util;

public class RefactoringAvailabilityTester {

	private static final ILog LOG = getLog(RefactoringAvailabilityTester.class);

	public static boolean isHybridizationAvailable(FunctionDef function) {
		LOG.info("Testing hybridization availability for: " + Util.getQualifiedName(function));
		// TODO
		return true;
	}

	private RefactoringAvailabilityTester() {
	}
}
