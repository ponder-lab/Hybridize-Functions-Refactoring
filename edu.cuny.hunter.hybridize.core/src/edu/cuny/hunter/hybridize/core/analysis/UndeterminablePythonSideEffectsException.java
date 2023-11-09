package edu.cuny.hunter.hybridize.core.analysis;

import com.ibm.wala.types.MethodReference;

public class UndeterminablePythonSideEffectsException extends Exception {

	private static final long serialVersionUID = -1229657254725226075L;

	public UndeterminablePythonSideEffectsException(MethodReference methodReference) {
		super("Can't find: " + methodReference + " in call graph.");
	}
}
