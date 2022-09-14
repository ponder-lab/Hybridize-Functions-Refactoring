package edu.cuny.hunter.hybridize.core.analysis;

import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;

public class Util {

	private Util() {
	}

	public static String getDeclaringModuleName(decoratorsType decorator) {
		// NOTE: __module__ gives us what we need. Either use dynamic analysis to get it or analyze imports?
		// Is there an import scope visitor? Module name getter?
		// Have a look at https://github.com/fabioz/Pydev/search?q=declared.
		// What module is thing declared in? __module__ is the name of the module the function was defined in,
		// or None if unavailable according to https://docs.python.org/3/reference/datamodel.html.
		return null;
	}

	public static String getFullyQualifiedName(decoratorsType decorator) {
		String declaringModuleName = getDeclaringModuleName(decorator);

		exprType decoratorFunction = decorator.func;
		String decoratorfullRepresentationString = NodeUtils.getRepresentationString(decoratorFunction);

		return declaringModuleName + "." + decoratorfullRepresentationString;
	}

}
