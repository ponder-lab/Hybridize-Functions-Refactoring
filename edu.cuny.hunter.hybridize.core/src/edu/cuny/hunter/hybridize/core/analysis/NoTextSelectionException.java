package edu.cuny.hunter.hybridize.core.analysis;

import org.python.pydev.parser.jython.SimpleNode;

public class NoTextSelectionException extends Exception {

	private static final long serialVersionUID = -5565002672946460853L;

	public NoTextSelectionException(SimpleNode expression, Throwable cause) {
		super("Can't get text selection from: " + expression + ".", cause);
	}

	public NoTextSelectionException(SimpleNode expression, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super("Can't get text selection from: " + expression + ".", cause, enableSuppression, writableStackTrace);
	}
}
