package edu.cuny.hunter.hybridize.core.analysis;

import org.python.pydev.parser.jython.SimpleNode;

public class NoTextSelectionException extends Exception {

	private static final long serialVersionUID = -5565002672946460853L;

	private static String getMessage(SimpleNode expression) {
		return "Can't get text selection from: " + expression + ".";
	}

	public NoTextSelectionException(SimpleNode expression) {
		super(getMessage(expression));
	}

	public NoTextSelectionException(SimpleNode expression, Throwable cause) {
		super(getMessage(expression), cause);
	}

	public NoTextSelectionException(SimpleNode expression, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(getMessage(expression), cause, enableSuppression, writableStackTrace);
	}
}
