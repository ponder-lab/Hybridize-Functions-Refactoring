package edu.cuny.hunter.hybridize.core.analysis;

public class NoDeclaringModuleException extends Exception {

	private static final long serialVersionUID = -677489832942987866L;

	public NoDeclaringModuleException() {
		super();
	}

	public NoDeclaringModuleException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}

	public NoDeclaringModuleException(String message, Throwable cause) {
		super(message, cause);
	}

	public NoDeclaringModuleException(String message) {
		super(message);
	}

	public NoDeclaringModuleException(Throwable cause) {
		super(cause);
	}
}
