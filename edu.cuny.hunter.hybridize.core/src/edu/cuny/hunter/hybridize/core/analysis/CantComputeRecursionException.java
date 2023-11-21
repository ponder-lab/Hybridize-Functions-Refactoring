package edu.cuny.hunter.hybridize.core.analysis;

public class CantComputeRecursionException extends Exception {

	private static final long serialVersionUID = 6647237480647044100L;

	public CantComputeRecursionException() {
	}

	public CantComputeRecursionException(String message) {
		super(message);
	}

	public CantComputeRecursionException(Throwable cause) {
		super(cause);
	}

	public CantComputeRecursionException(String message, Throwable cause) {
		super(message, cause);
	}

	public CantComputeRecursionException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}

}
