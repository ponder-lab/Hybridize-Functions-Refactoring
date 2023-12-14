package edu.cuny.hunter.hybridize.core.analysis;

public class CantInferPrimitiveParametersException extends Exception {

	private static final long serialVersionUID = 7274055899644724861L;

	public CantInferPrimitiveParametersException(String message) {
		super(message);
	}

	public CantInferPrimitiveParametersException(Throwable cause) {
		super(cause);
	}

	public CantInferPrimitiveParametersException(String message, Throwable cause) {
		super(message, cause);
	}

	public CantInferPrimitiveParametersException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}
}
