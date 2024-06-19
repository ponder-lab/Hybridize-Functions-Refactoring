package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Arrays;

public enum Information {

	/**
	 * Information related to (tensor) type inferencing.
	 */
	TYPE_INFERENCING(1),

	/**
	 * Context being used.
	 */
	SPECULATIVE_ANALYSIS(2);

	static {
		// check that the codes are unique.
		assert Arrays.stream(Information.values()).map(Information::getCode).distinct()
				.count() == Information.values().length : "Codes must be unique.";
	}

	private int code;

	private Information(int code) {
		this.code = code;
	}

	public int getCode() {
		return this.code;
	}

}
