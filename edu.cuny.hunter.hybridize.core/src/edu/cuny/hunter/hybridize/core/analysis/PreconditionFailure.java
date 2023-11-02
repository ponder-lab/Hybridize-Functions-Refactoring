package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Arrays;

public enum PreconditionFailure {
	CURRENTLY_NOT_HANDLED(1),

	/**
	 * Not a candidate.
	 */
	OPTIMIZATION_NOT_AVAILABLE(2),

	/**
	 * Either there is no call to the function, there is a call but don't handle it, or something about decorators?.
	 */
	UNDETERMINABLE_SIDE_EFFECTS(3),

	/**
	 * P1 failure.
	 */
	HAS_SIDE_EFFECTS(4),

	/**
	 * P2 "failure."
	 */
	ALREADY_OPTIMAL(5),

	/**
	 * P1 failure.
	 */
	HAS_NO_TENSOR_PARAMETERS(6);

	static {
		// check that the codes are unique.
		if (Arrays.stream(PreconditionFailure.values()).map(PreconditionFailure::getCode).distinct()
				.count() != PreconditionFailure.values().length)
			throw new IllegalStateException("Codes aren't unique.");
	}

	public static void main(String[] args) {
		System.out.println("code,name");
		for (PreconditionFailure failure : PreconditionFailure.values())
			System.out.println(failure.getCode() + "," + failure);
	}

	private int code;

	private PreconditionFailure(int code) {
		this.code = code;
	}

	public int getCode() {
		return this.code;
	}
}
