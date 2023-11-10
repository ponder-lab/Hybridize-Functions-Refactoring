package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Arrays;

public enum PreconditionFailure {
	CURRENTLY_NOT_HANDLED(1),

	/**
	 * Either there is no call to the function, there is a call but don't handle it, or something about decorators?.
	 */
	UNDETERMINABLE_SIDE_EFFECTS(3),

	HAS_PYTHON_SIDE_EFFECTS(4),

	HAS_NO_TENSOR_PARAMETERS(6),

	HAS_TENSOR_PARAMETERS(7),

	/**
	 * Can't determine whether the decorator actually refers to the real tf.function. It may not (and we've found cases of this in
	 * mead-baseline).
	 */
	UNDETERMINABLE_HYBRID_DECORATOR(8);

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
