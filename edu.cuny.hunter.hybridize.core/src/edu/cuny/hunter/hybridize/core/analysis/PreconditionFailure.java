package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Arrays;

public enum PreconditionFailure {
	CURRENTLY_NOT_HANDLED(1);

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
