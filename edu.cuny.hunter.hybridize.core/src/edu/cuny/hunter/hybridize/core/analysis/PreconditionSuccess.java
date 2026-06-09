package edu.cuny.hunter.hybridize.core.analysis;

public enum PreconditionSuccess {
	P1, P2, P3,

	/**
	 * A hybrid function that is already correctly hybrid but carries no {@code input_signature}: reconfigure its decorator to add the
	 * inferred input signature.
	 */
	P4,
}
