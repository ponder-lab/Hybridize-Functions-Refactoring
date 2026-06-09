package edu.cuny.hunter.hybridize.core.analysis;

public enum PreconditionSuccess {
	P1, P2, P3,

	/**
	 * A hybrid function that is already correctly hybrid but carries no {@code input_signature}: reconfigure its decorator to add the
	 * inferred input signature. This is the add-when-absent path into the {@link Transformation#RECONFIGURE} transformation; modifying an
	 * already-supplied signature (validate-then-overwrite) is a separate precondition path that reaches the same transformation, mirroring
	 * how {@link #P2} and {@link #P3} both reach {@link Transformation#CONVERT_TO_EAGER}.
	 */
	P4,
}
