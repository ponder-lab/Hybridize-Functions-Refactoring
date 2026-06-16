package edu.cuny.hunter.hybridize.core.analysis;

public enum PreconditionSuccess {
	P1, P2, P3,

	/**
	 * A hybrid function that is already correctly hybrid but carries no {@code input_signature}: reconfigure its decorator to add the
	 * inferred input signature. This is the add-when-absent path into the {@link Transformation#RECONFIGURE} transformation; overwriting an
	 * already-supplied signature ({@link #P5}) is a separate precondition path that reaches the same transformation, mirroring how
	 * {@link #P2} and {@link #P3} both reach {@link Transformation#CONVERT_TO_EAGER}.
	 */
	P4,

	/**
	 * A hybrid function whose existing {@code input_signature} disagrees with the inferred one in a way that warrants an overwrite—the
	 * supplied signature is strictly tighter than the inferred one (the call-site evidence admits inputs the supplied signature would
	 * reject), or the two are incomparable: reconfigure its decorator to replace the existing signature with the inferred one. This is the
	 * overwrite path into {@link Transformation#RECONFIGURE}; adding a signature when none is supplied is {@link #P4}. A supplied signature
	 * that is strictly broader than the inferred one is preserved (no transformation), and an agreeing signature is a no-op.
	 */
	P5,
}
