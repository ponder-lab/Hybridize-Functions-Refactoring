package edu.cuny.hunter.hybridize.core.analysis;

public enum PreconditionSuccess {
	P1, P2, P3,

	/**
	 * A hybrid function that is already correctly hybrid but carries no {@code input_signature}: reconfigure its decorator to add the
	 * inferred input signature. This is the add-when-absent path into the {@link Transformation#RECONFIGURE} transformation; the narrowing
	 * path is {@link #P5}.
	 */
	P4,

	/**
	 * A hybrid function whose existing {@code input_signature} is strictly broader than its reachable call sites require: narrow it to the
	 * inferred signature. Under the closed-world assumption the reachable call sites are all the callers, and every one of them already
	 * conforms to the inferred signature, so narrowing preserves behavior and yields a tighter signature. The narrowing is taken only for a
	 * literal signature (a named one may be shared; {@link PreconditionFailure#SUPPLIED_INPUT_SIGNATURE_SHARED_BY_NAME}) whose narrowing
	 * changes no statically read shape ({@link PreconditionFailure#NARROWING_CHANGES_STATICALLY_READ_SHAPE}).
	 * <p>
	 * The other relations are never rewritten. A supplied signature strictly tighter than the inferred one, or incomparable with it, means
	 * some reachable call already violates it, and rewriting to admit that call would repair the program rather than refactor it
	 * ({@link PreconditionFailure#SUPPLIED_INPUT_SIGNATURE_DISAGREES_WITH_CALLS}). An agreeing signature is a no-op. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/808.
	 */
	P5,

	/**
	 * A hybrid function with a tensor parameter that performs no tensor computation and has no Python side-effects: graph execution offers
	 * no benefit, only tracing overhead, so de-hybridize it. This is the hybrid-to-eager counterpart of the eager-to-hybrid
	 * {@link PreconditionFailure#NO_TENSOR_COMPUTATION} benefit precondition, and a peer of {@link #P2} and {@link #P3} in reaching
	 * {@link Transformation#CONVERT_TO_EAGER}. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/709.
	 */
	P6,
}
