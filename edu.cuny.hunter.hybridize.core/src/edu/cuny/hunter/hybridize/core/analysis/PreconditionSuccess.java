package edu.cuny.hunter.hybridize.core.analysis;

public enum PreconditionSuccess {
	P1, P2, P3,

	/**
	 * A hybrid function that is already correctly hybrid but carries no {@code input_signature}: reconfigure its decorator to add the
	 * inferred input signature. This is the add-when-absent path into the {@link Transformation#RECONFIGURE} transformation, and it is
	 * currently the only path into it that the refactoring takes. The overwrite path, {@link #P5}, is reserved rather than taken; see that
	 * constant for why.
	 */
	P4,

	/**
	 * Reserved, and not currently taken by the refactoring. It denotes a hybrid function whose existing {@code input_signature} disagrees
	 * with the inferred one in a way that would warrant an overwrite: the supplied signature is strictly tighter than the inferred one, so
	 * the call-site evidence admits inputs the supplied signature would reject, or the two are incomparable.
	 * <p>
	 * No relation rewrites a supplied signature. All four are adjudicated and reported instead, because the inferred signature is the join
	 * over the observed call sites, so a tighter or incomparable relation can only arise when some observed call already violates the
	 * supplied signature. Rewriting to admit those calls would repair the program rather than refactor it, and it would buy nothing: a
	 * signature that is already present pins one trace, so there is no retracing benefit to trade against the change in accepted inputs. A
	 * supplied signature that is strictly broader than the inferred one is likewise preserved, in case the breadth is intentional, and an
	 * agreeing signature is a no-op. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/808.
	 * <p>
	 * The constant is kept rather than removed because that rewrite is a sanctioned future find-and-fix transformation, distinct from this
	 * refactoring. Adding a signature when none is supplied is {@link #P4}, which is taken.
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
