package edu.cuny.hunter.hybridize.core.analysis;

/**
 * What became of a function's inferred input signature, as {@link Function#getInputSignatureOutcome()} reports it. Defined only for a
 * function with an inferred signature, and exactly one constant applies to each, so counting a project's functions by outcome partitions
 * those with an inferred signature. The {@link Transformation} counts cannot answer this on their own: a signature added by a conversion
 * and one added by a reconfiguration land under different kinds, a conversion can write a bare decorator, and a signature that agrees with
 * a supplied one, or that is never written, lands under no kind at all (issue 1028).
 *
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1028">Issue 1028</a>
 */
public enum InputSignatureOutcome {

	/**
	 * Added to a decorator that carried no signature, whether the tool adds the decorator to an eager function
	 * ({@link Transformation#CONVERT_TO_HYBRID}) or the hybrid function already had it ({@link Transformation#RECONFIGURE} with
	 * {@link PreconditionSuccess#P4}). Either way the function gains a signature it didn't have, so the two aren't told apart.
	 */
	WRITTEN_BY_ADDITION,

	/**
	 * Written over a broader supplied signature, by {@link Transformation#RECONFIGURE} with {@link PreconditionSuccess#P5}.
	 */
	WRITTEN_BY_NARROWING,

	/**
	 * Identical to the signature the hybrid function already supplies ({@link InputSignature.Relation#AGREEMENT}), so writing it changes
	 * nothing.
	 */
	AGREEMENT,

	/**
	 * Not written into a function that is hybrid: the signature was withheld, a supplied one disagrees with it, a precondition failed, or
	 * the decorator is removed.
	 */
	NOT_WRITTEN_HYBRID,

	/**
	 * Not written into a function that is eager: a precondition failed, or the function is converted with a bare decorator because the
	 * signature's names aren't in scope.
	 */
	NOT_WRITTEN_EAGER
}
