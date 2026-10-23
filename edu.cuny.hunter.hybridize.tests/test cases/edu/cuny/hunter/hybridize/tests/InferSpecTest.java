package edu.cuny.hunter.hybridize.tests;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.UNKNOWN;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Optional;
import java.util.Set;

import org.junit.Test;

import com.ibm.wala.cast.python.ml.types.NoneType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;

import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.InferenceResult.AbsenceReason;

/**
 * Synthesized-input tests for {@link Function#inferSpec(Set)}, the per-parameter multi-context reduction. Inputs are hand-built
 * {@link TensorType} sets so the reduction is exercised in isolation from upstream tensor-type precision.
 *
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/494">Issue 494</a>
 */
@SuppressWarnings("static-method")
public class InferSpecTest {

	/**
	 * A single context whose dtype is {@code UNKNOWN} (dtype-⊤) drops the spec. {@code tf.UNKNOWN} is not a valid runtime dtype for
	 * {@code tf.function(input_signature=...)}, so the conservative reduction is to bottom. This branch is fixture-unreachable (upstream no
	 * longer emits a {@code DType.UNKNOWN} singleton), so the synthesized input is the only way to exercise it.
	 */
	@Test
	public void testDtypeTopSingletonDrops() {
		Optional<TensorType> spec = Function.inferSpec(Set.of(new TensorType(UNKNOWN, List.of(new NumericDim(3)))));
		assertTrue("A dtype-⊤ (UNKNOWN) singleton should reduce to bottom.", spec.isEmpty());
	}

	/**
	 * Witness that the drop above is caused by the {@code UNKNOWN} dtype specifically, not by the singleton shape: an otherwise-identical
	 * context with a concrete dtype reduces to that same concrete type.
	 */
	@Test
	public void testConcreteDtypeSingletonReduces() {
		TensorType concrete = new TensorType(FLOAT32, List.of(new NumericDim(3)));
		Optional<TensorType> spec = Function.inferSpec(Set.of(concrete));
		assertFalse("A concrete-dtype singleton should reduce to a spec.", spec.isEmpty());
		assertEquals(concrete, spec.get());
	}

	/**
	 * A parameter that is sparse at one call site and dense at another (same dtype, same shape) abandons to bottom. The dtype consensus
	 * holds but the sparseness consensus fails, and no single spec is sound: a dense {@code TensorSpec} rejects the {@code SparseTensor}
	 * the function accepts at the sparse call site, while a {@code SparseTensorSpec} rejects the dense tensor it accepts at the dense call
	 * site. So the reduction drops (#642). This is the inversion of the former dense fall-through, which #650 pinned with a TODO.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/642">Issue 642</a>
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/533">Issue 533</a>
	 */
	@Test
	public void testMixedSparseDenseDrops() {
		TensorType dense = new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(3)));
		TensorType sparse = new TensorType(FLOAT32, List.of(new NumericDim(3), new NumericDim(3))).asSparse();
		Optional<TensorType> spec = Function.inferSpec(Set.of(dense, sparse));

		assertTrue("A mixed sparse/dense parameter has no sound single spec, so it abandons to bottom (#642).", spec.isEmpty());
	}

	/**
	 * A concrete extent at a position loses to a symbolic twin at that same position. The reduction takes per-position CONSENSUS, not the
	 * more precise member: a {@link NumericDim} beside a {@link SymbolicDim} is a disagreement, and a disagreement wildcards the axis.
	 * <p>
	 * This is the contract an upstream precision fix has to satisfy to reach the specification surface. A fix that ADDS the concrete member
	 * beside an existing placeholder changes nothing here, because the placeholder still disagrees with it; only a fix that ELIMINATES the
	 * placeholder moves the emitted axis. Pinning it so the distinction is not rediscovered from a whole-project run.
	 *
	 * @see <a href="https://github.com/wala/ML/issues/875">wala/ML issue 875</a>
	 */
	@Test
	public void testConcreteDimLosesToSymbolicTwin() {
		TensorType concrete = new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new NumericDim(46)));
		TensorType twin = new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new SymbolicDim("?")));
		Optional<TensorType> spec = Function.inferSpec(Set.of(concrete, twin));

		assertFalse("Dtype and rank agree, so the reduction yields a spec rather than bottom.", spec.isEmpty());
		assertEquals("The agreeing leading axes survive and the disputed trailing axis wildcards.",
				new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new SymbolicDim("?"))), spec.get());
	}

	/**
	 * The witness for the test above: with the symbolic twin removed, the same concrete extents survive the reduction. Without this, a
	 * wildcarded trailing axis is equally consistent with the reduction never keeping any extent, and the pin above would assert nothing
	 * about the twin specifically.
	 *
	 * @see <a href="https://github.com/wala/ML/issues/875">wala/ML issue 875</a>
	 */
	@Test
	public void testAgreeingConcreteDimsSurvive() {
		TensorType concrete = new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new NumericDim(46)));
		Optional<TensorType> spec = Function.inferSpec(Set.of(concrete));

		assertFalse("A concrete singleton reduces to a spec.", spec.isEmpty());
		assertEquals("Every axis is concrete and agreed, so every axis survives.", concrete, spec.get());
	}

	/**
	 * Contexts that disagree on a concrete extent wildcard only the disputed position. Distinguishes the twin case above from ordinary
	 * multi-context disagreement: both wildcard, but this one has no placeholder involved, so a fold that removes a placeholder cannot
	 * recover it. The trailing axis agrees across both contexts and survives, which is what makes an upstream fold worth making.
	 *
	 * @see <a href="https://github.com/wala/ML/issues/875">wala/ML issue 875</a>
	 */
	@Test
	public void testDisagreeingConcreteDimsWildcardOnlyThatAxis() {
		TensorType first = new TensorType(FLOAT32, List.of(new NumericDim(16), new NumericDim(100), new NumericDim(46)));
		TensorType second = new TensorType(FLOAT32, List.of(new NumericDim(8), new NumericDim(10), new NumericDim(46)));
		Optional<TensorType> spec = Function.inferSpec(Set.of(first, second));

		assertFalse("Dtype and rank agree, so the reduction yields a spec rather than bottom.", spec.isEmpty());
		assertEquals("The disputed leading axes wildcard and the agreed trailing extent survives.",
				new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new SymbolicDim("?"), new NumericDim(46))), spec.get());
	}

	/**
	 * One concrete dtype alongside an unresolved context is not a conflict among the callers. Before
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/969 the set was sized with {@code UNKNOWN} counted as a member,
	 * so this took the heterogeneous branch and the emitted reason asserted that the call sites disagreed about the dtype. They do not: one
	 * of them did not resolve, and resolving it collapses the set to {@code INT32}.
	 */
	@Test
	public void testOneConcreteDtypeWithUnknownIsPartialRatherThanConflicting() {
		AbsenceReason reason = Function.classifyDtypeBottom(
				Set.of(new TensorType(INT32, List.of(new NumericDim(3))), new TensorType(UNKNOWN, List.of(new NumericDim(3)))));
		assertEquals("One concrete dtype plus an unresolved context is a precision gap, not a caller conflict.",
				AbsenceReason.PARTIAL_DTYPE, reason);
	}

	/**
	 * The neighbouring branch must keep its meaning: two concrete dtypes genuinely conflict, and no engine improvement removes the drop.
	 * Witnesses that {@link #testOneConcreteDtypeWithUnknownIsPartialRatherThanConflicting()} narrowed the heterogeneous branch rather than
	 * emptying it.
	 */
	@Test
	public void testTwoConcreteDtypesRemainHeterogeneous() {
		AbsenceReason reason = Function.classifyDtypeBottom(
				Set.of(new TensorType(INT32, List.of(new NumericDim(3))), new TensorType(FLOAT32, List.of(new NumericDim(3)))));
		assertEquals("Two concrete dtypes are a genuine conflict.", AbsenceReason.HETEROGENEOUS_DTYPE, reason);
	}

	/**
	 * Two concrete dtypes stay heterogeneous even when a third context is unresolved. Excluding {@code UNKNOWN} from the set must not
	 * demote a real conflict that happens to be accompanied by an unresolved context.
	 */
	@Test
	public void testTwoConcreteDtypesWithUnknownRemainHeterogeneous() {
		AbsenceReason reason = Function.classifyDtypeBottom(Set.of(new TensorType(INT32, List.of(new NumericDim(3))),
				new TensorType(FLOAT32, List.of(new NumericDim(3))), new TensorType(UNKNOWN, List.of(new NumericDim(3)))));
		assertEquals("A real conflict is not demoted by an accompanying unresolved context.", AbsenceReason.HETEROGENEOUS_DTYPE, reason);
	}

	/**
	 * Every context unresolved remains {@link AbsenceReason#UNKNOWN_DTYPE}: there is no concrete dtype to agree on, so the new branch must
	 * not swallow this one.
	 */
	@Test
	public void testAllUnknownRemainsUnknownDtype() {
		AbsenceReason reason = Function.classifyDtypeBottom(
				Set.of(new TensorType(UNKNOWN, List.of(new NumericDim(3))), new TensorType(UNKNOWN, List.of(new NumericDim(4)))));
		assertEquals("No concrete dtype anywhere is still the dtype-top case.", AbsenceReason.UNKNOWN_DTYPE, reason);
	}

	/**
	 * A {@code None} member, which a call passing {@code None} contributes (wala/ML#1040), is its own reason: its dtype is meaningless, so
	 * read as {@code UNKNOWN} it would report an unresolved context when the call site in fact passes a value no tensor specification
	 * accepts.
	 */
	@Test
	public void testNoneMemberReceivesNone() {
		AbsenceReason reason = Function.classifyDtypeBottom(Set.of(new TensorType(INT32, List.of(new NumericDim(3))), NoneType.INSTANCE));
		assertEquals("A call passing None is not an unresolved dtype.", AbsenceReason.RECEIVES_NONE, reason);
	}

	/**
	 * {@code None} takes precedence over a dtype conflict: whatever the tensors' dtypes, a specification rejects the call passing
	 * {@code None}, which is the more specific cause.
	 */
	@Test
	public void testNoneMemberPrecedesHeterogeneousDtype() {
		AbsenceReason reason = Function.classifyDtypeBottom(Set.of(new TensorType(INT32, List.of(new NumericDim(3))),
				new TensorType(FLOAT32, List.of(new NumericDim(3))), NoneType.INSTANCE));
		assertEquals("None is reported before a dtype conflict.", AbsenceReason.RECEIVES_NONE, reason);
	}

	/**
	 * A single agreed concrete dtype reaches neither dtype branch, so the bottom is attributed to the remaining axis. Pins that excluding
	 * {@code UNKNOWN} did not make the dtype branches fire on a set they should not claim.
	 */
	@Test
	public void testSingleConcreteDtypeFallsThroughToSparseness() {
		AbsenceReason reason = Function.classifyDtypeBottom(
				Set.of(new TensorType(INT32, List.of(new NumericDim(3))), new TensorType(INT32, List.of(new NumericDim(4)))));
		assertEquals("One agreed concrete dtype leaves sparseness as the only remaining reason.", AbsenceReason.HETEROGENEOUS_SPARSITY,
				reason);
	}

	/**
	 * Each reason carries its own sentence, and the two that are easiest to confuse say different things. A reader of the diagnostic must
	 * be able to tell a caller conflict from an unresolved context, which is the whole point of
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/969, so the wording is pinned rather than left to drift.
	 */
	@Test
	public void testEachDropReasonHasItsOwnMessage() {
		String heterogeneous = Function.dropMessage(AbsenceReason.HETEROGENEOUS_DTYPE);
		String partial = Function.dropMessage(AbsenceReason.PARTIAL_DTYPE);
		String unknown = Function.dropMessage(AbsenceReason.UNKNOWN_DTYPE);
		String sparsity = Function.dropMessage(AbsenceReason.HETEROGENEOUS_SPARSITY);
		String none = Function.dropMessage(AbsenceReason.RECEIVES_NONE);

		assertEquals("Five reasons must produce five distinct sentences.", 5,
				Set.of(heterogeneous, partial, unknown, sparsity, none).size());

		assertTrue("Only a real conflict may say the call sites conflict.", heterogeneous.contains("conflicting dtypes"));
		assertFalse("An unresolved context must not be reported as a conflict.", partial.contains("conflicting dtypes"));
		assertTrue("The partial case must say the call sites do not disagree.", partial.contains("do not disagree"));
		assertTrue("The dtype-top case names an undeterminable dtype.", unknown.contains("cannot be determined"));
		assertTrue("The sparseness case names the layout.", sparsity.contains("sparse at some call sites"));
		assertTrue("The None case names None.", none.contains("receives None"));
	}

	/**
	 * A {@code None} member beside two concrete dtypes is two reasons, not one: a specification rejects the call passing {@code None}, and
	 * the tensor calls disagree about the dtype whatever the {@code None} call does. Both are reported, {@code None} first as before.
	 */
	@Test
	public void testNoneMemberAndHeterogeneousDtypeAreBothReported() {
		Set<TensorType> contexts = Set.of(new TensorType(INT32, List.of(new NumericDim(3))),
				new TensorType(FLOAT32, List.of(new NumericDim(3))), NoneType.INSTANCE);
		assertEquals("Both the None member and the dtype conflict hold.",
				List.of(AbsenceReason.RECEIVES_NONE, AbsenceReason.HETEROGENEOUS_DTYPE),
				List.copyOf(Function.classifyDtypeBottoms(contexts)));
		assertEquals("The single reason is the first of the set.", AbsenceReason.RECEIVES_NONE, Function.classifyDtypeBottom(contexts));
	}

	/**
	 * A {@code None} member beside one concrete dtype and an unresolved context: the {@code None} call is rejected, and separately one
	 * context did not resolve. {@code None}'s own {@code UNKNOWN} dtype is not what makes the dtype partial; the unresolved tensor is.
	 */
	@Test
	public void testNoneMemberAndPartialDtypeAreBothReported() {
		Set<TensorType> contexts = Set.of(new TensorType(INT32, List.of(new NumericDim(3))),
				new TensorType(UNKNOWN, List.of(new NumericDim(3))), NoneType.INSTANCE);
		assertEquals("Both the None member and the unresolved context hold.",
				List.of(AbsenceReason.RECEIVES_NONE, AbsenceReason.PARTIAL_DTYPE), List.copyOf(Function.classifyDtypeBottoms(contexts)));
	}

	/**
	 * A {@code None} member beside one concrete dtype is the {@code None} reason alone. {@code None}'s dtype is meaningless, so it must not
	 * be counted as the unresolved context that would make the dtype partial.
	 */
	@Test
	public void testNoneMemberAloneIsNotPartialDtype() {
		assertEquals("None's dtype does not make the dtype partial.", Set.of(AbsenceReason.RECEIVES_NONE),
				Function.classifyDtypeBottoms(Set.of(new TensorType(INT32, List.of(new NumericDim(3))), NoneType.INSTANCE)));
	}

	/** A {@code None} member beside a sparse tensor is not a mixed layout: {@code None} is neither sparse nor dense. */
	@Test
	public void testNoneMemberIsNotDense() {
		assertEquals("None does not make the layout mixed.", Set.of(AbsenceReason.RECEIVES_NONE),
				Function.classifyDtypeBottoms(Set.of(new TensorType(INT32, List.of(new NumericDim(3))).asSparse(), NoneType.INSTANCE)));
	}

	/** The dtype and sparseness axes are independent, so a dtype conflict does not hide a mixed layout. */
	@Test
	public void testHeterogeneousDtypeAndSparsityAreBothReported() {
		Set<TensorType> contexts = Set.of(new TensorType(INT32, List.of(new NumericDim(3))).asSparse(),
				new TensorType(FLOAT32, List.of(new NumericDim(3))));
		assertEquals("Both the dtype conflict and the mixed layout hold.",
				List.of(AbsenceReason.HETEROGENEOUS_DTYPE, AbsenceReason.HETEROGENEOUS_SPARSITY),
				List.copyOf(Function.classifyDtypeBottoms(contexts)));
	}

	/**
	 * When no other axis holds, the set is the sparseness reason alone, the remaining reason {@link Function#inferSpec(Set)} bottoms for,
	 * matching the single-reason classifier.
	 */
	@Test
	public void testNoOtherAxisFallsThroughToSparsenessAlone() {
		assertEquals("One agreed concrete dtype leaves sparseness alone.", Set.of(AbsenceReason.HETEROGENEOUS_SPARSITY),
				Function.classifyDtypeBottoms(
						Set.of(new TensorType(INT32, List.of(new NumericDim(3))), new TensorType(INT32, List.of(new NumericDim(4))))));
	}

	/** Wizard-facing diagnostic text must not cite an issue tracker, mirroring the fixture-level assertion. */
	@Test
	public void testDropMessagesCiteNoIssueTracker() {
		for (AbsenceReason reason : List.of(AbsenceReason.HETEROGENEOUS_DTYPE, AbsenceReason.PARTIAL_DTYPE, AbsenceReason.UNKNOWN_DTYPE,
				AbsenceReason.HETEROGENEOUS_SPARSITY, AbsenceReason.RECEIVES_NONE))
			assertFalse("Wizard-facing status text must not cite an issue tracker.", Function.dropMessage(reason).matches(".*#\\d+.*"));
	}
}
