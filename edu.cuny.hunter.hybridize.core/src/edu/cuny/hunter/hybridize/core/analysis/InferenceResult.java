package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Optional;

/**
 * The outcome of {@link Function#inferInputSignature}: either a successfully {@link Inferred} {@link InputSignature}, or an {@link Absent}
 * result carrying the {@link AbsenceReason} that blocked inference. Replacing a bare {@link Optional} lets downstream callers and tests
 * distinguish <em>why</em> a signature could not be inferred rather than collapsing every blocking condition to {@code Optional.empty}.
 * <p>
 * The result is total over the function's parameters: a signature is produced only when every non-{@code self} parameter contributes a
 * {@link com.ibm.wala.cast.python.ml.types.TensorType}; if any parameter blocks, the whole result is {@link Absent}. The degenerate case of
 * a function with no tensor parameter is a contract violation of {@link Function#inferInputSignature} (every refactoring call site is
 * already gated on {@link Function#getHasTensorParameter}), so it throws rather than producing a result.
 *
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/483">Issue 483</a>
 */
public sealed interface InferenceResult permits InferenceResult.Inferred, InferenceResult.Absent {

	/**
	 * The reason a single input signature could not be inferred for a parameter, in the order the per-parameter dispatch in
	 * {@link Function#inferInputSignature} encounters them. Each constant corresponds to one blocking branch of that dispatch.
	 */
	enum AbsenceReason {

		/**
		 * A parameter is not classified as tensor-typed by any phase, so no {@link com.ibm.wala.cast.python.ml.types.TensorType} is
		 * synthesized for it (#508).
		 */
		NON_TENSOR_PARAMETER,

		/**
		 * A parameter is classified as tensor-typed (via type hint or container detection) but has no call-site shape/dtype evidence to
		 * reduce into a concrete spec (#509).
		 */
		NO_SHAPE_OR_DTYPE_EVIDENCE,

		/**
		 * A parameter receives tensors with conflicting dtypes across call sites, so its dtype set has size {@code > 1} ({@code |D| ≠ 1}).
		 */
		HETEROGENEOUS_DTYPE,

		/**
		 * A parameter receives a tensor whose dtype cannot be determined (dtype-⊤: a single agreed {@code UNKNOWN}), which is not a valid
		 * runtime dtype for {@code tf.function(input_signature=...)} (#494).
		 */
		UNKNOWN_DTYPE
	}

	/**
	 * A successfully inferred input signature.
	 *
	 * @param inputSignature The inferred signature; never {@code null}.
	 */
	record Inferred(InputSignature inputSignature) implements InferenceResult {
	}

	/**
	 * Inference was blocked. Carries the first {@link AbsenceReason} encountered while dispatching over the function's parameters in
	 * declaration order; every blocking parameter still emits its own diagnostic INFO during inference.
	 *
	 * @param reason The first blocking reason; never {@code null}.
	 */
	record Absent(AbsenceReason reason) implements InferenceResult {
	}

	/**
	 * The inferred signature, or {@link Optional#empty} when inference was {@link Absent}. A projection for callers that only need the
	 * signature and not the absence reason (e.g. the source-write and the evaluator's content column).
	 *
	 * @return The signature for {@link Inferred}; {@link Optional#empty} for {@link Absent}.
	 */
	default Optional<InputSignature> signature() {
		return this instanceof Inferred inferred ? Optional.of(inferred.inputSignature()) : Optional.empty();
	}

	/**
	 * The absence reason, or {@link Optional#empty} when a signature was {@link Inferred}.
	 *
	 * @return The reason for {@link Absent}; {@link Optional#empty} for {@link Inferred}.
	 */
	default Optional<AbsenceReason> absenceReason() {
		return this instanceof Absent absent ? Optional.of(absent.reason()) : Optional.empty();
	}
}
