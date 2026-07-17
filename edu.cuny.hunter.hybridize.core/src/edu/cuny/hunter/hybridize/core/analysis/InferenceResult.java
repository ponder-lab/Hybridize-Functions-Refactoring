package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Optional;

/**
 * The outcome of {@link Function#inferInputSignature}: either a successfully {@link Inferred} {@link InputSignature}, or an {@link Absent}
 * result carrying the {@link AbsenceReason} that blocked inference. Replacing a bare {@link Optional} lets downstream callers and tests
 * distinguish <em>why</em> a signature could not be inferred rather than collapsing every blocking condition to {@code Optional.empty}. The
 * result is total over the function's parameters: a signature is produced only when every non-{@code self} parameter contributes a
 * {@link com.ibm.wala.cast.python.ml.types.TensorType}; if any parameter blocks, the whole result is {@link Absent}. The degenerate case of
 * a function with no non-{@code self} parameter (parameterless or {@code self}-only) is a contract violation of
 * {@link Function#inferInputSignature} (every refactoring call site is already gated on {@link Function#getHasTensorParameter}), so it
 * throws rather than producing a result. A non-tensor parameter is not degenerate: it yields an {@link Absent}.
 *
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/483">Issue 483</a>
 */
public sealed interface InferenceResult {

	/**
	 * The reason an input signature could not be inferred. {@link #SPECULATIVE_TENSOR_PARAMETER} blocks the whole function up front; the
	 * remaining constants each correspond to one blocking branch of the per-parameter dispatch in {@link Function#inferInputSignature}, in
	 * the order that dispatch encounters them.
	 */
	enum AbsenceReason {

		/**
		 * The function's tensor-parameter classification came from speculative context analysis (its name, and for a functor its
		 * {@code Model} inheritance) rather than from any parameter. Speculation identifies no particular parameter and carries no
		 * shape/dtype evidence, so no spec can be synthesized. Blocks the function before the per-parameter dispatch runs, hence no entry
		 * in {@link Function#getBlockingParameterReasons()} (#783).
		 */
		SPECULATIVE_TENSOR_PARAMETER,

		/**
		 * A parameter is not classified as tensor-typed by any phase and declares no default, so it is a <em>required</em> argument for
		 * which {@code tf.function(input_signature=...)} demands a {@code TensorSpec} that cannot be synthesized (#508). A non-tensor
		 * parameter that declares a default is not this case: TensorFlow requires a spec per required argument only, so it may be omittable
		 * instead ({@link #DEFAULTED_PARAMETER_SUPPLIED}, {@link #DEFAULTED_PARAMETER_PRECEDES_SPEC}, #787).
		 */
		NON_TENSOR_PARAMETER,

		/**
		 * A parameter declares a default and is not tensor-typed, so it could be omitted from the signature, except that some call site
		 * supplies an argument for it. Omitting it would drop the hybridized function's arity below what that call site passes, raising a
		 * {@code TypeError} there, so the parameter must be covered and no spec exists for it. Also reported when the call sites cannot be
		 * examined at all, since ignorance must not be read as evidence of absence (#787).
		 */
		DEFAULTED_PARAMETER_SUPPLIED,

		/**
		 * A parameter declares a default, is not tensor-typed, and no call site supplies it, yet it still cannot be omitted because a later
		 * parameter contributes a spec. {@code input_signature} covers a prefix of the parameter list positionally, so dropping this one
		 * would bind the later parameter's spec to this position. Arises for the {@code def call(self, x, training=False, mask=None)}
		 * shape, where {@code mask} is tensor-typed (#787).
		 */
		DEFAULTED_PARAMETER_PRECEDES_SPEC,

		/**
		 * A parameter is a container of tensors (classified by {@link Parameter#hasTensorContainer}) of a form the sequence reduction does
		 * not model (#781 models a list or tuple of tensors with a constant element structure): a dict or set, a container whose element
		 * structure is not a contiguous run of constant indices (e.g. built by an append loop the analysis cannot enumerate), an element
		 * that is itself a container, an element position with no tensor evidence, or a parameter whose call sites mix container and
		 * non-container values. Distinct from {@link #TYPE_HINT_WITHOUT_DTYPE} in what is recoverable: the analysis may hold more about the
		 * elements than the reduction consumes, so this can narrow further as more forms are modeled.
		 */
		TENSOR_CONTAINER_UNSUPPORTED,

		/**
		 * A sequence-of-tensors parameter whose element count disagrees across the container values reaching it, so its arity set has size
		 * {@code > 1}. No single nested spec admits both: TensorFlow enforces the declared structure, raising {@code ValueError} for a
		 * sequence of a different length, and no wildcard arity exists—the same {@code |X| ≠ 1 ⇒ ⊥} discipline the dtype and sparseness
		 * axes use (#781).
		 */
		HETEROGENEOUS_ARITY,

		/**
		 * A parameter is classified as tensor-typed by its type hint alone (Phase 1), with no call-site evidence. A bare
		 * {@code x: tf.Tensor} annotation carries no dtype, and {@code tf.function(input_signature=...)} admits no dtype-⊤ (#494), so no
		 * valid spec exists to synthesize from this signal. Unlike {@link #TENSOR_CONTAINER_UNSUPPORTED} the evidence is genuinely absent
		 * rather than discarded, so no tool-side recovery is possible; supplying it is a source-side change (pass {@code tf.constant(...)}
		 * at the call sites).
		 */
		TYPE_HINT_WITHOUT_DTYPE,

		/**
		 * A parameter receives tensors with conflicting dtypes across call sites, so its dtype set has size {@code > 1} ({@code |D| ≠ 1}).
		 */
		HETEROGENEOUS_DTYPE,

		/**
		 * A parameter receives a tensor whose dtype cannot be determined (dtype-⊤: a single agreed {@code UNKNOWN}), which is not a valid
		 * runtime dtype for {@code tf.function(input_signature=...)} (#494).
		 */
		UNKNOWN_DTYPE,

		/**
		 * A parameter is sparse at some call sites and dense at others. A {@code SparseTensorSpec} admits only sparse tensors and a dense
		 * {@code TensorSpec} admits only dense tensors, so no single spec accepts both layouts; emitting either would reject traffic the
		 * function accepts, so the reduction is bottom (#642). Checked after the dtype axis, mirroring {@link Function#inferSpec}.
		 */
		HETEROGENEOUS_SPARSITY
	}

	/**
	 * A successfully inferred input signature.
	 *
	 * @param inputSignature The inferred signature; never {@code null}.
	 */
	record Inferred(InputSignature inputSignature) implements InferenceResult {
	}

	/**
	 * Inference was blocked, either at the function level ({@link AbsenceReason#SPECULATIVE_TENSOR_PARAMETER}, which emits a single
	 * function-level diagnostic INFO) or by one or more parameters, in which case this carries the first {@link AbsenceReason} encountered
	 * while dispatching over them in declaration order and every blocking parameter emits its own diagnostic INFO.
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
