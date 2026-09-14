package edu.cuny.hunter.hybridize.core.analysis;

/**
 * What a parameter's tensor classification rests on, recorded alongside the verdict so that the verdict can be audited.
 * <p>
 * The verdict alone cannot be: {@link Parameter#classifyAsTensor} writes {@code FALSE} both when the analysis ran and concluded that the
 * parameter is not a tensor, and when the analysis did not run at all and the field took its default. Those two call for opposite work. A
 * determined non-tensor is outside the mechanism, since {@code tf.function(input_signature=...)} must cover every argument and no
 * {@code TensorSpec} describes a non-tensor, so no engine change recovers it. A parameter whose classification never ran may well be a
 * tensor, and the resulting absence is a precision limit rather than a property of the program.
 * <p>
 * Because a signature covers every parameter or none, a single such parameter suppresses a whole function's signature. Reading a default as
 * a determination therefore writes off an entire function as unreachable on the strength of a question that was never asked. See
 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/971.
 */
public enum TensorClassificationBasis {

	/** The parameter is {@code self}, which is never treated as a tensor. A determination, decided on the declaration alone. */
	SELF,

	/** A tensor type hint on the declaration classified the parameter (Phase 1). A determination. */
	TENSOR_TYPE_HINT,

	/** The tensor type analysis associated at least one {@code TensorType} with the parameter (Phase 2). A determination. */
	TENSOR_ANALYSIS,

	/** The parameter was classified as a container of tensors (Phase 3). A determination. */
	TENSOR_CONTAINER,

	/**
	 * Every phase ran and none classified the parameter as tensor-like. Under contract-compliant generators an empty tensor-type result is
	 * the analysis's bottom, meaning <em>not a tensor</em> rather than <em>nothing known</em>, so this is a determination and the absence
	 * it produces is outside the mechanism.
	 */
	ANALYZED_NOT_TENSOR,

	/**
	 * The enclosing function is absent from the call graph, so the tensor-type and container phases were <em>skipped</em> and the verdict
	 * took its default. Deliberately names the phases not running rather than a negative result: nothing was concluded about this
	 * parameter, and reading it as weak evidence of a non-tensor would repeat the conflation this enum exists to end.
	 */
	CLASSIFICATION_DID_NOT_RUN
}
