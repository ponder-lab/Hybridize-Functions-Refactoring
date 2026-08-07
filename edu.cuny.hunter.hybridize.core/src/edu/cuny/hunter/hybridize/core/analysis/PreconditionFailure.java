package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Arrays;

public enum PreconditionFailure {
	CURRENTLY_NOT_HANDLED(1),

	/**
	 * Either there is no call to the function, there is a call but don't handle it, or something about decorators?.
	 */
	UNDETERMINABLE_SIDE_EFFECTS(3),

	HAS_PYTHON_SIDE_EFFECTS(4),

	HAS_NO_TENSOR_PARAMETERS(6),

	HAS_TENSOR_PARAMETERS(7),

	/**
	 * Functions that are recursive can't be hybridized. Also, de-hybridizing hybrid recursive functions may alter semantics.
	 */
	IS_RECURSIVE(8),

	/**
	 * Can't find the CG node corresponding to the function.
	 */
	CANT_APPROXIMATE_RECURSION(9),

	/**
	 * Either there is no call to the function, there is a call but don't handle it, or something about decorators?.
	 */
	UNDETERMINABLE_TENSOR_PARAMETER(10),

	/**
	 * We need a call graph node.
	 */
	UNDETERMINABLE_PRIMITIVE_PARAMETER(11),

	HAS_PRIMITIVE_PARAMETERS(12),

	/**
	 * P3 failure.
	 */
	HAS_NO_PRIMITIVE_PARAMETERS(13),

	/**
	 * The function's body performs no (transitive) TensorFlow tensor computation, so graph execution yields no benefit and hybridization is
	 * unnecessary. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/709.
	 */
	NO_TENSOR_COMPUTATION(14),

	/**
	 * The function's body (transitively) invokes an eager-only API (e.g. {@code Tensor.numpy()}), which raises under {@code tf.function}
	 * tracing, so hybridization would not preserve semantics. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/363.
	 */
	HAS_EAGER_ONLY_CALLS(15),

	/**
	 * The function's body (transitively) applies a numpy/scipy API to a value flowing from its parameters, which raises under
	 * {@code tf.function} tracing once the parameters become symbolic tensors, so hybridization would not preserve semantics. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/740.
	 */
	HAS_NUMPY_CALLS_ON_PARAMETERS(16),

	/**
	 * The function's body passes a non-string constant where a TensorFlow API declares its {@code name} parameter (e.g.,
	 * {@code tf.sqrt(x, tf.float32)}). Eager execution never validates the name, but {@code tf.function} tracing opens a name scope with it
	 * and raises, so hybridization would not preserve semantics. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/814.
	 */
	HAS_INVALID_NAME_ARGUMENTS(17),

	/**
	 * The inferred input signature leaves unresolved (wildcard) a parameter axis that the function's body (transitively) reads statically
	 * and consumes where a Python integer is required (a weight shape, a reshape target, or integer arithmetic). Under the emitted
	 * signature such an axis is {@code None} at trace time, so the consumption raises or silently misbehaves; a dynamic read
	 * ({@code tf.shape(x)[i]}) is safe and does not disqualify. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/811. Since
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/864, this failure is emitted only on the reconfiguration path,
	 * where replacing an existing signature is the only action available; the conversion path withholds the unwritable signature
	 * ({@code InferenceResult.AbsenceReason#WITHHELD_STATICALLY_READ_AXES}) and hybridizes with a bare decorator, matching the tool's
	 * behavior with inference off.
	 */
	HAS_UNRESOLVED_STATICALLY_READ_AXES(18),

	/**
	 * The function snapshots a model's variable collection ({@code trainable_variables}/{@code trainable_weights}) before the model's first
	 * invocation in its body and feeds the snapshot to an optimizer or gradient computation. A subclassed Keras model's collection is
	 * silently empty before its first build, so under tracing the initial trace captures the empty snapshot, the in-trace build engages
	 * {@code tf.function}'s variable-lifting re-trace, and optimizer slot creation lands on a non-first trace, raising the
	 * singleton-variable {@code ValueError}. Reading the collection after the forward pass (the pervasive beneficial idiom) is untouched:
	 * the ordering is the discriminator. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/822.
	 */
	HAS_STALE_VARIABLE_READS(19),

	/**
	 * The function's body iterates a parameter-derived, tensor-typed value with a Python {@code for}. Eagerly the elements are tensors and
	 * the loop runs; under {@code tf.function} tracing the parameter is symbolic, and iterating a symbolic tensor raises
	 * {@code OperatorNotAllowedInGraphError} even with AutoGraph converting the loop. In-body {@code tf.range} loops are
	 * AutoGraph-supported and do not fire. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/830.
	 */
	HAS_TENSOR_PARAMETER_ITERATION(20),

	/**
	 * Every known call path to the function comes from hybridized code (the least-fixpoint caller coverage of issue 767), so its
	 * computation is already traced on every executed path and adding {@code tf.function} contributes only a redundant nested trace
	 * boundary. A benefit precondition with the allow-on-unknown polarity: unknown, module-level, or uncovered callers leave the function
	 * convertible, and only a determinate {@code TRUE} coverage blocks. Promoted from the phase-1 advisory on corpus evidence (four covered
	 * functions, each source-verified; the dead-caller semantics pinned first). See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/826.
	 */
	HAS_COVERED_CALLERS(21),

	/**
	 * A parameter's direct consumers impose more than one concrete eager-effective dtype (e.g., {@code W32 * x} beside {@code V64 * x}).
	 * Eager execution coerces a NumPy or Python argument at each op under the other operand's dtype, so the program runs; traced, the
	 * argument materializes at one dtype at the boundary, and any single {@code input_signature} breaks at least one op, so hybridization
	 * is declined. The singleton counterpart repairs instead of declining: the spec pins the one eager-effective dtype, whose boundary cast
	 * reproduces eager coercion (runtime-verified on the pinned TF 2.9.3). See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/861, Case 1.
	 */
	HAS_CONFLICTING_EAGER_DTYPE_COERCIONS(22);

	static {
		// check that the codes are unique.
		assert Arrays.stream(PreconditionFailure.values()).map(PreconditionFailure::getCode).distinct()
				.count() == PreconditionFailure.values().length : "Codes must be unique.";
	}

	public static void main(String[] args) {
		System.out.println("code,name");
		for (PreconditionFailure failure : PreconditionFailure.values())
			System.out.println(failure.getCode() + "," + failure);
	}

	private int code;

	private PreconditionFailure(int code) {
		this.code = code;
	}

	public int getCode() {
		return this.code;
	}
}
