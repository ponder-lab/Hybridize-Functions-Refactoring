package edu.cuny.hunter.hybridize.core.analysis;

import java.util.List;
import java.util.Objects;

import com.ibm.wala.cast.python.ml.types.TensorType;

/**
 * The inferred input signature of a hybridizable function: an ordered tuple of {@link TensorType}s in declaration order, one per
 * non-{@code self} parameter. Produced by {@link Function#inferInputSignature}.
 * <p>
 * `InputSignature` is total over the function's non-{@code self} parameters: every non-{@code self} parameter contributes a
 * {@link TensorType}, or {@link Function#inferInputSignature} returns {@link java.util.Optional#empty} instead of producing a partial
 * signature. The per-parameter dispatch (tensor-classified-with-evidence vs tensor-classified-without-evidence vs truly non-tensor) and its
 * diagnostics live in {@link Function#inferInputSignature}; see that method's Javadoc and #508 for the rules.
 * <p>
 * The list is unmodifiable. The constructor copies its argument defensively.
 */
public record InputSignature(List<TensorType> parameterTypes) {

	public InputSignature {
		Objects.requireNonNull(parameterTypes, "parameterTypes must not be null");
		parameterTypes = List.copyOf(parameterTypes);
	}
}
