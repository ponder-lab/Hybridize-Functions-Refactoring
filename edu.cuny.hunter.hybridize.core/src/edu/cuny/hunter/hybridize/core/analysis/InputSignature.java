package edu.cuny.hunter.hybridize.core.analysis;

import java.util.List;
import java.util.Objects;

import com.ibm.wala.cast.python.ml.types.TensorType;

/**
 * The inferred input signature of a hybridizable function: an ordered tuple of {@link TensorType}s, one per non-{@code self} parameter the
 * tensor-type analysis associated with at least one tensor type, in declaration order. Produced by {@link Function#inferInputSignature}.
 * <p>
 * Parameters the analysis classified as non-tensor (empty {@link Parameter#getTensorTypes()} set) are excluded from
 * {@link #parameterTypes}—the signature is by definition tensor-only. If all parameters are non-tensor, the function has no signature to
 * infer and {@link Function#inferInputSignature} returns {@link java.util.Optional#empty} rather than producing an empty
 * {@link InputSignature}.
 * <p>
 * The list is unmodifiable. The constructor copies its argument defensively.
 */
public record InputSignature(List<TensorType> parameterTypes) {

	public InputSignature {
		Objects.requireNonNull(parameterTypes, "parameterTypes must not be null");
		parameterTypes = List.copyOf(parameterTypes);
	}
}
