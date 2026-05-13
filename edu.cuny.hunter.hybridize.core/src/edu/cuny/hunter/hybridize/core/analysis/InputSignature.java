package edu.cuny.hunter.hybridize.core.analysis;

import java.util.List;
import java.util.Objects;

import com.ibm.wala.cast.python.ml.types.TensorType;

/**
 * The inferred input signature of a hybridizable function: an ordered tuple of {@link TensorType}s, one per non-{@code self}, non-excluded
 * parameter, in declaration order. Produced by {@link Function#inferInputSignature}, which implements Algorithm 2 of the approach.
 * <p>
 * The list is unmodifiable. The constructor copies its argument defensively.
 */
public record InputSignature(List<TensorType> parameterTypes) {

	public InputSignature {
		Objects.requireNonNull(parameterTypes, "parameterTypes must not be null");
		parameterTypes = List.copyOf(parameterTypes);
	}
}
