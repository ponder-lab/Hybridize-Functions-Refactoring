package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Map;
import java.util.Set;

import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;

/**
 * The Keras symbolic-argument analysis behind the precondition of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/887:
 * whether some call of a function passes a Keras <em>symbolic</em> tensor (a {@code KerasTensor}, the value {@code tf.keras.layers.Input}
 * produces and the Functional API threads through layer applications). {@code tf.function} is one of the APIs {@code KerasTensor}
 * explicitly refuses, so decorating such a function turns a working program into one that raises a {@code TypeError} on the first call,
 * before anything is traced. The hazard is independent of any {@code input_signature}: a bare decorator fails the same way.
 * <p>
 * Whether a value is symbolic is the engine's fact, not this analysis's: Ariadne's tensor-type analysis records it as the
 * {@link TensorOrigin#KERAS_SYMBOLIC} origin (https://github.com/wala/ML/issues/1101). The origin is seeded at {@code tf.keras.Input} and
 * carried across parameter barriers, type feeds, and TensorFlow calls, which covers a layer's or an operation's application to a symbolic
 * value, a value returned from a user function, and one read back out of a container, in whatever frame the {@code Input} was created
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1097). It is what the earlier, syntactic provenance walk over the
 * caller's frame approximated, and it replaces that walk.
 * <p>
 * The origin is read on the function's own parameters rather than on its callers' arguments. The engine carries it across the parameter
 * barrier exactly when some call passes the parameter a symbolic value, so a parameter holding it is the existential over call sites the
 * hazard needs (one call passing a {@code KerasTensor} is enough to break the decorated function), already resolved through keyword,
 * positional, and trampoline binding.
 * <p>
 * The origin is a may-fact: a value that is symbolic on only one path of a merge carries it, so it now declines, where the earlier walk
 * left a path-dependent argument allowing. Declining is the safe direction, and it is pinned in the tests as the engine's current answer
 * rather than corrected here. A layer's own {@code call} receives a placeholder rather than a {@code KerasTensor}, and the engine puts the
 * origin on the layer application's result rather than into {@code call}'s parameters (https://github.com/wala/ML/issues/1109), so a
 * function a layer body calls converts, and decorating it runs.
 * <p>
 * Ignorance is {@code null} rather than {@code FALSE}, the polarity of every sibling safety check: a node without IR leaves the verdict
 * undetermined, and only a determinate {@code TRUE} declines. A parameter the tensor-type analysis does not type carries no origin, which
 * is a determinate {@code FALSE}: the engine has no evidence that it is symbolic.
 */
class KerasSymbolicArgumentAnalysis {

	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/** The tensor-typed pointer keys mapped to their origins (see {@link Util#computeTensorTypedOrigins}). */
	private final Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys;

	KerasSymbolicArgumentAnalysis(PointerAnalysis<InstanceKey> pointerAnalysis, Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys) {
		this.pointerAnalysis = pointerAnalysis;
		this.tensorTypedKeys = tensorTypedKeys;
	}

	/**
	 * Whether some call of {@code nodes} passes a Keras symbolic tensor.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @return {@code TRUE} if some parameter carries the Keras symbolic origin, {@code FALSE} if none does, or {@code null} if a node has
	 *         no IR.
	 */
	Boolean hasKerasSymbolicArgument(Set<CGNode> nodes) {
		boolean symbolic = false;

		for (CGNode node : nodes) {
			IR ir = node.getIR();

			if (ir == null)
				// Undeterminable: this node's parameters are invisible, so no verdict is available for the function.
				return null;

			int[] parameters = ir.getSymbolTable().getParameterValueNumbers();

			// Slot 0 is the function itself, which no call passes as an argument.
			for (int i = 1; i < parameters.length; i++) {
				PointerKey key = this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, parameters[i]);
				Set<TensorOrigin> origins = this.tensorTypedKeys.get(key);

				if (origins != null && origins.contains(TensorOrigin.KERAS_SYMBOLIC))
					symbolic = true;
			}
		}

		return symbolic;
	}
}
