package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.KERAS_MODEL_TYPE;

import java.util.Optional;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;

/**
 * The layer-build analysis behind the {@link PreconditionFailure#MAY_BUILD_LAYERS} safety precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1172): a transitive walk over the call graph deciding whether a
 * function, or anything it calls, may create a Keras layer's variables through {@code Layer.add_weight}, and a points-to query deciding
 * whether the function's receiver is a Keras model. A layer creates its variables on its first call, when its {@code __call__} runs its
 * {@code build}. Under {@code tf.function}, that build runs inside the trace, outside the name scopes of the layers that enclose it, so
 * sibling layers' variables get the same names, and a save by name, such as to HDF5, fails. A Keras model's {@code call} is exempt:
 * decorating it drops the enclosing scopes from its weights' names too, but the save succeeds, whether or not the model is nested in
 * another one. So is a function that is no method, such as a training step: it runs inside no layer's name scope, and a model it calls
 * enters its own scope inside the trace.
 * <p>
 * Both facts come from the engine. {@link TensorFlowApis} names the call of {@code tensorflow.keras.layers.Layer.add_weight} both in a
 * program layer's {@code build} and in the engine's summary of a built-in layer's first call (wala/ML#1145). The latter lies inside a
 * TensorFlow library node, so the walk of {@link Util#reachesInstruction(CallGraph, CGNode, java.util.function.BiPredicate, boolean)} tests
 * library nodes too. The receiver's class is {@link com.ibm.wala.cast.python.util.Util#allocatedClass(InstanceKey)} of each object it may
 * be (wala/ML#1165), whose bases {@link Function#getClassAndBaseNames(IClassHierarchy, com.ibm.wala.types.TypeReference)} walks
 * transitively, through every declared Python base and every superclass, for the engine's modeled {@code tf.keras.Model}. The class
 * hierarchy's own subtype test would not do: the engine records a single superclass per class, so {@code class Net(Mixin, tf.keras.Model)}
 * would read as no model in either order of bases. In the engine's model, unlike at run time, {@code Model} does not extend {@code Layer},
 * so the walk names a model and not a layer.
 * <p>
 * It is a may-fact: a call graph can't tell which call to a layer is its first, and the engine does not model a layer's {@code built}
 * state, so a function calling a layer that an earlier call has already built is refused as well.
 */
class LayerBuildAnalysis {

	/** The fully-qualified name of {@code Layer.add_weight}, as the engine names it ({@link TensorFlowApis}). */
	private static final String ADD_WEIGHT_API = "tensorflow.keras.layers.Layer.add_weight";

	/** The call graph, whose successors the walk follows. */
	private final CallGraph callGraph;

	/** The call-graph builder, which the engine asks for the APIs a call invokes, the receiver's objects and the class hierarchy. */
	private final PythonSSAPropagationCallGraphBuilder builder;

	/**
	 * Creates the analysis.
	 *
	 * @param callGraph The call graph.
	 * @param builder The call-graph builder.
	 */
	LayerBuildAnalysis(CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder) {
		this.callGraph = callGraph;
		this.builder = builder;
	}

	/**
	 * True iff {@code node}, transitively over its call-graph successors, may call {@code Layer.add_weight}, whether in a program body or
	 * in a TensorFlow library node.
	 *
	 * @param node The call-graph node to check.
	 * @return True iff a layer's variable creation may be reachable from {@code node}.
	 */
	boolean mayBuildLayers(CGNode node) {
		return Util.reachesInstruction(this.callGraph, node, (n, instruction) -> instruction instanceof SSAAbstractInvokeInstruction invoke
				&& TensorFlowApis.of(this.builder, n, invoke).contains(ADD_WEIGHT_API), true);
	}

	/**
	 * Whether {@code node}'s receiver, the {@code self} of an instance method, is a Keras model: true iff the engine allocated every object
	 * the receiver may be by a constructor of a class reaching {@code tf.keras.Model} through its Python bases, false iff it allocated
	 * every one by a constructor of another class, and {@code null}, unknown, otherwise. Unknown covers a receiver with an object of
	 * unknown class, one mixing models with other objects, one with no object, and a function that is no instance method. A base the engine
	 * does not model is not recorded, so a class reaching no model may still be one; it reads as no model, and its method is refused, which
	 * is the conservative side.
	 *
	 * @param node The call-graph node to check.
	 * @param method True iff the function is an instance method, whose second parameter slot is its receiver.
	 * @return True iff the receiver is a Keras model, false iff it is not, {@code null} if unknown.
	 */
	Boolean receiverIsModel(CGNode node, boolean method) {
		if (!method)
			return null;

		IR ir = node.getIR();

		if (ir == null)
			return null;

		// Parameter value numbers: slot 0 is the function object itself; slot 1 is the receiver of an instance method.
		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();

		if (parameters.length < 2)
			return null;

		IClassHierarchy classHierarchy = this.builder.getClassHierarchy();

		if (classHierarchy.lookupClass(KERAS_MODEL_TYPE) == null)
			return null;

		PointerKey receiver = this.builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, parameters[1]);
		boolean anyModel = false;
		boolean anyOther = false;

		for (InstanceKey instanceKey : this.builder.getPointerAnalysis().getPointsToSet(receiver)) {
			Optional<IClass> allocatedClass = com.ibm.wala.cast.python.util.Util.allocatedClass(instanceKey);

			// An object of unknown class may be a model or not.
			if (allocatedClass.isEmpty())
				return null;

			if (Function.getClassAndBaseNames(classHierarchy, allocatedClass.get().getReference()).contains(KERAS_MODEL_TYPE.getName()))
				anyModel = true;
			else
				anyOther = true;
		}

		// No object, or models mixed with other objects, is unknown.
		if (anyModel == anyOther)
			return null;

		return anyModel;
	}
}
