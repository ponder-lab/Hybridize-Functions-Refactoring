package edu.cuny.hunter.hybridize.core.analysis;

import java.util.HashSet;
import java.util.Set;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Iterator2Iterable;

/**
 * Finds the values a program exports through a TensorFlow interface that outlives it: a SavedModel written by {@code tf.saved_model.save},
 * or a TensorFlow Lite model converted by {@code tf.lite.TFLiteConverter}. A function exported this way has its input signature fixed by
 * the exported interface as well as by its callers, so narrowing a supplied signature there is declined even under the closed-world
 * assumption (issue 808).
 * <p>
 * Only true exports count. A {@code get_concrete_function} call that only forces a trace fixes no external interface, so it is not an
 * export unless its result reaches a TensorFlow Lite conversion.
 * <p>
 * The analysis does not rely on call-graph edges into the export APIs, which the TensorFlow summaries do not model. It recognizes each call
 * in a reachable node by its attribute names ({@code saved_model.save}, {@code TFLiteConverter.from_concrete_functions},
 * {@code TFLiteConverter.from_keras_model}) and reads what each argument points to. When an export call's exported argument points to
 * nothing the analysis can see, as for an instance of a class the summaries do not model, what it exports is unknown, and every function
 * not otherwise found exported is reported as possibly exported. A {@code save} call on another receiver is recognized as a Keras model
 * saving itself only when the receiver resolves, since such calls are common on other objects.
 */
public class ExportAnalysis {

	/** The attribute naming the SavedModel module. */
	private static final String SAVED_MODEL = "saved_model";

	/** The SavedModel export function. */
	private static final String SAVE = "save";

	/** The Keras model-saving function, reached as {@code tf.keras.models.save_model}. */
	private static final String SAVE_MODEL = "save_model";

	/** The attribute naming the Keras models module. */
	private static final String MODELS = "models";

	/** The prefix the front end gives a global variable's name. */
	private static final String GLOBAL_PREFIX = "global ";

	/** The attribute naming the TensorFlow Lite converter class. */
	private static final String TFLITE_CONVERTER = "TFLiteConverter";

	/** The converter factory taking concrete functions. */
	private static final String FROM_CONCRETE_FUNCTIONS = "from_concrete_functions";

	/** The converter factory taking a Keras model. */
	private static final String FROM_KERAS_MODEL = "from_keras_model";

	/** The field in which the TensorFlow summaries' {@code tf.function} wrapper holds the function it wraps. */
	private static final String FUNC_FIELD = "func";

	/** The method turning a {@code tf.function} into a concrete function. */
	private static final String GET_CONCRETE_FUNCTION = "get_concrete_function";

	/** The pointer analysis. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/**
	 * The objects a SavedModel or a Keras-model conversion exports whole: every function stored on one of them is part of the exported
	 * interface. The front end stores an instance's methods, inherited ones included, on the instance, alongside any function assigned to
	 * one of its attributes.
	 */
	private final Set<InstanceKey> exportedObjects = new HashSet<>();

	/** The values exported as functions: a SavedModel's {@code signatures}, and the functions a concrete-function conversion was given. */
	private final Set<InstanceKey> exportedFunctions = new HashSet<>();

	/** The exported {@code signatures} containers, whose elements are exported functions too. */
	private final Set<InstanceKey> exportedContainers = new HashSet<>();

	/**
	 * The types of the exported functions' own objects, resolved once after the scan: each exported function, each function a
	 * {@code tf.function} wrapper among them wraps, each element of an exported container, and each function stored on an object exported
	 * whole.
	 */
	private final Set<String> exportedFunctionTypes = new HashSet<>();

	/** Whether some explicit export call's exported argument points to nothing the analysis can see, so what it exports is unknown. */
	private boolean unresolved;

	/**
	 * Scans every node of the call graph for export calls.
	 *
	 * @param callGraph The call graph.
	 * @param pointerAnalysis The pointer analysis.
	 */
	public ExportAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		this.pointerAnalysis = pointerAnalysis;

		for (CGNode node : callGraph) {
			IR ir = node.getIR();

			if (ir == null)
				continue;

			DefUse defUse = node.getDU();

			for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
				if (instruction instanceof PythonInvokeInstruction invoke)
					this.scan(node, invoke, defUse);
		}

		this.resolveTypes();
	}

	/**
	 * Resolves the exported instances to the types {@link #isExported} compares against. A single pass over the instance fields reaches
	 * both the function a {@code tf.function} wrapper holds in its {@code func} field (the TensorFlow summaries model
	 * {@code tf.function(fn, ...)} that way), the elements of an exported {@code signatures} container, and the functions stored on an
	 * object exported whole.
	 */
	private void resolveTypes() {
		for (InstanceKey instance : this.exportedFunctions)
			this.exportedFunctionTypes.add(typeName(instance.getConcreteType().getReference()));

		if (this.exportedFunctions.isEmpty() && this.exportedContainers.isEmpty() && this.exportedObjects.isEmpty())
			return;

		for (PointerKey key : this.pointerAnalysis.getPointerKeys())
			if (key instanceof InstanceFieldPointerKey field) {
				InstanceKey owner = field.getInstanceKey();
				boolean wrapper = this.exportedFunctions.contains(owner) && key instanceof InstanceFieldKey named
						&& FUNC_FIELD.equals(named.getField().getName().toString());

				// A function stored as an attribute of an object saved whole (for example, `module.serve = serve`) is exported with it. So
				// is a method the object's class inherits, which the front end stores on the instance.
				if (wrapper || this.exportedContainers.contains(owner) || this.exportedObjects.contains(owner))
					for (InstanceKey value : this.pointerAnalysis.getPointsToSet(key))
						this.exportedFunctionTypes.add(typeName(value.getConcreteType().getReference()));
			}
	}

	/**
	 * Records what an export call exports, if {@code invoke} is one.
	 *
	 * @param node The node containing the call.
	 * @param invoke The call.
	 * @param defUse The node's def-use information.
	 */
	private void scan(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		String member = this.memberName(node, defUse, invoke.getUse(0));
		String receiver = this.receiverName(node, defUse, invoke.getUse(0));

		if (SAVE.equals(member) && SAVED_MODEL.equals(receiver)) {
			// tf.saved_model.save(obj, export_dir, signatures=None, options=None)
			this.addPointsToOrFlag(node, argument(invoke, 1, "obj"), this.exportedObjects);
			this.addSignatures(node, argument(invoke, 3, "signatures"));
		} else if (SAVE.equals(member)) {
			// model.save(filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None, ...): a Keras
			// model saving itself. Any other object with a `save` method is swept in too, which only declines a narrowing.
			if (defUse.getDef(invoke.getUse(0)) instanceof PythonPropertyRead read)
				this.addPointsTo(node, read.getObjectRef(), this.exportedObjects);

			this.addSignatures(node, argument(invoke, 5, "signatures"));
		} else if (SAVE_MODEL.equals(member) && MODELS.equals(receiver)) {
			// tf.keras.models.save_model(model, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, ...)
			this.addPointsToOrFlag(node, argument(invoke, 1, "model"), this.exportedObjects);
			this.addSignatures(node, argument(invoke, 6, "signatures"));
		} else if (FROM_KERAS_MODEL.equals(member) && TFLITE_CONVERTER.equals(receiver))
			// tf.lite.TFLiteConverter.from_keras_model(model)
			this.addPointsToOrFlag(node, argument(invoke, 1, "model"), this.exportedObjects);
		else if (FROM_CONCRETE_FUNCTIONS.equals(member) && TFLITE_CONVERTER.equals(receiver)) {
			// tf.lite.TFLiteConverter.from_concrete_functions(funcs, trackable_obj=None): each element is the result of a
			// `get_concrete_function` call, whose receiver is the exported function.
			int funcs = argument(invoke, 1, "funcs");

			if (funcs != -1)
				for (int element : elementValues(defUse, funcs))
					if (!this.addConcreteFunctionReceiver(node, defUse, element))
						this.unresolved = true;
		}
	}

	/**
	 * Records a {@code signatures} argument: what it points to, and the entries of a container it is (such as a dictionary). A concrete
	 * function passed there needs no unwrapping: the summaries' {@code get_concrete_function} returns the function it is called on.
	 *
	 * @param node The node.
	 * @param signatures The argument value, or {@code -1} for none.
	 */
	private void addSignatures(CGNode node, int signatures) {
		if (signatures != -1 && !this.addPointsToAndElements(node, signatures, this.exportedFunctions))
			this.unresolved = true;
	}

	/**
	 * Records the receiver of the {@code get_concrete_function} call defining {@code value}, if it is defined by one.
	 *
	 * @param node The node.
	 * @param defUse The node's def-use information.
	 * @param value The value.
	 * @return True iff the receiver points to something.
	 */
	private boolean addConcreteFunctionReceiver(CGNode node, DefUse defUse, int value) {
		return defUse.getDef(value) instanceof PythonInvokeInstruction call
				&& GET_CONCRETE_FUNCTION.equals(this.memberName(node, defUse, call.getUse(0)))
				&& defUse.getDef(call.getUse(0)) instanceof PythonPropertyRead read
				&& this.addPointsTo(node, read.getObjectRef(), this.exportedFunctions);
	}

	/**
	 * The values written into the list or tuple literal {@code container}, or {@code container} itself when it is not written into in this
	 * node (for example, a single concrete function passed without a list).
	 *
	 * @param defUse The node's def-use information.
	 * @param container The container value.
	 * @return The element values.
	 */
	private static Set<Integer> elementValues(DefUse defUse, int container) {
		Set<Integer> ret = new HashSet<>();

		for (SSAInstruction use : Iterator2Iterable.make(defUse.getUses(container)))
			if (use instanceof PythonPropertyWrite write && write.getObjectRef() == container)
				ret.add(write.getValue());

		if (ret.isEmpty())
			ret.add(container);

		return ret;
	}

	/**
	 * The member name of the attribute read defining {@code value}, or {@code null} when it is not defined by one or the name is not a
	 * string constant.
	 *
	 * @param node The node.
	 * @param defUse The node's def-use information.
	 * @param value The value.
	 * @return The member name, or {@code null}.
	 */
	private String memberName(CGNode node, DefUse defUse, int value) {
		return defUse.getDef(value) instanceof PythonPropertyRead read
				? Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis)
				: null;
	}

	/**
	 * The name of the receiver of the attribute read defining {@code value}: {@code saved_model} for {@code tf.saved_model.save}, whether
	 * the receiver is itself an attribute read or a variable, as after {@code from tensorflow import saved_model}.
	 *
	 * @param node The node.
	 * @param defUse The node's def-use information.
	 * @param value The value.
	 * @return The receiver's name, or {@code null}.
	 */
	private String receiverName(CGNode node, DefUse defUse, int value) {
		if (!(defUse.getDef(value) instanceof PythonPropertyRead read))
			return null;

		SSAInstruction receiver = defUse.getDef(read.getObjectRef());

		if (receiver instanceof PythonPropertyRead)
			return this.memberName(node, defUse, read.getObjectRef());

		if (receiver instanceof AstGlobalRead global) {
			String name = global.getGlobalName();
			return name.startsWith(GLOBAL_PREFIX) ? name.substring(GLOBAL_PREFIX.length()) : name;
		}

		if (receiver instanceof AstLexicalRead lexical) {
			Access[] accesses = lexical.getAccesses();
			return accesses.length > 0 ? accesses[0].getName().fst : null;
		}

		return null;
	}

	/**
	 * The value passed for a parameter, by position or by keyword, or {@code -1} when the call does not pass it.
	 *
	 * @param invoke The call.
	 * @param position The parameter's positional slot (the callee occupies slot 0).
	 * @param keyword The parameter's name.
	 * @return The argument value, or {@code -1}.
	 */
	private static int argument(PythonInvokeInstruction invoke, int position, String keyword) {
		if (invoke.getKeywords().contains(keyword))
			return invoke.getUse(keyword);

		if (invoke.getNumberOfPositionalParameters() > position)
			return invoke.getUse(position);

		return -1;
	}

	/**
	 * Adds what {@code value} points to.
	 *
	 * @param node The node.
	 * @param value The value, or {@code -1} for none.
	 * @param into The set to add to.
	 * @return True iff {@code value} points to something.
	 */
	private boolean addPointsTo(CGNode node, int value, Set<InstanceKey> into) {
		if (value == -1)
			return false;

		PointerKey key = this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, value);
		boolean any = false;

		for (InstanceKey instance : this.pointerAnalysis.getPointsToSet(key)) {
			into.add(instance);
			any = true;
		}

		return any;
	}

	/**
	 * Adds what an explicit export call's exported argument points to, recording that what the call exports is unknown when it points to
	 * nothing.
	 *
	 * @param node The node.
	 * @param value The value, or {@code -1} when the call does not pass it.
	 * @param into The set to add to.
	 */
	private void addPointsToOrFlag(CGNode node, int value, Set<InstanceKey> into) {
		if (value != -1 && !this.addPointsTo(node, value, into))
			this.unresolved = true;
	}

	/**
	 * Adds what {@code value} points to, recording any container among it (such as a {@code signatures} dictionary) so that its elements
	 * are resolved as exported functions too.
	 *
	 * @param node The node.
	 * @param value The value, or {@code -1} for none.
	 * @param into The set to add to.
	 * @return True iff {@code value} points to something.
	 */
	private boolean addPointsToAndElements(CGNode node, int value, Set<InstanceKey> into) {
		PointerKey key = this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, value);
		boolean any = false;

		for (InstanceKey instance : this.pointerAnalysis.getPointsToSet(key)) {
			if (Util.isContainerType(instance.getConcreteType().getReference()))
				this.exportedContainers.add(instance);
			else
				into.add(instance);

			any = true;
		}

		return any;
	}

	/**
	 * Whether a function is part of an exported interface: its own object, or a {@code tf.function} wrapping it, is among the exported
	 * functions, or is stored on an object exported whole.
	 *
	 * @param functionType The type of the function's own object, i.e., its declaring-class reference.
	 * @return True iff the function is exported, false iff it is not, and {@code null} when it is not found exported but some export call's
	 *         exported value could not be resolved.
	 */
	Boolean isExported(TypeReference functionType) {
		if (this.exportedFunctionTypes.contains(typeName(functionType)))
			return Boolean.TRUE;

		return this.unresolved ? null : Boolean.FALSE;
	}

	/**
	 * A type's name with the front end's method-object marker removed. A function's own object is typed by its declaring-class reference
	 * ({@code Lscript A.py/M/f}), while reading it as an attribute of an instance yields a method object whose type carries a {@code $}
	 * after the {@code L} ({@code L$script A.py/M/f}); both name the same function.
	 *
	 * @param type The type.
	 * @return The normalized name.
	 */
	private static String typeName(TypeReference type) {
		String name = type.getName().toString();
		return name.startsWith("L$") ? "L" + name.substring(2) : name;
	}
}
