package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.util.Util.findAllocationSiteInNode;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.modref.DelegatingExtendedHeapModel;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.ipa.modref.ModRef;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Iterator2Iterable;

import edu.cuny.hunter.hybridize.core.wala.ml.PythonModRefWithBuiltinFunctions;

/**
 * The analysis behind the precondition of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1076: whether a function
 * reads state that another function rebinds between calls. Measured on TensorFlow 2.9.3, a method that reads a tensor from a field of
 * {@code self}, converted on its own, returns the first call's state once another method has rebound the field: the first trace captures
 * the tensor the field held, and later calls reuse the trace.
 * <p>
 * The reads are those of the function's body and of every node it reaches in the call graph, which a trace of the function runs as well. A
 * read counts when the value it defines is typed as a tensor by the tensor-type analysis, which is the only source of tensor-ness here. The
 * fields a read reads, and the fields a body writes, are the pointer keys {@link ModRef} gives for its instructions, so a read and a write
 * are of the same state exactly when they are the same instance key and field. A write counts when it is made outside the reader's closure,
 * whose own writes are the side-effect precondition's, and outside the methods that set the written object up once: the {@code __init__} of
 * its class or a base, and the {@code build} of a Keras {@code Layer} or {@code Model} subclass, resolved through the engine's class
 * hierarchy from the class whose constructor allocated the object, and invoked on the object itself, per the points-to set of their
 * {@code self} parameter. A field set only there is read live. An {@code __init__} or {@code build} of another class, a {@code build}
 * outside Keras, or a set-up method writing a field of an object other than its receiver, runs whenever it is called, so its writes count.
 * So do the writes of a helper that only a set-up method calls, a known over-refusal: the helper may be called elsewhere too. A
 * {@code tf.Variable} updated with {@code assign} is not rebound, since {@code assign} writes the variable, not the field holding it.
 * <p>
 * The elements of a list, tuple, dictionary, or set are not read as state: {@code tf.function} passes a container argument's tensors in
 * afresh on each call, and the code building a container writes its elements as fields. A container held in a field is missed with them,
 * though its element reads are stale too once another function rebuilds it. Objects are told apart by allocation site, so instances
 * allocated at one site are one object here: a getter of an object created afresh on each call is refused, though no instance it reads is
 * ever rebound between two of its reads.
 * <p>
 * The verdict is the function's, whatever its callers are: the guide documents that a trace is keyed on the arguments, so a change to a
 * Python global or free variable, which a field read through {@code self} is, does not retrace, and the trace-time value persists.
 * <p>
 * Only the closure depends on the function. A node's tensor-typed reads, and the writers of each field, are the same for every function
 * that asks, so both are computed once and shared: the reads per node, and the writers of every field in one pass over the call graph, the
 * first time a function reads a tensor-typed field. Each function then looks up the writers of the fields it reads, rather than scanning
 * the whole call graph again.
 *
 * @see <a href="https://www.tensorflow.org/guide/function#rules_of_tracing">Better performance with tf.function: Rules of tracing</a>
 * @see <a href="https://www.tensorflow.org/guide/function#changing_python_global_and_free_variables">Better performance with tf.function:
 *      Changing Python global and free variables</a>
 */
class ReboundStateReadAnalysis {

	/**
	 * Per-node tensor-typed field reads (see {@link #tensorFieldsRead(CGNode)}), shared across functions since their closures revisit the
	 * same nodes. A node belongs to one call graph, and its project's tensor-typed keys are fixed, so an entry never goes stale. Concurrent
	 * because functions may be processed in parallel.
	 */
	private static final Map<CGNode, Set<PointerKey>> tensorFieldsReadCache = new ConcurrentHashMap<>();

	/**
	 * The writers of each instance field, per call graph: the nodes whose direct mod set (see {@link Function#getDirectMod}) has the field.
	 * Built once per call graph, by identity, in one pass over its nodes. Concurrent because functions may be processed in parallel.
	 */
	private static final Map<CallGraph, Map<PointerKey, Set<CGNode>>> fieldWritersCache = new ConcurrentHashMap<>();

	/** Clears the caches shared across functions. */
	static void clearCaches() {
		tensorFieldsReadCache.clear();
		fieldWritersCache.clear();
	}

	private final CallGraph callGraph;

	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	private final Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys;

	private final ModRef<InstanceKey> modRef = new PythonModRefWithBuiltinFunctions();

	private final ExtendedHeapModel heapModel;

	/** The set-up method types of each class, by the class's type name (see {@link #getSetUpMethodTypes(InstanceKey)}). */
	private final Map<TypeName, Set<TypeReference>> setUpMethodTypes = new HashMap<>();

	/**
	 * Creates the analysis.
	 *
	 * @param callGraph The call graph, walked from the reader to its callees, and once over all its nodes for the writers of each field.
	 * @param pointerAnalysis The pointer analysis, whose heap model names the fields read and written.
	 * @param tensorTypedKeys The pointer keys the tensor-type analysis types as tensors (see {@link Util#computeTensorTypedOrigins}).
	 */
	ReboundStateReadAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
		this.tensorTypedKeys = tensorTypedKeys;
		// The Python mod visitor requires the pipeline's own AstHeapModel; wrap only a heap model that isn't already extended.
		this.heapModel = pointerAnalysis.getHeapModel() instanceof ExtendedHeapModel ehm ? ehm
				: new DelegatingExtendedHeapModel(pointerAnalysis.getHeapModel());
	}

	/**
	 * Whether the function of {@code nodes} reads a tensor-typed field that another function rebinds.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @return {@code TRUE} if a tensor-typed field the function reads is written by another function that doesn't set its object up,
	 *         {@code FALSE} otherwise.
	 */
	Boolean readsReboundState(Set<CGNode> nodes) {
		Set<CGNode> closure = new HashSet<>();
		Deque<CGNode> worklist = new ArrayDeque<>(nodes);
		closure.addAll(nodes);

		while (!worklist.isEmpty()) {
			CGNode current = worklist.remove();

			for (Iterator<CGNode> succNodes = this.callGraph.getSuccNodes(current); succNodes.hasNext();) {
				CGNode next = succNodes.next();

				if (closure.add(next))
					worklist.add(next);
			}
		}

		Set<PointerKey> tensorFieldsRead = new HashSet<>();

		for (CGNode node : closure)
			tensorFieldsRead.addAll(this.tensorFieldsRead(node));

		if (tensorFieldsRead.isEmpty())
			return Boolean.FALSE;

		Map<PointerKey, Set<CGNode>> fieldWriters = fieldWritersCache.computeIfAbsent(this.callGraph, this::computeFieldWriters);

		for (PointerKey read : tensorFieldsRead) {
			InstanceFieldPointerKey field = (InstanceFieldPointerKey) read;

			for (CGNode writer : fieldWriters.getOrDefault(field, Collections.emptySet()))
				if (!closure.contains(writer) && !this.setsUp(writer, field.getInstanceKey()))
					return Boolean.TRUE;
		}

		return Boolean.FALSE;
	}

	/**
	 * The writers of each instance field in {@code callGraph}, from every node's direct mod set.
	 *
	 * @param callGraph The call graph whose nodes are scanned, the key the result is cached under.
	 * @return Each instance-field pointer key written in the call graph, mapped to the nodes that write it directly.
	 */
	private Map<PointerKey, Set<CGNode>> computeFieldWriters(CallGraph callGraph) {
		Map<PointerKey, Set<CGNode>> ret = new HashMap<>();

		for (CGNode node : callGraph)
			for (PointerKey written : Function.getDirectMod(node, this.modRef, this.heapModel, this.pointerAnalysis))
				if (written instanceof InstanceFieldPointerKey)
					ret.computeIfAbsent(written, k -> new HashSet<>()).add(node);

		return Collections.unmodifiableMap(ret);
	}

	/**
	 * The fields of objects that {@code node}'s own instructions read where the value read is typed as a tensor, memoized in
	 * {@link #tensorFieldsReadCache}.
	 *
	 * @param node A node the function reaches.
	 * @return The instance-field pointer keys read into a tensor-typed value.
	 */
	private Set<PointerKey> tensorFieldsRead(CGNode node) {
		return tensorFieldsReadCache.computeIfAbsent(node, this::computeTensorFieldsRead);
	}

	/**
	 * Computes the fields of objects that {@code node}'s own instructions read where the value read is typed as a tensor.
	 *
	 * @param node A node the function reaches.
	 * @return The instance-field pointer keys read into a tensor-typed value.
	 */
	private Set<PointerKey> computeTensorFieldsRead(CGNode node) {
		Set<PointerKey> ret = new HashSet<>();
		IR ir = node.getIR();

		if (ir == null)
			return ret;

		for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions())) {
			if (!this.definesTensor(node, instruction))
				continue;

			for (PointerKey pointerKey : this.modRef.getRef(node, this.heapModel, this.pointerAnalysis, instruction, null))
				if (pointerKey instanceof InstanceFieldPointerKey field
						&& !Util.isContainerType(field.getInstanceKey().concreteType().getReference()))
					ret.add(field);
		}

		return Collections.unmodifiableSet(ret);
	}

	/**
	 * True iff {@code instruction} defines a value the tensor-type analysis types as a tensor.
	 *
	 * @param node The node of {@code instruction}.
	 * @param instruction An instruction of {@code node}.
	 * @return True iff one of its defs is tensor-typed.
	 */
	private boolean definesTensor(CGNode node, SSAInstruction instruction) {
		for (int i = 0; i < instruction.getNumberOfDefs(); i++)
			if (this.tensorTypedKeys.containsKey(this.heapModel.getPointerKeyForLocal(node, instruction.getDef(i))))
				return true;

		return false;
	}

	/**
	 * True iff {@code node} sets up {@code instanceKey}: its method is the {@code __init__} of the object's class or of one of its bases,
	 * or, when the class is a Keras {@code Layer} or {@code Model} subclass, its {@code build}, and the object is its receiver. Both run
	 * once per object, the first as the object is constructed and the second before the first {@code call}, so a field set only there holds
	 * one value for the object's life.
	 *
	 * @param node A node writing a field of {@code instanceKey}.
	 * @param instanceKey The object whose field {@code node} writes.
	 * @return True iff {@code node}'s method is a set-up method of the object's class.
	 */
	private boolean setsUp(CGNode node, InstanceKey instanceKey) {
		return this.getSetUpMethodTypes(instanceKey).contains(node.getMethod().getDeclaringClass().getReference())
				&& this.isReceiver(node, instanceKey);
	}

	/**
	 * True iff {@code instanceKey} is in the points-to set of {@code node}'s {@code self} parameter, the first formal after the function
	 * object. A set-up method sets up only the object it's invoked on: a write it makes to another object's field, such as a child's
	 * {@code __init__} writing its parent's, rebinds that field each time the method runs.
	 *
	 * @param node A set-up method's node.
	 * @param instanceKey The object whose field {@code node} writes.
	 * @return True iff {@code node}'s receiver may be {@code instanceKey}.
	 */
	private boolean isReceiver(CGNode node, InstanceKey instanceKey) {
		IR ir = node.getIR();

		if (ir == null || ir.getNumberOfParameters() < 2)
			return false;

		return this.pointerAnalysis.getPointsToSet(this.heapModel.getPointerKeyForLocal(node, ir.getParameter(1))).contains(instanceKey);
	}

	/**
	 * The types of the functions that set up an object, resolved through the engine's class hierarchy. The object's class is the one whose
	 * constructor allocated it: the engine allocates an instance of a class the program defines in a {@link PythonConstructorFunction}
	 * declared by that class. Its {@code __init__}, and its {@code build} for a Keras {@code Layer} or {@code Model} subclass, are the
	 * function types the engine names under the class and each of its bases, as the engine resolves a constructor's {@code __init__}. An
	 * object allocated elsewhere has no set-up methods here, so every write to its fields counts.
	 *
	 * @param instanceKey An object whose field is written.
	 * @return The type references of the functions that set the object up.
	 */
	private Set<TypeReference> getSetUpMethodTypes(InstanceKey instanceKey) {
		AllocationSiteInNode site = findAllocationSiteInNode(instanceKey);

		if (site == null || !(site.getNode().getMethod() instanceof PythonConstructorFunction constructor))
			return Collections.emptySet();

		return this.setUpMethodTypes.computeIfAbsent(constructor.getDeclaringClass().getName(), className -> {
			Set<TypeName> classAndBaseNames = Function.getClassAndBaseNames(this.callGraph.getClassHierarchy(), className);
			// Keras invokes `build` once, before the first `call`, on a layer or a model; elsewhere `build` is an ordinary method.
			boolean keras = !Collections.disjoint(classAndBaseNames, Function.KERAS_CALLABLE_TYPE_NAMES);
			Set<TypeReference> ret = new HashSet<>();

			for (TypeName name : classAndBaseNames) {
				ret.add(TypeReference.findOrCreate(PythonTypes.pythonLoader, name + "/" + PythonTypes.INIT_METHOD_NAME));

				if (keras)
					ret.add(TypeReference.findOrCreate(PythonTypes.pythonLoader, name + "/" + PythonTypes.KERAS_BUILD_METHOD_NAME));
			}

			return ret;
		});
	}
}
