package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.util.Util.findAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.isPythonScalarKey;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.modref.DelegatingExtendedHeapModel;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.ipa.modref.ModRef;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Iterator2Iterable;

import edu.cuny.hunter.hybridize.core.wala.ml.PythonModRefWithBuiltinFunctions;

/**
 * The analysis behind the precondition of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1076: whether a function
 * reads state that another function rebinds between calls. Measured on TensorFlow 2.9.3, a method that reads a tensor from a field of
 * {@code self}, converted on its own, returns the first call's state once another method has rebound the field: the first trace captures
 * the tensor the field held, and later calls reuse the trace. A Python scalar read from a field is captured the same way: a function that
 * reads an {@code int} field of a plain object returns the first call's value after another function rewrites the field, with one trace.
 * <p>
 * The reads are those of the function's body and of every node it reaches in the call graph, which a trace of the function runs as well. A
 * read counts when the value it defines is typed as a tensor by the tensor-type analysis, which is the only source of tensor-ness here, or
 * when its points-to set holds a Python scalar: a constant {@code int}, {@code float}, {@code bool} or {@code str}, or the key the engine
 * gives a scalar of unknown value. Scalar reads count since https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1048,
 * under which a primitive field of an object keyed by its identity no longer makes a primitive parameter: such a field never causes a
 * retrace, so a rewrite of it after tracing is read stale, and this is the check that refuses it. A scalar read counts whatever it's used
 * for, so a string field used only as an op's name is refused though the values the function computes don't change. The fields a read
 * reads, and the fields a body writes, are the pointer keys {@link ModRef} gives for its instructions, so a read and a write are of the
 * same state exactly when they are the same instance key and field. A write counts when it is made outside the reader's closure, whose own
 * writes are the side-effect precondition's, and outside the methods that set the written object up once: the {@code __init__} of its class
 * or a base, and the {@code build} of a Keras {@code Layer} or {@code Model} subclass, resolved through the engine's class hierarchy from
 * the class whose constructor allocated the object, and invoked on the object itself, per the points-to set of their {@code self}
 * parameter. A field set only there is read live. So is a field set only by the engine summary that allocates the object, such as the
 * engine's constructor of a {@code NamedTuple} without an {@code __init__}, or the summary of a TensorFlow operation populating the tensor
 * it returns: these are the objects {@code tf.function} flattens into its inputs on every call, and a read through an argument holding one
 * is live. An {@code __init__} or {@code build} of another class, a {@code build} outside Keras, or a set-up method writing a field of an
 * object other than its receiver, runs whenever it is called, so its writes count. So do the writes of a helper that only a set-up method
 * calls, a known over-refusal: the helper may be called elsewhere too. A {@code tf.Variable} updated with {@code assign} is not rebound,
 * since {@code assign} writes the variable, not the field holding it.
 * <p>
 * The elements of a list, tuple, dictionary, or set are not read as state: {@code tf.function} passes a container argument's tensors in
 * afresh on each call, and the code building a container writes its elements as fields. A container held in a field is missed with them,
 * though its element reads are stale too once another function rebuilds it. A field holding an object a summary allocated is read as state
 * whatever its type, so a {@code NamedTuple} or a tensor held in a field that another function rebinds is refused. Objects are told apart
 * by allocation site, so instances allocated at one site are one object here: a getter of an object created afresh on each call is refused,
 * though no instance it reads is ever rebound between two of its reads.
 * <p>
 * The verdict is the function's, whatever its callers are: the guide documents that a trace is keyed on the arguments, so a change to a
 * Python global or free variable, which a field read through {@code self} is, does not retrace, and the trace-time value persists.
 *
 * @see <a href="https://www.tensorflow.org/guide/function#rules_of_tracing">Better performance with tf.function: Rules of tracing</a>
 * @see <a href="https://www.tensorflow.org/guide/function#changing_python_global_and_free_variables">Better performance with tf.function:
 *      Changing Python global and free variables</a>
 */
class ReboundStateReadAnalysis {

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
	 * @param callGraph The call graph, walked from the reader to its callees and over the whole program for the writers.
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
	 * A write that rebinds a tensor-typed or scalar-valued field the function reads: the field, the node writing it, and the write's source
	 * position.
	 *
	 * @param field The field read and rebound.
	 * @param writer The node making the write, outside the reader's closure and outside the methods setting the object up.
	 * @param position The write's source position, {@code null} when the writer has no source.
	 */
	record ReboundWrite(InstanceFieldPointerKey field, CGNode writer, Position position) {

		/**
		 * Describes the write for a status message: the field, the class of the object holding it, the writing function, and the write's
		 * file and line when known.
		 *
		 * @return A description naming the field and the write's location.
		 */
		String describe() {
			StringBuilder ret = new StringBuilder();
			Object name = this.field instanceof InstanceFieldKey named ? named.getField().getName() : this.field;
			ret.append("field `").append(name).append("` of `").append(getClassName(this.field.getInstanceKey())).append("`, rebound by `")
					.append(this.writer.getMethod().getDeclaringClass().getName()).append('`');

			if (this.position != null && this.position.getURL() != null) {
				String file = this.position.getURL().getPath();
				ret.append(" at ").append(file.substring(file.lastIndexOf('/') + 1)).append(':').append(this.position.getFirstLine());
			}

			return ret.toString();
		}
	}

	/**
	 * The engine constructor that allocated {@code instanceKey}, if one did: the engine allocates an instance of a class the program
	 * defines in a {@link PythonConstructorFunction} declared by that class.
	 *
	 * @param instanceKey An object.
	 * @return The node of the constructor allocating {@code instanceKey}, or {@code null} if no engine constructor allocated it.
	 */
	private static CGNode getConstructorNode(InstanceKey instanceKey) {
		AllocationSiteInNode site = findAllocationSiteInNode(instanceKey);
		return site != null && site.getNode().getMethod() instanceof PythonConstructorFunction ? site.getNode() : null;
	}

	/**
	 * The name of {@code instanceKey}'s class: the class whose constructor allocated it, else its concrete type.
	 *
	 * @param instanceKey An object.
	 * @return The name of the object's class.
	 */
	private static TypeName getClassName(InstanceKey instanceKey) {
		CGNode constructor = getConstructorNode(instanceKey);
		return constructor == null ? instanceKey.concreteType().getName() : constructor.getMethod().getDeclaringClass().getName();
	}

	/**
	 * Whether the function of {@code nodes} reads a tensor-typed or scalar-valued field that another function rebinds.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @return A write rebinding a tensor-typed or scalar-valued field the function reads, made by another function that doesn't set its
	 *         object up, or empty if there is none.
	 */
	Optional<ReboundWrite> readsReboundState(Set<CGNode> nodes) {
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
		Set<PointerKey> holderFieldsRead = new HashSet<>();

		for (CGNode node : closure)
			this.addStateFieldsRead(node, tensorFieldsRead, holderFieldsRead);

		if (tensorFieldsRead.isEmpty() && holderFieldsRead.isEmpty())
			return Optional.empty();

		for (CGNode node : this.callGraph)
			if (!closure.contains(node))
				for (PointerKey written : Function.getDirectMod(node, this.modRef, this.heapModel, this.pointerAnalysis))
					if (written instanceof InstanceFieldPointerKey field
							&& (tensorFieldsRead.contains(field)
									|| (holderFieldsRead.contains(field) && node.getMethod() instanceof AstMethod))
							&& !this.setsUp(node, field))
						return Optional.of(new ReboundWrite(field, node, this.getWritePosition(node, field)));

		return Optional.empty();
	}

	/**
	 * The source position of an instruction of {@code node} writing {@code field}.
	 *
	 * @param node A node writing {@code field}.
	 * @param field The field written.
	 * @return The position of the first instruction of {@code node} writing {@code field}, or {@code null} if {@code node} has no source.
	 */
	private Position getWritePosition(CGNode node, PointerKey field) {
		IR ir = node.getIR();

		if (ir == null || !(node.getMethod() instanceof AstMethod method) || method.debugInfo() == null)
			return null;

		for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
			if (this.modRef.getMod(node, this.heapModel, this.pointerAnalysis, instruction, null).contains(field))
				return method.debugInfo().getInstructionPosition(instruction.iIndex());

		return null;
	}

	/**
	 * Adds the fields of objects that {@code node}'s own instructions read as state. A field read into a value typed as a tensor, or into a
	 * value that may be a Python scalar (see {@link #definesPythonScalar(CGNode, SSAInstruction)}), is a tensor field, rebound by any write
	 * that doesn't set its object up. A field read into a value that may be an object a summary allocated is a holder field (see
	 * {@link #isAllocatingSummary(CGNode, InstanceFieldPointerKey)}): a summary's writes to such an object are excused as its
	 * initialization, so a rebinding of the field holding it, say {@code self.g = G(v)} for a {@code NamedTuple} {@code G}, is what changes
	 * what a read through the field sees, tensor-typed or not. A holder field is rebound only by program source. A summary writing one
	 * either binds a method object, as the summary of {@code tf.GradientTape} binds {@code gradient} on the tape, or models module state
	 * such as a file system or a generator, which reads of module state own
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1077). An object passed in as an argument is read through no
	 * field, so its own fields stay excused.
	 *
	 * @param node A node the function reaches.
	 * @param tensorFields The tensor fields read, added to.
	 * @param holderFields The holder fields read, added to.
	 */
	private void addStateFieldsRead(CGNode node, Set<PointerKey> tensorFields, Set<PointerKey> holderFields) {
		IR ir = node.getIR();

		if (ir == null)
			return;

		for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions())) {
			Set<PointerKey> fields;

			if (this.definesTensor(node, instruction) || this.definesPythonScalar(node, instruction))
				fields = tensorFields;
			else if (this.readsSummaryAllocation(node, instruction))
				fields = holderFields;
			else
				continue;

			for (PointerKey pointerKey : this.modRef.getRef(node, this.heapModel, this.pointerAnalysis, instruction, null))
				if (pointerKey instanceof InstanceFieldPointerKey field
						&& !Util.isContainerType(field.getInstanceKey().concreteType().getReference())
						&& !isSuperProxy(field.getInstanceKey()))
					fields.add(field);
		}
	}

	/**
	 * True iff {@code instanceKey} is the proxy the front end lowers a {@code super()} call to. Every method calling {@code super()}
	 * allocates a fresh {@code superfun} object in its prologue and sets its {@code $class} and {@code $self} fields there, and the
	 * engine's {@code super} model reads them back. The proxy is a lowering artifact, initialized on each invocation before any use, so its
	 * fields are no program state, though the writes are program source's: another invocation of a method calling {@code super()}, outside
	 * the reader's closure, writes the same abstract proxy.
	 *
	 * @param instanceKey The object whose field is read.
	 * @return True iff {@code instanceKey} is a {@code super()} proxy.
	 */
	private static boolean isSuperProxy(InstanceKey instanceKey) {
		return PythonTypes.superfun.equals(instanceKey.concreteType().getReference());
	}

	/**
	 * True iff {@code instruction} reads a field into a value that may be an object a summary, rather than program source, allocated.
	 *
	 * @param node The node of {@code instruction}.
	 * @param instruction An instruction of {@code node}.
	 * @return True iff {@code instruction} is a field read whose value's points-to set holds an object a summary allocated.
	 */
	private boolean readsSummaryAllocation(CGNode node, SSAInstruction instruction) {
		if (!(instruction instanceof SSAGetInstruction) && !(instruction instanceof AstPropertyRead))
			return false;

		for (int i = 0; i < instruction.getNumberOfDefs(); i++)
			for (InstanceKey instanceKey : this.pointerAnalysis
					.getPointsToSet(this.heapModel.getPointerKeyForLocal(node, instruction.getDef(i)))) {
				AllocationSiteInNode site = findAllocationSiteInNode(instanceKey);

				if (site != null && !(site.getNode().getMethod() instanceof AstMethod))
					return true;
			}

		return false;
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
	 * True iff {@code instruction} defines a value that may be a Python scalar: its points-to set holds a constant {@code int},
	 * {@code float}, {@code bool} or {@code str}, or the key the engine gives a Python scalar of unknown value (wala/ML#1148).
	 *
	 * @param node The node of {@code instruction}.
	 * @param instruction An instruction of {@code node}.
	 * @return True iff one of its defs may be a Python scalar.
	 */
	private boolean definesPythonScalar(CGNode node, SSAInstruction instruction) {
		for (int i = 0; i < instruction.getNumberOfDefs(); i++)
			for (InstanceKey instanceKey : this.pointerAnalysis
					.getPointsToSet(this.heapModel.getPointerKeyForLocal(node, instruction.getDef(i))))
				if (isPythonScalarKey(instanceKey)
						|| (instanceKey instanceof ConstantKey<?> constantKey && (constantKey.getValue() instanceof Number
								|| constantKey.getValue() instanceof Boolean || constantKey.getValue() instanceof String)))
					return true;

		return false;
	}

	/**
	 * True iff {@code node} sets up the object whose field {@code field} is: it is the summary that allocated the object and the write is
	 * to that allocation (see {@link #isAllocatingSummary(CGNode, InstanceFieldPointerKey)}), or its method is the {@code __init__} of the
	 * object's class or of one of its bases, or, when the class is a Keras {@code Layer} or {@code Model} subclass, its {@code build}, and
	 * the object is its receiver. Each sets the field before the object's first use, the summary and {@code __init__} as the object is
	 * created and {@code build} before the first {@code call}, so a field set only there holds one value for the object's life.
	 *
	 * @param node A node writing {@code field}.
	 * @param field The field {@code node} writes.
	 * @return True iff {@code node} initializes the object in the summary allocating it or is a set-up method of its class.
	 */
	private boolean setsUp(CGNode node, InstanceFieldPointerKey field) {
		InstanceKey instanceKey = field.getInstanceKey();
		return this.isAllocatingSummary(node, field)
				|| (this.getSetUpMethodTypes(instanceKey).contains(node.getMethod().getDeclaringClass().getReference())
						&& this.isReceiver(node, instanceKey));
	}

	/**
	 * True iff {@code node} is a function the engine synthesizes, not program source, that allocated the object whose field {@code field}
	 * is, and each of its writes to {@code field} is to the object its own allocation of that site defines. Such writes are initialization:
	 * each concrete object's field is set by the invocation that creates it, before any other code sees the object, and a later write by
	 * any other node still counts. The anchoring on the allocation's own value excludes a write the summary makes to an argument that
	 * points to an object the same node allocated on an earlier call. Two kinds of summary write such fields. The engine's constructor for
	 * a {@code NamedTuple} with no {@code __init__} stores each positional argument into its declared field
	 * (https://github.com/wala/ML/issues/579), and the summary of a TensorFlow operation such as {@code tf.concat} populates the tensor it
	 * returns, whose contents a body reads when it iterates or indexes the tensor. A read of these fields through an argument is live
	 * because {@code tf.function} flattens a {@code NamedTuple} or a tensor argument into its inputs on every call; held in a field, the
	 * object is read through that field, which {@link #addStateFieldsRead(CGNode, Set, Set)} counts.
	 *
	 * @param node A node writing {@code field}.
	 * @param field The field {@code node} writes.
	 * @return True iff {@code node} is a summary that allocated the object and writes {@code field} only on the object it allocates.
	 */
	private boolean isAllocatingSummary(CGNode node, InstanceFieldPointerKey field) {
		if (node.getMethod() instanceof AstMethod)
			return false;

		AllocationSiteInNode site = findAllocationSiteInNode(field.getInstanceKey());
		IR ir = node.getIR();

		if (site == null || !node.equals(site.getNode()) || ir == null)
			return false;

		DefUse du = node.getDU();
		boolean wrote = false;

		for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
			if (this.modRef.getMod(node, this.heapModel, this.pointerAnalysis, instruction, null).contains(field)) {
				int ref = instruction instanceof SSAPutInstruction put ? put.getRef()
						: instruction instanceof AstPropertyWrite write ? write.getObjectRef() : -1;

				if (ref == -1 || !(du.getDef(ref) instanceof SSANewInstruction allocation)
						|| !allocation.getNewSite().equals(site.getSite()))
					return false;

				wrote = true;
			}

		return wrote;
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
		CGNode constructor = getConstructorNode(instanceKey);

		if (constructor == null)
			return Collections.emptySet();

		return this.setUpMethodTypes.computeIfAbsent(constructor.getMethod().getDeclaringClass().getName(), className -> {
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
