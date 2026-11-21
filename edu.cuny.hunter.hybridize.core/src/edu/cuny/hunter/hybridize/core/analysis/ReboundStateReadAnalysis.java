package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.util.Util.findAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.isPythonScalarKey;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.CallSiteReference;
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
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSACFG;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Iterator2Iterable;
import com.ibm.wala.util.intset.IntIterator;
import com.ibm.wala.util.intset.IntSet;
import com.ibm.wala.util.intset.OrdinalSet;

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
 * for, so a string field used only as an op's name is refused though the values the function computes don't change. It counts only when the
 * field may hold more than one value, though: a field whose points-to set is exactly one constant, such as a class-body constant or a field
 * rebound only to the same literal, is read live whatever writes it. A value from a source the engine doesn't model adds nothing to that
 * set, so a field holding one literal and such a value reads as holding the literal alone, and its read is missed. A tensor-typed read
 * counts whatever its field holds. The fields a read reads, and the fields a body writes, are the pointer keys {@link ModRef} gives for its
 * instructions, so a read and a write are of the same state exactly when they are the same instance key and field. A write counts when it
 * is made outside the reader's closure, whose own writes are the side-effect precondition's, and outside the methods that set the written
 * object up once: the {@code __init__} of its class or a base, and the {@code build} of a Keras {@code Layer} or {@code Model} subclass,
 * resolved through the engine's class hierarchy from the class whose constructor allocated the object, and invoked on the object itself,
 * per the points-to set of their {@code self} parameter. A field set only there is read live. So is a field set only by the engine summary
 * that allocates the object, such as the engine's constructor of a {@code NamedTuple} without an {@code __init__}, or the summary of a
 * TensorFlow operation populating the tensor it returns: these are the objects {@code tf.function} flattens into its inputs on every call,
 * and a read through an argument holding one is live. An {@code __init__} or {@code build} of another class, a {@code build} outside Keras,
 * or a set-up method writing a field of an object other than its receiver, runs whenever it is called, so its writes count. So do the
 * writes of a helper that only a set-up method calls, a known over-refusal: the helper may be called elsewhere too. A {@code tf.Variable}
 * updated with {@code assign} is not rebound, since {@code assign} writes the variable, not the field holding it.
 * <p>
 * A write outside those methods doesn't count either when it is part of the written object's construction
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1165). Either the written object is one the writing function has
 * just created, by a call whose every target is one of the engine's constructors, or the writer is a method that is only ever called as a
 * bound method of an object its caller has just created. In both cases, no call that reaches the reader lies on a path of the creating
 * function's control-flow graph between the creation and the write, or from the creation through the call making it, so that a method that
 * itself reaches the reader isn't such a writer. Each object's field is then written only before the reader first runs on that object, so a
 * trace of the reader that a new object brings reads it live: {@code tf.function} traces a method once per instance, and keys a trace on
 * the identity of an object passed as an argument. The reader must therefore meet the object as its receiver, an argument, or a field of
 * one. An object its closure may reach through a global, through a variable of a function outside the closure, or through a container,
 * directly or through fields, may be replaced under a trace that stays, so a write to it counts however it is made. So does a write the
 * rule doesn't follow: one in a generator, whose caller runs between two of its instructions, one through a value that is neither the
 * object created nor the writer's receiver, one in a method reached other than through the engine's bound-method trampoline, such as a call
 * through the class, and one in a method that may be another object's. Such a method is one read from a field that program source also
 * writes, from a field that the engine's constructor sets to anything but a method it binds to the new object, such as an attribute it
 * copies from the class, or from a field holding a method that was given anything but the function its class defines, as when the program
 * has replaced that function on the class.
 * <p>
 * That rule rests on the call graph being the whole program. The functions that may write a field, and the calls that may run between a
 * creation and a write, are those the call graph holds, so a caller it lacks, such as a client of a library that fills an object again
 * after reading it, makes a converted reader return the value it was traced with, silently. The writes that count without this rule need no
 * such assumption about callers.
 * <p>
 * The elements of a list, tuple, dictionary, or set are not read as state: {@code tf.function} passes a container argument's tensors in
 * afresh on each call, and the code building a container writes its elements as fields. A container held in a field is missed with them,
 * though its element reads are stale too once another function rebuilds it. A field holding an object a summary allocated is read as state
 * whatever its type, so a {@code NamedTuple} or a tensor held in a field that another function rebinds is refused. Objects are told apart
 * by allocation site, so instances allocated at one site are one object here: a getter of an object created afresh on each call is refused
 * unless the writes to it are part of its construction, though no instance it reads is ever rebound between two of its reads.
 * <p>
 * The verdict is the function's, and its callers enter it only in deciding which writes are part of an object's construction: the guide
 * documents that a trace is keyed on the arguments, so a change to a Python global or free variable, which a field read through
 * {@code self} is, does not retrace, and the trace-time value persists.
 *
 * @see <a href="https://www.tensorflow.org/guide/function#rules_of_tracing">Better performance with tf.function: Rules of tracing</a>
 * @see <a href="https://www.tensorflow.org/guide/function#changing_python_global_and_free_variables">Better performance with tf.function:
 *      Changing Python global and free variables</a>
 */
class ReboundStateReadAnalysis {

	/** The field in which the engine's method trampoline holds the function it calls. */
	private static final String TRAMPOLINE_FUNCTION_FIELD = "$function";

	private final CallGraph callGraph;

	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	private final Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys;

	private final ModRef<InstanceKey> modRef = new PythonModRefWithBuiltinFunctions();

	private final ExtendedHeapModel heapModel;

	/** The set-up method types of each class, by the class's type name (see {@link #getSetUpMethodTypes(InstanceKey)}). */
	private final Map<TypeName, Set<TypeReference>> setUpMethodTypes = new HashMap<>();

	/** Each summary node's fields, with whether all its writes to each are to its own allocation (see {@link #getOwnAllocationWrites}). */
	private final Map<CGNode, Map<InstanceFieldPointerKey, Boolean>> ownAllocationWrites = new HashMap<>();

	/** The fields some node writes outside the methods setting their objects up, over the whole program; computed on first use. */
	private Set<InstanceFieldPointerKey> reboundFields;

	/** The fields that program source writes, over the whole program; computed on first use. */
	private Set<PointerKey> sourceWrittenFields;

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
	 * Orders rebinding writes for the status message: a write with a source position before one without, then by file, line and column,
	 * then by the description, which names the field and the writer.
	 */
	private static final Comparator<ReboundWrite> WITNESS_ORDER = Comparator
			.comparing(ReboundWrite::position,
					Comparator.nullsLast(Comparator.comparing((Position p) -> String.valueOf(p.getURL()))
							.thenComparingInt(Position::getFirstLine).thenComparingInt(Position::getFirstCol)))
			.thenComparing(ReboundWrite::describe);

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
	 * @return The first write in source order rebinding a tensor-typed or scalar-valued field the function reads, made by another function
	 *         that doesn't set its object up, or empty if there is none.
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

		// Every rebinding write is a witness; the message names the first in source order, as the hash order of the fields varies between
		// runs.
		Optional<ReboundWrite> ret = Optional.empty();
		Reader reader = new Reader(nodes, closure);

		for (CGNode node : this.callGraph)
			if (!closure.contains(node))
				for (PointerKey written : Function.getDirectMod(node, this.modRef, this.heapModel, this.pointerAnalysis))
					if (written instanceof InstanceFieldPointerKey field
							&& (tensorFieldsRead.contains(field)
									|| (holderFieldsRead.contains(field) && node.getMethod() instanceof AstMethod))
							&& !this.setsUp(node, field) && !this.isConstructionWrite(reader, node, field)) {
						ReboundWrite write = new ReboundWrite(field, node, this.getWritePosition(node, field));

						if (ret.isEmpty() || WITNESS_ORDER.compare(write, ret.get()) < 0)
							ret = Optional.of(write);
					}

		return ret;
	}

	/**
	 * True iff {@code read}, a field read of {@code node}, reads only fields that nothing writes outside the methods setting their objects
	 * up ({@link #setsUp(CGNode, InstanceFieldPointerKey)}): each object's field then holds one value for the object's life. A read whose
	 * fields the heap model doesn't name, or that reads a container's element, isn't.
	 *
	 * @param node The node of {@code read}.
	 * @param read A field read of {@code node}.
	 * @return True iff {@code read} reads at least one field, and every field it reads is written only while its object is set up.
	 */
	boolean readsSetUpStateOnly(CGNode node, SSAInstruction read) {
		if (this.reboundFields == null) {
			this.reboundFields = new HashSet<>();

			for (CGNode writer : this.callGraph)
				for (PointerKey written : Function.getDirectMod(writer, this.modRef, this.heapModel, this.pointerAnalysis))
					if (written instanceof InstanceFieldPointerKey field && !this.setsUp(writer, field))
						this.reboundFields.add(field);
		}

		boolean any = false;

		for (PointerKey pointerKey : this.modRef.getRef(node, this.heapModel, this.pointerAnalysis, read, null)) {
			if (!(pointerKey instanceof InstanceFieldPointerKey field)
					|| Util.isContainerType(field.getInstanceKey().concreteType().getReference()) || this.reboundFields.contains(field))
				return false;

			any = true;
		}

		return any;
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
	 * value that may be a Python scalar (see {@link #definesPythonScalar(CGNode, SSAInstruction)}) when the field doesn't hold just one
	 * constant (see {@link #holdsOneConstant(InstanceFieldPointerKey)}), is a tensor field, rebound by any write that doesn't set its
	 * object up. A field read into a value that may be an object a summary allocated is a holder field (see
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
			// A read counted only for its scalar value, not for a tensor's.
			boolean scalarOnly = false;

			if (this.definesTensor(node, instruction))
				fields = tensorFields;
			else if (this.definesPythonScalar(node, instruction)) {
				fields = tensorFields;
				scalarOnly = true;
			} else if (this.readsSummaryAllocation(node, instruction))
				fields = holderFields;
			else
				continue;

			for (PointerKey pointerKey : this.modRef.getRef(node, this.heapModel, this.pointerAnalysis, instruction, null))
				if (pointerKey instanceof InstanceFieldPointerKey field
						&& !Util.isContainerType(field.getInstanceKey().concreteType().getReference())
						&& !isSuperProxy(field.getInstanceKey()) && !(scalarOnly && this.holdsOneConstant(field)))
					fields.add(field);
		}
	}

	/**
	 * True iff the points-to set of {@code field} is exactly one {@link ConstantKey}: the field only ever holds one literal, so whatever
	 * writes it, a read of it can't be stale.
	 *
	 * @param field A field read for its scalar value.
	 * @return True iff {@code field} points to a single constant and nothing else.
	 */
	private boolean holdsOneConstant(InstanceFieldPointerKey field) {
		OrdinalSet<InstanceKey> pointsToSet = this.pointerAnalysis.getPointsToSet(field);
		return pointsToSet.size() == 1 && pointsToSet.iterator().next() instanceof ConstantKey<?>;
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

		return this.getOwnAllocationWrites(node, ir).getOrDefault(field, false);
	}

	/**
	 * For each field {@code node}'s instructions write, whether every write to it is to an object that {@code node}'s own allocation of
	 * that field's object's site defines: one pass over the node's instructions, memoized, since a summary may write many fields and the
	 * call graph and pointer analysis are final here.
	 *
	 * @param node A node the engine synthesizes.
	 * @param ir The node's IR.
	 * @return Each field written, mapped to whether all its writes are anchored on the node's own allocation.
	 */
	private Map<InstanceFieldPointerKey, Boolean> getOwnAllocationWrites(CGNode node, IR ir) {
		return this.ownAllocationWrites.computeIfAbsent(node, k -> {
			Map<InstanceFieldPointerKey, Boolean> ret = new HashMap<>();
			DefUse du = node.getDU();

			for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
				for (PointerKey written : this.modRef.getMod(node, this.heapModel, this.pointerAnalysis, instruction, null)) {
					if (!(written instanceof InstanceFieldPointerKey field))
						continue;

					AllocationSiteInNode site = findAllocationSiteInNode(field.getInstanceKey());
					int ref = instruction instanceof SSAPutInstruction put ? put.getRef()
							: instruction instanceof AstPropertyWrite write ? write.getObjectRef() : -1;
					boolean anchored = site != null && ref != -1 && du.getDef(ref) instanceof SSANewInstruction allocation
							&& allocation.getNewSite().equals(site.getSite());
					ret.merge(field, anchored, Boolean::logicalAnd);
				}

			return ret;
		});
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

	/**
	 * What the check keeps about the function whose reads it decides, for the writes it takes as part of an object's construction.
	 */
	private static final class Reader {

		/** The call-graph nodes of the function. */
		private final Set<CGNode> nodes;

		/** The nodes the function reaches in the call graph, its own included. */
		private final Set<CGNode> closure;

		/** The nodes that reach the function in the call graph, its own included; computed on first use. */
		private Set<CGNode> reaching;

		/** The objects the function may reach through a place that can be rebound between its calls; computed on first use. */
		private Set<InstanceKey> shared;

		/**
		 * Creates the record of a function.
		 *
		 * @param nodes The call-graph nodes of the function.
		 * @param closure The nodes the function reaches in the call graph, its own included.
		 */
		Reader(Set<CGNode> nodes, Set<CGNode> closure) {
			this.nodes = nodes;
			this.closure = closure;
		}
	}

	/**
	 * True iff each write {@code writer} makes to {@code field} is part of the construction of the object written, as the function of
	 * {@code reader} sees it: the write is to an object that the same invocation of {@code writer} has created, or {@code writer} is a
	 * method whose every call is on an object its caller has just created, and nothing that reaches the function runs between the creation
	 * and the write.
	 *
	 * @param reader The function whose reads are decided.
	 * @param writer A node of program source writing {@code field}.
	 * @param field The field {@code writer} writes.
	 * @return True iff every write of {@code field} by {@code writer} precedes every run of the function that may read it.
	 */
	private boolean isConstructionWrite(Reader reader, CGNode writer, InstanceFieldPointerKey field) {
		IR ir = writer.getIR();

		if (!(writer.getMethod() instanceof AstMethod) || ir == null || isGenerator(writer))
			return false;

		SSAInstruction[] instructions = ir.getInstructions();
		Boolean calledOnCreated = null;
		boolean any = false;

		for (int index = 0; index < instructions.length; index++) {
			SSAInstruction instruction = instructions[index];

			if (instruction == null || !this.modRef.getMod(writer, this.heapModel, this.pointerAnalysis, instruction, null).contains(field))
				continue;

			int ref = instruction instanceof SSAPutInstruction put ? put.getRef()
					: instruction instanceof AstPropertyWrite write ? write.getObjectRef() : -1;
			SSAAbstractInvokeInstruction creation = ref == -1 ? null : this.getCreation(writer, ref);

			if (creation != null) {
				if (!this.runsNoReaderBetween(reader, writer, creation.iIndex(), index))
					return false;
			} else if (ir.getNumberOfParameters() > 1 && ref == ir.getParameter(1)) {
				if (calledOnCreated == null)
					calledOnCreated = this.isCalledOnCreatedObjects(reader, writer, field);

				if (!calledOnCreated)
					return false;
			} else
				return false;

			any = true;
		}

		return any && !this.getShared(reader).contains(field.getInstanceKey());
	}

	/**
	 * The call of {@code node} that creates the object {@code value} names: the call defining {@code value} whose every target is an engine
	 * constructor, which allocates an instance of a class the program defines.
	 *
	 * @param node A node.
	 * @param value A value of {@code node}.
	 * @return The call creating the object, or {@code null} if {@code value} isn't the result of a call to constructors only.
	 */
	private SSAAbstractInvokeInstruction getCreation(CGNode node, int value) {
		if (!(node.getDU().getDef(value) instanceof SSAAbstractInvokeInstruction invoke) || invoke.getDef() != value)
			return null;

		Set<CGNode> targets = this.callGraph.getPossibleTargets(node, invoke.getCallSite());

		for (CGNode target : targets)
			if (!(target.getMethod() instanceof PythonConstructorFunction))
				return null;

		return targets.isEmpty() ? null : invoke;
	}

	/**
	 * True iff every call of {@code method} is a bound-method call on an object that the calling function has just created, with nothing
	 * that reaches the function of {@code reader} run between the creation and the call. The call itself counts as run there, so a
	 * {@code method} that reaches that function is never one.
	 *
	 * @param reader The function whose reads are decided.
	 * @param method A method writing {@code field} through its receiver.
	 * @param field The field {@code method} writes.
	 * @return True iff each call of {@code method} fills an object as it is created.
	 */
	private boolean isCalledOnCreatedObjects(Reader reader, CGNode method, InstanceFieldPointerKey field) {
		boolean any = false;

		for (CGNode trampoline : Iterator2Iterable.make(this.callGraph.getPredNodes(method))) {
			if (!(trampoline.getMethod().getDeclaringClass() instanceof PythonInstanceMethodTrampoline))
				return false;

			for (CGNode caller : Iterator2Iterable.make(this.callGraph.getPredNodes(trampoline))) {
				IR ir = caller.getIR();

				if (!(caller.getMethod() instanceof AstMethod) || ir == null || isGenerator(caller))
					return false;

				for (CallSiteReference site : Iterator2Iterable.make(this.callGraph.getPossibleSites(caller, trampoline)))
					for (SSAAbstractInvokeInstruction call : ir.getCalls(site)) {
						int receiver = call instanceof PythonInvokeInstruction ? this.getBoundReceiver(caller, call.getUse(0)) : -1;
						SSAAbstractInvokeInstruction creation = receiver == -1 ? null : this.getCreation(caller, receiver);

						if (creation == null
								|| !this.pointerAnalysis.getPointsToSet(this.heapModel.getPointerKeyForLocal(caller, receiver))
										.contains(field.getInstanceKey())
								|| !this.runsNoReaderBetween(reader, caller, creation.iIndex(), call.iIndex()))
							return false;

						any = true;
					}
			}
		}

		return any;
	}

	/**
	 * The receiver of the bound method that {@code function} names: the object whose field the value was read from, when that field holds
	 * only the methods the engine's constructor bound to the object itself. No program source writes such a field, the constructor stores
	 * in it only the methods it binds (see {@link #isSetToBoundMethods(InstanceFieldKey)}), and each of them runs the function its class
	 * defines (see {@link #holdsDefinedFunctions(InstanceFieldKey)}). Instances allocated at one site are one object here, so the objects
	 * the field's methods are bound to don't tell a method of the receiver from the same method of an earlier instance; where the field's
	 * value comes from does.
	 *
	 * @param node A node.
	 * @param function The value a call of {@code node} invokes.
	 * @return The value of the object {@code function} was read from, or -1 if {@code function} isn't a read of a field holding only the
	 *         object's own methods.
	 */
	private int getBoundReceiver(CGNode node, int function) {
		SSAInstruction def = node.getDU().getDef(function);
		int ret = def instanceof AstPropertyRead read ? read.getObjectRef()
				: def instanceof SSAGetInstruction get && !get.isStatic() ? get.getRef() : -1;

		if (ret == -1)
			return -1;

		Set<PointerKey> fields = new HashSet<>(this.modRef.getRef(node, this.heapModel, this.pointerAnalysis, def, null));

		if (fields.isEmpty() || !Collections.disjoint(fields, this.getSourceWrittenFields()))
			return -1;

		for (PointerKey field : fields)
			if (!(field instanceof InstanceFieldKey bound) || !isSetToBoundMethods(bound) || !this.holdsDefinedFunctions(bound))
				return -1;

		return ret;
	}

	/**
	 * True iff the engine's constructor allocating the object of {@code field} stores in that field only methods it binds there: each value
	 * it stores is a method trampoline the constructor itself allocates. The constructor also copies onto the object each attribute the
	 * class body assigns, read from the class, and a class attribute may come to hold a method bound to another object, which a call
	 * through the new object then runs on that other object.
	 *
	 * @param field A field of an object.
	 * @return True iff an engine constructor allocated the object and every value it stores in {@code field} is its own allocation.
	 */
	private static boolean isSetToBoundMethods(InstanceFieldKey field) {
		CGNode constructor = getConstructorNode(field.getInstanceKey());

		if (constructor == null || constructor.getIR() == null)
			return false;

		boolean any = false;

		for (SSAInstruction instruction : Iterator2Iterable.make(constructor.getIR().iterateNormalInstructions()))
			if (instruction instanceof SSAPutInstruction put && put.getDeclaredField().getName().equals(field.getField().getName())) {
				if (!(constructor.getDU().getDef(put.getVal()) instanceof SSANewInstruction))
					return false;

				any = true;
			}

		return any;
	}

	/**
	 * True iff each value of {@code field} is a method trampoline whose function is the one its class defines. The engine's constructor
	 * reads a method's function from the class attribute of the method's name, and its trampoline calls only the function the class
	 * defines, so when the program has replaced that attribute, such as by a method bound to another object, the call graph still shows the
	 * defined function run on the new object, while Python runs the replacement.
	 *
	 * @param field A field of an object.
	 * @return True iff {@code field} holds only method trampolines, each of whose functions is an instance of the trampoline's function
	 *         class, and at least one.
	 */
	private boolean holdsDefinedFunctions(InstanceFieldKey field) {
		Map<InstanceKey, Set<InstanceFieldPointerKey>> parts = LayerBuildAnalysis.getFieldIndex(this.pointerAnalysis);
		boolean any = false;

		for (InstanceKey method : this.pointerAnalysis.getPointsToSet(field)) {
			if (!(method.concreteType() instanceof PythonInstanceMethodTrampoline trampoline))
				return false;

			for (InstanceFieldPointerKey part : parts.getOrDefault(method, Collections.emptySet()))
				if (part instanceof InstanceFieldKey named && TRAMPOLINE_FUNCTION_FIELD.equals(named.getField().getName().toString()))
					for (InstanceKey function : this.pointerAnalysis.getPointsToSet(named)) {
						if (!function.concreteType().getReference().equals(trampoline.getRealClass().getReference()))
							return false;

						any = true;
					}
		}

		return any;
	}

	/**
	 * The fields that program source writes, over the whole program.
	 *
	 * @return The fields written by the nodes of program source.
	 */
	private Set<PointerKey> getSourceWrittenFields() {
		if (this.sourceWrittenFields == null) {
			this.sourceWrittenFields = new HashSet<>();

			for (CGNode node : this.callGraph)
				if (node.getMethod() instanceof AstMethod)
					this.sourceWrittenFields.addAll(Function.getDirectMod(node, this.modRef, this.heapModel, this.pointerAnalysis));
		}

		return this.sourceWrittenFields;
	}

	/**
	 * True iff {@code node} is a generator. The engine lowers each {@code yield} into a write of the function object's content field. A
	 * generator's body runs when it is iterated, not when it is called, and stops at each {@code yield} while its caller runs, so what runs
	 * between two of its instructions isn't only what it calls.
	 *
	 * @param node A node.
	 * @return True iff {@code node}'s body yields.
	 */
	private static boolean isGenerator(CGNode node) {
		IR ir = node.getIR();

		if (ir == null)
			return false;

		SymbolTable symbolTable = ir.getSymbolTable();

		for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
			if (instruction instanceof AstPropertyWrite write && symbolTable.isStringConstant(write.getMemberRef())
					&& PythonTypes.GENERATOR_CONTENT_FIELD_NAME.equals(symbolTable.getStringValue(write.getMemberRef())))
				return true;

		return false;
	}

	/**
	 * True iff no call that reaches the function of {@code reader} may run in {@code frame} after its instruction {@code from} and before
	 * its instruction {@code to}, the two included: no such call lies on a path of the control-flow graph, exceptional edges included, that
	 * leads from {@code from} to {@code to} without passing {@code from} again. A call the IR doesn't place counts as lying on one.
	 *
	 * @param reader The function whose reads are decided.
	 * @param frame A node with an IR.
	 * @param from The index of the instruction creating an object.
	 * @param to The index of a later instruction of {@code frame}.
	 * @return True iff the function can't run between the two instructions.
	 */
	private boolean runsNoReaderBetween(Reader reader, CGNode frame, int from, int to) {
		IR ir = frame.getIR();
		BitSet window = getWindow(ir.getControlFlowGraph(), from, to);

		for (CGNode callee : Iterator2Iterable.make(this.callGraph.getSuccNodes(frame)))
			if (this.getReaching(reader).contains(callee))
				for (CallSiteReference site : Iterator2Iterable.make(this.callGraph.getPossibleSites(frame, callee))) {
					IntSet indices = ir.getCallInstructionIndices(site);

					if (indices == null || indices.isEmpty())
						return false;

					for (IntIterator iterator = indices.intIterator(); iterator.hasNext();)
						if (window.get(iterator.next()))
							return false;
				}

		return true;
	}

	/**
	 * The instructions of {@code cfg} on a path from {@code from} to {@code to} that doesn't pass {@code from} again, the two included.
	 *
	 * @param cfg A control-flow graph.
	 * @param from The index of an instruction.
	 * @param to The index of another instruction.
	 * @return The indices of the instructions between the two.
	 */
	private static BitSet getWindow(SSACFG cfg, int from, int to) {
		BitSet ret = getReached(cfg, from, from, true);
		ret.and(getReached(cfg, to, from, false));
		ret.set(from);
		ret.set(to);
		return ret;
	}

	/**
	 * The instructions of {@code cfg} reached from {@code start}, not going on from {@code stop}.
	 *
	 * @param cfg A control-flow graph.
	 * @param start The index of the instruction to start after, or before.
	 * @param stop The index of the instruction to stop at.
	 * @param forward Whether to follow the graph forward.
	 * @return The indices of the instructions reached, without {@code start} unless a cycle leads back to it.
	 */
	private static BitSet getReached(SSACFG cfg, int start, int stop, boolean forward) {
		BitSet ret = new BitSet();
		Deque<Integer> worklist = new ArrayDeque<>(getNeighbors(cfg, start, forward));

		while (!worklist.isEmpty()) {
			int index = worklist.remove();

			if (!ret.get(index)) {
				ret.set(index);

				if (index != stop)
					worklist.addAll(getNeighbors(cfg, index, forward));
			}
		}

		return ret;
	}

	/**
	 * The instructions of {@code cfg} that may run right after, or right before, the instruction at {@code index}: its neighbor in its
	 * block, or, at the block's end, the nearest instruction of each following block, looking through blocks without instructions.
	 *
	 * @param cfg A control-flow graph.
	 * @param index The index of an instruction.
	 * @param forward Whether to follow the graph forward.
	 * @return The indices of the neighboring instructions.
	 */
	private static List<Integer> getNeighbors(SSACFG cfg, int index, boolean forward) {
		ISSABasicBlock block = cfg.getBlockForInstruction(index);

		if (forward ? index < block.getLastInstructionIndex() : index > block.getFirstInstructionIndex())
			return List.of(forward ? index + 1 : index - 1);

		List<Integer> ret = new ArrayList<>();
		Set<ISSABasicBlock> seen = new HashSet<>();
		Deque<ISSABasicBlock> worklist = new ArrayDeque<>();
		(forward ? cfg.getSuccNodes(block) : cfg.getPredNodes(block)).forEachRemaining(worklist::add);

		while (!worklist.isEmpty()) {
			ISSABasicBlock next = worklist.remove();

			if (!seen.add(next))
				continue;

			if (next.getFirstInstructionIndex() >= 0 && next.getFirstInstructionIndex() <= next.getLastInstructionIndex())
				ret.add(forward ? next.getFirstInstructionIndex() : next.getLastInstructionIndex());
			else
				(forward ? cfg.getSuccNodes(next) : cfg.getPredNodes(next)).forEachRemaining(worklist::add);
		}

		return ret;
	}

	/**
	 * The nodes that reach the function of {@code reader} in the call graph, its own included.
	 *
	 * @param reader The function whose reads are decided.
	 * @return The nodes from which a run of the function may follow.
	 */
	private Set<CGNode> getReaching(Reader reader) {
		if (reader.reaching == null) {
			reader.reaching = new HashSet<>(reader.nodes);
			Deque<CGNode> worklist = new ArrayDeque<>(reader.nodes);

			while (!worklist.isEmpty())
				for (CGNode caller : Iterator2Iterable.make(this.callGraph.getPredNodes(worklist.remove())))
					if (reader.reaching.add(caller))
						worklist.add(caller);
		}

		return reader.reaching;
	}

	/**
	 * The objects the function of {@code reader} may reach through a place that can be rebound between its calls without a new trace: a
	 * global, a variable of a function outside its closure, or a container it handles, and from there through fields.
	 *
	 * @param reader The function whose reads are decided.
	 * @return The objects reached from such a place.
	 */
	private Set<InstanceKey> getShared(Reader reader) {
		if (reader.shared == null) {
			reader.shared = new HashSet<>();
			Deque<InstanceKey> worklist = new ArrayDeque<>();
			Set<String> types = new HashSet<>();

			for (CGNode node : reader.closure)
				types.add(NumpyParameterFlowAnalysis.stripTypePrefix(node.getMethod().getDeclaringClass().getName().toString()));

			for (CGNode node : reader.closure) {
				IR ir = node.getIR();

				if (ir == null)
					continue;

				for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
					if (instruction instanceof AstGlobalRead global)
						this.pointerAnalysis.getPointsToSet(this.heapModel.getPointerKeyForLocal(node, global.getDef()))
								.forEach(worklist::add);
					else if (instruction instanceof AstLexicalRead lexical) {
						Access[] accesses = lexical.getAccesses();

						for (int index = 0; index < accesses.length; index++) {
							String definer = accesses[index].getName().snd;

							if (definer == null || !types.contains(NumpyParameterFlowAnalysis.stripTypePrefix(definer)))
								this.pointerAnalysis.getPointsToSet(this.heapModel.getPointerKeyForLocal(node, lexical.getDef(index)))
										.forEach(worklist::add);
						}
					}

				for (int value = 1; value <= ir.getSymbolTable().getMaxValueNumber(); value++)
					for (InstanceKey instanceKey : this.pointerAnalysis.getPointsToSet(this.heapModel.getPointerKeyForLocal(node, value)))
						if (Util.isContainerType(instanceKey.concreteType().getReference()))
							worklist.add(instanceKey);
			}

			Map<InstanceKey, Set<InstanceFieldPointerKey>> fields = LayerBuildAnalysis.getFieldIndex(this.pointerAnalysis);

			while (!worklist.isEmpty()) {
				InstanceKey instanceKey = worklist.remove();

				if (reader.shared.add(instanceKey))
					for (InstanceFieldPointerKey field : fields.getOrDefault(instanceKey, Collections.emptySet()))
						this.pointerAnalysis.getPointsToSet(field).forEach(worklist::add);
			}
		}

		return reader.shared;
	}
}
