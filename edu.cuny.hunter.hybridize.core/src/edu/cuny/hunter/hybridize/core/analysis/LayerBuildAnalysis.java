package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.KERAS_LAYER_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.KERAS_MODEL_TYPE;
import static com.ibm.wala.cast.python.util.Util.allocatedClass;
import static com.ibm.wala.cast.python.util.Util.findAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.isPythonScalarKey;
import static org.eclipse.core.runtime.Platform.getLog;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

import org.eclipse.core.runtime.ILog;

import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.ir.ssa.EachElementHasNextInstruction;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.loader.IPythonClass;
import com.ibm.wala.cast.python.loader.StarFormalDeclaration;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.cfg.cdg.ControlDependenceGraph;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cfg.ExceptionPrunedCFG;
import com.ibm.wala.ipa.cfg.PrunedCFG;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSACFG;
import com.ibm.wala.ssa.SSAConditionalBranchInstruction;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPhiInstruction;
import com.ibm.wala.ssa.SSASwitchInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Iterator2Iterable;
import com.ibm.wala.util.intset.OrdinalSet;

/**
 * The layer-build analysis behind the {@link PreconditionFailure#MAY_BUILD_LAYERS} safety precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1172 and
 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1224). A layer creates its variables on its first call, when its
 * {@code __call__} runs its {@code build}. Under {@code tf.function}, Keras builds a sublayer that a {@code call} invokes on every call
 * under {@code init_scope} during the first trace, and a retrace builds nothing, so such a {@code call} is safe to decorate. Three shapes
 * change behavior, and a method is refused only for one of them ({@link LayerBuildShape}):
 * <ol>
 * <li>{@link LayerBuildShape#CONSTRUCTED_IN_BODY}: the engine allocates an object of a class reaching {@code tf.keras.layers.Layer} or
 * {@code tf.keras.Model} through its Python bases, at an allocation site in the method or in a callee it reaches other than a layer's own
 * {@code __init__} or {@code build}, which set a layer up once. Each trace creates new variables.</li>
 * <li>{@link LayerBuildShape#FIRST_CALL_ON_RETRACE}: a call whose callees may reach {@code Layer.add_weight} lies in a block of the method,
 * or of a program function it reaches, that is control dependent, transitively, on a branch or loop condition that can differ between
 * traces, and the branch decides whether the callee is first called: a trace can reach the branch without calling the callee, and an arm of
 * it leads to the exit without calling it either. The calls of one callee object are taken together, so a layer called on both arms of a
 * branch, or before it, isn't first called under it. Only the paths that return normally count, since a trace that raises keeps no graph;
 * but a call in an exception handler that returns normally is refused, since a retrace whose protected code raises first calls it. The
 * condition takes the same path on every trace when the engine shows it, as one constant in every context or a tensor in every context
 * (AutoGraph traces both arms of a tensor's branch), or when it is built only from constants, the receiver, fields written only while their
 * objects are set up in {@code __init__} or a layer's {@code build}, operators over these, a loop over such an iterable, such as
 * {@code self.blocks}, and {@code range} or {@code len} of them, followed back through the IR's def-use chains. A tensor in one context or
 * call and a Python value in another is a mix, which varies. Anything else, such as a parameter, a field another method rebinds, another
 * call's result, or a value the walk can't trace, may vary, and the method is refused. A layer chosen by data rather than by control counts
 * too: a call whose callee value may be more than one layer is refused unless a phi merges them under branches that are fixed so, as
 * {@code (self.a if flag else self.b)(x)} is refused for a parameter {@code flag}. One layer through a local alias is one value. One
 * allocation that may make several layers, as a comprehension does, counts as several unless a fixed key selects one, as in
 * {@code self.blocks[0]}; {@code self.blocks[i]} for a varying {@code i} is refused, in the method or in a function it calls. The receiver
 * choosing, as {@code self.d} does in a class constructed twice, is fixed, since each instance traces its own function.</li>
 * <li>{@link LayerBuildShape#COLLIDING_NAMES}: decorating the method names the variables built in its trace without the scopes of the
 * receiver and the layers enclosing it, so two sublayers that carry the same name build variables with the same names, and a save keyed by
 * name, such as to HDF5, fails or mixes weights (issue 1172). The sublayers are the layer objects reachable from the receiver through
 * fields, at any depth, told apart by the node allocating them, a constructor or a summary in its context, since the engine allocates a
 * program layer's bound methods in its constructor too. The receiver's own allocation is left out: its name is dropped with its scope. A
 * sublayer's name is the {@code name} argument of the call constructing it, or of the {@code __init__} call of a base that its class's
 * {@code __init__} makes, read as the constant strings of its points-to set, which holds a parameter's default too, unless that
 * {@code __init__} forwards its own {@code name} parameter and every constructing call passes one. A sublayer's variables are named under
 * its parent's scope, the nearest layer holding it, so only siblings may collide: two allocations under one parent that may carry the same
 * constant name, whatever their classes, or one allocation that may run more than once in a run, such as one in a loop or a comprehension,
 * in a function invoked from two sites that one program run reaches, or in the {@code build} of a layer that has more than one instance,
 * unless each run is named apart. A run is named apart when its one constructing call runs once per iteration of the one loop whose index
 * fills its name, {@code "h{}".format(i)} for {@code i in range(n)}, or once per run of a parent's {@code __init__} or {@code build} whose
 * layers are named apart in turn or carry unique names from Keras. The receiver must be one object: decorating the method drops its scope,
 * so two receivers' sublayers collide. Two templates under one parent may fill a name alike, as {@code "h{}"} and {@code "h1{}"} do with
 * {@code h10}, and refuse. A sublayer's parent is the layer holding it; Keras scopes it under the layer whose {@code __call__} first builds
 * it, which is the same whenever a layer is called by its holder, but not for a layer passed to another's {@code call} and built there. Any
 * other name that is not one constant string may collide with any other and refuses. A sublayer constructed without a name, or with
 * {@code None}, gets a unique one from Keras and never collides. The front end reads an f-string as the one constant string of its source
 * text, opening quote included, which counts as computed until the engine reads it so (wala/ML#1231). A constructing call that unpacks
 * {@code **} keywords, as {@code Dense(**config)} does, may pass any name, which counts as computed too, and so does a base's
 * {@code __init__} call unpacking anything but the calling {@code __init__}'s own {@code **kwargs}: {@code super().__init__(**kwargs)}
 * forwards only the keywords the constructing call passed, which are read there.</li>
 * </ol>
 * A call with no targets in the call graph, such as one on a layer read from a dictionary under a computed key
 * (https://github.com/wala/ML/issues/1222), is invisible to {@link #mayBuildLayers(CGNode)} and to every shape here. A Keras model's
 * {@code call} is exempt: decorating it drops the enclosing scopes from its weights' names too, but the save succeeds, whether or not the
 * model is nested in another one. So is a function that is no method, such as a training step: it runs inside no layer's name scope, and a
 * model it calls enters its own scope inside the trace.
 * <p>
 * Every type fact comes from the engine. {@link TensorFlowApis} names the call of {@code tensorflow.keras.layers.Layer.add_weight} both in
 * a program layer's {@code build} and in the engine's summary of a built-in layer's first call (wala/ML#1145). The latter lies inside a
 * TensorFlow library node, so the walk of {@link Util#reachesInstruction(CallGraph, CGNode, java.util.function.BiPredicate, boolean)} tests
 * library nodes too. An object's class is {@link com.ibm.wala.cast.python.util.Util#allocatedClass(InstanceKey)} (wala/ML#1165), whose
 * bases {@link Function#getClassAndBaseNames(IClassHierarchy, com.ibm.wala.types.TypeReference)} walks transitively, through every declared
 * Python base and every superclass, for the engine's modeled {@code tf.keras.layers.Layer} and {@code tf.keras.Model}. The class
 * hierarchy's own subtype test would not do: the engine records a single superclass per class, so {@code class Net(Mixin, tf.keras.Model)}
 * would read as no model in either order of bases. In the engine's model, unlike at run time, {@code Model} does not extend {@code Layer},
 * so the walk tells a model from a layer. Allocation sites, points-to sets, field keys, call sites, and the control dependence of a node's
 * own control-flow graph are WALA's; the conditions, names and counts over them are this analysis's logic.
 */
class LayerBuildAnalysis {

	/**
	 * A reason a method is refused: the shape and a description of where it occurs.
	 *
	 * @param shape The shape.
	 * @param detail A description of the occurrence, for a status message.
	 */
	record Verdict(LayerBuildShape shape, String detail) {
	}

	/**
	 * The names a sublayer may carry.
	 *
	 * @param constants The constant names it may carry.
	 * @param templates The templates of the names it may carry that a loop's index fills ({@link #getIndexedName(CGNode, int)}).
	 * @param computed True iff it may carry a name that is neither one constant string nor filled from a loop's index.
	 */
	private record Names(Set<String> constants, Set<String> templates, boolean computed) {

		/**
		 * True iff the sublayer may carry a name at all, rather than one Keras makes unique.
		 *
		 * @return True iff a constant or indexed name may be given.
		 */
		boolean named() {
			return !this.constants.isEmpty() || !this.templates.isEmpty();
		}
	}

	/**
	 * A name filled from a loop's index, as {@code "h{}".format(i)} in {@code for i in range(n)} is.
	 *
	 * @param template The string constant formatted, with one {@code {}} field.
	 * @param index The {@code next} call defining the loop's index.
	 */
	private record IndexedName(String template, PythonInvokeInstruction index) {
	}

	/**
	 * The layers reachable from a method's receiver through fields, by the node allocating them, and the parent of each: the nearest layer
	 * holding it, by its allocating node, or {@code null} for the receiver, whose own scope a decorated method drops.
	 *
	 * @param objects The sublayers' objects by the node allocating them.
	 * @param parents The parents of each sublayer, by the node allocating it.
	 */
	private record Sublayers(Map<CGNode, Set<InstanceKey>> objects, Map<CGNode, Set<CGNode>> parents) {
	}

	private static final ILog LOG = getLog(LayerBuildAnalysis.class);

	/**
	 * The engine's modeled {@code tf.keras.layers.Layer} and {@code tf.keras.Model}, which a layer's class reaches through its bases.
	 */
	private static final Set<TypeName> KERAS_TYPE_NAMES = Set.of(KERAS_LAYER_TYPE.getName(), KERAS_MODEL_TYPE.getName());

	/** The fully-qualified name of {@code Layer.add_weight}, as the engine names it ({@link TensorFlowApis}). */
	private static final String ADD_WEIGHT_API = "tensorflow.keras.layers.Layer.add_weight";

	/** The keyword the front end gives a {@code **} argument of a call, the string of its missing keyword name. */
	private static final String UNPACKED_KEYWORDS = String.valueOf((Object) null);

	/** The keyword of a layer's name. */
	private static final String NAME_KEYWORD = "name";

	/** The string method that fills a template's fields, as {@code "h{}".format(i)} does. */
	private static final String FORMAT_METHOD_NAME = "format";

	/** The builtins whose result is the same on every trace when their arguments are, including those a {@code for} loop is lowered to. */
	private static final Set<String> FIXED_BUILTIN_NAMES = Set.of("iter", "len", "next", "range");

	/**
	 * The fields of each object, by object, of each pointer analysis. Built once per pointer analysis, since every method's sublayer walk
	 * reads it; concurrent because functions may be processed in parallel.
	 */
	private static final Map<PointerAnalysis<InstanceKey>, Map<InstanceKey, Set<InstanceFieldPointerKey>>> fieldIndexCache = new ConcurrentHashMap<>();

	/**
	 * The entry points reaching each node, by node, of each call graph ({@link #getEntryPointsReaching(CallGraph)}). Built once per call
	 * graph; concurrent because functions may be processed in parallel.
	 */
	private static final Map<CallGraph, Map<CGNode, BitSet>> entryPointsCache = new ConcurrentHashMap<>();

	/** The call graph, whose successors the walk follows. */
	private final CallGraph callGraph;

	/** The call-graph builder, which the engine asks for the APIs a call invokes, the receiver's objects and the class hierarchy. */
	private final PythonSSAPropagationCallGraphBuilder builder;

	/** Whether each node may reach {@code Layer.add_weight}, memoized. */
	private final Map<CGNode, Boolean> reachesAddWeight = new HashMap<>();

	/** Whether each node may run more than once, memoized; a node being decided reads as running more than once, as recursion may. */
	private final Map<CGNode, Boolean> runsMoreThanOnce = new HashMap<>();

	/** The pointer keys the tensor-type analysis types as tensors, which the engine publishes ({@link Util#computeTensorTypedOrigins}). */
	private final Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys;

	/** Which fields are written only while their objects are set up, for {@link #isFixed(CGNode, int)}; created on first use. */
	private ReboundStateReadAnalysis reboundState;

	/**
	 * Creates the analysis.
	 *
	 * @param callGraph The call graph.
	 * @param builder The call-graph builder.
	 * @param tensorTypedKeys The pointer keys the tensor-type analysis types as tensors.
	 */
	LayerBuildAnalysis(CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder,
			Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys) {
		this.callGraph = callGraph;
		this.builder = builder;
		this.tensorTypedKeys = tensorTypedKeys;
	}

	/**
	 * Releases the field index of each pointer analysis and the entry points reaching each node of each call graph.
	 */
	static void clearCaches() {
		fieldIndexCache.clear();
		entryPointsCache.clear();
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

		PointerKey receiver = this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, parameters[1]);
		boolean anyModel = false;
		boolean anyOther = false;

		for (InstanceKey instanceKey : this.getPointerAnalysis().getPointsToSet(receiver)) {
			Optional<IClass> allocatedClass = allocatedClass(instanceKey);

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

	/**
	 * The shape of layer building, if any, for which the method of {@code nodes} is refused, checked in the order of
	 * {@link LayerBuildShape}'s constants.
	 *
	 * @param nodes The call-graph nodes of an instance method that may build a layer.
	 * @return The first shape that holds, with where it occurs, or empty if none does.
	 */
	Optional<Verdict> getShape(Set<CGNode> nodes) {
		Set<CGNode> traced = this.getTracedNodes(nodes);
		Optional<Verdict> ret = this.findConstruction(traced);

		if (ret.isEmpty())
			ret = this.findConditionalBuild(traced);

		if (ret.isEmpty())
			ret = this.findCollidingNames(nodes);

		return ret;
	}

	/**
	 * The nodes a trace of the method of {@code nodes} runs: those nodes and every node they transitively call, except the subtrees of a
	 * layer's own {@code __init__} and {@code build} ({@link #isLayerSetUp(CGNode)}), which set the layer up once, the latter under
	 * {@code init_scope}.
	 *
	 * @param nodes The method's nodes.
	 * @return The nodes the method's trace runs.
	 */
	private Set<CGNode> getTracedNodes(Set<CGNode> nodes) {
		Set<CGNode> ret = new LinkedHashSet<>(nodes);
		Deque<CGNode> worklist = new ArrayDeque<>(nodes);

		while (!worklist.isEmpty())
			for (Iterator<CGNode> succNodes = this.callGraph.getSuccNodes(worklist.remove()); succNodes.hasNext();) {
				CGNode next = succNodes.next();

				if (!this.isLayerSetUp(next) && ret.add(next))
					worklist.add(next);
			}

		return ret;
	}

	/**
	 * Finds an allocation of a layer or a model at a site in one of {@code traced}.
	 *
	 * @param traced The nodes a trace runs ({@link #getTracedNodes(Set)}).
	 * @return {@link LayerBuildShape#CONSTRUCTED_IN_BODY} with the class allocated, or empty if no traced node allocates a layer.
	 */
	private Optional<Verdict> findConstruction(Set<CGNode> traced) {
		for (CGNode node : traced) {
			IR ir = node.getIR();

			if (ir == null)
				continue;

			for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
				if (instruction instanceof SSANewInstruction allocation) {
					InstanceKey instanceKey = this.getPointerAnalysis().getHeapModel().getInstanceKeyForAllocation(node,
							allocation.getNewSite());

					// An allocation whose object the engine doesn't name may be a layer.
					if (instanceKey == null)
						return Optional.of(new Verdict(LayerBuildShape.CONSTRUCTED_IN_BODY,
								"an object it allocates in `" + getFunctionName(node) + "` can't be determined"));

					Optional<IClass> allocatedClass = allocatedClass(instanceKey);

					if (allocatedClass.isPresent() && this.isLayerClass(allocatedClass.get().getReference()))
						return Optional.of(new Verdict(LayerBuildShape.CONSTRUCTED_IN_BODY,
								"it constructs `" + getSimpleName(allocatedClass.get().getName()) + "`"));
				}
		}

		return Optional.empty();
	}

	/**
	 * Finds a call that may build a layer under a condition that can differ between traces, in a program function among {@code traced}.
	 *
	 * @param traced The nodes a trace runs ({@link #getTracedNodes(Set)}).
	 * @return {@link LayerBuildShape#FIRST_CALL_ON_RETRACE} with the call's position, or empty if every call that may build a layer runs
	 *         under conditions that are fixed across traces.
	 */
	private Optional<Verdict> findConditionalBuild(Set<CGNode> traced) {
		for (CGNode node : traced) {
			// Only program code: a summary's branches, such as `__call__`'s test of whether the layer is built, are the engine's model.
			if (!(node.getMethod() instanceof AstMethod))
				continue;

			IR ir = node.getIR();

			if (ir == null)
				continue;

			List<CallSiteReference> sites = new ArrayList<>();

			for (CallSiteReference site : Iterator2Iterable.make(ir.iterateCallSites()))
				if (this.callGraph.getPossibleTargets(node, site).stream().anyMatch(this::reachesAddWeight))
					sites.add(site);

			if (sites.isEmpty())
				continue;

			PrunedCFG<SSAInstruction, ISSABasicBlock> cfg;
			ControlDependenceGraph<ISSABasicBlock> controlDependence;

			try {
				// A trace that raises builds no graph, so only the paths that return normally count: the control-flow graph without
				// exceptional edges, restricted to the blocks that reach the exit.
				cfg = ExceptionPrunedCFG.make(ir.getControlFlowGraph());
				controlDependence = new ControlDependenceGraph<>(cfg);
			} catch (RuntimeException e) {
				// Without the control dependence, the call may run under any condition.
				LOG.warn("Can't compute the control dependence of " + node + ".", e);
				return Optional.of(new Verdict(LayerBuildShape.FIRST_CALL_ON_RETRACE,
						"the conditions of its calls in `" + getFunctionName(node) + "` can't be determined"));
			}

			// The blocks calling each callee object, since a layer is first called by whichever of its calls a trace reaches first. A call
			// whose callee the engine can't name is its own group.
			Map<Object, Set<ISSABasicBlock>> callingBlocks = new LinkedHashMap<>();
			Map<Object, SSAAbstractInvokeInstruction> firstCalls = new HashMap<>();

			for (CallSiteReference site : sites)
				for (SSAAbstractInvokeInstruction call : ir.getCalls(site)) {
					ISSABasicBlock block = ir.getControlFlowGraph().getBlockForInstruction(call.iIndex());

					// A call on no path that returns normally builds nothing a trace keeps. A call in an exception handler that returns
					// normally does: a retrace whose protected code raises, as a layer does for an input it wasn't built for, first calls
					// it.
					if (!cfg.containsNode(block)) {
						if (reachesExitNormally(ir.getControlFlowGraph(), block))
							return Optional.of(new Verdict(LayerBuildShape.FIRST_CALL_ON_RETRACE, "its call at "
									+ describePosition(node, call) + " runs in an exception handler, which a retrace may reach"));

						continue;
					}

					// A callee chosen among several layers by a value that may differ between traces may first call one on a retrace.
					if (!this.isSelectedFixedly(node, call.getUse(0), ir, cfg, controlDependence, new HashSet<>()))
						return Optional.of(new Verdict(LayerBuildShape.FIRST_CALL_ON_RETRACE,
								"its call at " + describePosition(node, call) + " may call a different layer on a retrace"));

					for (Object callee : this.getCallees(node, call)) {
						callingBlocks.computeIfAbsent(callee, k -> new LinkedHashSet<>()).add(block);
						firstCalls.putIfAbsent(callee, call);
					}
				}

			for (Map.Entry<Object, Set<ISSABasicBlock>> entry : callingBlocks.entrySet()) {
				Set<ISSABasicBlock> blocks = entry.getValue();
				Set<ISSABasicBlock> reachedFirst = getReachedAvoiding(cfg, cfg.entry(), blocks, false);
				Set<ISSABasicBlock> leaveWithout = getReachedAvoiding(cfg, cfg.exit(), blocks, true);

				// A controlling branch decides the first call when a trace can reach it without the call and leave it on a path to the exit
				// that avoids the call: one trace may then call the layer and another not. A branch whose every arm calls it decides
				// nothing.
				for (ISSABasicBlock block : blocks)
					for (ISSABasicBlock controller : getControllers(controlDependence, block))
						if (reachedFirst.contains(controller) && anySuccessorIn(cfg, controller, leaveWithout)
								&& this.varies(node, controller, ir))
							return Optional.of(new Verdict(LayerBuildShape.FIRST_CALL_ON_RETRACE,
									"its call at " + describePosition(node, firstCalls.get(entry.getKey()))
											+ " runs under a condition that can differ between traces"));
			}
		}

		return Optional.empty();
	}

	/**
	 * True iff {@code callee}, a call's callee value, calls the same layer on every trace: the engine allocates at most one object it may
	 * be, a layer or another, such as a bound method of an object holding one, told apart by allocating node, and not several, or it is
	 * read by a fixed selection ({@link #isReadFixedly(CGNode, int)}), or it merges several at a phi whose choosing branches are fixed
	 * ({@link #varies(CGNode, ISSABasicBlock, IR)}) and each of whose operands is selected fixedly in turn. A value that may be several
	 * layers and that no phi selects, such as a field or an element holding several, has no selecting value the engine shows fixed, so it
	 * isn't. A single layer called through a local alias is the same value and is.
	 *
	 * @param node The node of {@code callee}.
	 * @param callee A value number of {@code node}.
	 * @param ir The node's IR.
	 * @param cfg The node's control-flow graph, without exceptional edges.
	 * @param controlDependence The control dependence graph of {@code cfg}.
	 * @param seen The values already visited, along a cycle of phis.
	 * @return True iff the value is one layer on every trace.
	 */
	private boolean isSelectedFixedly(CGNode node, int callee, IR ir, PrunedCFG<SSAInstruction, ISSABasicBlock> cfg,
			ControlDependenceGraph<ISSABasicBlock> controlDependence, Set<Integer> seen) {
		if (!seen.add(callee))
			return true;

		Set<CGNode> allocators = new HashSet<>();
		boolean several = false;

		// Every object the callee may be, not only a layer: the call reaches `add_weight`, so a bound method of a plain object holding a
		// layer, such as `self.heads[i].run`, selects the layer as well.
		for (InstanceKey instanceKey : this.getPointerAnalysis()
				.getPointsToSet(this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, callee))) {
			AllocationSiteInNode site = findAllocationSiteInNode(instanceKey);

			if (site == null) {
				// A constant or a Python scalar has no allocation and holds no layer. Any other object whose allocation the engine
				// doesn't place can't be told apart from the others.
				if (instanceKey instanceof ConstantKey<?> || isPythonScalarKey(instanceKey))
					continue;

				return false;
			}

			allocators.add(site.getNode());
			several |= this.mayBeSeveral(instanceKey);
		}

		// One layer. One allocation that may make several, as a comprehension does, needs a fixed selection, in a callee too: a
		// callee's receiver is fixed, since the call into it had its own callee value checked so in its caller.
		if (allocators.isEmpty() || allocators.size() == 1 && !several)
			return true;

		if (!(node.getDU().getDef(callee) instanceof SSAPhiInstruction phi))
			return allocators.size() == 1 && this.isReadFixedly(node, callee);

		ISSABasicBlock merge = null;

		for (ISSABasicBlock block : ir.getControlFlowGraph())
			for (Iterator<SSAPhiInstruction> phis = block.iteratePhis(); phis.hasNext();)
				if (phis.next() == phi)
					merge = block;

		if (merge == null || !cfg.containsNode(merge))
			return false;

		// The branches choosing an incoming edge: each predecessor, and its controllers, other than the merge's own.
		Set<ISSABasicBlock> choosing = new LinkedHashSet<>();

		for (Iterator<ISSABasicBlock> predNodes = cfg.getPredNodes(merge); predNodes.hasNext();) {
			ISSABasicBlock pred = predNodes.next();
			choosing.add(pred);
			choosing.addAll(getControllers(controlDependence, pred));
		}

		choosing.removeAll(getControllers(controlDependence, merge));

		for (ISSABasicBlock branch : choosing)
			if (this.varies(node, branch, ir))
				return false;

		for (int i = 0; i < phi.getNumberOfUses(); i++)
			if (!this.isSelectedFixedly(node, phi.getUse(i), ir, cfg, controlDependence, seen))
				return false;

		return true;
	}

	/**
	 * True iff {@code value} of a program node reads one layer of an allocation that makes several by a fixed selection: the node's
	 * receiver, since each instance of the method traces its own function, and a callee's receiver is the callee value of a call checked so
	 * in its caller, a field or element of a value read so, under a key that is a literal or fixed ({@link #isFixed(CGNode, int)}), or the
	 * element of a loop over a fixed iterable. {@code self.blocks[i]} for a varying {@code i} isn't, in the method or in a function it
	 * calls; {@code self.d}, {@code self.blocks[0]}, {@code block} in {@code for block in self.blocks} and {@code self.blocks[i]} in
	 * {@code for i in range(self.n)} are.
	 *
	 * @param node A program node the method's trace runs.
	 * @param value A value number of {@code node}.
	 * @return True iff the value's selection is fixed across traces.
	 */
	private boolean isReadFixedly(CGNode node, int value) {
		SymbolTable symbolTable = node.getIR().getSymbolTable();
		int[] parameters = symbolTable.getParameterValueNumbers();

		// The receiver chooses: each instance traces its own function, so each first calls its own layer on its first trace.
		if (parameters.length > 1 && parameters[1] == value && this.getDeclaringClass(node).isPresent())
			return true;

		SSAInstruction def = node.getDU().getDef(value);

		if (def instanceof SSAGetInstruction get && !get.isStatic())
			return this.isReadFixedly(node, get.getRef());

		if (def instanceof AstPropertyRead read)
			return (symbolTable.isConstant(read.getMemberRef()) || this.isFixed(node, read.getMemberRef()))
					&& this.isReadFixedly(node, read.getObjectRef());

		// A loop over a fixed iterable, `next` of `iter` of it, calls every element in the same order on every trace, so each first call is
		// on the first trace.
		if (def instanceof PythonInvokeInstruction invoke && invoke.getKeywords().isEmpty() && this.callsFixedBuiltin(node, invoke))
			return this.isFixed(node, value);

		return false;
	}

	/**
	 * True iff {@code invoke} calls one of the builtins whose result is the same on every trace when its arguments are:
	 * {@link #FIXED_BUILTIN_NAMES}, through which the front end also lowers a {@code for} loop, as {@code next} of {@code iter} of the
	 * iterable.
	 *
	 * @param node The node of {@code invoke}.
	 * @param invoke A call.
	 * @return True iff the call's only callee is one of those builtins.
	 */
	private boolean callsFixedBuiltin(CGNode node, PythonInvokeInstruction invoke) {
		return FIXED_BUILTIN_NAMES.stream().anyMatch(name -> Util.callsBuiltin(this.builder, node, invoke, name));
	}

	/**
	 * True iff the exit of {@code cfg} is reachable from {@code block} along normal edges: a block reached only through an exception, such
	 * as a handler's, may still return normally.
	 *
	 * @param cfg A control-flow graph, with its exceptional edges.
	 * @param block One of its blocks.
	 * @return True iff a path of normal edges leads from {@code block} to the exit.
	 */
	private static boolean reachesExitNormally(SSACFG cfg, ISSABasicBlock block) {
		Set<ISSABasicBlock> seen = new HashSet<>(List.of(block));
		Deque<ISSABasicBlock> worklist = new ArrayDeque<>(List.of(block));

		while (!worklist.isEmpty()) {
			ISSABasicBlock current = worklist.remove();

			if (current.equals(cfg.exit()))
				return true;

			for (ISSABasicBlock next : cfg.getNormalSuccessors(current))
				if (seen.add(next))
					worklist.add(next);
		}

		return false;
	}

	/**
	 * The objects {@code call} may call, as the points-to set of its callee value gives them, or the call itself when that set is empty.
	 *
	 * @param node The node of {@code call}.
	 * @param call A call.
	 * @return The callee objects, or the call when the engine names none.
	 */
	private Set<Object> getCallees(CGNode node, SSAAbstractInvokeInstruction call) {
		Set<Object> ret = new LinkedHashSet<>();
		this.getPointerAnalysis().getPointsToSet(this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, call.getUse(0)))
				.forEach(ret::add);

		if (ret.isEmpty())
			ret.add(call);

		return ret;
	}

	/**
	 * The blocks reached from {@code start} without passing through {@code avoided}, forward along successors or backward along
	 * predecessors. A block of {@code avoided} is not reached.
	 *
	 * @param cfg A control-flow graph.
	 * @param start The block to start from.
	 * @param avoided The blocks not to pass through.
	 * @param backward True to follow predecessors, false to follow successors.
	 * @return The blocks reached.
	 */
	private static Set<ISSABasicBlock> getReachedAvoiding(PrunedCFG<SSAInstruction, ISSABasicBlock> cfg, ISSABasicBlock start,
			Set<ISSABasicBlock> avoided, boolean backward) {
		Set<ISSABasicBlock> ret = new HashSet<>();

		if (avoided.contains(start))
			return ret;

		Deque<ISSABasicBlock> worklist = new ArrayDeque<>(List.of(start));
		ret.add(start);

		while (!worklist.isEmpty()) {
			ISSABasicBlock current = worklist.remove();

			for (Iterator<ISSABasicBlock> next = backward ? cfg.getPredNodes(current) : cfg.getSuccNodes(current); next.hasNext();) {
				ISSABasicBlock block = next.next();

				if (!avoided.contains(block) && ret.add(block))
					worklist.add(block);
			}
		}

		return ret;
	}

	/**
	 * True iff a successor of {@code block} is in {@code blocks}.
	 *
	 * @param cfg A control-flow graph.
	 * @param block A block of {@code cfg}.
	 * @param blocks Some blocks.
	 * @return True iff {@code block} has a successor in {@code blocks}.
	 */
	private static boolean anySuccessorIn(PrunedCFG<SSAInstruction, ISSABasicBlock> cfg, ISSABasicBlock block, Set<ISSABasicBlock> blocks) {
		for (Iterator<ISSABasicBlock> succNodes = cfg.getSuccNodes(block); succNodes.hasNext();)
			if (blocks.contains(succNodes.next()))
				return true;

		return false;
	}

	/**
	 * The blocks {@code block} is transitively control dependent on.
	 *
	 * @param controlDependence The control dependence graph of the block's control-flow graph.
	 * @param block A block.
	 * @return The blocks deciding whether {@code block} runs, directly or through others.
	 */
	private static Set<ISSABasicBlock> getControllers(ControlDependenceGraph<ISSABasicBlock> controlDependence, ISSABasicBlock block) {
		Set<ISSABasicBlock> ret = new LinkedHashSet<>();
		Deque<ISSABasicBlock> worklist = new ArrayDeque<>(List.of(block));

		while (!worklist.isEmpty())
			for (Iterator<ISSABasicBlock> predNodes = controlDependence.getPredNodes(worklist.remove()); predNodes.hasNext();) {
				ISSABasicBlock controller = predNodes.next();

				if (ret.add(controller))
					worklist.add(controller);
			}

		return ret;
	}

	/**
	 * True iff {@code controller} ends in a branch whose condition may differ between traces. A block that ends otherwise, such as one
	 * ending in a call that may raise, decides nothing a retrace changes. The branch's literal operands, the constants of the IR it
	 * compares against, are fixed; each other operand, the condition value, is fixed when {@link #isFixed(CGNode, int)} shows it.
	 *
	 * @param node The node of {@code controller}.
	 * @param controller A block a call is control dependent on.
	 * @param ir The node's IR.
	 * @return True iff the block ends in a conditional branch or switch one of whose operands may vary.
	 */
	private boolean varies(CGNode node, ISSABasicBlock controller, IR ir) {
		int last = controller.getLastInstructionIndex();

		if (last < 0)
			return false;

		SSAInstruction instruction = ir.getInstructions()[last];

		if (!(instruction instanceof SSAConditionalBranchInstruction) && !(instruction instanceof SSASwitchInstruction))
			return false;

		for (int i = 0; i < instruction.getNumberOfUses(); i++) {
			int value = instruction.getUse(i);

			if (!ir.getSymbolTable().isConstant(value) && !this.isFixed(node, value))
				return true;
		}

		return false;
	}

	/**
	 * True iff {@code value} of {@code node} is the same on every trace of the function: the engine shows it
	 * ({@link #isFixedByEngine(CGNode, int)}), or it is built only from constants, the method's receiver, fields of fixed objects that are
	 * written only while their objects are set up, in {@code __init__} or a layer's {@code build}, operators over fixed operands, and
	 * {@code range}, {@code len}, {@code iter} or {@code next} of fixed arguments, the last two being how a loop over a fixed iterable
	 * steps, following the value back through the IR's def-use chains. A parameter other than the receiver, a field another method rebinds,
	 * any other call, a merge of values at a phi, and a value the walk can't trace may vary.
	 *
	 * @param node The node of {@code value}.
	 * @param value A value number of {@code node}.
	 * @return True iff the value is the same on every trace.
	 */
	private boolean isFixed(CGNode node, int value) {
		return this.isFixed(node, value, new HashSet<>());
	}

	/**
	 * Decides {@link #isFixed(CGNode, int)}.
	 *
	 * @param node The node of {@code value}.
	 * @param value A value number of {@code node}.
	 * @param seen The values already being decided, along a cycle of definitions, which is taken to vary.
	 * @return True iff the value is the same on every trace.
	 */
	private boolean isFixed(CGNode node, int value, Set<Integer> seen) {
		if (this.isFixedByEngine(node, value))
			return true;

		IR ir = node.getIR();

		if (ir == null || !seen.add(value))
			return false;

		SymbolTable symbolTable = ir.getSymbolTable();

		if (symbolTable.isConstant(value))
			return true;

		int[] parameters = symbolTable.getParameterValueNumbers();

		for (int i = 0; i < parameters.length; i++)
			if (parameters[i] == value)
				// Slot 1 of a method is its receiver, the same object on every trace of its own traced function.
				return i == 1 && this.isReceiver(node, value);

		SSAInstruction def = node.getDU().getDef(value);

		if (def instanceof AstPropertyRead read)
			return this.isFixed(node, read.getObjectRef(), seen)
					&& (symbolTable.isConstant(read.getMemberRef()) || this.isFixed(node, read.getMemberRef(), seen))
					&& this.getReboundState().readsSetUpStateOnly(node, read);

		if (def instanceof SSAGetInstruction get)
			return !get.isStatic() && this.isFixed(node, get.getRef(), seen) && this.getReboundState().readsSetUpStateOnly(node, get);

		if (def instanceof SSABinaryOpInstruction || def instanceof SSAUnaryOpInstruction)
			return this.usesAreFixed(node, def, seen);

		// A loop's test reads the next element of its iterable, which is the same sequence on every trace when the iterable is fixed.
		if (def instanceof EachElementGetInstruction || def instanceof EachElementHasNextInstruction)
			return this.isFixed(node, def.getUse(0), seen);

		if (def instanceof PythonInvokeInstruction invoke && invoke.getKeywords().isEmpty() && this.callsFixedBuiltin(node, invoke)) {
			for (int i = 1; i < invoke.getNumberOfPositionalParameters(); i++)
				if (!this.isFixed(node, invoke.getUse(i), seen))
					return false;

			return true;
		}

		return false;
	}

	/**
	 * True iff every operand of {@code instruction} is fixed ({@link #isFixed(CGNode, int, Set)}).
	 *
	 * @param node The node of {@code instruction}.
	 * @param instruction An instruction of {@code node}.
	 * @param seen The values already being decided.
	 * @return True iff each use of {@code instruction} is fixed.
	 */
	private boolean usesAreFixed(CGNode node, SSAInstruction instruction, Set<Integer> seen) {
		for (int i = 0; i < instruction.getNumberOfUses(); i++)
			if (!this.isFixed(node, instruction.getUse(i), seen))
				return false;

		return true;
	}

	/**
	 * True iff {@code value}, the second parameter slot of {@code node}, is the receiver of a method: the function is declared in a class,
	 * and every object the slot may be is an instance of that class or a subclass, per the engine's allocated classes and their bases.
	 *
	 * @param node A node.
	 * @param value The value number of {@code node}'s second parameter slot.
	 * @return True iff the slot holds the method's receiver.
	 */
	private boolean isReceiver(CGNode node, int value) {
		Optional<TypeName> declaringClass = this.getDeclaringClass(node);

		if (declaringClass.isEmpty())
			return false;

		OrdinalSet<InstanceKey> receivers = this.getPointerAnalysis()
				.getPointsToSet(this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, value));

		if (receivers.isEmpty())
			return false;

		for (InstanceKey receiver : receivers) {
			Optional<IClass> allocatedClass = allocatedClass(receiver);

			if (allocatedClass.isEmpty()
					|| !Function.getClassAndBaseNames(this.builder.getClassHierarchy(), allocatedClass.get().getReference())
							.contains(declaringClass.get()))
				return false;
		}

		return true;
	}

	/**
	 * The analysis of which fields are written outside their objects' set-up methods, created on first use.
	 *
	 * @return The rebound-state analysis over this call graph.
	 */
	private ReboundStateReadAnalysis getReboundState() {
		if (this.reboundState == null)
			this.reboundState = new ReboundStateReadAnalysis(this.callGraph, this.getPointerAnalysis(), this.tensorTypedKeys);

		return this.reboundState;
	}

	/**
	 * True iff the engine shows that {@code value}, a branch's condition, takes the same path on every trace: its points-to set is one
	 * constant, the same in every context of the node's function, so it is one literal or module constant whatever the call, or the engine
	 * types it as a tensor, and no Python value, in every context, whose branch AutoGraph traces both arms of, so the first trace makes the
	 * first call. Anything else may vary: a value whose points-to set is empty or holds two constants, and a mix, such as a tensor in one
	 * context or call and {@code None} or a Python {@code bool} in another, which takes a Python branch on one trace and a tensor branch on
	 * another. A context whose points-to set is empty but which the engine types as a tensor counts as a tensor with no Python value, so a
	 * Python value the engine doesn't model, merged there, is hidden; that is rare, since a Python {@code bool} mostly has a key.
	 *
	 * @param node The node of {@code value}.
	 * @param value A value number of {@code node}.
	 * @return True iff the engine shows the value fixed across traces.
	 */
	private boolean isFixedByEngine(CGNode node, int value) {
		boolean anyTensor = false;
		Set<InstanceKey> constants = new HashSet<>();

		for (CGNode context : this.callGraph.getNodes(node.getMethod().getReference())) {
			PointerKey pointerKey = this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(context, value);
			OrdinalSet<InstanceKey> pointsTo = this.getPointerAnalysis().getPointsToSet(pointerKey);
			boolean pythonValue = false;

			for (InstanceKey instanceKey : pointsTo)
				if (instanceKey instanceof ConstantKey<?> || isPythonScalarKey(instanceKey))
					pythonValue = true;

			if (this.tensorTypedKeys.containsKey(pointerKey) && !pythonValue)
				anyTensor = true;
			else if (pointsTo.size() == 1 && pointsTo.iterator().next() instanceof ConstantKey<?> constant)
				constants.add(constant);
			else
				// Empty in a context is unknown there; a tensor alongside a Python value is a mix.
				return false;
		}

		// Every context a tensor, or every context the same one constant.
		return anyTensor ? constants.isEmpty() : constants.size() == 1;
	}

	/**
	 * The class declaring {@code node}'s function, if it is a program function declared in a class: the engine names a function by its
	 * enclosing scope's type followed by its own name.
	 *
	 * @param node A node.
	 * @return The type name of the class declaring the node's function, or empty if the function isn't program code declared in a class.
	 */
	private Optional<TypeName> getDeclaringClass(CGNode node) {
		if (!(node.getMethod() instanceof AstMethod))
			return Optional.empty();

		String name = node.getMethod().getDeclaringClass().getName().toString();
		int slash = name.lastIndexOf('/');

		if (slash < 0)
			return Optional.empty();

		TypeName ret = TypeName.string2TypeName(name.substring(0, slash));
		IClass cls = this.builder.getClassHierarchy()
				.lookupClass(TypeReference.findOrCreate(node.getMethod().getDeclaringClass().getClassLoader().getReference(), ret));
		return cls instanceof IPythonClass ? Optional.of(ret) : Optional.empty();
	}

	/**
	 * True iff {@code node} is a layer's own {@code __init__} or {@code build}: a program function of that name declared in a class
	 * reaching {@code tf.keras.layers.Layer} or {@code tf.keras.Model}. Each sets the layer up once, before the layer's first call.
	 *
	 * @param node A node.
	 * @return True iff {@code node} sets a layer up.
	 */
	private boolean isLayerSetUp(CGNode node) {
		String name = getFunctionName(node);
		return (name.equals(PythonTypes.INIT_METHOD_NAME) || name.equals(PythonTypes.KERAS_BUILD_METHOD_NAME))
				&& this.getDeclaringClass(node).map(this::isLayerClass).orElse(false);
	}

	/**
	 * True iff {@code node} is a layer's own {@code build}, which Keras runs once per layer.
	 *
	 * @param node A node.
	 * @return True iff {@code node} is a layer's {@code build}.
	 */
	private boolean isLayerBuild(CGNode node) {
		return getFunctionName(node).equals(PythonTypes.KERAS_BUILD_METHOD_NAME) && this.isLayerSetUp(node);
	}

	/**
	 * The name of {@code node}'s function, the last segment of its type's name.
	 *
	 * @param node A node.
	 * @return The function's name.
	 */
	private static String getFunctionName(CGNode node) {
		return getSimpleName(node.getMethod().getDeclaringClass().getName());
	}

	/**
	 * The last segment of {@code name}.
	 *
	 * @param name A type name.
	 * @return The segment after the last slash.
	 */
	private static String getSimpleName(TypeName name) {
		String string = name.toString();
		return string.substring(string.lastIndexOf('/') + 1);
	}

	/**
	 * True iff the class {@code type} names reaches {@code tf.keras.layers.Layer} or {@code tf.keras.Model} through its bases.
	 *
	 * @param type A class.
	 * @return True iff the class is a layer or a model.
	 */
	private boolean isLayerClass(TypeReference type) {
		return !Collections.disjoint(Function.getClassAndBaseNames(this.builder.getClassHierarchy(), type), KERAS_TYPE_NAMES);
	}

	/**
	 * True iff the class {@code name} names reaches {@code tf.keras.layers.Layer} or {@code tf.keras.Model} through its bases.
	 *
	 * @param name A class's type name.
	 * @return True iff the class is a layer or a model.
	 */
	private boolean isLayerClass(TypeName name) {
		return !Collections.disjoint(Function.getClassAndBaseNames(this.builder.getClassHierarchy(), name), KERAS_TYPE_NAMES);
	}

	/**
	 * True iff {@code node} may reach {@code Layer.add_weight} ({@link #mayBuildLayers(CGNode)}), memoized.
	 *
	 * @param node A node.
	 * @return True iff a layer's variable creation may be reachable from {@code node}.
	 */
	private boolean reachesAddWeight(CGNode node) {
		return this.reachesAddWeight.computeIfAbsent(node, this::mayBuildLayers);
	}

	/**
	 * Finds two sublayers reachable from the receiver of {@code nodes} that may carry the same name. A sublayer is told apart by the node
	 * allocating it, a constructor or a summary in its context: the engine allocates the bound methods of a program layer in the layer's
	 * constructor too, and they stand for the same layer.
	 *
	 * @param nodes The method's nodes.
	 * @return {@link LayerBuildShape#COLLIDING_NAMES} with the name, or empty if every name is unique.
	 */
	private Optional<Verdict> findCollidingNames(Set<CGNode> nodes) {
		Set<InstanceKey> receivers = new HashSet<>();

		for (CGNode node : nodes) {
			IR ir = node.getIR();

			if (ir != null && ir.getSymbolTable().getParameterValueNumbers().length > 1)
				this.getPointerAnalysis()
						.getPointsToSet(
								this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, ir.getSymbolTable().getParameter(1)))
						.forEach(receivers::add);
		}

		Sublayers sublayers = this.getSublayers(receivers);

		if (sublayers == null)
			return Optional
					.of(new Verdict(LayerBuildShape.COLLIDING_NAMES, "a sublayer's allocation, and so its name, can't be determined"));

		Map<CGNode, Names> names = new HashMap<>();

		for (Map.Entry<CGNode, Set<InstanceKey>> sublayer : sublayers.objects().entrySet()) {
			Names sublayerNames = this.getNames(sublayer.getKey(), sublayer.getValue());

			if (sublayerNames.computed())
				return Optional.of(new Verdict(LayerBuildShape.COLLIDING_NAMES,
						"a `" + getSimpleName(allocatedClass(sublayer.getValue().iterator().next()).orElseThrow().getName())
								+ "` sublayer's name is computed"));

			names.put(sublayer.getKey(), sublayerNames);
		}

		// A sublayer's variables are named under its parent's scope, the receiver's dropped, so only siblings may collide.
		Map<CGNode, Map<String, CGNode>> constantsByParent = new HashMap<>();
		Map<CGNode, Map<String, CGNode>> templatesByParent = new HashMap<>();

		for (Map.Entry<CGNode, Set<InstanceKey>> sublayer : sublayers.objects().entrySet()) {
			CGNode allocator = sublayer.getKey();
			Names sublayerNames = names.get(allocator);

			for (CGNode parent : sublayers.parents().get(allocator)) {
				Map<String, CGNode> constants = constantsByParent.computeIfAbsent(parent, k -> new HashMap<>());
				Map<String, CGNode> templates = templatesByParent.computeIfAbsent(parent, k -> new HashMap<>());

				for (String name : sublayerNames.constants()) {
					CGNode other = constants.putIfAbsent(name, allocator);

					if (other != null && !other.equals(allocator))
						return Optional.of(new Verdict(LayerBuildShape.COLLIDING_NAMES, "several sublayers may be named `" + name + "`"));
				}

				for (String template : sublayerNames.templates()) {
					CGNode other = templates.putIfAbsent(template, allocator);

					if (other != null && !other.equals(allocator))
						return Optional
								.of(new Verdict(LayerBuildShape.COLLIDING_NAMES, "several sublayers may be named `" + template + "`"));
				}
			}

			// One allocator stands for several layers when one of its objects may be several, which collide unless each is named apart.
			if (sublayerNames.named() && sublayer.getValue().stream().anyMatch(this::mayBeSeveral)
					&& !this.isNamedApartPerRun(allocator, sublayers, names, new HashSet<>()))
				return Optional.of(new Verdict(LayerBuildShape.COLLIDING_NAMES, "several sublayers may be named `"
						+ (sublayerNames.constants().isEmpty() ? sublayerNames.templates() : sublayerNames.constants()).iterator().next()
						+ "`"));
		}

		// A template a loop's index fills may produce a sibling's constant name, or a name another template fills.
		for (Map.Entry<CGNode, Map<String, CGNode>> parent : templatesByParent.entrySet()) {
			List<String> templates = new ArrayList<>(parent.getValue().keySet());

			for (String template : templates)
				for (String constant : constantsByParent.getOrDefault(parent.getKey(), Collections.emptyMap()).keySet())
					if (fills(template, constant))
						return Optional
								.of(new Verdict(LayerBuildShape.COLLIDING_NAMES, "several sublayers may be named `" + constant + "`"));

			for (int i = 0; i < templates.size(); i++)
				for (int j = i + 1; j < templates.size(); j++)
					if (mayFillAlike(templates.get(i), templates.get(j)))
						return Optional.of(new Verdict(LayerBuildShape.COLLIDING_NAMES,
								"sublayers named `" + templates.get(i) + "` and `" + templates.get(j) + "` may share a name"));
		}

		return Optional.empty();
	}

	/**
	 * True iff each layer {@code allocator} allocates is named apart from the others it allocates: its one constructing call runs once per
	 * iteration of the one loop whose index fills its name ({@link #getIndexedName(CGNode, int)}), or once per run of a node that runs
	 * once, or once per run of a layer's {@code __init__} or {@code build}, which runs once per layer, every one of whose layers is a
	 * sublayer named apart in turn or carrying a unique name Keras makes. The receiver isn't such a layer: decorating the method drops its
	 * scope, so when it may be several objects, their sublayers collide.
	 *
	 * @param allocator The node allocating a sublayer.
	 * @param sublayers The receiver's sublayers.
	 * @param names The names each sublayer may carry, by its allocating node.
	 * @param seen The allocators already visited, along a cycle.
	 * @return True iff the allocator's layers are named apart.
	 */
	private boolean isNamedApartPerRun(CGNode allocator, Sublayers sublayers, Map<CGNode, Names> names, Set<CGNode> seen) {
		if (!seen.add(allocator))
			return false;

		List<CGNode> callers = new ArrayList<>();
		List<PythonInvokeInstruction> calls = new ArrayList<>();

		for (CGNode caller : Iterator2Iterable.make(this.callGraph.getPredNodes(allocator)))
			for (CallSiteReference site : Iterator2Iterable.make(this.callGraph.getPossibleSites(caller, allocator)))
				for (SSAAbstractInvokeInstruction call : caller.getIR().getCalls(site))
					if (call instanceof PythonInvokeInstruction invoke) {
						callers.add(caller);
						calls.add(invoke);
					} else
						return false;

		if (calls.size() != 1 || calls.get(0).isComprehension())
			return false;

		CGNode caller = callers.get(0);
		PythonInvokeInstruction call = calls.get(0);

		// In a loop, the name must be filled from that loop's index, and the call must lie in no other loop.
		if (isInLoop(caller.getIR(), call.iIndex())) {
			Optional<IndexedName> indexed = call.getUse(NAME_KEYWORD) < 0 ? Optional.empty()
					: this.getIndexedName(caller, call.getUse(NAME_KEYWORD));

			if (indexed.isEmpty() || !names.get(allocator).constants().isEmpty()
					|| !isInOnlyLoopOf(caller.getIR(), call, indexed.get().index()))
				return false;
		}

		if (!this.mayRunMoreThanOnce(caller))
			return true;

		// A layer's set-up runs once per layer, each of which must be a sublayer named apart, or one Keras names uniquely.
		if (!this.isLayerSetUp(caller))
			return false;

		for (InstanceKey layer : this.getPointerAnalysis().getPointsToSet(
				this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, caller.getIR().getSymbolTable().getParameter(1)))) {
			AllocationSiteInNode site = findAllocationSiteInNode(layer);

			if (site == null || !sublayers.objects().containsKey(site.getNode()))
				return false;

			if (names.get(site.getNode()).named() && !this.isNamedApartPerRun(site.getNode(), sublayers, names, seen))
				return false;
		}

		return true;
	}

	/**
	 * The name {@code value} of {@code node} if a loop's index fills it: {@code template.format(i)}, a call of the {@code format} attribute
	 * of a string constant with one {@code {}} field and no other brace, with {@code i} its only argument, the {@code next} of an
	 * {@code iter} of a {@code range}, as the front end lowers {@code for i in range(n)}. Distinct values of {@code i} fill distinct names.
	 * A reassigned index, an {@code enumerate}, or an f-string, which the front end reads as the constant of its source text, isn't
	 * matched.
	 *
	 * @param node The node of {@code value}.
	 * @param value A value number of {@code node}, a {@code name} argument.
	 * @return The indexed name, or empty if the name isn't filled from a loop's index.
	 */
	private Optional<IndexedName> getIndexedName(CGNode node, int value) {
		SymbolTable symbolTable = node.getIR().getSymbolTable();

		if (!(node.getDU().getDef(value) instanceof PythonInvokeInstruction format) || !format.getKeywords().isEmpty()
				|| format.getNumberOfPositionalParameters() != 2)
			return Optional.empty();

		if (!(node.getDU().getDef(format.getUse(0)) instanceof AstPropertyRead read) || !symbolTable.isStringConstant(read.getMemberRef())
				|| !symbolTable.getStringValue(read.getMemberRef()).equals(FORMAT_METHOD_NAME)
				|| !symbolTable.isStringConstant(read.getObjectRef()))
			return Optional.empty();

		String template = symbolTable.getStringValue(read.getObjectRef());

		if (!template.replaceFirst("\\{\\}", "").chars().noneMatch(c -> c == '{' || c == '}') || !template.contains("{}"))
			return Optional.empty();

		if (!(node.getDU().getDef(format.getUse(1)) instanceof PythonInvokeInstruction next) || !this.callsBuiltinOnly(node, next, "next")
				|| !(node.getDU().getDef(next.getUse(1)) instanceof PythonInvokeInstruction iter)
				|| !this.callsBuiltinOnly(node, iter, "iter")
				|| !(node.getDU().getDef(iter.getUse(1)) instanceof PythonInvokeInstruction range)
				|| !this.callsBuiltinOnly(node, range, "range"))
			return Optional.empty();

		// The iterator must be made once, outside the loop, so each `next` advances it.
		if (isInLoop(node.getIR(), iter.iIndex()) && sameCycle(node.getIR(), iter.iIndex(), next.iIndex()))
			return Optional.empty();

		return Optional.of(new IndexedName(template, next));
	}

	/**
	 * True iff two templates, each with one {@code {}} field a loop's index fills, may fill the same name, as {@code "h{}"} with {@code 10}
	 * and {@code "h1{}"} with {@code 0} both fill {@code h10}. A fill is the string of an integer, so a shared name requires one template's
	 * text before its field to start the other's, and one's text after its field to end the other's. Templates failing either, as
	 * {@code "h{}"} and {@code "ln_{}"} do, never share a name.
	 *
	 * @param first A template with one {@code {}} field.
	 * @param second Another template with one {@code {}} field.
	 * @return True iff the templates may fill the same name.
	 */
	private static boolean mayFillAlike(String first, String second) {
		int firstField = first.indexOf("{}");
		int secondField = second.indexOf("{}");
		String firstPrefix = first.substring(0, firstField);
		String secondPrefix = second.substring(0, secondField);
		String firstSuffix = first.substring(firstField + 2);
		String secondSuffix = second.substring(secondField + 2);
		return (firstPrefix.startsWith(secondPrefix) || secondPrefix.startsWith(firstPrefix))
				&& (firstSuffix.endsWith(secondSuffix) || secondSuffix.endsWith(firstSuffix));
	}

	/**
	 * True iff {@code invoke} calls the builtin {@code name} with one argument and no keywords.
	 *
	 * @param node The node of {@code invoke}.
	 * @param invoke A call.
	 * @param name The builtin's name.
	 * @return True iff the call is {@code name(x)}.
	 */
	private boolean callsBuiltinOnly(CGNode node, PythonInvokeInstruction invoke, String name) {
		return invoke.getKeywords().isEmpty() && invoke.getNumberOfPositionalParameters() == 2
				&& Util.callsBuiltin(this.builder, node, invoke, name);
	}

	/**
	 * True iff {@code name} is a name {@code template} may be filled into: the template's text around its one {@code {}} field, with any
	 * text between.
	 *
	 * @param template A template with one {@code {}} field.
	 * @param name A constant name.
	 * @return True iff the name may be one the template produces.
	 */
	private static boolean fills(String template, String name) {
		int field = template.indexOf("{}");
		String prefix = template.substring(0, field);
		String suffix = template.substring(field + 2);
		return name.length() >= prefix.length() + suffix.length() && name.startsWith(prefix) && name.endsWith(suffix);
	}

	/**
	 * True iff {@code call} lies in the loop of {@code index}, the {@code next} defining a loop's index, and in no other: the blocks on a
	 * cycle through the index's block hold the call's, and, without the loop's one entry, hold no cycle.
	 *
	 * @param ir An IR.
	 * @param call A call of {@code ir}.
	 * @param index The {@code next} call of {@code ir} defining a loop's index.
	 * @return True iff the call runs once per iteration of that loop alone.
	 */
	private static boolean isInOnlyLoopOf(IR ir, SSAInstruction call, SSAInstruction index) {
		SSACFG cfg = ir.getControlFlowGraph();
		Set<ISSABasicBlock> loop = getCycle(cfg, cfg.getBlockForInstruction(index.iIndex()));

		if (!loop.contains(cfg.getBlockForInstruction(call.iIndex())))
			return false;

		// The loop's entries: its blocks with a predecessor outside it.
		Set<ISSABasicBlock> entries = new HashSet<>();

		for (ISSABasicBlock block : loop)
			for (Iterator<ISSABasicBlock> preds = cfg.getPredNodes(block); preds.hasNext();)
				if (!loop.contains(preds.next()))
					entries.add(block);

		if (entries.size() != 1)
			return false;

		// Without its entry, the loop holds another loop iff a cycle remains.
		ISSABasicBlock entry = entries.iterator().next();

		for (ISSABasicBlock block : loop)
			if (!block.equals(entry)) {
				Set<ISSABasicBlock> seen = new HashSet<>();
				Deque<ISSABasicBlock> worklist = new ArrayDeque<>();
				cfg.getSuccNodes(block).forEachRemaining(worklist::add);

				while (!worklist.isEmpty()) {
					ISSABasicBlock next = worklist.remove();

					if (next.equals(block))
						return false;

					if (!next.equals(entry) && loop.contains(next) && seen.add(next))
						cfg.getSuccNodes(next).forEachRemaining(worklist::add);
				}
			}

		return true;
	}

	/**
	 * True iff the instructions at {@code first} and {@code second} of {@code ir} lie on one cycle.
	 *
	 * @param ir An IR.
	 * @param first An instruction index.
	 * @param second An instruction index.
	 * @return True iff some cycle holds both instructions' blocks.
	 */
	private static boolean sameCycle(IR ir, int first, int second) {
		SSACFG cfg = ir.getControlFlowGraph();
		return getCycle(cfg, cfg.getBlockForInstruction(first)).contains(cfg.getBlockForInstruction(second));
	}

	/**
	 * The blocks of {@code cfg} on a cycle through {@code block}: those it reaches that reach it.
	 *
	 * @param cfg A control-flow graph.
	 * @param block One of its blocks.
	 * @return The blocks of the strongly connected component holding {@code block}, empty if it lies on no cycle.
	 */
	private static Set<ISSABasicBlock> getCycle(SSACFG cfg, ISSABasicBlock block) {
		Set<ISSABasicBlock> forward = new HashSet<>();
		Deque<ISSABasicBlock> worklist = new ArrayDeque<>();
		cfg.getSuccNodes(block).forEachRemaining(worklist::add);

		while (!worklist.isEmpty()) {
			ISSABasicBlock next = worklist.remove();

			if (forward.add(next))
				cfg.getSuccNodes(next).forEachRemaining(worklist::add);
		}

		if (!forward.contains(block))
			return Collections.emptySet();

		Set<ISSABasicBlock> backward = new HashSet<>();
		cfg.getPredNodes(block).forEachRemaining(worklist::add);

		while (!worklist.isEmpty()) {
			ISSABasicBlock next = worklist.remove();

			if (backward.add(next))
				cfg.getPredNodes(next).forEachRemaining(worklist::add);
		}

		forward.retainAll(backward);
		return forward;
	}

	/**
	 * The layers and models reachable from {@code receivers} through fields, at any depth, by the node allocating them, in a stable order,
	 * with the parent of each: the nearest layer holding it, or {@code null} for the receiver. The node allocating a receiver is left out.
	 *
	 * @param receivers The objects the receiver may be.
	 * @return The sublayers, or {@code null} if the engine doesn't place a sublayer's allocation.
	 */
	private Sublayers getSublayers(Set<InstanceKey> receivers) {
		Map<InstanceKey, Set<InstanceFieldPointerKey>> fields = getFieldIndex(this.getPointerAnalysis());
		Set<CGNode> receiverAllocators = new HashSet<>();

		for (InstanceKey receiver : receivers) {
			AllocationSiteInNode site = findAllocationSiteInNode(receiver);

			if (site != null)
				receiverAllocators.add(site.getNode());
		}

		// The nearest layers holding each object reached, by their allocating nodes; null is the receiver.
		Map<InstanceKey, Set<CGNode>> holders = new HashMap<>();
		Deque<InstanceKey> worklist = new ArrayDeque<>();

		for (InstanceKey receiver : receivers) {
			holders.computeIfAbsent(receiver, k -> new HashSet<>()).add(null);
			worklist.add(receiver);
		}

		Map<CGNode, Set<InstanceKey>> objects = new TreeMap<>(Comparator.comparing(Object::toString));
		Map<CGNode, Set<CGNode>> parents = new HashMap<>();

		while (!worklist.isEmpty()) {
			InstanceKey holder = worklist.remove();

			// What a sublayer holds is under its scope; what any other object holds is under its holders'.
			Set<CGNode> scopes = holders.get(holder);
			AllocationSiteInNode holderSite = findAllocationSiteInNode(holder);

			if (holderSite != null && objects.containsKey(holderSite.getNode()))
				scopes = Collections.singleton(holderSite.getNode());

			for (InstanceFieldPointerKey field : fields.getOrDefault(holder, Collections.emptySet()))
				for (InstanceKey value : this.getPointerAnalysis().getPointsToSet(field)) {
					Optional<IClass> allocatedClass = allocatedClass(value);
					boolean isLayer = allocatedClass.isPresent() && this.isLayerClass(allocatedClass.get().getReference());

					if (isLayer) {
						AllocationSiteInNode site = findAllocationSiteInNode(value);

						// A sublayer whose allocation the engine doesn't place has names that can't be read.
						if (site == null)
							return null;

						// A layer of the receiver's own class that the receiver's allocator also builds, as a recursive layer's
						// children are, is left out with it, and so are its children's names. It's rare.
						if (!receiverAllocators.contains(site.getNode())) {
							objects.computeIfAbsent(site.getNode(), k -> new HashSet<>()).add(value);
							parents.computeIfAbsent(site.getNode(), k -> new HashSet<>()).addAll(scopes);
						}
					}

					// An object is revisited only when it gains a holder; a sublayer's own fields are under its scope whatever holds it.
					if (holders.computeIfAbsent(value, k -> new HashSet<>()).addAll(scopes))
						worklist.add(value);
				}
		}

		return new Sublayers(objects, parents);
	}

	/**
	 * The fields of each object of {@code pointerAnalysis}, built once per pointer analysis.
	 *
	 * @param pointerAnalysis A pointer analysis.
	 * @return The field keys of each object.
	 */
	private static Map<InstanceKey, Set<InstanceFieldPointerKey>> getFieldIndex(PointerAnalysis<InstanceKey> pointerAnalysis) {
		return fieldIndexCache.computeIfAbsent(pointerAnalysis, pa -> {
			Map<InstanceKey, Set<InstanceFieldPointerKey>> ret = new HashMap<>();

			for (PointerKey pointerKey : pa.getPointerKeys())
				if (pointerKey instanceof InstanceFieldPointerKey field)
					ret.computeIfAbsent(field.getInstanceKey(), k -> new HashSet<>()).add(field);

			return ret;
		});
	}

	/**
	 * The names the sublayer {@code allocator} allocates may carry: the {@code name} argument of each call constructing it, and, for a
	 * program class, the {@code name} argument of each {@code __init__} call its class's {@code __init__} makes, such as
	 * {@code super().__init__(name=name)}, whose points-to set holds the constructing call's argument or the parameter's default.
	 *
	 * @param allocator The node allocating the sublayer.
	 * @param objects The sublayer's objects {@code allocator} allocates.
	 * @return The names the sublayer may carry.
	 */
	private Names getNames(CGNode allocator, Set<InstanceKey> objects) {
		Set<String> constants = new TreeSet<>();
		Set<String> templates = new TreeSet<>();
		boolean computed = false;

		// Whether every call constructing it passes its name, which an `__init__` forwarding its own `name` parameter then carries.
		boolean everyCallNames = true;

		// The calls constructing it. One unpacking keywords, as `Dense(**config)` does, may pass any name.
		for (CGNode caller : Iterator2Iterable.make(this.callGraph.getPredNodes(allocator)))
			for (CallSiteReference callSite : Iterator2Iterable.make(this.callGraph.getPossibleSites(caller, allocator)))
				for (SSAAbstractInvokeInstruction call : caller.getIR().getCalls(callSite))
					if (call instanceof PythonInvokeInstruction invoke)
						if (unpacksKeywords(invoke)) {
							computed = true;
							everyCallNames = false;
						} else if (invoke.getUse(NAME_KEYWORD) >= 0) {
							Optional<IndexedName> indexed = this.getIndexedName(caller, invoke.getUse(NAME_KEYWORD));

							if (indexed.isPresent())
								templates.add(indexed.get().template());
							else
								computed |= addNames(caller, invoke.getUse(NAME_KEYWORD), constants);
						} else
							everyCallNames = false;
					else
						everyCallNames = false;

		// The `__init__` calls of its class's `__init__`, for a program class. One unpacking keywords may pass any name, unless it
		// unpacks the `__init__`'s own `**kwargs`, as `super().__init__(**kwargs)` does: that forwards only the keywords the constructing
		// calls passed, read above, and the forwarding chain's own, read here.
		if (allocator.getMethod() instanceof PythonConstructorFunction)
			for (CGNode init : this.getInitializers(allocator, objects))
				for (SSAInstruction instruction : init.getIR().getInstructions())
					if (instruction instanceof PythonInvokeInstruction invoke && callsInit(init, invoke)) {
						if (unpacksKeywords(invoke) && !forwardsOwnKeywords(init, invoke))
							computed = true;

						// An `__init__` forwarding its own `name` parameter carries what the constructing calls pass, read above.
						if (invoke.getUse(NAME_KEYWORD) >= 0 && !(everyCallNames && forwardsOwnName(init, invoke.getUse(NAME_KEYWORD))))
							computed |= addNames(init, invoke.getUse(NAME_KEYWORD), constants);
					}

		return new Names(constants, templates, computed);
	}

	/**
	 * True iff {@code value} of {@code node} is the function's own parameter named {@code name}.
	 *
	 * @param node An {@code __init__} node.
	 * @param value A value number of {@code node}, a {@code name} argument it passes.
	 * @return True iff the value is the {@code name} parameter.
	 */
	private static boolean forwardsOwnName(CGNode node, int value) {
		IR ir = node.getIR();

		if (!ir.getSymbolTable().isParameter(value))
			return false;

		String[] localNames = ir.getLocalNames(0, value);
		return localNames != null && List.of(localNames).contains(NAME_KEYWORD);
	}

	/**
	 * The program {@code __init__} nodes the constructor {@code allocator} reaches that are invoked on one of {@code objects}: its class's,
	 * and those of the bases they call.
	 *
	 * @param allocator The constructor allocating {@code objects}.
	 * @param objects The objects.
	 * @return The {@code __init__} nodes whose receiver may be one of {@code objects}.
	 */
	private Set<CGNode> getInitializers(CGNode allocator, Set<InstanceKey> objects) {
		Set<CGNode> ret = new LinkedHashSet<>();
		Deque<CGNode> worklist = new ArrayDeque<>(List.of(allocator));
		Set<CGNode> seen = new HashSet<>(List.of(allocator));

		while (!worklist.isEmpty())
			for (Iterator<CGNode> succNodes = this.callGraph.getSuccNodes(worklist.remove()); succNodes.hasNext();) {
				CGNode next = succNodes.next();

				if (next.getMethod() instanceof AstMethod && getFunctionName(next).equals(PythonTypes.INIT_METHOD_NAME) && seen.add(next)
						&& next.getIR() != null && next.getIR().getSymbolTable().getParameterValueNumbers().length > 1
						&& objects.stream().anyMatch(this.getPointerAnalysis().getPointsToSet(this.getPointerAnalysis().getHeapModel()
								.getPointerKeyForLocal(next, next.getIR().getSymbolTable().getParameter(1)))::contains)) {
					ret.add(next);
					worklist.add(next);
				}
			}

		return ret;
	}

	/**
	 * True iff {@code invoke} unpacks a mapping into its keyword arguments, {@code **kwargs}, whose keys, a {@code name} among them, it
	 * doesn't show. The front end lowers such an argument as a keyword whose name is the string of a missing one.
	 *
	 * @param invoke A call.
	 * @return True iff the call has a {@code **} argument.
	 */
	private static boolean unpacksKeywords(PythonInvokeInstruction invoke) {
		// TODO: This reads the front end's lowering of `**`; an engine query for whether a call unpacks `**` would retire it.
		return invoke.getKeywords().contains(UNPACKED_KEYWORDS);
	}

	/**
	 * True iff the {@code **} argument of {@code invoke} is the variable-keyword parameter, {@code **kwargs}, of {@code node}'s own
	 * function, as the engine declares it ({@link StarFormalDeclaration#getKeywordsParameter()}).
	 *
	 * @param node The node of {@code invoke}.
	 * @param invoke A call unpacking keywords ({@link #unpacksKeywords(PythonInvokeInstruction)}).
	 * @return True iff the call forwards the function's own {@code **kwargs}.
	 */
	private static boolean forwardsOwnKeywords(CGNode node, PythonInvokeInstruction invoke) {
		if (!(node.getMethod() instanceof StarFormalDeclaration declaration) || declaration.getKeywordsParameter() < 0)
			return false;

		int[] parameters = node.getIR().getSymbolTable().getParameterValueNumbers();
		int index = declaration.getKeywordsParameter();
		return index < parameters.length && invoke.getUse(UNPACKED_KEYWORDS) == parameters[index];
	}

	/**
	 * True iff {@code invoke} calls an {@code __init__}: its callee is read as the field {@code __init__} of an object, as
	 * {@code super().__init__} and {@code Base.__init__} are.
	 *
	 * @param node The node of {@code invoke}.
	 * @param invoke A call.
	 * @return True iff the call's callee is an {@code __init__} attribute.
	 */
	private static boolean callsInit(CGNode node, PythonInvokeInstruction invoke) {
		SSAInstruction calleeDef = node.getDU().getDef(invoke.getUse(0));

		if (calleeDef instanceof SSAGetInstruction get)
			return get.getDeclaredField().getName().toString().equals(PythonTypes.INIT_METHOD_NAME);

		if (calleeDef instanceof AstPropertyRead read) {
			SymbolTable symbolTable = node.getIR().getSymbolTable();
			return symbolTable.isStringConstant(read.getMemberRef())
					&& symbolTable.getStringValue(read.getMemberRef()).equals(PythonTypes.INIT_METHOD_NAME);
		}

		return false;
	}

	/**
	 * Adds the constant names {@code value} of {@code node} may be to {@code constants}: each constant string of its points-to set. A
	 * {@code None} adds nothing, since Keras then generates a unique name.
	 *
	 * @param node A node.
	 * @param value A value number of {@code node}, a {@code name} argument.
	 * @param constants The constant names, added to.
	 * @return True iff the value may be a name that isn't one constant string: its points-to set is empty or holds another object.
	 */
	private boolean addNames(CGNode node, int value, Set<String> constants) {
		SymbolTable symbolTable = node.getIR().getSymbolTable();

		if (symbolTable.isStringConstant(value))
			return addName(symbolTable.getStringValue(value), constants);

		if (symbolTable.isNullConstant(value))
			return false;

		OrdinalSet<InstanceKey> pointsToSet = this.getPointerAnalysis()
				.getPointsToSet(this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, value));

		if (pointsToSet.isEmpty())
			return true;

		boolean ret = false;

		for (InstanceKey instanceKey : pointsToSet)
			if (instanceKey instanceof ConstantKey<?> constant && constant.getValue() instanceof String string)
				ret |= addName(string, constants);
			else if (!(instanceKey instanceof ConstantKey<?> constant && constant.getValue() == null))
				ret = true;

		return ret;
	}

	/**
	 * Adds {@code name}, a constant string the engine publishes, to {@code constants}, unless it is the source text of an f-string, which
	 * the front end leaves as a constant starting with the opening quote, and whose value is computed.
	 *
	 * @param name A constant string.
	 * @param constants The constant names, added to.
	 * @return True iff the name is computed.
	 */
	private static boolean addName(String name, Set<String> constants) {
		// TODO: Read an f-string's value as computed from the engine once it stops lowering it to its source text
		// (https://github.com/wala/ML/issues/1231).
		if (name.startsWith("\"") || name.startsWith("'"))
			return true;

		constants.add(name);
		return false;
	}

	/**
	 * True iff {@code object} may stand for more than one object at run time: the site allocating it lies in a loop, or its node may run
	 * more than once ({@link #mayRunMoreThanOnce(CGNode)}).
	 *
	 * @param object An object.
	 * @return True iff the object's allocation may run more than once.
	 */
	private boolean mayBeSeveral(InstanceKey object) {
		AllocationSiteInNode site = findAllocationSiteInNode(object);

		if (site == null)
			return true;

		CGNode allocator = site.getNode();
		IR ir = allocator.getIR();
		SSANewInstruction allocation = ir == null ? null : ir.getNew(site.getSite());
		return allocation == null || isInLoop(ir, allocation.iIndex()) || this.mayRunMoreThanOnce(allocator);
	}

	/**
	 * True iff {@code node} may run more than once in one run of the program. An entry point the fake root calls runs once. A layer's
	 * {@code build} runs once per layer, so it may run more than once when its receiver may be several objects
	 * ({@link #mayBeSeveral(InstanceKey)}). Any other node runs more than once when a site calling it is a comprehension's call, lies in a
	 * loop, or lies in a node that may run more than once, or when two sites call it that one entry point reaches. Sites reached only from
	 * different entry points, such as two scripts, belong to different runs.
	 *
	 * @param node A node.
	 * @return True iff the node may run more than once.
	 */
	private boolean mayRunMoreThanOnce(CGNode node) {
		Boolean memo = this.runsMoreThanOnce.get(node);

		if (memo != null)
			return memo;

		// Recursion may run the node more than once.
		this.runsMoreThanOnce.put(node, true);
		boolean ret = this.computeRunsMoreThanOnce(node);
		this.runsMoreThanOnce.put(node, ret);
		return ret;
	}

	/**
	 * Decides {@link #mayRunMoreThanOnce(CGNode)}.
	 *
	 * @param node A node.
	 * @return True iff the node may run more than once.
	 */
	private boolean computeRunsMoreThanOnce(CGNode node) {
		if (this.isLayerBuild(node)) {
			IR ir = node.getIR();
			OrdinalSet<InstanceKey> layers = this.getPointerAnalysis().getPointsToSet(
					this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, ir.getSymbolTable().getParameter(1)));
			return layers.size() != 1 || this.mayBeSeveral(layers.iterator().next());
		}

		// The entry points reaching each caller with a site calling the node, once per site.
		List<BitSet> sites = new ArrayList<>();

		for (CGNode caller : Iterator2Iterable.make(this.callGraph.getPredNodes(node))) {
			if (caller.equals(this.callGraph.getFakeRootNode()) || caller.equals(this.callGraph.getFakeWorldClinitNode()))
				continue;

			IR ir = caller.getIR();

			for (CallSiteReference site : Iterator2Iterable.make(this.callGraph.getPossibleSites(caller, node))) {
				// A site that may run more than once, in a comprehension, a loop, or a caller that may run more than once.
				if (ir == null)
					return true;

				for (SSAAbstractInvokeInstruction call : ir.getCalls(site))
					if (call instanceof PythonInvokeInstruction invoke && invoke.isComprehension() || isInLoop(ir, call.iIndex()))
						return true;

				if (this.mayRunMoreThanOnce(caller))
					return true;

				// Two sites that one entry point reaches. Sites that only different entry points, such as two scripts, reach are of
				// different runs.
				BitSet entryPoints = getEntryPointsReaching(this.callGraph).get(caller);

				for (BitSet other : sites)
					if (entryPoints == null || other == null || entryPoints.intersects(other))
						return true;

				sites.add(entryPoints);
			}
		}

		return false;
	}

	/**
	 * The entry points of {@code callGraph}, the nodes its fake root calls, that reach each node, by their index among the fake root's
	 * successors. Built once per call graph.
	 *
	 * @param callGraph A call graph.
	 * @return The indices of the entry points reaching each node.
	 */
	private static Map<CGNode, BitSet> getEntryPointsReaching(CallGraph callGraph) {
		return entryPointsCache.computeIfAbsent(callGraph, cg -> {
			Map<CGNode, BitSet> ret = new HashMap<>();
			int index = 0;

			for (Iterator<CGNode> entryPoints = cg.getSuccNodes(cg.getFakeRootNode()); entryPoints.hasNext(); index++) {
				CGNode entryPoint = entryPoints.next();
				Deque<CGNode> worklist = new ArrayDeque<>(List.of(entryPoint));
				Set<CGNode> seen = new HashSet<>(List.of(entryPoint));

				while (!worklist.isEmpty()) {
					CGNode current = worklist.remove();
					ret.computeIfAbsent(current, k -> new BitSet()).set(index);

					for (Iterator<CGNode> succNodes = cg.getSuccNodes(current); succNodes.hasNext();) {
						CGNode next = succNodes.next();

						if (seen.add(next))
							worklist.add(next);
					}
				}
			}

			return ret;
		});
	}

	/**
	 * True iff the instruction at {@code index} of {@code ir} lies in a loop: its block can reach itself in the control-flow graph.
	 *
	 * @param ir An IR.
	 * @param index An instruction index of {@code ir}.
	 * @return True iff the instruction's block lies on a cycle.
	 */
	private static boolean isInLoop(IR ir, int index) {
		SSACFG cfg = ir.getControlFlowGraph();
		ISSABasicBlock block = cfg.getBlockForInstruction(index);
		Set<ISSABasicBlock> seen = new HashSet<>();
		Deque<ISSABasicBlock> worklist = new ArrayDeque<>();
		cfg.getSuccNodes(block).forEachRemaining(worklist::add);

		while (!worklist.isEmpty()) {
			ISSABasicBlock next = worklist.remove();

			if (next.equals(block))
				return true;

			if (seen.add(next))
				cfg.getSuccNodes(next).forEachRemaining(worklist::add);
		}

		return false;
	}

	/**
	 * Describes the source position of {@code instruction} of {@code node}.
	 *
	 * @param node A node.
	 * @param instruction One of its instructions.
	 * @return The function's name with the file and line, when known.
	 */
	private static String describePosition(CGNode node, SSAInstruction instruction) {
		String ret = "`" + getFunctionName(node) + "`";

		if (node.getMethod() instanceof AstMethod method && method.debugInfo() != null) {
			Position position = method.debugInfo().getInstructionPosition(instruction.iIndex());

			if (position != null && position.getURL() != null) {
				String file = position.getURL().getPath();
				ret += " (" + file.substring(file.lastIndexOf('/') + 1) + ":" + position.getFirstLine() + ")";
			}
		}

		return ret;
	}

	/**
	 * The pointer analysis.
	 *
	 * @return The builder's pointer analysis.
	 */
	private PointerAnalysis<InstanceKey> getPointerAnalysis() {
		return this.builder.getPointerAnalysis();
	}

}
