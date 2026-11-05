package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.KERAS_LAYER_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.KERAS_MODEL_TYPE;
import static com.ibm.wala.cast.python.util.Util.allocatedClass;
import static com.ibm.wala.cast.python.util.Util.findAllocationSiteInNode;
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
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.loader.IPythonClass;
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
import com.ibm.wala.ssa.SSACFG;
import com.ibm.wala.ssa.SSAConditionalBranchInstruction;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSASwitchInstruction;
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
 * branch, or before it, isn't first called under it. Only the paths that return normally count, since a trace that raises keeps no graph.
 * The condition takes the same path on every trace only on an engine fact: its points-to set is one constant, the same in every context, as
 * a literal or a module constant is, or the engine types it as a tensor, whose branch AutoGraph traces both arms of, so the first trace
 * makes the first call. Anything else, such as a parameter or field that may hold two values, an iterator over {@code self.blocks}, or a
 * value whose points-to set is empty, may vary, and the method is refused; whether a condition is fixed at every call is a fact the engine
 * doesn't publish yet (wala/ML#1230). A layer chosen by data rather than by control, such as {@code (self.a if flag else self.b)(x)}, isn't
 * covered.</li>
 * <li>{@link LayerBuildShape#COLLIDING_NAMES}: decorating the method names the variables built in its trace without the scopes of the
 * receiver and the layers enclosing it, so two sublayers that carry the same name build variables with the same names, and a save keyed by
 * name, such as to HDF5, fails or mixes weights (issue 1172). The sublayers are the layer objects reachable from the receiver through
 * fields, at any depth, told apart by the node allocating them, a constructor or a summary in its context, since the engine allocates a
 * program layer's bound methods in its constructor too. The receiver's own allocation is left out: its name is dropped with its scope. A
 * sublayer's name is the {@code name} argument of the call constructing it, or of the {@code __init__} call of a base that its class's
 * {@code __init__} makes, read as the constant strings of its points-to set, which holds a parameter's default too. Two sublayers collide
 * when they may carry the same constant name, whatever their classes: two allocations, or one that may run more than once in a run, such as
 * one in a loop or a comprehension, in a function invoked from two sites that one program run reaches, or in the {@code build} of a layer
 * that has more than one instance. A name that is not one constant string, such as {@code "h{}".format(i)}, may collide with any other and
 * refuses. A sublayer constructed without a name, or with {@code None}, gets a unique one from Keras and never collides. The front end
 * reads an f-string as the one constant string of its source text, so an f-string name counts as that constant.</li>
 * </ol>
 * A Keras model's {@code call} is exempt: decorating it drops the enclosing scopes from its weights' names too, but the save succeeds,
 * whether or not the model is nested in another one. So is a function that is no method, such as a training step: it runs inside no layer's
 * name scope, and a model it calls enters its own scope inside the trace.
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
	 * @param computed True iff it may carry a name that isn't one constant string.
	 */
	private record Names(Set<String> constants, boolean computed) {
	}

	private static final ILog LOG = getLog(LayerBuildAnalysis.class);

	/**
	 * The engine's modeled {@code tf.keras.layers.Layer} and {@code tf.keras.Model}, which a layer's class reaches through its bases.
	 */
	// TODO: Ask the engine whether a class reaches Model or Layer through any base (https://github.com/wala/ML/issues/1228).
	private static final Set<TypeName> KERAS_TYPE_NAMES = Set.of(KERAS_LAYER_TYPE.getName(), KERAS_MODEL_TYPE.getName());

	/** The fully-qualified name of {@code Layer.add_weight}, as the engine names it ({@link TensorFlowApis}). */
	private static final String ADD_WEIGHT_API = "tensorflow.keras.layers.Layer.add_weight";

	/** The keyword of a layer's name. */
	private static final String NAME_KEYWORD = "name";

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

		// TODO: Ask the engine whether a class reaches Model or Layer through any base (https://github.com/wala/ML/issues/1228).
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

					if (instanceKey == null)
						continue;

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

					// A call on no path that returns normally builds nothing a trace keeps.
					if (!cfg.containsNode(block))
						continue;

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
	 * compares against, are fixed; each other operand, the condition value, is fixed only on an engine fact
	 * ({@link #isFixed(CGNode, int)}).
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

			// TODO: Read whether the condition is fixed at every call from the engine (https://github.com/wala/ML/issues/1230).
			if (!ir.getSymbolTable().isConstant(value) && !this.isFixed(node, value))
				return true;
		}

		return false;
	}

	/**
	 * True iff the engine shows that {@code value}, a branch's condition, takes the same path on every trace: its points-to set is one
	 * constant, the same in every context of the node's function, so it is one literal or module constant whatever the call, or the engine
	 * types it as a tensor, whose branch AutoGraph traces both arms of, so the first trace makes the first call. Anything else, including a
	 * value whose points-to set is empty or holds two constants, may vary.
	 *
	 * @param node The node of {@code value}.
	 * @param value A value number of {@code node}.
	 * @return True iff the value is fixed across traces on an engine fact.
	 */
	private boolean isFixed(CGNode node, int value) {
		Set<InstanceKey> pointsTo = new HashSet<>();

		for (CGNode context : this.callGraph.getNodes(node.getMethod().getReference())) {
			PointerKey pointerKey = this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(context, value);

			if (this.tensorTypedKeys.containsKey(pointerKey))
				return true;

			OrdinalSet<InstanceKey> contextPointsTo = this.getPointerAnalysis().getPointsToSet(pointerKey);

			// Empty in a context is unknown there.
			if (contextPointsTo.isEmpty())
				return false;

			contextPointsTo.forEach(pointsTo::add);
		}

		return pointsTo.size() == 1 && pointsTo.iterator().next() instanceof ConstantKey<?>;
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

		Map<String, CGNode> named = new HashMap<>();

		for (Map.Entry<CGNode, Set<InstanceKey>> sublayer : this.getSublayers(receivers).entrySet()) {
			CGNode allocator = sublayer.getKey();
			Names names = this.getNames(allocator, sublayer.getValue());

			if (names.computed())
				return Optional.of(new Verdict(LayerBuildShape.COLLIDING_NAMES,
						"a `" + getSimpleName(allocatedClass(sublayer.getValue().iterator().next()).orElseThrow().getName())
								+ "` sublayer's name is computed"));

			for (String name : names.constants()) {
				CGNode other = named.putIfAbsent(name, allocator);

				// Two allocations collide; one allocator stands for several layers when one of its objects may be several.
				if (other != null && !other.equals(allocator) || sublayer.getValue().stream().anyMatch(this::mayBeSeveral))
					return Optional.of(new Verdict(LayerBuildShape.COLLIDING_NAMES, "several sublayers may be named `" + name + "`"));
			}
		}

		return Optional.empty();
	}

	/**
	 * The layers and models reachable from {@code receivers} through fields, at any depth, by the node allocating them, in a stable order.
	 * The node allocating a receiver is left out.
	 *
	 * @param receivers The objects the receiver may be.
	 * @return The sublayers' objects by the node allocating them.
	 */
	private Map<CGNode, Set<InstanceKey>> getSublayers(Set<InstanceKey> receivers) {
		Map<InstanceKey, Set<InstanceFieldPointerKey>> fields = getFieldIndex(this.getPointerAnalysis());
		Set<CGNode> receiverAllocators = new HashSet<>();

		for (InstanceKey receiver : receivers) {
			AllocationSiteInNode site = findAllocationSiteInNode(receiver);

			if (site != null)
				receiverAllocators.add(site.getNode());
		}

		Set<InstanceKey> seen = new HashSet<>(receivers);
		Deque<InstanceKey> worklist = new ArrayDeque<>(receivers);
		Map<CGNode, Set<InstanceKey>> ret = new TreeMap<>(Comparator.comparing(Object::toString));

		while (!worklist.isEmpty())
			for (InstanceFieldPointerKey field : fields.getOrDefault(worklist.remove(), Collections.emptySet()))
				for (InstanceKey value : this.getPointerAnalysis().getPointsToSet(field))
					if (seen.add(value)) {
						worklist.add(value);
						Optional<IClass> allocatedClass = allocatedClass(value);

						if (allocatedClass.isPresent() && this.isLayerClass(allocatedClass.get().getReference())) {
							CGNode allocator = findAllocationSiteInNode(value).getNode();

							if (!receiverAllocators.contains(allocator))
								ret.computeIfAbsent(allocator, k -> new HashSet<>()).add(value);
						}
					}

		return ret;
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
		boolean computed = false;

		// The calls constructing it.
		for (CGNode caller : Iterator2Iterable.make(this.callGraph.getPredNodes(allocator)))
			for (CallSiteReference callSite : Iterator2Iterable.make(this.callGraph.getPossibleSites(caller, allocator)))
				for (SSAAbstractInvokeInstruction call : caller.getIR().getCalls(callSite))
					if (call instanceof PythonInvokeInstruction invoke && invoke.getUse(NAME_KEYWORD) >= 0)
						computed |= addNames(caller, invoke.getUse(NAME_KEYWORD), constants);

		// The `__init__` calls of its class's `__init__`, for a program class.
		if (allocator.getMethod() instanceof PythonConstructorFunction)
			for (CGNode init : this.getInitializers(allocator, objects))
				for (SSAInstruction instruction : init.getIR().getInstructions())
					if (instruction instanceof PythonInvokeInstruction invoke && invoke.getUse(NAME_KEYWORD) >= 0
							&& callsInit(init, invoke))
						computed |= addNames(init, invoke.getUse(NAME_KEYWORD), constants);

		return new Names(constants, computed);
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

		if (symbolTable.isStringConstant(value)) {
			constants.add(symbolTable.getStringValue(value));
			return false;
		}

		if (symbolTable.isNullConstant(value))
			return false;

		OrdinalSet<InstanceKey> pointsToSet = this.getPointerAnalysis()
				.getPointsToSet(this.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, value));

		if (pointsToSet.isEmpty())
			return true;

		boolean ret = false;

		for (InstanceKey instanceKey : pointsToSet)
			if (instanceKey instanceof ConstantKey<?> constant && constant.getValue() instanceof String string)
				constants.add(string);
			else if (!(instanceKey instanceof ConstantKey<?> constant && constant.getValue() == null))
				ret = true;

		return ret;
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
