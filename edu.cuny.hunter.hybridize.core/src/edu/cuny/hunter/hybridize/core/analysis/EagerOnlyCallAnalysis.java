package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Iterator;
import java.util.Set;

import com.google.common.collect.Sets;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.Iterator2Iterable;

/**
 * The eager-only-call analysis behind the {@link PreconditionFailure#HAS_EAGER_ONLY_CALLS} safety precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/363,
 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/836): a transitive walk over the call graph deciding whether a
 * function (or anything it reaches) invokes an API that is only valid in eager execution and therefore raises under {@code tf.function}
 * tracing.
 * <p>
 * Two name populations with different disciplines. The unconditional names ({@link #EAGER_ONLY_METHOD_NAMES}, e.g. {@code Tensor.numpy()})
 * block on the callee attribute name alone: the receiver's tensor typing is frequently unavailable, missing a real call would hybridize a
 * function that crashes on first call, and over-matching only declines an optimization. The Keras training-surface names
 * ({@link #EAGER_ONLY_MODEL_METHOD_NAMES}) cannot block bare, since they are generic ML verbs with demonstrated in-corpus collisions; they
 * require the dispatch evidence of {@link #invokesGuardedModelEndpoint(CGNode, PythonInvokeInstruction, PythonPropertyRead)}.
 */
class EagerOnlyCallAnalysis {

	/** The call graph, used to follow callees transitively and to resolve training-surface dispatches. */
	private final CallGraph callGraph;

	/** The pointer analysis, used to resolve callee attribute names and to type unresolved receivers. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	EagerOnlyCallAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
	}

	/** Method names whose invocation is only valid in eager execution (e.g. {@code Tensor.numpy()}). */
	private static final Set<String> EAGER_ONLY_METHOD_NAMES = Set.of("numpy");

	/** The Keras {@code Model.fit} member. */
	private static final String FIT_MEMBER_NAME = "fit";

	/** The Keras {@code Model.predict} member. */
	private static final String PREDICT_MEMBER_NAME = "predict";

	/** The Keras {@code Model.evaluate} member. */
	private static final String EVALUATE_MEMBER_NAME = "evaluate";

	/** The Keras {@code Model.train_on_batch} member. */
	private static final String TRAIN_ON_BATCH_MEMBER_NAME = "train_on_batch";

	/** The Keras {@code Model.test_on_batch} member. */
	private static final String TEST_ON_BATCH_MEMBER_NAME = "test_on_batch";

	/** The Keras {@code Model.predict_on_batch} member. */
	private static final String PREDICT_ON_BATCH_MEMBER_NAME = "predict_on_batch";

	/** The deprecated {@code Model.fit_generator} wrapper, forwarding to {@link #FIT_MEMBER_NAME}. */
	private static final String FIT_GENERATOR_MEMBER_NAME = "fit_generator";

	/** The deprecated {@code Model.evaluate_generator} wrapper, forwarding to {@link #EVALUATE_MEMBER_NAME}. */
	private static final String EVALUATE_GENERATOR_MEMBER_NAME = "evaluate_generator";

	/** The deprecated {@code Model.predict_generator} wrapper, forwarding to {@link #PREDICT_MEMBER_NAME}. */
	private static final String PREDICT_GENERATOR_MEMBER_NAME = "predict_generator";

	/**
	 * The Keras training-surface member names guarded by {@code _disallow_inside_tf_function}: each raises {@code RuntimeError} when
	 * invoked inside a {@code tf.function} trace (every public member runtime-verified on the pinned TF 2.9.3; the deprecated
	 * {@code *_generator} wrappers forward to the guarded members), because they are training-loop orchestrators that manage their own
	 * traces rather than tensor operations. Unlike {@link #EAGER_ONLY_METHOD_NAMES}, these names cannot block bare: they are generic ML
	 * verbs with demonstrated in-corpus collisions (a test-framework base class's {@code evaluate}; user-defined {@code predict} overrides
	 * on model subclasses), so a hit requires the dispatch evidence of
	 * {@link #invokesGuardedModelEndpoint(CGNode, PythonInvokeInstruction, PythonPropertyRead)}. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/836.
	 */
	private static final Set<String> EAGER_ONLY_MODEL_METHOD_NAMES = Set.of(FIT_MEMBER_NAME, PREDICT_MEMBER_NAME, EVALUATE_MEMBER_NAME,
			TRAIN_ON_BATCH_MEMBER_NAME, TEST_ON_BATCH_MEMBER_NAME, PREDICT_ON_BATCH_MEMBER_NAME, FIT_GENERATOR_MEMBER_NAME,
			EVALUATE_GENERATOR_MEMBER_NAME, PREDICT_GENERATOR_MEMBER_NAME);

	/**
	 * True iff {@code node}, transitively over its call-graph successors, invokes an eager-only API, which raises under {@code tf.function}
	 * tracing. Only user-defined bodies are scanned, and the traversal walks through TensorFlow library nodes to reach user callbacks, both
	 * mirroring {@code Util.performsTensorFlowOp}.
	 *
	 * @param node The call-graph node to check.
	 * @return True iff an eager-only API call is reachable from {@code node}.
	 */
	boolean callsEagerOnlyApi(CGNode node) {
		return this.callsEagerOnlyApi(node, Sets.newHashSet());
	}

	private boolean callsEagerOnlyApi(CGNode node, Set<CGNode> seen) {
		if (!seen.add(node))
			return false;

		if (!Util.isTensorFlowNode(node)) {
			IR ir = node.getIR();

			if (ir != null) {
				DefUse defUse = node.getDU();

				for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
					if (instruction instanceof PythonInvokeInstruction invoke && this.invokesEagerOnlyApi(node, invoke, defUse))
						return true;
			}
		}

		for (Iterator<CGNode> succNodes = this.callGraph.getSuccNodes(node); succNodes.hasNext();) {
			CGNode succNode = succNodes.next();

			if (this.callsEagerOnlyApi(succNode, seen))
				return true;
		}

		return false;
	}

	/**
	 * True iff {@code invoke}'s callee is an attribute read whose member name is an eager-only method name. The unconditional names
	 * ({@link #EAGER_ONLY_METHOD_NAMES}) block on the name alone; the training-surface names ({@link #EAGER_ONLY_MODEL_METHOD_NAMES})
	 * additionally require the dispatch evidence of
	 * {@link #invokesGuardedModelEndpoint(CGNode, PythonInvokeInstruction, PythonPropertyRead)}.
	 */
	private boolean invokesEagerOnlyApi(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		SSAInstruction def = defUse.getDef(invoke.getUse(0));

		if (def instanceof PythonPropertyRead read) {
			String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);

			if (member == null)
				return false;

			if (EAGER_ONLY_METHOD_NAMES.contains(member))
				return true;

			if (EAGER_ONLY_MODEL_METHOD_NAMES.contains(member))
				return this.invokesGuardedModelEndpoint(node, invoke, read);
		}

		return false;
	}

	/**
	 * True iff the training-surface member call {@code invoke} dispatches to the framework's own guarded endpoint rather than to user code.
	 * A resolved call-graph target in the TensorFlow namespace is the endpoint itself; a resolved user-defined target is an override, whose
	 * body the transitive walk of {@link #callsEagerOnlyApi(CGNode)} already analyzes on its own merits, so it does not block here. When
	 * the call site resolves to no target at all, the receiver decides: a points-to set holding an instance of a summarized TensorFlow
	 * class is the endpoint. Everything else is the unresolved residue and does not block (allow-on-unresolved): the front end does not yet
	 * record a user model class's framework base class (wala/ML#571), so a class-hierarchy walk cannot positively identify user model
	 * subclasses, and blocking the residue would re-import the bare-name false positives this discipline exists to avoid.
	 *
	 * @param node The node containing the call site.
	 * @param invoke The training-surface member call.
	 * @param read The attribute read defining the callee (the bound method), whose object is the receiver.
	 * @return True iff the call dispatches to the guarded framework endpoint.
	 */
	private boolean invokesGuardedModelEndpoint(CGNode node, PythonInvokeInstruction invoke, PythonPropertyRead read) {
		Set<CGNode> targets = this.callGraph.getPossibleTargets(node, invoke.getCallSite());

		for (CGNode target : targets)
			if (Util.isTensorFlowNode(target))
				return true;

		if (!targets.isEmpty())
			return false;

		return Util.pointsToType(node, read.getObjectRef(), this.pointerAnalysis, Util.TENSORFLOW_MODULE_TYPE_NAME, false);
	}
}
