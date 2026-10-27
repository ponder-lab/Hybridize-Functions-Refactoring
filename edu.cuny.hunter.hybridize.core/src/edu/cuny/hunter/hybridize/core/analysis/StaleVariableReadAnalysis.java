package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;

/**
 * The stale-variable-read analysis behind the optimizer slot-creation precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/822): a body-local scan deciding whether a function snapshots a
 * model's variable collection ({@code trainable_variables}/{@code trainable_weights}) <em>before</em> the model's first invocation in the
 * same body and feeds the snapshot to an optimizer or gradient computation.
 * <p>
 * The hazard is trace-time, not eager: a subclassed Keras model silently returns an empty collection before its first build, so the
 * snapshot taken on the initial trace is empty; the forward pass then creates the weights inside the trace, engaging {@code tf.function}'s
 * variable-lifting re-trace, on which the same Python-level read yields the populated list and {@code apply_gradients} creates optimizer
 * slot variables on a non-first trace, raising the singleton-variable {@code ValueError}. The pervasive beneficial idiom, reading the
 * collection <em>after</em> the forward pass, is untouched by construction: the ordering is the discriminator. Whether the model is already
 * built before the first trace is runtime state the scan cannot see, so a stale-ordered read is declined regardless, the safety polarity
 * (the prebuilt shape is a knowing, rare over-approximation).
 * <p>
 * Receiver identity is matched by the attribute chain's root token (a global, lexical, or local anchor), ordering by IR instruction
 * position (a loop's back edge only makes the first iteration stale, which is the iteration that traces), and the snapshot's consumption by
 * a forward def-use closure from the read to an {@code apply_gradients}, {@code minimize}, or {@code gradient} member call. The read and
 * the consumer flow are body-local, but the receiver-invocation scan follows call-graph edges: a call site counts as an invocation of every
 * stably-anchored root (global or lexical, never a node-relative local anchor) that its transitive callees invoke, since a forward pass one
 * call down still builds inside the same trace ({@code run_optimization} reaches its model only through {@code backprop};
 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/861).
 */
class StaleVariableReadAnalysis {

	/** The call graph, used to resolve call sites for the transitive receiver-invocation scan. */
	private final CallGraph callGraph;

	/** The pointer analysis, used to resolve attribute member names. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/** Per-node memo of the stably-anchored receiver roots the node's own body invokes. */
	private final Map<CGNode, Set<String>> locallyInvokedRootsByNode = new HashMap<>();

	StaleVariableReadAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
	}

	/** Member names whose read snapshots a model's variable collection. */
	private static final Set<String> VARIABLE_COLLECTION_MEMBER_NAMES = Set.of("trainable_variables", "trainable_weights");

	/** Member names whose invocation consumes a variable collection where staleness breaks tracing. */
	private static final Set<String> CONSUMER_MEMBER_NAMES = Set.of("apply_gradients", "minimize", "gradient");

	/**
	 * Member names whose invocation populates a model's variable collection (the lazy build sites), beside a direct call of the receiver.
	 * Beyond the core four, the whole Keras training surface triggers the lazy build: {@code predict}, {@code evaluate}, the
	 * {@code *_on_batch} family, and the deprecated {@code *_generator} wrappers were each runtime-verified to flip {@code built} on the
	 * pinned TF 2.9.3 (#825), so a variable-collection read ordered after any of them is fresh. Other member invocations ({@code compile},
	 * {@code summary}, ...) do not build, so they must not register as receiver invocations: counting them would suppress a genuinely stale
	 * read ordered after them ({@code compile} is pinned in that direction by {@code compiled_stale}).
	 */
	private static final Set<String> BUILDING_MEMBER_NAMES = Set.of("call", "__call__", "build", "fit", "predict", "evaluate",
			"train_on_batch", "test_on_batch", "predict_on_batch", "fit_generator", "evaluate_generator", "predict_generator");

	/**
	 * True iff {@code node}'s body snapshots a receiver's variable collection before that receiver's first invocation in the body, invokes
	 * the receiver later (the in-trace build that engages the lifting re-trace), and feeds the snapshot to a consumer.
	 *
	 * @param node The call-graph node to scan.
	 * @return True iff a stale variable-collection read reaches a consumer.
	 */
	boolean hasStaleVariableRead(CGNode node) {
		IR ir = node.getIR();

		if (ir == null)
			return false;

		DefUse defUse = node.getDU();
		SSAInstruction[] instructions = ir.getInstructions();

		// Receiver-root token -> IR positions at which that root is invoked.
		Map<String, List<Integer>> invocationPositionsByRoot = new HashMap<>();
		record CandidateRead(int position, String root, int def) {
		}
		List<CandidateRead> candidates = new ArrayList<>();

		for (int i = 0; i < instructions.length; i++) {
			SSAInstruction instruction = instructions[i];

			if (instruction == null)
				continue;

			if (instruction instanceof PythonInvokeInstruction invoke) {
				String root = this.receiverInvocationRoot(node, invoke, defUse);

				if (root != null)
					invocationPositionsByRoot.computeIfAbsent(root, r -> new ArrayList<>()).add(i);

				// The callee may build the receiver even when this site is no direct invocation of it (a forward pass one call
				// down builds inside the same trace), so the site also counts as an invocation of every stably-anchored root its
				// transitive callees invoke.
				for (String calleeRoot : this.transitivelyInvokedRoots(node, invoke))
					invocationPositionsByRoot.computeIfAbsent(calleeRoot, r -> new ArrayList<>()).add(i);
			} else if (instruction instanceof PythonPropertyRead read) {
				String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);

				if (member != null && VARIABLE_COLLECTION_MEMBER_NAMES.contains(member)) {
					String root = this.rootToken(node, read.getObjectRef(), defUse);

					if (root != null)
						for (int d = 0; d < read.getNumberOfDefs(); d++)
							candidates.add(new CandidateRead(i, root, read.getDef(d)));
				}
			}
		}

		for (CandidateRead candidate : candidates) {
			List<Integer> invocations = invocationPositionsByRoot.getOrDefault(candidate.root(), List.of());
			boolean invokedBefore = invocations.stream().anyMatch(p -> p < candidate.position());
			boolean invokedAfter = invocations.stream().anyMatch(p -> p > candidate.position());

			// Stale iff the receiver is first invoked only after the read: the initial trace's snapshot precedes the in-trace
			// build. An earlier invocation populates the collection on every trace; no later invocation means this body does not
			// build the receiver, so the lifting re-trace is not engaged by it.
			if (invokedBefore || !invokedAfter)
				continue;

			if (this.reachesConsumer(node, candidate.def(), defUse))
				return true;
		}

		return false;
	}

	/**
	 * True iff {@code def} flows, through intra-body def-use closure, into an argument of a consumer member call (see
	 * {@link #CONSUMER_MEMBER_NAMES}).
	 */
	private boolean reachesConsumer(CGNode node, int def, DefUse defUse) {
		Set<Integer> reached = new TreeSet<>();
		Deque<Integer> worklist = new ArrayDeque<>();
		reached.add(def);
		worklist.add(def);

		while (!worklist.isEmpty()) {
			int value = worklist.pop();

			for (Iterator<SSAInstruction> uses = defUse.getUses(value); uses.hasNext();) {
				SSAInstruction use = uses.next();

				if (use instanceof PythonInvokeInstruction invoke && this.isConsumerCall(node, invoke, defUse))
					for (int j = 1; j < invoke.getNumberOfUses(); j++)
						if (invoke.getUse(j) == value)
							return true;

				for (int d = 0; d < use.getNumberOfDefs(); d++) {
					int next = use.getDef(d);

					if (reached.add(next))
						worklist.push(next);
				}
			}
		}

		return false;
	}

	/**
	 * The stably-anchored receiver roots invoked anywhere in the call-graph closure of {@code invoke}'s possible targets. Reachability over
	 * successor nodes with a visited set, so recursion terminates; each node's own body is scanned once and memoized
	 * ({@link #locallyInvokedRootsByNode}). Node-relative local anchors are excluded: a callee's {@code l:} root names the callee's own
	 * value numbers and carries no identity in this node.
	 */
	private Set<String> transitivelyInvokedRoots(CGNode node, PythonInvokeInstruction invoke) {
		Set<String> roots = new HashSet<>();
		Set<CGNode> visited = new HashSet<>();
		Deque<CGNode> worklist = new ArrayDeque<>();

		for (CGNode target : this.callGraph.getPossibleTargets(node, invoke.getCallSite()))
			if (visited.add(target))
				worklist.add(target);

		while (!worklist.isEmpty()) {
			CGNode next = worklist.pop();
			roots.addAll(this.locallyInvokedRoots(next));

			for (Iterator<CGNode> successors = this.callGraph.getSuccNodes(next); successors.hasNext();) {
				CGNode successor = successors.next();

				if (visited.add(successor))
					worklist.add(successor);
			}
		}

		return roots;
	}

	/** The stably-anchored (global or lexical) receiver roots {@code node}'s own body invokes; see {@link #receiverInvocationRoot}. */
	private Set<String> locallyInvokedRoots(CGNode node) {
		return this.locallyInvokedRootsByNode.computeIfAbsent(node, n -> {
			IR ir = n.getIR();

			if (ir == null)
				return Set.of();

			DefUse defUse = n.getDU();
			Set<String> roots = new HashSet<>();

			for (SSAInstruction instruction : ir.getInstructions())
				if (instruction instanceof PythonInvokeInstruction invoke) {
					String root = this.receiverInvocationRoot(n, invoke, defUse);

					if (root != null && !root.startsWith("l:"))
						roots.add(root);
				}

			return roots;
		});
	}

	/**
	 * The receiver-root token of an invocation that populates the receiver's variable collection: a direct call of the receiver, or a
	 * building member ({@link #BUILDING_MEMBER_NAMES}). {@code null} for any other member invocation, which builds nothing and therefore
	 * neither suppresses a later stale read nor counts as the in-trace build.
	 */
	private String receiverInvocationRoot(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		SSAInstruction def = defUse.getDef(invoke.getUse(0));

		if (def instanceof PythonPropertyRead read) {
			String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);
			return member != null && BUILDING_MEMBER_NAMES.contains(member) ? this.rootToken(node, read.getObjectRef(), defUse) : null;
		}

		return this.rootToken(node, invoke.getUse(0), defUse);
	}

	/** True iff {@code invoke}'s callee is a property read of a consumer member name. */
	private boolean isConsumerCall(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		SSAInstruction def = defUse.getDef(invoke.getUse(0));

		if (!(def instanceof PythonPropertyRead read))
			return false;

		String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);
		return member != null && CONSUMER_MEMBER_NAMES.contains(member);
	}

	/**
	 * The identity token anchoring an attribute chain: the global or lexical name at its root, or a local anchor. {@code null} when no
	 * stable anchor exists (e.g., the root is itself a call result), in which case ordering cannot be established and the value does not
	 * participate, the allowing direction for receiver matching (a candidate without an anchor never fires).
	 */
	private String rootToken(CGNode node, int value, DefUse defUse) {
		SSAInstruction def = defUse.getDef(value);

		if (def instanceof PythonPropertyRead read)
			return this.rootToken(node, read.getObjectRef(), defUse);

		if (def instanceof AstGlobalRead global)
			return "g:" + global.getGlobalName();

		if (def instanceof AstLexicalRead lexical) {
			Access[] accesses = lexical.getAccesses();
			return accesses.length > 0 ? "x:" + accesses[0].getName().fst : null;
		}

		if (def == null && value <= node.getIR().getSymbolTable().getMaxValueNumber())
			// A parameter or constant: anchor by value number (a model passed as a parameter still orders consistently).
			return "l:" + value;

		return null;
	}

	/** The set of nodes' verdicts, any-match; see {@link #hasStaleVariableRead(CGNode)}. */
	boolean hasStaleVariableRead(Set<CGNode> nodes) {
		return nodes.stream().anyMatch(this::hasStaleVariableRead);
	}
}
