package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSACFG;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.Iterator2Iterable;

/**
 * The expected-failure analysis behind the negative-test exclusion of
 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/888: which of a function's call-graph nodes are reached
 * <em>only</em> from call sites the developer has declared must fail.
 * <p>
 * The tool infers an input signature from observed usage, and observed usage includes uses that are wrong on purpose. A call inside
 * {@code with self.assertRaises(...)} is not a hint: {@code unittest} enforces it, and the test fails if the call succeeds. So the site is
 * an executable assertion that the callee rejects that argument, which makes it the one kind of evidence a specification must not be
 * derived from — a signature anchored there admits exactly the input the body refuses and rejects every conforming caller.
 * <p>
 * Reading the developer's own assertion is why this is a marker rather than a heuristic. The general property is "this call raises," which
 * would require reasoning about the callee's own guards; the guard forms below are where a developer has already written that property
 * down.
 * <p>
 * A node is excluded only when <em>every</em> call site reaching it is guarded, since a node shared between a guarded and a conforming site
 * carries evidence from both. Ariadne interposes a synthesized trampoline between a caller and an instance method, so the walk hops through
 * any predecessor that is not user code (an {@link AstMethod}) to reach the frame the guard is written in; without that hop a Keras
 * {@code call} override would never see its test method. A node with no resolvable call site is not excluded, the allow-on-unknown polarity
 * of every sibling analysis.
 * <p>
 * The guard test is dominance rather than lexical nesting, which is what the IR preserves: the {@code with} body sits on the straight-line
 * path after the {@code assertRaises(...)} invoke, so the invoke's block dominates it. The {@code assertRaises(Exception, f, x)} call-form
 * is deliberately out of scope: there the callee is passed as a value and applied inside {@code unittest}, which is unmodeled, so no
 * call-graph edge carries evidence from it in the first place.
 */
class ExpectedFailureContextAnalysis {

	/**
	 * Member names of the {@code unittest} expected-failure context managers. Distinctive enough to match on the name alone: nothing else
	 * in this ecosystem is called {@code assertRaises}.
	 */
	private static final Set<String> UNITTEST_GUARD_MEMBER_NAMES = Set.of("assertRaises", "assertRaisesRegex", "assertRaisesRegexp");

	/** The pytest spelling. Matched only on a {@code pytest}-rooted receiver, since {@code raises} alone is not distinctive. */
	private static final String PYTEST_GUARD_MEMBER_NAME = "raises";

	/** Global-read names identifying the pytest module by import alias, mirroring {@link Util#NUMPY_MODULE_GLOBAL_NAMES}. */
	private static final Set<String> PYTEST_MODULE_GLOBAL_NAMES = Set.of("global pytest", "global py");

	private final CallGraph callGraph;

	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	ExpectedFailureContextAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
	}

	/**
	 * Returns the subset of {@code nodes} reached only from expected-failure call sites.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @return Those nodes every call site of which is guarded; empty when none is.
	 */
	Set<CGNode> guardedOnlyNodes(Set<CGNode> nodes) {
		Set<CGNode> ret = new HashSet<>();

		for (CGNode node : nodes)
			if (this.isGuardedOnly(node))
				ret.add(node);

		return ret;
	}

	/** True iff every resolvable call site reaching {@code node} is guarded by an expected-failure context manager. */
	private boolean isGuardedOnly(CGNode node) {
		boolean sawSite = false;

		for (Site site : this.originatingSites(node, new HashSet<>())) {
			sawSite = true;

			if (!this.isGuarded(site))
				return false;
		}

		// No resolvable call site is ignorance rather than evidence, so the node keeps its evidence.
		return sawSite;
	}

	/** A call site in the frame that wrote it: the caller's node paired with the site reference. */
	private record Site(CGNode caller, CallSiteReference reference) {
	}

	/**
	 * The call sites reaching {@code node} from user code, hopping any predecessor that is not an {@link AstMethod} so a synthesized
	 * trampoline resolves to the frame that actually contains the call. {@code seen} guards against a cycle among synthetic frames.
	 */
	private Set<Site> originatingSites(CGNode node, Set<CGNode> seen) {
		Set<Site> ret = new HashSet<>();

		if (!seen.add(node))
			return ret;

		for (CGNode predecessor : Iterator2Iterable.make(this.callGraph.getPredNodes(node)))
			if (predecessor.getMethod() instanceof AstMethod)
				for (CallSiteReference reference : Iterator2Iterable.make(this.callGraph.getPossibleSites(predecessor, node)))
					ret.add(new Site(predecessor, reference));
			else
				// A trampoline forwards the originating call, so the guard is written one frame further up.
				ret.addAll(this.originatingSites(predecessor, seen));

		return ret;
	}

	/** True iff every invoke at {@code site} is dominated by an expected-failure guard in the same frame. */
	private boolean isGuarded(Site site) {
		IR ir = site.caller().getIR();

		if (ir == null)
			return false;

		Set<ISSABasicBlock> guards = this.guardBlocks(site.caller(), ir);

		if (guards.isEmpty())
			return false;

		for (SSAAbstractInvokeInstruction instruction : ir.getCalls(site.reference())) {
			ISSABasicBlock block = ir.getBasicBlockForInstruction(instruction);

			if (block == null)
				return false;

			boolean dominated = false;

			for (ISSABasicBlock guard : guards)
				if (!guard.equals(block) && dominates(ir.getControlFlowGraph(), guard, block)) {
					dominated = true;
					break;
				}

			if (!dominated)
				return false;
		}

		return true;
	}

	/**
	 * True iff {@code guard} dominates {@code block}: every path from the entry to {@code block} passes through {@code guard}. Computed as
	 * unreachability from the entry with {@code guard} removed, which is the definition read directly, rather than through WALA's
	 * {@code Dominators}, whose package the bundle does not export.
	 */
	private static boolean dominates(SSACFG cfg, ISSABasicBlock guard, ISSABasicBlock block) {
		Set<ISSABasicBlock> reached = new HashSet<>();
		Deque<ISSABasicBlock> worklist = new ArrayDeque<>();

		reached.add(guard);
		worklist.add(cfg.entry());
		reached.add(cfg.entry());

		while (!worklist.isEmpty()) {
			ISSABasicBlock current = worklist.pop();

			if (current.equals(block))
				// Reached without going through the guard, so the guard does not dominate it.
				return false;

			for (ISSABasicBlock successor : Iterator2Iterable.make(cfg.getSuccNodes(current)))
				if (reached.add(successor))
					worklist.push(successor);
		}

		// The entry itself being the guard leaves `block` unreached, which is dominance only if they differ; the caller has excluded
		// equality.
		return true;
	}

	/** The blocks containing an expected-failure guard invocation in {@code ir}. */
	private Set<ISSABasicBlock> guardBlocks(CGNode node, IR ir) {
		Set<ISSABasicBlock> ret = new HashSet<>();
		DefUse defUse = node.getDU();

		for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
			if (instruction instanceof PythonInvokeInstruction invoke && this.isGuardCall(node, invoke, defUse))
				ret.add(ir.getBasicBlockForInstruction(invoke));

		ret.remove(null);
		return ret;
	}

	/** True iff {@code invoke} calls an expected-failure context manager. */
	private boolean isGuardCall(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		if (!(defUse.getDef(invoke.getUse(0)) instanceof PythonPropertyRead read))
			return false;

		String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);

		if (member == null)
			return false;

		if (UNITTEST_GUARD_MEMBER_NAMES.contains(member))
			return true;

		return PYTEST_GUARD_MEMBER_NAME.equals(member) && this.isPytestModule(node, read.getObjectRef(), defUse);
	}

	/** True iff {@code use} refers to the pytest module, by points-to or by the import alias on a global read. */
	private boolean isPytestModule(CGNode node, int use, DefUse defUse) {
		if (Util.pointsToType(node, use, this.pointerAnalysis, "Lpytest", false))
			return true;

		return defUse.getDef(use) instanceof AstGlobalRead global && PYTEST_MODULE_GLOBAL_NAMES.contains(global.getGlobalName());
	}
}
