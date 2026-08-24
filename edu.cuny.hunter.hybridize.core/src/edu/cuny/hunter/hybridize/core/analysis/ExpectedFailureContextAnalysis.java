package edu.cuny.hunter.hybridize.core.analysis;

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
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.Iterator2Iterable;
import com.ibm.wala.util.graph.dominators.Dominators;

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
 * The guard test delimits the {@code with} body rather than testing dominance by the guard alone. Dominance alone was wrong in the
 * permissive direction: the body sits on the straight-line path after the {@code assertRaises(...)} invoke, but so does everything
 * following the block, so an ordinary call written after a guard was read as guarded and its conforming evidence discarded (#898). What
 * makes the region expressible is that the front end materializes it: {@code with} lowers to {@code __begin__} and {@code __end__} invokes
 * on the context manager the guard call produced, with the body between them and a second {@code __end__} on the exception path. A call is
 * therefore inside the body iff its block is dominated by that manager's {@code __begin__} and is <em>not</em> dominated by any of its
 * {@code __end__}s, which the normal-completion one supplies for everything after the block. A guard whose region cannot be recovered
 * guards nothing, keeping the allow-on-unknown polarity: failing to recognize a shape leaves evidence in rather than discarding it. The
 * {@code assertRaises(Exception, f, x)} call-form is deliberately out of scope: there the callee is passed as a value and applied inside
 * {@code unittest}, which is unmodeled, so no call-graph edge carries evidence from it in the first place.
 */
class ExpectedFailureContextAnalysis {

	/**
	 * Member names of the {@code unittest} expected-failure context managers. Distinctive enough to match on the name alone: nothing else
	 * in this ecosystem is called {@code assertRaises}.
	 */
	private static final Set<String> UNITTEST_GUARD_MEMBER_NAMES = Set.of("assertRaises", "assertRaisesRegex", "assertRaisesRegexp");

	/** The pytest spelling. Matched only on a {@code pytest}-rooted receiver, since {@code raises} alone is not distinctive. */
	private static final String PYTEST_GUARD_MEMBER_NAME = "raises";

	/** The member the front end invokes on a context manager where a {@code with} body opens. */
	private static final String REGION_BEGIN_MEMBER_NAME = "__begin__";

	/** The member it invokes where the body closes, once on normal completion and once on the exception path. */
	private static final String REGION_END_MEMBER_NAME = "__end__";

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

	/** True iff every invoke at {@code site} lies inside the body of an expected-failure guard in the same frame. */
	private boolean isGuarded(Site site) {
		IR ir = site.caller().getIR();

		if (ir == null)
			return false;

		Set<GuardRegion> regions = this.guardRegions(site.caller(), ir);

		if (regions.isEmpty())
			return false;

		Dominators<ISSABasicBlock> dominators = Dominators.make(ir.getControlFlowGraph(), ir.getControlFlowGraph().entry());

		for (SSAAbstractInvokeInstruction instruction : ir.getCalls(site.reference())) {
			ISSABasicBlock block = ir.getBasicBlockForInstruction(instruction);

			if (block == null)
				return false;

			boolean inside = false;

			for (GuardRegion region : regions)
				if (region.contains(block, dominators)) {
					inside = true;
					break;
				}

			if (!inside)
				return false;
		}

		return true;
	}

	/**
	 * One guard's {@code with} body, as the blocks that open and close it. A block lies inside the body when the opening dominates it and
	 * no closing does; the closing on normal completion is what puts everything after the block outside (#898).
	 *
	 * @param begins The blocks invoking {@code __begin__} on the guard's context manager.
	 * @param ends The blocks invoking {@code __end__} on it, on normal completion and on the exception path.
	 */
	private record GuardRegion(Set<ISSABasicBlock> begins, Set<ISSABasicBlock> ends) {

		/** True iff {@code block} lies within this body. */
		boolean contains(ISSABasicBlock block, Dominators<ISSABasicBlock> dominators) {
			boolean opened = false;

			for (ISSABasicBlock begin : this.begins())
				if (!begin.equals(block) && dominators.isDominatedBy(block, begin)) {
					opened = true;
					break;
				}

			if (!opened)
				return false;

			for (ISSABasicBlock end : this.ends())
				if (end.equals(block) || dominators.isDominatedBy(block, end))
					return false;

			return true;
		}
	}

	/**
	 * The {@code with} bodies of the expected-failure guards in {@code ir}, one region per guard call whose context manager the front end
	 * opened and closed. A guard whose region is not recoverable is left out, so it guards nothing.
	 */
	private Set<GuardRegion> guardRegions(CGNode node, IR ir) {
		Set<GuardRegion> ret = new HashSet<>();
		DefUse defUse = node.getDU();

		for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions())) {
			if (!(instruction instanceof PythonInvokeInstruction invoke) || !this.isGuardCall(node, invoke, defUse))
				continue;

			// The guard call's result is the context manager the `with` opens, and the region is delimited by the members invoked on it.
			int manager = invoke.getDef();
			Set<ISSABasicBlock> begins = this.regionBlocks(node, ir, defUse, manager, REGION_BEGIN_MEMBER_NAME);
			Set<ISSABasicBlock> ends = this.regionBlocks(node, ir, defUse, manager, REGION_END_MEMBER_NAME);

			if (!begins.isEmpty() && !ends.isEmpty())
				ret.add(new GuardRegion(begins, ends));
		}

		return ret;
	}

	/** The blocks invoking {@code member} on the value {@code manager} in {@code ir}. */
	private Set<ISSABasicBlock> regionBlocks(CGNode node, IR ir, DefUse defUse, int manager, String member) {
		Set<ISSABasicBlock> ret = new HashSet<>();

		for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions())) {
			if (!(instruction instanceof PythonInvokeInstruction invoke))
				continue;

			if (!(defUse.getDef(invoke.getUse(0)) instanceof PythonPropertyRead read) || read.getObjectRef() != manager)
				continue;

			if (member.equals(Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis)))
				ret.add(ir.getBasicBlockForInstruction(invoke));
		}

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
