package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
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
import com.ibm.wala.ssa.SSANewInstruction;
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
 * {@code call} override would never see its test method. It hops a user frame too where no guard or {@code except} clause of that frame
 * encloses the call, since what the call raises leaves the frame unhandled: a user {@code __call__} override forwarding to
 * {@code super().__call__} sits between the test and {@code call}. A node with no resolvable call site is not excluded, the
 * allow-on-unknown polarity of every sibling analysis.
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
 * <p>
 * A {@code try} statement's body is a guard as well, for the conversion only (#1014): a call there is one whose behavior depends on which
 * exception it raises, but the statement declares nothing about whether it raises, so its evidence is never set aside. The front end lowers
 * the statement without the classes its {@code except} clauses name, so the clauses are read from the module's source at the call's line
 * ({@link ExceptionHandlerAnalysis}).
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

	/** The method the front end invokes on a class to bind it by name, as in {@code invokestatic LTypeError.import()}. */
	private static final String IMPORT_METHOD_NAME = "import";

	/** Global-read names identifying the pytest module by import alias, mirroring {@link Util#NUMPY_MODULE_GLOBAL_NAMES}. */
	private static final Set<String> PYTEST_MODULE_GLOBAL_NAMES = Set.of("global pytest", "global py");

	/**
	 * The exception classes one guard admits, as written at the guard: {@code assertRaises(TypeError)}, a tuple of classes, or a
	 * {@code tf.errors} class. A class is named by its simple name. A guard is an expected-failure context manager, which declares that the
	 * call raises what it admits, or one {@code except} clause of a {@code try} statement, which declares nothing about whether the call
	 * raises (#1014).
	 *
	 * @param names The simple names of the classes the guard admits; empty when unresolved.
	 * @param resolved False when some admitted class could not be named, in which case what the guard admits is unknown.
	 * @param handler True iff the guard is an {@code except} clause rather than an expected-failure context manager.
	 */
	record Guard(Set<String> names, boolean resolved, boolean handler) {

		/** The unresolved expected-failure guard: what it admits is unknown. */
		static final Guard UNRESOLVED = new Guard(Set.of(), false, false);

		/** The unresolved {@code except} clause: what it catches is unknown. */
		static final Guard UNRESOLVED_HANDLER = new Guard(Set.of(), false, true);

		/** The classes that admit every exception a call can raise. */
		private static final Set<String> BROAD_EXCEPTION_NAMES = Set.of("Exception", "BaseException");

		/**
		 * The simple names of {@code tf.errors.OpError} and its subclasses, the errors a TensorFlow kernel raises when it runs (#1014).
		 */
		private static final Set<String> OP_ERROR_NAMES = Set.of("OpError", "CancelledError", "UnknownError", "InvalidArgumentError",
				"DeadlineExceededError", "NotFoundError", "AlreadyExistsError", "PermissionDeniedError", "UnauthenticatedError",
				"ResourceExhaustedError", "FailedPreconditionError", "AbortedError", "OutOfRangeError", "UnimplementedError",
				"InternalError", "UnavailableError", "DataLossError");

		/**
		 * The simple names of the {@code tf.errors} classes a static check's failure is raised as eagerly: {@code InvalidArgumentError} and
		 * its ancestor {@code OpError} (#1014). A shape or dtype that an operation rejects raised {@code InvalidArgumentError} in every
		 * case measured on TensorFlow 2.9.3, including an operation with no kernel for the dtype.
		 */
		private static final Set<String> STATIC_OP_ERROR_NAMES = Set.of("InvalidArgumentError", "OpError");

		/**
		 * True iff this guard is known to admit the exception named {@code exception}: it names that class, {@code Exception}, or
		 * {@code BaseException}. An unresolved guard admits nothing.
		 *
		 * @param exception The simple name of an exception class.
		 * @return Whether the guard admits it.
		 */
		boolean admits(String exception) {
			return this.resolved() && (this.names().contains(exception) || this.admitsAny());
		}

		/**
		 * True iff this guard is known to admit every exception, by naming {@code Exception} or {@code BaseException}.
		 *
		 * @return Whether the guard admits any exception.
		 */
		boolean admitsAny() {
			return this.resolved() && this.names().stream().anyMatch(BROAD_EXCEPTION_NAMES::contains);
		}

		/**
		 * True iff this guard may be declaring the error a failed static check is raised as eagerly: it names {@code InvalidArgumentError}
		 * or {@code OpError}, or what it admits is unknown (#1014). Another {@code tf.errors} class, such as {@code OutOfRangeError}, is
		 * raised where tracing raises it too.
		 *
		 * @return Whether the declared exception may be a failed static check's.
		 */
		boolean mayAdmitStaticOpError() {
			return !this.resolved() || this.names().stream().anyMatch(STATIC_OP_ERROR_NAMES::contains);
		}

		/**
		 * True iff this guard is known to admit the error a failed static check is raised as eagerly: it names
		 * {@code InvalidArgumentError}, {@code OpError}, {@code Exception}, or {@code BaseException} (#1014).
		 *
		 * @return Whether the guard admits that error; false when what it admits is unknown.
		 */
		boolean admitsStaticOpError() {
			return this.resolved() && (this.admitsAny() || this.names().stream().anyMatch(STATIC_OP_ERROR_NAMES::contains));
		}

		/**
		 * True iff this guard is known to admit some error a TensorFlow kernel raises: it names a {@code tf.errors} class,
		 * {@code Exception}, or {@code BaseException} (#1014). A bare {@code except} admits every exception.
		 *
		 * @return Whether the guard admits a kernel's error; false when what it admits is unknown.
		 */
		boolean admitsOpError() {
			return this.resolved() && (this.admitsAny() || this.names().stream().anyMatch(OP_ERROR_NAMES::contains));
		}
	}

	private final CallGraph callGraph;

	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	/**
	 * The modules whose {@code try} statements are read, by the absolute path of their file, from which a call's {@code except} clauses are
	 * found (#1014). A call in a module that is not here has none.
	 */
	private final Map<String, ExceptionHandlerAnalysis> modules;

	ExpectedFailureContextAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			Map<String, ExceptionHandlerAnalysis> modules) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
		this.modules = modules;
	}

	/**
	 * Returns the subset of {@code nodes} reached only from expected-failure call sites.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @return Those nodes every call site of which is guarded; empty when none is.
	 */
	Set<CGNode> guardedOnlyNodes(Set<CGNode> nodes) {
		return this.guardedOnlyNodes(nodes, false);
	}

	/**
	 * Returns the subset of {@code nodes} reached only from call sites inside a guard: an expected-failure context manager, or, when
	 * {@code handlers} is true, also a {@code try} statement with an {@code except} clause (#1014).
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @param handlers Whether a call inside a {@code try} statement's body counts as guarded.
	 * @return Those nodes every call site of which is guarded; empty when none is.
	 */
	Set<CGNode> guardedOnlyNodes(Set<CGNode> nodes, boolean handlers) {
		Set<CGNode> ret = new HashSet<>();

		for (CGNode node : nodes)
			if (this.isGuardedOnly(node, handlers))
				ret.add(node);

		return ret;
	}

	/**
	 * Returns, for each of {@code nodes}, the guards around each guarded call reaching it: one list per call, holding every
	 * expected-failure guard whose {@code with} body contains that call, innermost or not, followed by the {@code except} clauses whose
	 * {@code try} body contains it, in the order Python tries them (#1014). A call's exception is admitted by an expected-failure guard
	 * when any of them admits it (#1005).
	 *
	 * @param nodes Nodes reached only from guarded call sites, as {@link #guardedOnlyNodes(Set, boolean)} returns.
	 * @return The guards around each call reaching each node.
	 */
	Map<CGNode, List<List<Guard>>> guardsOf(Set<CGNode> nodes) {
		Map<CGNode, List<List<Guard>>> ret = new HashMap<>();

		for (CGNode node : nodes) {
			List<List<Guard>> calls = new ArrayList<>();

			for (Site site : this.originatingSites(node, new HashMap<>())) {
				IR ir = site.caller().getIR();

				if (ir == null)
					continue;

				Set<GuardRegion> regions = this.guardRegions(site.caller(), ir);
				Dominators<ISSABasicBlock> dominators = Dominators.make(ir.getControlFlowGraph(), ir.getControlFlowGraph().entry());

				for (SSAAbstractInvokeInstruction instruction : ir.getCalls(site.reference())) {
					ISSABasicBlock block = ir.getBasicBlockForInstruction(instruction);
					List<Guard> around = new ArrayList<>();

					if (block != null)
						for (GuardRegion region : regions)
							if (region.contains(block, dominators))
								around.add(region.guard());

					around.addAll(this.handlersAround(site.caller(), instruction));
					calls.add(around);
				}
			}

			ret.put(node, calls);
		}

		return ret;
	}

	/**
	 * True iff every resolvable call site reaching {@code node} is guarded by an expected-failure context manager, or, when
	 * {@code handlers} is true, by that or an {@code except} clause.
	 */
	private boolean isGuardedOnly(CGNode node, boolean handlers) {
		boolean sawSite = false;

		for (Site site : this.originatingSites(node, new HashMap<>())) {
			sawSite = true;

			if (!this.isGuarded(site, handlers))
				return false;
		}

		// No resolvable call site is ignorance rather than evidence, so the node keeps its evidence.
		return sawSite;
	}

	/**
	 * The {@code except} clauses around {@code instruction} in {@code caller}'s source, innermost first (#1014). The front end drops the
	 * classes an {@code except} clause names, so they are read from the module's AST at the call's line.
	 *
	 * @param caller The node making the call.
	 * @param instruction The call.
	 * @return The clauses; empty when there are none, or when the call's position or module is not known.
	 */
	List<Guard> handlersAround(CGNode caller, SSAAbstractInvokeInstruction instruction) {
		if (!(caller.getMethod() instanceof AstMethod method) || method.debugInfo() == null)
			return List.of();

		ExceptionHandlerAnalysis module = this.modules.get(method.getDeclaringClass().getSourceFileName());
		Position position = method.debugInfo().getInstructionPosition(instruction.iIndex());

		if (module == null || position == null)
			return List.of();

		return module.handlersAround(position.getFirstLine());
	}

	/** A call site in the frame that wrote it: the caller's node paired with the site reference. */
	private record Site(CGNode caller, CallSiteReference reference) {
	}

	/**
	 * The call sites reaching {@code node} from user code, hopping any predecessor that is not an {@link AstMethod} so a synthesized
	 * trampoline resolves to the frame that actually contains the call. A call in user code that no guard or {@code except} clause of its
	 * own frame encloses is hopped too, to the sites calling that frame, since an exception the call raises propagates out of the frame
	 * unhandled: a user {@code __call__} override forwarding to {@code super().__call__} sits between a guarded test and the layer's
	 * {@code call}. A frame with no call site of its own keeps its call, so a root's unguarded call remains evidence. A hopped call counts
	 * as guarded whether or not its argument comes from the guarded call's, such as a warm-up call in a helper the guard calls, which is
	 * the imprecision of two calls in one guarded {@code with} body. {@code memo} holds each node's sites, so a frame the walk reaches
	 * along two paths, as when contexts merge above it, yields its sites both times; a node still being walked has none, which stops a
	 * cycle at the call inside it.
	 */
	private Set<Site> originatingSites(CGNode node, Map<CGNode, Set<Site>> memo) {
		Set<Site> known = memo.get(node);

		if (known != null)
			return known;

		memo.put(node, Set.of());
		Set<Site> ret = new HashSet<>();

		for (CGNode predecessor : Iterator2Iterable.make(this.callGraph.getPredNodes(node)))
			if (predecessor.getMethod() instanceof AstMethod)
				for (CallSiteReference reference : Iterator2Iterable.make(this.callGraph.getPossibleSites(predecessor, node))) {
					Set<Site> outer = this.isEnclosed(predecessor, reference) ? Set.of() : this.originatingSites(predecessor, memo);

					if (outer.isEmpty())
						ret.add(new Site(predecessor, reference));
					else
						// The frame does not handle what the call raises, so the guard, if any, is written further up.
						ret.addAll(outer);
				}
			else
				// A trampoline forwards the originating call, so the guard is written one frame further up.
				ret.addAll(this.originatingSites(predecessor, memo));

		memo.put(node, ret);
		return ret;
	}

	/**
	 * True iff some invoke at {@code reference} in {@code caller} lies inside an expected-failure guard's body or a {@code try} statement
	 * with an {@code except} clause, or its frame cannot be read, in which case the walk stops at the call.
	 */
	private boolean isEnclosed(CGNode caller, CallSiteReference reference) {
		IR ir = caller.getIR();

		if (ir == null)
			return true;

		Set<GuardRegion> regions = this.guardRegions(caller, ir);
		Dominators<ISSABasicBlock> dominators = regions.isEmpty() ? null
				: Dominators.make(ir.getControlFlowGraph(), ir.getControlFlowGraph().entry());

		for (SSAAbstractInvokeInstruction instruction : ir.getCalls(reference)) {
			ISSABasicBlock block = ir.getBasicBlockForInstruction(instruction);

			if (block == null || !this.handlersAround(caller, instruction).isEmpty())
				return true;

			for (GuardRegion region : regions)
				if (region.contains(block, dominators))
					return true;
		}

		return false;
	}

	/**
	 * True iff every invoke at {@code site} lies inside the body of an expected-failure guard in the same frame, or, when {@code handlers}
	 * is true, inside that or the body of a {@code try} statement with an {@code except} clause.
	 */
	private boolean isGuarded(Site site, boolean handlers) {
		IR ir = site.caller().getIR();

		if (ir == null)
			return false;

		Set<GuardRegion> regions = this.guardRegions(site.caller(), ir);

		if (regions.isEmpty() && !handlers)
			return false;

		Dominators<ISSABasicBlock> dominators = Dominators.make(ir.getControlFlowGraph(), ir.getControlFlowGraph().entry());

		for (SSAAbstractInvokeInstruction instruction : ir.getCalls(site.reference())) {
			ISSABasicBlock block = ir.getBasicBlockForInstruction(instruction);

			if (block == null)
				return false;

			boolean inside = handlers && !this.handlersAround(site.caller(), instruction).isEmpty();

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
	 * @param guard The exception classes the guard admits.
	 */
	private record GuardRegion(Set<ISSABasicBlock> begins, Set<ISSABasicBlock> ends, Guard guard) {

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
				ret.add(new GuardRegion(begins, ends, this.guardOf(node, invoke, defUse)));
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

	/**
	 * The exception classes the guard call {@code invoke} admits: its first argument other than the receiver, which is the class the
	 * {@code with} block expects ({@code assertRaises(TypeError)}, {@code pytest.raises((TypeError, ValueError))}).
	 */
	private Guard guardOf(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		int receiver = defUse.getDef(invoke.getUse(0)) instanceof PythonPropertyRead read ? read.getObjectRef() : -1;

		for (int slot = 1; slot < invoke.getNumberOfPositionalParameters(); slot++) {
			int use = invoke.getUse(slot);

			if (use != receiver) {
				Set<String> names = new HashSet<>();
				return this.addExceptionNames(node, defUse, use, names, new HashSet<>()) ? new Guard(Set.copyOf(names), true, false)
						: Guard.UNRESOLVED;
			}
		}

		return Guard.UNRESOLVED;
	}

	/**
	 * Adds the simple names of the exception classes {@code value} denotes to {@code names}: a bare name ({@code TypeError}), an attribute
	 * ({@code tf.errors.InvalidArgumentError}), or a tuple literal of either.
	 *
	 * @return False iff some class could not be named.
	 */
	private boolean addExceptionNames(CGNode node, DefUse defUse, int value, Set<String> names, Set<Integer> seen) {
		if (!seen.add(value))
			return false;

		SSAInstruction def = defUse.getDef(value);

		// A builtin class read at module scope is bound by the front end's import of that class: `invokestatic LTypeError.import()`.
		if (def instanceof SSAAbstractInvokeInstruction imported
				&& IMPORT_METHOD_NAME.equals(imported.getDeclaredTarget().getName().toString())) {
			String type = imported.getDeclaredTarget().getDeclaringClass().getName().toString();
			names.add(type.substring(type.lastIndexOf('/') + 1).replaceFirst("^L", ""));
			return true;
		}

		if (def instanceof AstGlobalRead global) {
			String name = global.getGlobalName();
			names.add(name.startsWith("global ") ? name.substring("global ".length()) : name);
			return true;
		}

		if (def instanceof AstLexicalRead lexical) {
			Access[] accesses = lexical.getAccesses();

			if (accesses.length == 0)
				return false;

			names.add(accesses[0].getName().fst);
			return true;
		}

		if (def instanceof PythonPropertyRead read) {
			String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);

			if (member == null)
				return false;

			names.add(member);
			return true;
		}

		if (def instanceof SSANewInstruction) {
			boolean any = false;

			for (SSAInstruction use : Iterator2Iterable.make(defUse.getUses(value)))
				if (use instanceof PythonPropertyWrite write && write.getObjectRef() == value) {
					if (!this.addExceptionNames(node, defUse, write.getValue(), names, seen))
						return false;

					any = true;
				}

			return any;
		}

		return false;
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
