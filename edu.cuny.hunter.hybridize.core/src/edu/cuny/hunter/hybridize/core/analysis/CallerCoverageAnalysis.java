package edu.cuny.hunter.hybridize.core.analysis;

import static java.lang.Boolean.TRUE;
import static org.eclipse.core.runtime.Platform.getLog;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.types.MethodReference;

/**
 * The caller-coverage analysis behind the redundant-hybridization advisory
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/767): which functions have every known call path dominated by a
 * hybridized caller, so their tensor computation is already traced ({@code tf.function} inlines its transitive callees) and decorating them
 * adds trace-time cost without additional speedup.
 * <p>
 * The condition is a path cut, computed as a <em>least</em> fixpoint: {@code covered(f)} iff {@code f} has at least one known caller and
 * every caller {@code g} satisfies {@code hybrid(g)} or {@code covered(g)}. Cycles and unknowns stay uncovered, so mutually-recursive eager
 * functions never cover each other and incomplete evidence never claims coverage. The polarity is deliberately inverted from the safety
 * preconditions: this is a benefit signal, and a wrong application deters an otherwise helpful refactoring, so every indeterminate state
 * (no call-graph node, an unresolvable predecessor, a synthetic predecessor with no real caller behind it, a library or module-level
 * caller) blocks coverage rather than granting it.
 * <p>
 * Predecessor classification hops through Ariadne's synthesized trampolines (a method's immediate predecessors bind {@code self} rather
 * than being user code); a synthetic hop that reaches no known function blocks, which also handles the synthetic root. Coverage is only as
 * sound as call-site enumeration: a function value that escapes to library code (a dataset mapper, a callback) may be invoked on paths the
 * graph never sees, which is why this analysis feeds an advisory and a measurement column, not a precondition failure (the two-phase plan
 * on the issue).
 */
class CallerCoverageAnalysis {

	private static final ILog LOG = getLog(CallerCoverageAnalysis.class);

	private CallerCoverageAnalysis() {
	}

	/**
	 * The subset of {@code functions} whose every known call path is dominated by a hybridized caller, per the least fixpoint above.
	 * Hybridization must already be computed for every function.
	 *
	 * @param functions Every function under analysis in the project.
	 * @param callGraph The call graph.
	 * @return The covered subset.
	 */
	static Set<Function> computeCovered(Set<Function> functions, CallGraph callGraph) {
		Map<MethodReference, Function> functionsByReference = new HashMap<>();

		for (Function function : functions)
			try {
				functionsByReference.put(function.getMethodReference(), function);
			} catch (CoreException _) {
				LOG.info("Can't resolve a method reference for " + function + "; it will not participate in caller coverage.");
			}

		// Memoized caller sets; null marks a function whose callers cannot all be resolved to known functions (blocked).
		Map<Function, Set<Function>> callersByFunction = new HashMap<>();

		for (Function function : functions)
			callersByFunction.put(function, callersOf(function, callGraph, functionsByReference));

		Set<Function> covered = new HashSet<>();
		boolean changed = true;

		while (changed) {
			changed = false;

			for (Function function : functions) {
				if (covered.contains(function))
					continue;

				Set<Function> callers = callersByFunction.get(function);

				// Blocked (unresolvable provenance) or no known caller: positive evidence is required.
				if (callers == null || callers.isEmpty())
					continue;

				if (callers.stream().allMatch(g -> TRUE.equals(g.isHybrid()) || covered.contains(g))) {
					covered.add(function);
					changed = true;
				}
			}
		}

		return covered;
	}

	/**
	 * The known user-function callers of {@code function}, hopping through synthetic trampolines, or {@code null} when any predecessor
	 * cannot be resolved to a known function (a module-level or library caller, a synthetic with no real caller behind it, or no call-graph
	 * node at all), the blocking direction.
	 */
	private static Set<Function> callersOf(Function function, CallGraph callGraph, Map<MethodReference, Function> functionsByReference) {
		MethodReference reference;

		try {
			reference = function.getMethodReference();
		} catch (CoreException _) {
			return null;
		}

		Set<CGNode> nodes = callGraph.getNodes(reference);

		if (nodes.isEmpty())
			return null;

		Set<Function> callers = new HashSet<>();

		for (CGNode node : nodes)
			for (Iterator<CGNode> predecessors = callGraph.getPredNodes(node); predecessors.hasNext();) {
				CGNode predecessor = predecessors.next();

				if (predecessor.equals(node))
					// A self-loop adds the function as its own caller; the fixpoint never covers it (conservative on recursion).
					continue;

				Function caller = functionsByReference.get(predecessor.getMethod().getReference());

				if (caller != null) {
					callers.add(caller);
					continue;
				}

				if (!predecessor.getMethod().isSynthetic())
					// A non-synthetic unknown caller: module-level code or a library frame. Blocks.
					return null;

				// A trampoline: hop to its predecessors, which must all be known functions.
				boolean sawRealCaller = false;

				for (Iterator<CGNode> hops = callGraph.getPredNodes(predecessor); hops.hasNext();) {
					CGNode hop = hops.next();
					Function hopCaller = functionsByReference.get(hop.getMethod().getReference());

					if (hopCaller == null)
						// The synthetic's own provenance is unknown (another synthetic, module code, the root). Blocks.
						return null;

					callers.add(hopCaller);
					sawRealCaller = true;
				}

				if (!sawRealCaller)
					// A synthetic with no predecessors (the root): entry-reachable with unknown provenance. Blocks.
					return null;
			}

		return callers;
	}
}
