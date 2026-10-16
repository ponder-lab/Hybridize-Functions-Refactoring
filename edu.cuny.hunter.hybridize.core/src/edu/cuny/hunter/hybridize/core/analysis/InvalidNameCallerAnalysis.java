package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;

import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.util.collections.Iterator2Iterable;

/**
 * The caller half of the invalid-name-argument safety precondition ({@link PreconditionFailure#HAS_INVALID_NAME_ARGUMENTS},
 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1087): tracing a function traces the functions it calls, so a caller
 * of a function whose name argument raises under tracing raises too.
 * <p>
 * A callee's own body decides it in one of two ways. Either it passes an invalid name argument whatever its callers pass, and every call to
 * it raises, or it forwards a parameter defaulting to {@code None} as the name of an API whose own default name is a string, and only a
 * call omitting that parameter raises. A call passes the parameter when it names it by keyword or supplies enough positional arguments to
 * reach it; a call that spreads a starred argument is read as passing it. The set of callers is a least fixpoint over the call graph, so a
 * caller of a caller raises too.
 * <p>
 * A call's arguments line up with the callee's parameters only when the call reaches the callee itself, or reaches it through an
 * instance-method trampoline ({@link PythonInstanceMethodTrampoline}), which binds the method's first parameter, whatever its name, and
 * forwards the rest. A callee the call reaches through any other node, such as a library summary calling user code back, receives arguments
 * the call does not show, so whether the call omits a forwarded parameter is undetermined and does not block. Whether that callee raises
 * for every call still does.
 */
class InvalidNameCallerAnalysis {

	private static final ILog LOG = getLog(InvalidNameCallerAnalysis.class);

	/** How a call's arguments line up with a callee's parameters. */
	private enum Alignment {
		/** The call reaches the callee itself, so its arguments are the callee's. */
		DIRECT,

		/** The call reaches the callee through an instance-method trampoline, which binds the callee's first parameter. */
		BOUND,

		/** The call reaches the callee through another node, which passes it arguments the call does not show. */
		UNDETERMINED
	}

	private InvalidNameCallerAnalysis() {
	}

	/**
	 * The subset of {@code functions} that call, directly or through other such callers, a function whose name argument raises under
	 * tracing. Every function's own {@link Function#computeInvalidNameArguments()} must already have run.
	 *
	 * @param functions Every function under analysis in the project.
	 * @param callGraph The call graph.
	 * @return The callers.
	 */
	static Set<Function> computeCallers(Set<Function> functions, CallGraph callGraph) {
		Map<MethodReference, Function> functionsByReference = new HashMap<>();

		for (Function function : functions)
			try {
				functionsByReference.put(function.getMethodReference(), function);
			} catch (CoreException _) {
				LOG.info("Can't resolve a method reference for " + function + "; its callers are not checked for its name arguments.");
			}

		Set<Function> callers = new HashSet<>();
		boolean changed = true;

		while (changed) {
			changed = false;

			for (Function function : functions)
				if (!callers.contains(function) && callsRaisingFunction(function, callGraph, functionsByReference, callers)) {
					callers.add(function);
					changed = true;
				}
		}

		return callers;
	}

	/**
	 * True iff a call in {@code function}'s body reaches a function whose name argument raises for that call.
	 *
	 * @param function The function whose calls are read.
	 * @param callGraph The call graph.
	 * @param functionsByReference The functions under analysis, by method reference.
	 * @param callers The callers found so far, whose every call raises.
	 * @return True iff such a call is in the body.
	 */
	private static boolean callsRaisingFunction(Function function, CallGraph callGraph, Map<MethodReference, Function> functionsByReference,
			Set<Function> callers) {
		MethodReference reference;

		try {
			reference = function.getMethodReference();
		} catch (CoreException _) {
			return false;
		}

		for (CGNode node : callGraph.getNodes(reference)) {
			IR ir = node.getIR();

			if (ir != null)
				for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
					if (instruction instanceof PythonInvokeInstruction invoke)
						for (Map.Entry<CGNode, Alignment> target : calleesOf(callGraph, node, invoke).entrySet()) {
							Function callee = functionsByReference.get(target.getKey().getMethod().getReference());

							if (callee != null && callee != function
									&& (callee.passesInvalidNameArgumentsUnconditionally() || callers.contains(callee)
											|| target.getValue() != Alignment.UNDETERMINED
													&& omitsForwardedNoneParameter(invoke, callee, target.getValue() == Alignment.BOUND)))
								return true;
						}
		}

		return false;
	}

	/**
	 * The nodes {@code invoke} reaches from {@code node}, each with how the call's arguments line up with its parameters: each call-graph
	 * target, and each node a target calls in turn.
	 *
	 * @param callGraph The call graph.
	 * @param node The node containing the call.
	 * @param invoke The call.
	 * @return The reached nodes and their alignment.
	 */
	private static Map<CGNode, Alignment> calleesOf(CallGraph callGraph, CGNode node, PythonInvokeInstruction invoke) {
		Map<CGNode, Alignment> callees = new HashMap<>();

		for (CGNode target : callGraph.getPossibleTargets(node, invoke.getCallSite())) {
			callees.put(target, Alignment.DIRECT);
			Alignment forwarding = target.getMethod().getDeclaringClass() instanceof PythonInstanceMethodTrampoline ? Alignment.BOUND
					: Alignment.UNDETERMINED;

			for (Iterator<CGNode> forwarded = callGraph.getSuccNodes(target); forwarded.hasNext();)
				callees.putIfAbsent(forwarded.next(), forwarding);
		}

		return callees;
	}

	/**
	 * True iff {@code invoke} omits a parameter that {@code callee} forwards, defaulting to {@code None}, as the name of an API whose own
	 * default name is a string.
	 *
	 * @param invoke The call.
	 * @param callee The function the call reaches.
	 * @param bound Whether the call reaches {@code callee} through a trampoline that binds its first parameter.
	 * @return True iff the call omits a forwarded parameter.
	 */
	private static boolean omitsForwardedNoneParameter(PythonInvokeInstruction invoke, Function callee, boolean bound) {
		Set<String> forwarded = callee.getForwardedNoneNameParameters();

		if (forwarded.isEmpty() || invoke.getStarredPositions().length > 0)
			return false;

		// The callee is the invoke's first use, so the call's positional arguments are the rest.
		int positionalArguments = invoke.getNumberOfPositionalParameters() - 1;
		List<String> positional = callee.getParameters().stream().filter(p -> !p.isKeywordOnly()).map(Parameter::getName).toList();

		if (bound && !positional.isEmpty())
			positional = positional.subList(1, positional.size());

		for (String name : forwarded) {
			if (invoke.getKeywords().contains(name))
				continue;

			int index = positional.indexOf(name);

			if (index < 0 || positionalArguments <= index)
				return true;
		}

		return false;
	}
}
