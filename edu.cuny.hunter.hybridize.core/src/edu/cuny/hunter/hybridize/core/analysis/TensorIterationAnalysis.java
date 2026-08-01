package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.Pair;

/**
 * The tensor-iteration analysis behind the symbolic-iteration precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/830): whether a function's body iterates a parameter-derived,
 * tensor-typed value with a Python {@code for}. Eagerly such iteration works (the elements are tensors); under {@code tf.function} tracing,
 * the decorated function's inputs are symbolic, and iterating a symbolic tensor raises {@code OperatorNotAllowedInGraphError} even with
 * AutoGraph converting the loop (runtime-verified on the pinned framework). The scope is deliberately anchored on parameter-derived values:
 * an in-body {@code tf.range} loop is AutoGraph-supported and must not fire, and it is the decoration that makes a parameter symbolic at
 * entry.
 * <p>
 * The scan is body-local: parameter value numbers seed a forward def-use taint, and a {@link EachElementGetInstruction} (the IR form of a
 * Python {@code for}-each) whose iterated object is tainted <em>and</em> tensor-typed by the tensor-type analysis fires. The tensor-typed
 * gate keeps ordinary Python-collection parameters (lists, dicts, dataset objects) out: only a value the analysis types as a tensor can
 * become a symbolic tensor at trace time.
 */
class TensorIterationAnalysis {

	/** The (node, value number) index of the tensor-type analysis, mirroring {@link NumpyParameterFlowAnalysis}'s index. */
	private final Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex;

	TensorIterationAnalysis(TensorTypeAnalysis tensorTypeAnalysis) {
		Map<CGNode, Map<Integer, Set<TensorType>>> index = new HashMap<>();

		for (Pair<PointerKey, TensorVariable> pair : tensorTypeAnalysis)
			if (pair.fst instanceof LocalPointerKey local && pair.snd != null)
				index.computeIfAbsent(local.getNode(), n -> new HashMap<>()).computeIfAbsent(local.getValueNumber(), v -> new HashSet<>())
						.addAll(pair.snd.getTypes());

		this.tensorTypeIndex = index;
	}

	/**
	 * True iff {@code node}'s body iterates a parameter-derived, tensor-typed value.
	 *
	 * @param node The call-graph node to scan.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a source.
	 * @return True iff a Python {@code for} iterates a parameter-derived tensor.
	 */
	boolean iteratesTensorParameter(CGNode node, boolean method) {
		IR ir = node.getIR();

		if (ir == null)
			return false;

		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();
		Set<Integer> tainted = new HashSet<>();
		Deque<Integer> worklist = new ArrayDeque<>();

		for (int i = method ? 2 : 1; i < parameters.length; i++) {
			tainted.add(parameters[i]);
			worklist.add(parameters[i]);
		}

		if (tainted.isEmpty())
			return false;

		DefUse defUse = node.getDU();

		while (!worklist.isEmpty()) {
			int value = worklist.pop();

			for (Iterator<SSAInstruction> uses = defUse.getUses(value); uses.hasNext();) {
				SSAInstruction use = uses.next();

				if (use instanceof EachElementGetInstruction each && each.getUse(0) == value
						&& !this.tensorTypeIndex.getOrDefault(node, Map.of()).getOrDefault(value, Set.of()).isEmpty())
					return true;

				for (int d = 0; d < use.getNumberOfDefs(); d++) {
					int def = use.getDef(d);

					if (tainted.add(def))
						worklist.push(def);
				}
			}
		}

		return false;
	}
}
