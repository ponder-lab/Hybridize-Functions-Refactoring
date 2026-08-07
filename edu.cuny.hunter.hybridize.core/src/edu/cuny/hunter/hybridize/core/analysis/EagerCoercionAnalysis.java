package edu.cuny.hunter.hybridize.core.analysis;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.Pair;

/**
 * The eager-coercion analysis behind the implicitly-cast NumPy argument hazard
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/861, Case 1): the dtypes eager execution would impose on a
 * parameter's argument at the ops that consume the parameter <em>directly</em>, read from the other operand's tensor typing.
 * <p>
 * The hazard is a coercion divergence. Eagerly, a NumPy (or Python) operand is converted at each op under the other operand's dtype;
 * traced, the argument materializes as a tensor of its own dtype at the boundary, and the op then raises on a genuine mismatch. A signature
 * pinned to the eager-effective dtype reproduces the eager coercion at the boundary (runtime-verified on the pinned TF 2.9.3), so a
 * singleton dtype set repairs the divergence; a plural set (parallel direct consumptions under different dtypes) admits no single signature
 * and declines. Chains need no traversal: an op consuming the result of an earlier op sees a tensor rather than the argument, and eager
 * already raises on tensor-tensor dtype mismatches, so a program that runs eagerly is dtype-consistent downstream of the first coercion.
 * <p>
 * The same premise makes the predicate self-discriminating on the argument's kind: were the argument a genuine tensor whose dtype differs
 * from a partner operand's, the <em>eager</em> program would already raise at that op, so a running program whose parameter dtype evidence
 * disagrees with the eager-effective set implies a weak-convertible (NumPy or Python) argument.
 * <p>
 * Polarity is allowing: a partner operand without tensor typing contributes nothing, and a partner whose dtype is ⊤ ({@link DType#UNKNOWN})
 * makes the whole parameter indeterminate, firing neither the pin nor the decline (wala/ML#827 tracks the scalar-initializer derivation
 * that leaves the reduced subject's variables ⊤ today). A partner that is itself a parameter likewise contributes nothing
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/878): each side of such a pair would take its dtype from the other,
 * a circular decision with no fixed point, which is also the exclusion the upstream parameter coercion applies (wala/ML#828).
 */
class EagerCoercionAnalysis {

	/** The (node, value number) index of the tensor-type analysis, mirroring {@link TensorIterationAnalysis}'s index. */
	private final Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex;

	EagerCoercionAnalysis(TensorTypeAnalysis tensorTypeAnalysis) {
		Map<CGNode, Map<Integer, Set<TensorType>>> index = new HashMap<>();

		for (Pair<PointerKey, TensorVariable> pair : tensorTypeAnalysis)
			if (pair.fst instanceof LocalPointerKey local && pair.snd != null)
				index.computeIfAbsent(local.getNode(), n -> new HashMap<>()).computeIfAbsent(local.getValueNumber(), v -> new HashSet<>())
						.addAll(pair.snd.getTypes());

		this.tensorTypeIndex = index;
	}

	/**
	 * The eager-effective dtype outcome for one parameter: the concrete dtypes its direct consumers' partner operands impose, and whether
	 * any partner's ⊤ dtype leaves the collection indeterminate.
	 */
	record Outcome(Set<DType> dtypes, boolean indeterminate) {
	}

	/**
	 * Collects, over every {@link SSABinaryOpInstruction} in {@code node}'s body consuming the raw parameter value {@code parameterValue}
	 * directly, the concrete dtypes of the other operand's tensor typing.
	 *
	 * @param node The call-graph node to scan.
	 * @param parameterValue The parameter's IR value number.
	 * @return The parameter's {@link Outcome}.
	 */
	Outcome eagerEffectiveDtypes(CGNode node, int parameterValue) {
		Set<DType> dtypes = new HashSet<>();
		boolean indeterminate = false;
		IR ir = node.getIR();

		if (ir == null)
			return new Outcome(dtypes, false);

		DefUse defUse = node.getDU();

		for (Iterator<SSAInstruction> uses = defUse.getUses(parameterValue); uses.hasNext();) {
			SSAInstruction use = uses.next();

			if (!(use instanceof SSABinaryOpInstruction binary))
				continue;

			int other = binary.getUse(0) == parameterValue ? binary.getUse(1) : binary.getUse(0);

			// A partner that is itself a parameter (including a self-combination like x * x) imposes no dtype: each side of such a
			// pair would take its dtype from the other, a circular decision with no fixed point whose orientation is arbitrary
			// (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/878). The upstream parameter coercion excludes
			// the same case (wala/ML#828), keeping the two implementations in agreement until #875 collapses this one into a read.
			if (ir.getSymbolTable().isParameter(other))
				continue;

			Set<TensorType> partnerTypes = this.tensorTypeIndex.getOrDefault(node, Map.of()).getOrDefault(other, Set.of());

			for (TensorType partnerType : partnerTypes) {
				DType dtype = partnerType.getDType();

				if (dtype == null || dtype == DType.UNKNOWN)
					indeterminate = true;
				else
					dtypes.add(dtype);
			}
		}

		return new Outcome(dtypes, indeterminate);
	}
}
