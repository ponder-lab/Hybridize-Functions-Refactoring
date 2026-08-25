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
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
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
 * a circular decision with no fixed point, which is also the exclusion the upstream parameter coercion applies (wala/ML#828). The pair is
 * not thereby exempt: when both parameters' own evidence is a single concrete dtype and the two differ, the eager program's survival
 * implies a weak operand whose identity the pair alone cannot decide, either-orientation pin breaks one reading, a spec naming the fed
 * dtypes raises at the op, and the bare decorator materializes the weak argument at its own dtype and raises the same way, so the consumer
 * declines (the plural-set outcome).
 */
class EagerCoercionAnalysis {

	/**
	 * The operations that convert a non-tensor operand against their tensor partner's dtype, by member name. The criterion is that behavior
	 * rather than the list, which is only its current extension: an operation belongs here iff eager execution converts a non-tensor
	 * operand through {@code convert_to_tensor} against the other operand's dtype.
	 * <p>
	 * The upstream declaration covers the operator spellings alone, reached through {@link SSABinaryOpInstruction}, so the call spellings
	 * of the very same operations are unaccounted on both sides: {@code V * x} is covered where {@code tf.multiply(V, x)} is not (#907).
	 * Matching on the member name covers the {@code tf.math} aliases with the plain ones.
	 * <p>
	 * The list's canonical home is upstream (wala/ML#837); it lives here only while the declaration does not cover the call spellings.
	 */
	private static final Set<String> COERCING_MEMBER_NAMES = Set.of("matmul", "tensordot", "einsum", "matvec", "add", "subtract",
			"multiply", "divide", "truediv", "floordiv", "mod", "floormod", "divide_no_nan", "multiply_no_nan", "pow", "maximum", "minimum",
			"squared_difference");

	/**
	 * The recognized operations whose first positional argument is not an operand. {@code einsum} leads with its equation string, so its
	 * operands start one slot later; reading its arity as though the equation were an operand would decline it always, which is the
	 * listed-but-unreachable shape that looks like coverage and is not.
	 */
	private static final Set<String> EQUATION_LED_MEMBER_NAMES = Set.of("einsum");

	/**
	 * The keyword arguments that carry no operand and so do not make a call unaccounted. An operation's {@code name} is metadata for the
	 * graph, not a value the operation combines.
	 */
	private static final Set<String> NON_OPERAND_KEYWORDS = Set.of("name");

	/** The (node, value number) index of the tensor-type analysis, mirroring {@link TensorIterationAnalysis}'s index. */
	private final Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex;

	/** Resolves a callee's fully-qualified name, which is how a coercing call is told from a same-named method on anything else. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	EagerCoercionAnalysis(TensorTypeAnalysis tensorTypeAnalysis, PointerAnalysis<InstanceKey> pointerAnalysis) {
		this.pointerAnalysis = pointerAnalysis;
		Map<CGNode, Map<Integer, Set<TensorType>>> index = new HashMap<>();

		for (Pair<PointerKey, TensorVariable> pair : tensorTypeAnalysis)
			if (pair.fst instanceof LocalPointerKey local && pair.snd != null)
				index.computeIfAbsent(local.getNode(), n -> new HashMap<>()).computeIfAbsent(local.getValueNumber(), v -> new HashSet<>())
						.addAll(pair.snd.getTypes());

		this.tensorTypeIndex = index;
	}

	/**
	 * The eager-effective dtype outcome for one parameter: the concrete dtypes its direct consumers' partner operands impose, whether any
	 * partner's ⊤ dtype leaves the collection indeterminate, and the value numbers of partner operands that are themselves parameters
	 * (self-combinations excluded), whose evidence the caller compares for the divergent-pair decline (issue 878).
	 */
	record Outcome(Set<DType> dtypes, boolean indeterminate, Set<Integer> parameterPartners) {
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
		Set<Integer> parameterPartners = new HashSet<>();
		IR ir = node.getIR();

		if (ir == null)
			return new Outcome(dtypes, false, parameterPartners);

		DefUse defUse = node.getDU();

		for (Iterator<SSAInstruction> uses = defUse.getUses(parameterValue); uses.hasNext();) {
			SSAInstruction use = uses.next();
			int other;

			if (use instanceof SSABinaryOpInstruction binary)
				other = binary.getUse(0) == parameterValue ? binary.getUse(1) : binary.getUse(0);
			else if (use instanceof PythonInvokeInstruction invoke
					&& this.coercingMemberName(node, invoke, defUse) instanceof String member) {
				// The call spelling of the same coercion. Only the binary shape is read: a call carrying a further operand, or a keyword
				// that is not mere metadata, is unaccounted, and incompleteness declines the whole parameter rather than pinning from the
				// operands it did understand.
				int firstOperand = EQUATION_LED_MEMBER_NAMES.contains(member) ? 2 : 1;

				if (invoke.getNumberOfPositionalParameters() != firstOperand + 2
						|| !NON_OPERAND_KEYWORDS.containsAll(invoke.getKeywords())) {
					indeterminate = true;
					continue;
				}

				int first = invoke.getUse(firstOperand);
				int second = invoke.getUse(firstOperand + 1);

				if (first != parameterValue && second != parameterValue)
					continue; // Reaches the call other than as a direct operand, so no dtype is imposed on it here.

				other = first == parameterValue ? second : first;
			} else
				continue;

			// A partner that is itself a parameter (including a self-combination like x * x) imposes no dtype: each side of such a
			// pair would take its dtype from the other, a circular decision with no fixed point whose orientation is arbitrary
			// (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/878). The upstream parameter coercion excludes
			// the same case (wala/ML#828), keeping the two implementations in agreement until #875 collapses this one into a read.
			// A pair that is not a self-combination is recorded for the caller's evidence comparison: a definite dtype divergence
			// between the two is a definite hazard no single signature (and no bare decorator) survives.
			if (ir.getSymbolTable().isParameter(other)) {
				if (other != parameterValue)
					parameterPartners.add(other);

				continue;
			}

			Set<TensorType> partnerTypes = this.tensorTypeIndex.getOrDefault(node, Map.of()).getOrDefault(other, Set.of());

			for (TensorType partnerType : partnerTypes) {
				DType dtype = partnerType.getDType();

				if (dtype == null || dtype == DType.UNKNOWN)
					indeterminate = true;
				else
					dtypes.add(dtype);
			}
		}

		return new Outcome(dtypes, indeterminate, parameterPartners);
	}

	/**
	 * The member name under which {@code invoke} calls an operation that converts a non-tensor operand against its tensor partner's dtype,
	 * or {@code null} if it calls no such operation. The callee's fully-qualified name is resolved and required to root at the TensorFlow
	 * module, so a same-named method on an unrelated object imposes nothing. The name is returned rather than a verdict because the operand
	 * positions depend on it ({@link #EQUATION_LED_MEMBER_NAMES}).
	 *
	 * @param node The call-graph node containing the call.
	 * @param invoke The call.
	 * @param defUse The node's def-use chains.
	 * @return The coercing operation's member name, or {@code null} if the call coerces no operand.
	 */
	private String coercingMemberName(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		String fqn = Util.resolveCalleeFullyQualifiedName(node, invoke.getUse(0), defUse, this.pointerAnalysis);

		if (fqn == null)
			return null;

		int lastDot = fqn.lastIndexOf('.');

		if (lastDot < 0)
			return null;

		String member = fqn.substring(lastDot + 1);

		return COERCING_MEMBER_NAMES.contains(member) ? member : null;
	}
}
