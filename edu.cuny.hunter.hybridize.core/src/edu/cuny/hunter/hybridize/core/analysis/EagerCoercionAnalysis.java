package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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
	private static final Set<String> COERCING_MEMBER_NAMES = Set.of("matmul", "einsum", "add", "subtract", "multiply", "divide", "truediv",
			"floordiv", "mod", "floormod", "divide_no_nan", "multiply_no_nan", "pow", "maximum", "minimum", "squared_difference");

	/**
	 * The operations that require their operands to agree rather than converting one against the other. They convert a non-tensor operand
	 * without passing the partner's dtype as a hint and then raise on the mismatch, measured on the pinned TensorFlow with a NumPy operand
	 * and with a Python list alike, so they fail {@link #COERCING_MEMBER_NAMES}'s criterion (wala/ML#837).
	 * <p>
	 * For the signature question the two classes converge on the same imposition, by a different rationale. A program that ran cannot have
	 * carried a mismatch through one of these, because the mismatch raises eagerly, so the operand's eager-effective dtype at that consumer
	 * <em>equals</em> the partner's and imposing it is sound. Leaving them unaccounted instead would be worse than either: it would make
	 * the whole parameter indeterminate and discard the coercions its other consumers did impose.
	 */
	private static final Set<String> EQUALITY_ENFORCING_MEMBER_NAMES = Set.of("tensordot", "matvec");

	/** The operations from which a partner's dtype may be imposed, whether by conversion or by the run premise. */
	private static final Set<String> IMPOSING_MEMBER_NAMES = Stream
			.concat(COERCING_MEMBER_NAMES.stream(), EQUALITY_ENFORCING_MEMBER_NAMES.stream()).collect(Collectors.toUnmodifiableSet());

	/**
	 * Where an imposing operation's operands sit among its arguments, so a call can be read rather than refused for carrying arguments the
	 * operation has always taken.
	 * <p>
	 * Refusing is not free. An unreadable consumer is a consumer whose coercion is real and whose result cannot be named, and no
	 * disposition available then preserves the function: a specification naming the fed dtype raises at the operation, and so does the bare
	 * decorator, which materializes the argument at that same dtype (the reasoning
	 * {@link PreconditionFailure#HAS_CONFLICTING_EAGER_DTYPE_COERCIONS} already records). Every argument shape read here is therefore a
	 * decline that never has to fire, which is why the shapes are described rather than counted (#909).
	 *
	 * @param firstOperand The positional index of the first operand ({@code 0} is the callee).
	 * @param variadic Whether every remaining positional argument is an operand, as {@code einsum}'s are.
	 * @param trailingNonOperands How many non-operand positional arguments may follow the operands when not variadic.
	 * @param nonOperandKeywords The keyword arguments that carry no operand.
	 */
	private record OperandShape(int firstOperand, boolean variadic, int trailingNonOperands, Set<String> nonOperandKeywords) {
	}

	/** An operation's {@code name} is graph metadata rather than a value it combines, so every shape tolerates it. */
	private static final Set<String> UNIVERSAL_NON_OPERAND_KEYWORDS = Set.of("name");

	/** Two operands, then whatever trailing metadata the operation declares. The shape of the elementwise family. */
	private static final OperandShape BINARY_SHAPE = new OperandShape(1, false, 1, UNIVERSAL_NON_OPERAND_KEYWORDS);

	/**
	 * The operations whose operands do not sit in the two slots straight after the callee. Anything absent takes {@link #BINARY_SHAPE}.
	 * <p>
	 * {@code einsum} leads with its equation string and then takes any number of operands, so its two-operand form is not its only form.
	 * The rest declare trailing arguments that are not operands and that may be passed positionally: {@code tensordot}'s contraction axes,
	 * and the transposition and sparsity flags of the two matrix products. None of them changes the dtype an operand is read at.
	 */
	private static final Map<String, OperandShape> OPERAND_SHAPES = Map.of("einsum",
			new OperandShape(2, true, 0, Set.of("name", "optimize")), "tensordot", new OperandShape(1, false, 2, Set.of("name", "axes")),
			"matmul",
			new OperandShape(1, false, 8,
					Set.of("name", "transpose_a", "transpose_b", "adjoint_a", "adjoint_b", "a_is_sparse", "b_is_sparse", "output_type")),
			"matvec", new OperandShape(1, false, 5, Set.of("name", "transpose_a", "adjoint_a", "a_is_sparse", "b_is_sparse")));

	/** The (node, value number) index of the tensor-type analysis, mirroring {@link TensorIterationAnalysis}'s index. */
	private final Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex;

	/** Resolves a callee's fully-qualified name, which is how an imposing call is told from a same-named method on anything else. */
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
			List<Integer> partners;

			if (use instanceof SSABinaryOpInstruction binary)
				partners = List.of(binary.getUse(0) == parameterValue ? binary.getUse(1) : binary.getUse(0));
			else if (use instanceof PythonInvokeInstruction invoke
					&& this.imposingMemberName(node, invoke, defUse) instanceof String member) {
				// The call spelling of the same coercion, read at the operand positions the operation declares.
				List<Integer> operands = operands(invoke, OPERAND_SHAPES.getOrDefault(member, BINARY_SHAPE));

				if (operands == null) {
					// A shape this reader does not account for. No pin is formed from the operands it did understand: a skipped
					// consumer could leave a lone dtype unopposed.
					indeterminate = true;
					continue;
				}

				if (!operands.contains(parameterValue))
					continue; // Reaches the call other than as a direct operand, so no dtype is imposed on it here.

				partners = operands.stream().filter(operand -> operand != parameterValue).toList();
			} else
				continue;

			for (int other : partners) {
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
		}

		return new Outcome(dtypes, indeterminate, parameterPartners);
	}

	/**
	 * The value numbers of {@code invoke}'s operands under {@code shape}, or {@code null} if the call does not fit it. A call that does not
	 * fit carries something this reader cannot place, and placing it wrongly would impose a dtype from a value the operation never
	 * combines, so it is refused rather than guessed at.
	 *
	 * @param invoke The call.
	 * @param shape Where the operation's operands sit among its arguments.
	 * @return The operands' value numbers, or {@code null} if the call does not fit the shape.
	 */
	private static List<Integer> operands(PythonInvokeInstruction invoke, OperandShape shape) {
		if (!shape.nonOperandKeywords().containsAll(invoke.getKeywords()))
			return null;

		int positional = invoke.getNumberOfPositionalParameters();
		int last = shape.variadic() ? positional : shape.firstOperand() + 2;

		// Variadic operations take every remaining positional argument as an operand and need at least two to combine anything. The
		// rest take exactly two, optionally followed by the non-operand arguments they declare, which may be passed positionally.
		if (last - shape.firstOperand() < 2 || (!shape.variadic() && positional > last + shape.trailingNonOperands()))
			return null;

		List<Integer> operands = new ArrayList<>(last - shape.firstOperand());

		for (int operand = shape.firstOperand(); operand < last; operand++)
			operands.add(invoke.getUse(operand));

		return operands;
	}

	/**
	 * The member name under which {@code invoke} calls an operation that imposes its tensor partner's dtype on the other operand, or
	 * {@code null} if it calls no such operation. The callee's fully-qualified name is resolved and required to root at the TensorFlow
	 * module, so a same-named method on an unrelated object imposes nothing. The name is returned rather than a verdict because the operand
	 * positions depend on it ({@link #OPERAND_SHAPES}).
	 *
	 * @param node The call-graph node containing the call.
	 * @param invoke The call.
	 * @param defUse The node's def-use chains.
	 * @return The imposing operation's member name, or {@code null} if the call imposes nothing.
	 */
	private String imposingMemberName(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		String fqn = Util.resolveCalleeFullyQualifiedName(node, invoke.getUse(0), defUse, this.pointerAnalysis);

		if (fqn == null)
			return null;

		int lastDot = fqn.lastIndexOf('.');

		if (lastDot < 0)
			return null;

		String member = fqn.substring(lastDot + 1);

		return IMPOSING_MEMBER_NAMES.contains(member) ? member : null;
	}
}
