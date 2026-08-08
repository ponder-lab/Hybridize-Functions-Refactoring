package edu.cuny.hunter.hybridize.core.analysis;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAPhiInstruction;
import com.ibm.wala.util.collections.Iterator2Iterable;

/**
 * The Keras symbolic-argument analysis behind the precondition of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/887:
 * whether some call site of a function passes a Keras <em>symbolic</em> tensor (a {@code KerasTensor}, the value
 * {@code tf.keras.layers.Input} produces and the Functional API threads through layer applications). {@code tf.function} is one of the APIs
 * {@code KerasTensor} explicitly refuses, so decorating such a function turns a working program into one that raises a {@code TypeError} on
 * the first call, before anything is traced. The hazard is independent of any {@code input_signature}: a bare decorator fails the same way.
 * <p>
 * Unlike the other safety analyses, which ask what a function's body reaches, this one asks what its callers pass, so it walks the call
 * graph in the caller direction like {@link Function#inferSuppliedParameters}. The marker is syntactic and needs no dynamic reasoning: the
 * producing API is the fixed, summarized constructor {@code tf.keras.layers.Input} (which {@code tf.keras.Input} and
 * {@code keras.engine.input_layer.Input} alias to the same summary object), so the question reduces to the provenance of the argument value
 * in the caller's own IR.
 * <p>
 * Provenance is resolved backwards from the argument value:
 * <ul>
 * <li>An {@code Input(...)} result is symbolic. This is the direct form, and the one the reproduction takes: YOLOV3's only call site passes
 * {@code tf.keras.layers.Input([416, 416, 3])} straight through.
 * <li>A built-in Keras layer applied to a symbolic value is symbolic, since the Functional API's layer application maps
 * {@code KerasTensor}s to {@code KerasTensor}s. The recursion into the application's own arguments is what keeps this from over-blocking:
 * the same layer on an eager tensor yields an eager tensor and does not fire.
 * <li>A phi is symbolic iff <em>every</em> operand is. A merge of a symbolic value with an eager one makes the argument's symbolicness
 * depend on the executed path, which is ignorance rather than evidence, and this analysis's polarity — like the eager-coercion collection's
 * — leaves the indeterminate case allowing.
 * </ul>
 * Everything else — a parameter, a constant, an unmodeled call — is not symbolic. The walk stays inside the caller's frame, which is also
 * where the distinction the hazard turns on lives: a layer's own {@code call} receives a placeholder rather than a {@code KerasTensor}, so
 * a value derived from an {@code Input} <em>inside</em> a layer body is not symbolic, and no {@code Input} allocation is visible in such a
 * frame to seed the walk in the first place.
 * <p>
 * Across call sites the quantifier flips to existential, which the path-merge rule above does not contradict: a call site is a call that
 * happens, so one call site passing a symbolic tensor is enough to break the decorated function, while one phi operand being symbolic is
 * only a path that may not be taken.
 * <p>
 * Ignorance is {@code null} rather than {@code FALSE}, the polarity of every sibling safety check: no call-graph node, no IR for a caller,
 * or a non-Python invoke leaves the verdict undetermined, and only a determinate {@code TRUE} declines. A method whose callers reach it
 * through Ariadne's synthesized trampoline is a determinate {@code FALSE} rather than a {@code TRUE}: the trampoline forwards its own
 * parameters, whose defs are absent, so no provenance is visible one hop up. That is the incompleteness-safe direction and matches how the
 * Functional API is written in practice — an {@code Input} is passed to a module-level function or to a layer, not to a method whose
 * receiver Ariadne must bind.
 */
class KerasSymbolicArgumentAnalysis {

	/**
	 * The summarized Keras symbolic-input constructor. Every source-level spelling ({@code tf.keras.Input}, {@code tf.keras.layers.Input},
	 * {@code keras.engine.input_layer.Input}, and a {@code from ... import Input} binding of any of them) reads the one summary object of
	 * this type, so an exact points-to test on the invoked callee covers them all without matching names.
	 */
	private static final String INPUT_TYPE_NAME = "Ltensorflow/keras/layers/Input";

	/**
	 * Type-name prefixes of the built-in Keras layer <em>instances</em> the Functional API applies to symbolic values. Ariadne names an
	 * instance after the summary class without the {@code /class/} segment its class object carries ({@code tf.keras.layers.Dense(3)}
	 * yields an {@code Ltensorflow/keras/layers/Dense}), so excluding that segment separates applying a layer from constructing one: only
	 * the application propagates symbolicness. The {@code $}-prefixed form is the trampoline's, mirroring {@link StaticShapeReadAnalysis}'s
	 * endpoint prefixes.
	 */
	private static final Set<String> LAYER_TYPE_NAME_PREFIXES = Set.of("Ltensorflow/keras/layers/", "L$tensorflow/keras/layers/");

	/** The namespace segment holding class objects (constructors), excluded from the layer-application predicate. */
	private static final String CLASS_NAMESPACE_SEGMENT = "/class/";

	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	KerasSymbolicArgumentAnalysis(PointerAnalysis<InstanceKey> pointerAnalysis) {
		this.pointerAnalysis = pointerAnalysis;
	}

	/**
	 * Whether some call site of {@code nodes} passes a Keras symbolic tensor.
	 *
	 * @param nodes The call-graph nodes of the function in question.
	 * @param callGraph The call graph, queried in the caller direction.
	 * @return {@code TRUE} if some call site passes a symbolic tensor, {@code FALSE} if every examined call site passes none, or
	 *         {@code null} if a caller could not be examined at all.
	 */
	Boolean hasKerasSymbolicArgument(Set<CGNode> nodes, CallGraph callGraph) {
		for (CGNode node : nodes)
			for (CGNode predecessor : Iterator2Iterable.make(callGraph.getPredNodes(node))) {
				IR ir = predecessor.getIR();

				if (ir == null)
					// Undeterminable: this caller's arguments are invisible, so no verdict is available for the function.
					return null;

				DefUse defUse = predecessor.getDU();

				for (CallSiteReference site : Iterator2Iterable.make(callGraph.getPossibleSites(predecessor, node)))
					for (SSAAbstractInvokeInstruction instruction : ir.getCalls(site)) {
						if (!(instruction instanceof PythonInvokeInstruction invoke))
							return null;

						// Positional slot 0 is the callee itself; the arguments start at 1.
						for (int slot = 1; slot < invoke.getNumberOfPositionalParameters(); slot++)
							if (this.isKerasSymbolic(predecessor, invoke.getUse(slot), defUse, new HashMap<>()))
								return Boolean.TRUE;

						for (String keyword : invoke.getKeywords())
							if (this.isKerasSymbolic(predecessor, invoke.getUse(keyword), defUse, new HashMap<>()))
								return Boolean.TRUE;
					}
			}

		return Boolean.FALSE;
	}

	/**
	 * True iff {@code value} in {@code node}'s frame holds a Keras symbolic tensor, per the provenance rules documented on this class.
	 * {@code memo} both caches answers and guards the recursion against loops: a value is marked non-symbolic on entry, so a cycle
	 * contributes nothing, and its real answer replaces the marker on exit, so a sub-value two operands of one phi share is decided once
	 * rather than being skipped as already-visited by the second operand (which would report an all-symbolic merge as allowing).
	 */
	private boolean isKerasSymbolic(CGNode node, int value, DefUse defUse, Map<Integer, Boolean> memo) {
		// A phi operand is -1 where the variable is undefined on that path, which the def-use chains cannot be asked about. Not symbolic,
		// which under the universal phi rule is also enough to settle the merge.
		if (value < 0)
			return false;

		Boolean cached = memo.get(value);

		if (cached != null)
			return cached;

		memo.put(value, false);

		boolean result = this.computeKerasSymbolic(node, value, defUse, memo);
		memo.put(value, result);

		return result;
	}

	/** The provenance rules themselves; {@link #isKerasSymbolic} wraps this with the memo and the cycle guard. */
	private boolean computeKerasSymbolic(CGNode node, int value, DefUse defUse, Map<Integer, Boolean> memo) {
		SSAInstruction def = defUse.getDef(value);

		if (def instanceof PythonInvokeInstruction invoke) {
			int callee = invoke.getUse(0);

			if (Util.pointsToType(node, callee, this.pointerAnalysis, INPUT_TYPE_NAME, true))
				return true;

			// A built-in layer application propagates symbolicness from its inputs rather than producing it: the Functional API's
			// `Dense(...)(kt)` is a KerasTensor, while the same layer on an eager tensor is an eager tensor.
			if (this.appliesBuiltInKerasLayer(node, callee))
				for (int slot = 1; slot < invoke.getNumberOfPositionalParameters(); slot++)
					if (this.isKerasSymbolic(node, invoke.getUse(slot), defUse, memo))
						return true;

			return false;
		}

		// A merge is symbolic only when it cannot be anything else; see the class comment on why the quantifier is universal here and
		// existential across call sites.
		if (def instanceof SSAPhiInstruction phi) {
			for (int i = 0; i < phi.getNumberOfUses(); i++)
				if (!this.isKerasSymbolic(node, phi.getUse(i), defUse, memo))
					return false;

			// A phi always has operands, so reaching here means every one of them was symbolic.
			return true;
		}

		return false;
	}

	/** True iff {@code callee} refers to an instance of a built-in Keras layer rather than to a layer's class object. */
	private boolean appliesBuiltInKerasLayer(CGNode node, int callee) {
		for (InstanceKey instanceKey : this.pointerAnalysis
				.getPointsToSet(this.pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, callee))) {
			String name = instanceKey.concreteType().getReference().getName().toString();

			if (LAYER_TYPE_NAME_PREFIXES.stream().anyMatch(name::startsWith) && !name.contains(CLASS_NAMESPACE_SEGMENT))
				return true;
		}

		return false;
	}
}
