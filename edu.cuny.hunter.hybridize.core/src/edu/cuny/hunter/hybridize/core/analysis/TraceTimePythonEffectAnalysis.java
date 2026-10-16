package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Sets;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.Iterator2Iterable;

/**
 * The trace-time Python effect analysis behind the {@link PreconditionFailure#HAS_TRACE_TIME_PYTHON_EFFECTS} safety precondition
 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1077): a transitive walk over the call graph deciding whether a
 * function (or anything it reaches) does Python I/O, reads the clock, or draws Python-level randomness. Under {@code tf.function} such a
 * call runs once, while the function is traced, so its effect happens and its result is fixed on the first call only.
 * <p>
 * A call is recognized by the module its callee's attribute chain roots at, never by a bare member name: {@code random.uniform} draws once
 * at trace time, while {@code tf.random.uniform}, ending in the same two names, is a graph op that draws on every call, and a {@code time}
 * method on the program's own object is not the clock. The builtin {@code open} is recognized as the builtin only where the script does not
 * rebind the name. The walk mirrors {@link EagerOnlyCallAnalysis}.
 */
class TraceTimePythonEffectAnalysis {

	/** The builtin that opens a file. */
	private static final String OPEN_BUILTIN_NAME = "open";

	/** The fully-qualified name the TensorFlow module roots a chain at. */
	private static final String TENSORFLOW_MODULE_NAME = "tensorflow";

	/**
	 * The Python modules, other than TensorFlow, whose calls are recognized, each with the names an import binds it to in a script's scope
	 * ({@code import numpy as np} binds {@code np}). A module is also recognized by its object's type, so another alias is found where the
	 * points-to set of the read is available. Over-recognizing an alias, a script binding {@code random} to something else, over-blocks, as
	 * with {@link Util#NUMPY_MODULE_GLOBAL_NAMES}.
	 */
	private static final Map<String, Set<String>> MODULE_ALIASES = Map.of("time", Set.of("time"), "random", Set.of("random"), "numpy",
			Set.of("numpy", "np"), "cv2", Set.of("cv2"));

	/**
	 * Fully-qualified names of the recognized calls that are not recognized by module: the clock reads of the {@code time} module, whose
	 * other functions (such as {@code sleep}, which has its own trace-time problem, or the formatting ones) are left out, and OpenCV's file
	 * reads and writes, OpenCV's image operations being pure computations over arrays.
	 */
	private static final Set<String> TRACE_TIME_EFFECT_NAMES = Set.of("time.time", "time.time_ns", "time.perf_counter",
			"time.perf_counter_ns", "time.monotonic", "time.monotonic_ns", "time.process_time", "time.process_time_ns", "cv2.imread",
			"cv2.imreadmulti", "cv2.imwrite");

	/**
	 * Fully-qualified name prefixes of the recognized modules every call of which is recognized: Python's {@code random} and NumPy's
	 * {@code numpy.random}, whose every function draws from or reseeds a Python-level generator, and TensorFlow's {@code gfile}, whose file
	 * operations run in Python rather than as graph ops, under its TensorFlow 2 and compatibility spellings.
	 */
	private static final Set<String> TRACE_TIME_EFFECT_NAME_PREFIXES = Set.of("random.", "numpy.random.", "tensorflow.io.gfile.",
			"tensorflow.compat.v1.gfile.", "tensorflow.compat.v1.io.gfile.");

	/** The call graph, used to follow callees transitively and to tell a builtin from a rebinding of its name. */
	private final CallGraph callGraph;

	/** The pointer analysis, used to resolve callee attribute names and module objects. */
	private final PointerAnalysis<InstanceKey> pointerAnalysis;

	TraceTimePythonEffectAnalysis(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		this.callGraph = callGraph;
		this.pointerAnalysis = pointerAnalysis;
	}

	/**
	 * True iff {@code node}, transitively over its call-graph successors, calls a recognized Python I/O, clock, or random API. Only bodies
	 * outside the TensorFlow namespace are scanned, and the traversal walks through TensorFlow library nodes to reach user callbacks, both
	 * mirroring {@link EagerOnlyCallAnalysis#callsEagerOnlyApi(CGNode)}.
	 *
	 * @param node The call-graph node to check.
	 * @return True iff a recognized call is reachable from {@code node}.
	 */
	boolean hasTraceTimePythonEffects(CGNode node) {
		return this.hasTraceTimePythonEffects(node, Sets.newHashSet());
	}

	private boolean hasTraceTimePythonEffects(CGNode node, Set<CGNode> seen) {
		if (!seen.add(node))
			return false;

		if (!Util.isTensorFlowNode(node)) {
			IR ir = node.getIR();

			if (ir != null) {
				DefUse defUse = node.getDU();

				for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
					if (instruction instanceof PythonInvokeInstruction invoke && this.invokesTraceTimeEffect(node, invoke, defUse))
						return true;
			}
		}

		for (Iterator<CGNode> succNodes = this.callGraph.getSuccNodes(node); succNodes.hasNext();)
			if (this.hasTraceTimePythonEffects(succNodes.next(), seen))
				return true;

		return false;
	}

	/** True iff {@code invoke} calls the builtin {@code open} or a recognized module function. */
	private boolean invokesTraceTimeEffect(CGNode node, PythonInvokeInstruction invoke, DefUse defUse) {
		if (Util.callsBuiltin(node, invoke, defUse, this.callGraph, OPEN_BUILTIN_NAME))
			return true;

		String name = this.resolveModuleQualifiedName(node, invoke.getUse(0), defUse);

		return name != null
				&& (TRACE_TIME_EFFECT_NAMES.contains(name) || TRACE_TIME_EFFECT_NAME_PREFIXES.stream().anyMatch(name::startsWith));
	}

	/**
	 * The fully-qualified name of the value {@code use} refers to (e.g. {@code numpy.random.choice}), by walking its attribute chain back
	 * to a recognized module, or {@code null} when the chain does not root at one.
	 */
	private String resolveModuleQualifiedName(CGNode node, int use, DefUse defUse) {
		String module = this.moduleOf(node, use, defUse);

		if (module != null)
			return module;

		if (defUse.getDef(use) instanceof PythonPropertyRead read) {
			String member = Util.resolveStringConstant(node, read.getMemberRef(), this.pointerAnalysis);

			if (member == null)
				return null;

			String base = this.resolveModuleQualifiedName(node, read.getObjectRef(), defUse);
			return base == null ? null : base + "." + member;
		}

		return null;
	}

	/**
	 * The name of the recognized module {@code use} refers to, or {@code null} if it refers to none. Points-to is preferred; an import
	 * alias read from a script's scope is the fallback, as in {@link Util#isNumpyModule}. The fallback is needed for the modules no summary
	 * models, such as {@code random} and {@code time}, whose reads have empty points-to sets.
	 */
	private String moduleOf(CGNode node, int use, DefUse defUse) {
		if (Util.isTensorFlowModule(node, use, defUse, this.pointerAnalysis))
			return TENSORFLOW_MODULE_NAME;

		String alias = Util.scriptScopeReadName(defUse.getDef(use));

		for (Map.Entry<String, Set<String>> entry : MODULE_ALIASES.entrySet()) {
			String module = entry.getKey();

			if (Util.pointsToType(node, use, this.pointerAnalysis, "L" + module, true) || alias != null && entry.getValue().contains(alias))
				return module;
		}

		return null;
	}
}
