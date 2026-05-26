package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static edu.cuny.hunter.hybridize.core.analysis.Information.INPUT_SIGNATURE_INFERENCE;
import static edu.cuny.hunter.hybridize.core.analysis.Information.SPECULATIVE_ANALYSIS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_PRIMITIVE_PARAMETERS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P1;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P2;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P3;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.OPTIMIZE_HYBRID_FUNCTION;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.CONVERT_TO_EAGER;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.CONVERT_TO_HYBRID;
import static edu.cuny.hunter.hybridize.core.analysis.Util.getAllParentNames;
import static edu.cuny.hunter.hybridize.core.utils.Util.getPythonPath;
import static edu.cuny.hunter.hybridize.core.wala.ml.PythonModRefWithBuiltinFunctions.PythonModVisitorWithBuiltinFunctions.GLOBAL_OUTPUT_STREAM_POINTER_KEY;
import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;
import static org.eclipse.core.runtime.Platform.getLog;
import static org.python.pydev.parser.visitors.NodeUtils.getFullRepresentationString;
import static org.python.pydev.parser.visitors.NodeUtils.getOffset;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.IRegion;
import org.eclipse.jface.text.Region;
import org.eclipse.ltk.core.refactoring.FileStatusContext;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.RefactoringStatusContext;
import org.eclipse.ltk.core.refactoring.RefactoringStatusEntry;
import org.eclipse.text.edits.DeleteEdit;
import org.eclipse.text.edits.InsertEdit;
import org.eclipse.text.edits.MalformedTreeException;
import org.eclipse.text.edits.MultiTextEdit;
import org.eclipse.text.edits.TextEdit;
import org.osgi.framework.FrameworkUtil;
import org.python.pydev.ast.refactoring.AbstractPyRefactoring;
import org.python.pydev.ast.refactoring.HierarchyNodeModel;
import org.python.pydev.ast.refactoring.IPyRefactoring2;
import org.python.pydev.ast.refactoring.RefactoringRequest;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.ImportHandle;
import org.python.pydev.core.docutils.ImportHandle.ImportHandleInfo;
import org.python.pydev.core.docutils.PyImportsHandling;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.jython.ast.keywordType;
import org.python.pydev.parser.visitors.NodeUtils;

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.ibm.wala.cast.ipa.callgraph.AstGlobalPointerKey;
import com.ibm.wala.cast.ipa.callgraph.ScopeMappingInstanceKeys.ScopeMappingInstanceKey;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.StaticFieldKey;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import com.python.pydev.analysis.refactoring.refactorer.Refactorer;

import edu.cuny.hunter.hybridize.core.utils.RefactoringAvailabilityTester;

/**
 * A representation of a (syntactic) Python function.
 *
 * @author <a href="mailto:rk1424@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @author <a href="mailto:tcastrovelez@gradcenter.cuny.edu">Tatiana Castro Vélez</a>
 */
public class Function {

	/**
	 * Used for speculative analysis of the function name.
	 */
	private static final String FUNCTION_NAME_CONTEXT_REGEX = ".*(train|test).*_step|call|__call__|run_model|.*inference";

	/**
	 * Parameters that may be passed to a tf.fuction decorator. Parameter descriptions found at:
	 * https://tensorflow.org/versions/r2.9/api_docs/python/tf/function Note: We are also parsing the deprecated parameters specified in the
	 * documentation. Users can still use these deprecated parameters. Therefore we need to be able to account for them. Please refer to
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki/tf.function-parameter's-version-information to see more
	 * information about the tf.function parameters according to the versions.
	 */
	public class HybridizationParameters {

		private static final String AUTOGRAPH = "autograph";

		private static final String EXPERIMENTAL_AUTOGRAPH_OPTIONS = "experimental_autograph_options";

		private static final String EXPERIMENTAL_COMPILE = "experimental_compile";

		private static final String EXPERIMENTAL_FOLLOW_TYPE_HINTS = "experimental_follow_type_hints";

		private static final String EXPERIMENTAL_IMPLEMENTS = "experimental_implements";

		private static final String EXPERIMENTAL_RELAX_SHAPES = "experimental_relax_shapes";

		private static final String FUNC = "func";

		private static final String INPUT_SIGNATURE = "input_signature";

		private static final String JIT_COMPILE = "jit_compile";

		private static final String REDUCE_RETRACING = "reduce_retracing";

		/**
		 * The positional parameter order of {@code tf.function} as of TensorFlow 2.9 (the version this tool's tests target). When a user
		 * writes {@code @tf.function(some_callable, [tf.TensorSpec(...)])} the second argument binds to {@code input_signature} by
		 * position, etc. This array lets us map a positional index back to the parameter name without consulting PyDev's symbol-resolution
		 * machinery (which is brittle across PyDev versions and TF stub variants). The TF API is stable across the [2.0, 2.11] range we
		 * support; if a future TF version shuffles parameters, this array (along with `Util.isHybrid`'s acceptance window) would need an
		 * update. Tracks #108.
		 */
		private static final String[] TF_FUNCTION_POSITIONAL_PARAMS = { FUNC, INPUT_SIGNATURE, AUTOGRAPH, JIT_COMPILE, REDUCE_RETRACING,
				EXPERIMENTAL_IMPLEMENTS, EXPERIMENTAL_AUTOGRAPH_OPTIONS, EXPERIMENTAL_RELAX_SHAPES, EXPERIMENTAL_COMPILE,
				EXPERIMENTAL_FOLLOW_TYPE_HINTS };

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter autograph.
		 */
		private boolean autoGraphParam;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_follow_type_hints.
		 */
		private boolean experimentalFollowTypeHintsParam;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_autograph_options.
		 */
		private boolean experimentalAutographOptionsParam;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_implements.
		 */
		private boolean experimentalImplementsParam;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter func.
		 */
		private boolean funcParam;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter input_signature.
		 */
		private boolean inputSignatureParam;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter jit_compile.
		 */
		private boolean jitCompileParam;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter reduce_retracing.
		 */
		private boolean reduceRetracingParam;

		private void computeParameters() {
			// Use the hybrid decorator cached by `computeHybridization` (#118). That method already iterated every
			// decorator on this function and stored the hybrid hit in `Function.this.hybridDecorator`; we no
			// longer need to re-run the per-decorator `isHybrid` probe here.
			decoratorsType tfFunctionDecorator = Function.this.hybridDecorator;

			if (tfFunctionDecorator == null)
				throw new IllegalStateException(
						"No hybrid decorator was cached on " + Function.this + ". computeHybridization must run before computeParameters.");
			// tfFunctionDecorator must be an instance of Call, because that's the only way we have parameters.
			if (tfFunctionDecorator.func instanceof Call) {
				Call callFunction = (Call) tfFunctionDecorator.func;

				// Process positional arguments (#108). `tf.function`'s parameter order is hardcoded above in
				// `TF_FUNCTION_POSITIONAL_PARAMS`; arg[i] binds to that array's i-th name. Excess positional args
				// past the array length are silently ignored (Python would raise `TypeError` at decoration time,
				// which we don't try to mirror; the precondition framework would later flag the function as
				// non-hybridizable for unrelated reasons).
				exprType[] positionalArgs = callFunction.args;
				if (positionalArgs != null) {
					int limit = Math.min(positionalArgs.length, TF_FUNCTION_POSITIONAL_PARAMS.length);
					for (int i = 0; i < limit; i++)
						this.markParam(TF_FUNCTION_POSITIONAL_PARAMS[i]);
				}

				// Process keyword arguments. Keyword args are unordered; each carries its parameter name
				// directly. A user can mix positional and keyword in the same call (e.g.
				// `@tf.function(my_func, autograph=False)`); both branches mark the same fields.
				keywordType[] keywords = callFunction.keywords;
				for (keywordType keyword : keywords)
					if (keyword.arg instanceof NameTok) {
						NameTok name = (NameTok) keyword.arg;
						this.markParam(name.id);
					}
			} // else, tf.function is used without parameters.
		}

		/**
		 * Set the appropriate {@code *Param} field for the given {@code tf.function} parameter name. Recognizes both current names and the
		 * deprecated aliases ({@code experimental_compile} → {@code jit_compile}, {@code experimental_relax_shapes} →
		 * {@code reduce_retracing}). Unknown names are logged at {@code WARNING} level but otherwise ignored; they may belong to a future
		 * TF version we don't model yet. Intermediate step toward the original ask in #204 (custom exception + test); the log surfaces the
		 * signal without sacrificing forward-compatibility.
		 *
		 * @param paramName The parameter name passed to {@code @tf.function(...)}, exactly as it appears in the call.
		 */
		private void markParam(String paramName) {
			switch (paramName) {
			case FUNC -> this.funcParam = true;
			case INPUT_SIGNATURE -> this.inputSignatureParam = true;
			case AUTOGRAPH -> this.autoGraphParam = true;
			case JIT_COMPILE, EXPERIMENTAL_COMPILE -> this.jitCompileParam = true;
			case REDUCE_RETRACING, EXPERIMENTAL_RELAX_SHAPES -> this.reduceRetracingParam = true;
			case EXPERIMENTAL_IMPLEMENTS -> this.experimentalImplementsParam = true;
			case EXPERIMENTAL_AUTOGRAPH_OPTIONS -> this.experimentalAutographOptionsParam = true;
			case EXPERIMENTAL_FOLLOW_TYPE_HINTS -> this.experimentalFollowTypeHintsParam = true;
			default -> LOG.warn("Unknown @tf.function argument: " + paramName + " on " + Function.this + ".");
			}
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter autograph.
		 *
		 * @return True iff this {@link decoratorsType} has parameter autograph.
		 */
		public boolean hasAutoGraphParam() {
			return this.autoGraphParam;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_autograph_options.
		 *
		 * @return True iff this {@link decoratorsType} has parameter experimental_autograph_options.
		 */
		public boolean hasExperimentalAutographOptionsParam() {
			return this.experimentalAutographOptionsParam;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_follow_type_hints.
		 *
		 * @return True iff this {@link decoratorsType} has parameter experimental_follow_type_hints.
		 */
		public boolean hasExperimentalFollowTypeHintsParam() {
			return this.experimentalFollowTypeHintsParam;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_implements.
		 *
		 * @return True iff this {@link decoratorsType} has parameter experimental_implements.
		 */
		public boolean hasExperimentalImplementsParam() {
			return this.experimentalImplementsParam;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter func.
		 *
		 * @return True iff this {@link decoratorsType} has parameter func.
		 */
		public boolean hasFuncParam() {
			return this.funcParam;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter input_signature.
		 *
		 * @return True iff this {@link decoratorsType} has parameter input_signature.
		 */
		public boolean hasInputSignatureParam() {
			return this.inputSignatureParam;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter jit_compile.
		 *
		 * @return True iff this {@link decoratorsType} has parameter jit_compile.
		 */
		public boolean hasJitCompileParam() {
			return this.jitCompileParam;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter reduce_retracing.
		 *
		 * @return True iff this {@link Function} has parameter reduce_retracing.
		 */
		public boolean hasReduceRetracingParam() {
			return this.reduceRetracingParam;
		}
	}

	private static Map<MethodReference, Map<InstanceKey, Map<CallGraph, Boolean>>> creationsCache = Maps.newHashMap();

	private static final ILog LOG = getLog(Function.class);

	public static final String PLUGIN_ID = FrameworkUtil.getBundle(Function.class).getSymbolicName();

	/**
	 * Containing {@link File}s that have had import statements added to them during transformation.
	 */
	private static Set<File> filesWithAddedImport = new HashSet<>();

	private static final String TF_FUNCTION_FQN = "tensorflow.python.eager.def_function.function";

	/**
	 * True iff verbose output is desired.
	 */
	private static final boolean VERBOSE = false;

	/**
	 * True iff verbose output for an empty CG node set is desired.
	 */
	private static final boolean VERBOSE_NO_NODES = false;

	private static boolean allCreationsWithin(MethodReference methodReference, InstanceKey instanceKey, CallGraph callGraph) {
		int numCreations = 0;

		// for each creation site of the given instance.
		for (Iterator<Pair<CGNode, NewSiteReference>> it = instanceKey.getCreationSites(callGraph); it.hasNext();) {
			Pair<CGNode, NewSiteReference> creationSite = it.next();
			CGNode creationNode = creationSite.fst;
			NewSiteReference newSiteReference = creationSite.snd;

			// is this instance being created outside this function?
			if ((!creationNode.getMethod().getReference().equals(methodReference)
					&& !newSiteReference.getDeclaredType().equals(methodReference.getDeclaringClass())))
				return false;

			++numCreations;
		}

		if (numCreations == 0) // if there are no creations.
			// then, they can't be within this method.
			return false;

		return true;
	}

	private static boolean allCreationsWithinClosure(MethodReference methodReference, InstanceKey instanceKey, CallGraph callGraph) {
		Set<MethodReference> seen = Sets.newHashSet();
		return allCreationsWithinClosureInteral(methodReference, instanceKey, callGraph, seen);

	}

	private static boolean allCreationsWithinClosureInteral(MethodReference methodReference, InstanceKey instanceKey, CallGraph callGraph,
			Set<MethodReference> seen) {
		Map<InstanceKey, Map<CallGraph, Boolean>> cache2 = creationsCache.get(methodReference);

		if (cache2 != null) {
			Map<CallGraph, Boolean> cache3 = cache2.get(instanceKey);

			if (cache3 != null) {
				Boolean result = cache3.get(callGraph);

				if (result != null)
					return result;
			}
		}

		boolean result = allCreationsWithinClosureInteral2(methodReference, instanceKey, callGraph, seen);

		if (cache2 == null) {
			cache2 = Maps.newHashMap();
			creationsCache.put(methodReference, cache2);
		}

		Map<CallGraph, Boolean> cache3 = cache2.get(instanceKey);

		if (cache3 == null) {
			cache3 = Maps.newHashMap();
			cache2.put(instanceKey, cache3);
		}

		cache3.put(callGraph, result);

		return result;
	}

	private static boolean allCreationsWithinClosureInteral2(MethodReference methodReference, InstanceKey instanceKey, CallGraph callGraph,
			Set<MethodReference> seen) {
		seen.add(methodReference);

		// check this function.
		if (allCreationsWithin(methodReference, instanceKey, callGraph))
			return true;

		// otherwise, check its callees.
		Set<CGNode> cgNodes = getNodes(methodReference, callGraph);

		if (cgNodes.isEmpty())
			throw new IllegalArgumentException("Can't find call graph nodes corresponding to: " + methodReference + ".");

		// Only consider the first node. The only difference should be the calling context, which shouldn't make a difference for us.
		CGNode node = cgNodes.iterator().next();

		// check the callees.
		for (Iterator<CGNode> succNodes = callGraph.getSuccNodes(node); succNodes.hasNext();) {
			CGNode next = succNodes.next();
			MethodReference reference = next.getMethod().getReference();

			if (!seen.contains(reference) && allCreationsWithinClosureInteral(reference, instanceKey, callGraph, seen))
				return true;
		}

		return false;
	}

	public static void clearCaches() {
		creationsCache.clear();
		Parameter.clearCaches();
		filesWithAddedImport.clear();
	}

	/**
	 * Returns true iff the given {@link InstanceKey} takes on primitive values.
	 *
	 * @param instanceKey The {@link InstanceKey} in question.
	 * @param ignoreBooleans True iff boolean values should not be considered.
	 * @param pointerAnalysis The {@link PointerAnalysis} corresponding to the given {@link InstanceKey}.
	 * @param monitor To monitor progress.
	 * @return True iff the given {@link InstanceKey} takes on primitive values according to the given {@link PointerAnalysis}.
	 */
	private static boolean containsPrimitive(InstanceKey instanceKey, boolean ignoreBooleans, PointerAnalysis<InstanceKey> pointerAnalysis,
			IProgressMonitor monitor) {
		return containsPrimitive(instanceKey, ignoreBooleans, pointerAnalysis, new HashSet<>(), monitor);
	}

	private static boolean containsPrimitive(InstanceKey instanceKey, boolean ignoreBooleans, PointerAnalysis<InstanceKey> pointerAnalysis,
			Set<InstanceKey> seen, IProgressMonitor monitor) {
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Examining instance...", 1);

		seen.add(instanceKey);

		if (instanceKey instanceof ConstantKey<?>) {
			ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
			Object constantValue = constantKey.getValue();

			if (constantValue != null) {
				LOG.info("Found constant value: " + constantValue + ".");

				boolean foundBooleanValue = constantValue.equals(TRUE) || constantValue.equals(FALSE);

				// If it's not the case that we found a boolean value and we are ignoring booleans.
				if ((!foundBooleanValue || !ignoreBooleans)) {
					// We have found a primitive.
					subMonitor.done();
					return true;
				}
			}
		} else if (instanceKey instanceof AllocationSiteInNode || instanceKey instanceof ScopeMappingInstanceKey
				|| instanceKey instanceof ConcreteTypeKey) {
			InstanceKey instanceKeyToProcess;

			if (instanceKey instanceof AllocationSiteInNode || instanceKey instanceof ScopeMappingInstanceKey)
				instanceKeyToProcess = getAllocationSiteInNode(instanceKey);
			else // it's a ConcreteTypeKey.
				instanceKeyToProcess = instanceKey; // use the original.

			IClass concreteType = instanceKeyToProcess.getConcreteType();
			Collection<IField> allInstanceFields = concreteType.getAllInstanceFields();

			subMonitor.beginTask("Examining fields...", allInstanceFields.size());

			for (IField field : allInstanceFields) {
				InstanceFieldPointerKey instanceFieldKey = (InstanceFieldPointerKey) pointerAnalysis.getHeapModel()
						.getPointerKeyForInstanceField(instanceKeyToProcess, field);
				OrdinalSet<InstanceKey> instanceFieldPointsToSet = pointerAnalysis.getPointsToSet(instanceFieldKey);

				subMonitor.beginTask("Examining instance field instances...", instanceFieldPointsToSet.size());

				for (InstanceKey key : instanceFieldPointsToSet)
					if (!seen.contains(key) && containsPrimitive(key, ignoreBooleans, pointerAnalysis, seen, subMonitor.split(1))) {
						subMonitor.done();
						return true;
					}

				subMonitor.worked(1);
			}
		} else
			throw new IllegalArgumentException("Not expecting: " + instanceKey.getClass());

		subMonitor.done();
		return false;
	}

	/**
	 * Get the {@link CallGraph} nodes corresponding to the given {@link MethodReference}.
	 *
	 * @param methodReference The method to search for.
	 * @param callGraph The {@link CallGraph} to search.
	 * @return The nodes in the {@link CallGraph} corresponding to this {@link Function}.
	 * @apiNote There can be multiple nodes for a single {@link Function} under the current representation.
	 */
	private static Set<CGNode> getNodes(MethodReference methodReference, CallGraph callGraph) {
		Set<CGNode> nodes = callGraph.getNodes(methodReference);

		if (nodes.isEmpty()) {
			LOG.error("Can't get call graph nodes for: " + methodReference + ".");

			if (VERBOSE_NO_NODES) {
				LOG.info("Method reference is: " + methodReference + ".");
				LOG.info("Call graph nodes:\n" + callGraph.stream().map(Objects::toString).collect(Collectors.joining("\n")));
			}
		}

		LOG.info("Found " + nodes.size() + " node(s) corresponding to: " + methodReference + ".");

		if (VERBOSE)
			LOG.info("Nodes:\n" + nodes.stream().map(Objects::toString).collect(Collectors.joining("\n")));

		return nodes;
	}

	/**
	 * True iff the given decorator is a hybridization decorator.
	 *
	 * @param decorator The {@link decoratorsType} in question.
	 * @param containingModuleName The name of the module where the decorator is used.
	 * @param containingFile The {@link File} where the containingModuleName is defined.
	 * @param selection The {@link PySelection} containing the decorator.
	 * @param nature The {@link IPythonNature} to use.
	 * @param monitor The IProgressMonitor to use.
	 * @return The FQN of the given {@link decoratorsType}.
	 * @throws AmbiguousDeclaringModuleException If the definition of the decorator is ambiguous.
	 * @throws BadLocationException When the containing entities cannot be parsed.
	 * @throws NoDeclaringModuleException When a declaring module can't be found.
	 */
	private static boolean isHybrid(decoratorsType decorator, String containingModuleName, File containingFile, PySelection selection,
			IPythonNature nature, IProgressMonitor monitor)
			throws BadLocationException, AmbiguousDeclaringModuleException, NoDeclaringModuleException {
		String decoratorFQN = Util.getFullyQualifiedName(decorator, containingModuleName, containingFile, selection, nature, monitor);

		LOG.info("Found decorator: " + decoratorFQN + ".");

		// if this function is decorated with "tf.function."
		if (decoratorFQN.equals(TF_FUNCTION_FQN))
			return true;

		LOG.info(decoratorFQN + " does not equal " + TF_FUNCTION_FQN + ".");
		return false;
	}

	private boolean alwaysFollowTypeHints;

	/**
	 * True iff tensor contexts should be considered.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229">Issue 229</a>
	 */
	private boolean useSpeculativeAnalysis;

	/**
	 * The {@link FunctionDefinition} representing this {@link Function}.
	 */
	private FunctionDefinition functionDefinition;

	/**
	 * True iff this {@link Function} has Python side-effects.
	 */
	private Boolean hasPythonSideEffects;

	/**
	 * This {@link Function}'s associated hybridization parameters.
	 */
	private Function.HybridizationParameters hybridizationParameters;

	/**
	 * The hybrid decorator found on this {@link Function} during {@link #computeHybridization(IProgressMonitor)}, or {@code null} if no
	 * hybrid decorator was found (or hybridization has not yet been computed). Cached so that {@code
	 * HybridizationParameters.computeParameters} can reuse the result rather than re-running the per-decorator {@code isHybrid} probe
	 * (which is the slow part of decorator analysis: it walks selections, modules, and natures). If the function carries multiple hybrid
	 * decorators (unusual; stacking {@code @tf.function} is not semantically valid in TF), the last one in source order wins, matching the
	 * legacy behaviour of {@code computeParameters}. Tracks #118.
	 */
	private decoratorsType hybridDecorator;

	private boolean ignoreBooleans;

	/**
	 * True iff this {@link Function} is decorated with tf.function.
	 */
	private Boolean hybrid;

	private Boolean recursive;

	/**
	 * True iff this {@link Function} has at least one parameter that is likely a primitive.
	 */
	private Boolean hasPrimitiveParameter;

	/**
	 * True iff this {@link Function} has at least one parameter that is a tf.Tensor (https://bit.ly/3vYG7iP).
	 */
	private Boolean hasTensorParameter;

	private PreconditionSuccess passingPrecondition;

	/**
	 * The refactoring that this {@link Function} qualifies for. There should be only one as the refactorings are mutually exclusive.
	 */
	private Refactoring refactoring;

	private RefactoringStatus status = new RefactoringStatus();

	private Set<Transformation> transformations = new HashSet<>();

	/**
	 * Positional parameters wrapped as {@link Parameter}s. Built once in the constructor; never re-assigned. Empty if the underlying Jython
	 * {@code args} array is null or has zero entries.
	 */
	private final List<Parameter> parameters;

	public Function(FunctionDefinition fd, boolean ignoreBooleans, boolean alwaysFollowTypeHints, boolean useSpeculativeAnalysis) {
		this.functionDefinition = fd;
		this.ignoreBooleans = ignoreBooleans;
		this.alwaysFollowTypeHints = alwaysFollowTypeHints;
		this.useSpeculativeAnalysis = useSpeculativeAnalysis;

		// Jython's `argumentsType` is the whole parameter-list node; its `.args` field is the positional/positional-or-keyword name array.
		// `vararg`, `kwarg`, and `kwonlyargs` are sibling fields on the same node that we don't currently wrap.
		// Tracked at https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/465.
		argumentsType args = fd.getFunctionDef().args;
		List<Parameter> built = new ArrayList<>();
		if (args != null && args.args != null)
			for (int i = 0; i < args.args.length; i++)
				built.add(new Parameter(args, i, this));
		this.parameters = Collections.unmodifiableList(built);
	}

	public void addFailure(PreconditionFailure failure, String message) {
		// If is side-effects is filled, we can't set a precondition failure that we can't determine them.
		assert this.getHasPythonSideEffects() == null
				|| failure != PreconditionFailure.UNDETERMINABLE_SIDE_EFFECTS : "Can't both have side-effects filled and have tem undterminable.";

		this.addStatus(RefactoringStatus.ERROR, message, failure.getCode());
	}

	public void addInfo(Information information, String message) {
		this.addInfo(message, information.getCode());
	}

	public void addInfo(String message) {
		this.addInfo(message, RefactoringStatusEntry.NO_CODE);
	}

	private void addInfo(String message, int code) {
		this.addStatus(RefactoringStatus.INFO, message, code);
	}

	private void addStatus(int status, String message, int code) {
		FunctionDef functionDef = this.getFunctionDefinition().getFunctionDef();

		// https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/369.
		// Adding a very simply context here; only highlights the function name.
		int offset = getOffset(this.getContainingDocument(), functionDef.name);
		int length = getFullRepresentationString(functionDef).length();

		IRegion region = new Region(offset, length);
		RefactoringStatusContext context = new FileStatusContext(this.getContainingActualFile(), region);

		this.getStatus().addEntry(status, message, context, PLUGIN_ID, code, this);
	}

	protected void addTransformation(Transformation transformation) {
		assert (transformation != CONVERT_TO_EAGER || !this.getTransformations().contains(CONVERT_TO_HYBRID))
				&& (transformation != CONVERT_TO_HYBRID
						|| !this.getTransformations().contains(CONVERT_TO_EAGER)) : "Conversion transformations are mutually exclusive.";

		this.transformations.add(transformation);
	}

	public void addWarning(String message) {
		this.addStatus(RefactoringStatus.WARNING, message, RefactoringStatusEntry.NO_CODE);
	}

	/**
	 * Check refactoring preconditions. The status is added to this {@link Function}.
	 *
	 * @see #getStatus()
	 */
	public void check() {
		if (!this.isHybrid()) { // Eager. Table 1.
			this.setRefactoring(CONVERT_EAGER_FUNCTION_TO_HYBRID);

			if (this.getHasTensorParameter() != null && this.getHasTensorParameter()) {
				this.addInfo("This eager function likely has a tensor parameter.");
				if (this.getHasPrimitiveParameter() != null && !this.getHasPrimitiveParameter()) {
					this.addInfo("This eager function likely does not have a primitive parameter.");
					if (this.getHasPythonSideEffects() != null && !this.getHasPythonSideEffects()) {
						this.addInfo("This eager function does not have Python side-effects.");
						if (this.isRecursive() != null && !this.isRecursive()) {
							this.addInfo("This eager function is not recursive.");
							this.addTransformation(Transformation.CONVERT_TO_HYBRID);
							this.setPassingPrecondition(P1);
						} else if (this.isRecursive() != null) // it's recursive.
							this.addFailure(PreconditionFailure.IS_RECURSIVE, "Can't hybridize a recursive function.");
					} else if (this.getHasPythonSideEffects() != null) { // it has side-effects.
						this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
								"Can't hybridize a function with Python side-effects.");

						if (this.isRecursive() != null && this.isRecursive())
							this.addFailure(PreconditionFailure.IS_RECURSIVE, "Can't hybridize a recursive function.");
					}
				} else if (this.getHasPrimitiveParameter() != null) { // it has primitive parameters.
					this.addFailure(HAS_PRIMITIVE_PARAMETERS, "Hybridizing a function with primitive parameters may induce retracing.");

					if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects())
						this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
								"Can't hybridize a function with Python side-effects.");

					if (this.isRecursive() != null && this.isRecursive())
						this.addFailure(PreconditionFailure.IS_RECURSIVE, "Can't hybridize a recursive function.");
				}
			} else if (this.getHasTensorParameter() != null) { // no tensor parameters.
				this.addFailure(PreconditionFailure.HAS_NO_TENSOR_PARAMETERS,
						"This function has no tensor parameters and may not benefit from hybridization.");

				if (this.getHasPrimitiveParameter() != null && this.getHasPrimitiveParameter())
					this.addFailure(HAS_PRIMITIVE_PARAMETERS, "Hybridizing a function with primitive parameters may induce retracing.");

				if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects())
					this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS, "Can't hybridize a function with Python side-effects.");

				if (this.isRecursive() != null && this.isRecursive())
					this.addFailure(PreconditionFailure.IS_RECURSIVE, "Can't hybridize a recursive function.");
			}
		} else { // Hybrid. Use table 2.
			this.setRefactoring(OPTIMIZE_HYBRID_FUNCTION);

			if (this.getHasTensorParameter() != null && !this.getHasTensorParameter()) {
				this.addInfo("This hybrid function does not likely have a tensor parameter from tensor analysis.");

				if (this.getHasPythonSideEffects() != null && !this.getHasPythonSideEffects()) {
					this.addInfo("This hybrid function does not have Python side-effects.");
					this.addTransformation(CONVERT_TO_EAGER);
					this.setPassingPrecondition(P2);

				} else if (this.getHasPythonSideEffects() != null) // it has side-effects.
					this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
							"De-hybridizing a function with Python side-effects may alter semantics.");
			} else if (this.getHasTensorParameter() != null) { // it has a tensor parameter.
				this.addInfo("This hybrid function likely has a tensor parameter.");
				// if it has primitive parameters.
				if (this.getHasPrimitiveParameter() != null && this.getHasPrimitiveParameter()) {
					this.addInfo("This hybrid function likely has a primitive parameter.");
					// if it does not have side-effects.
					if (this.getHasPythonSideEffects() != null && !this.getHasPythonSideEffects()) {
						this.addInfo("This hybrid function does not have Python side-effects.");
						this.addTransformation(CONVERT_TO_EAGER);
						this.setPassingPrecondition(P3);
					} else if (this.getHasPythonSideEffects() != null) // it has side-effects.
						this.addFailure(HAS_PYTHON_SIDE_EFFECTS, "De-hybridizing a function with Python side-effects may alter semantics.");
				} else if (this.getHasPrimitiveParameter() != null) { // no primitive parameters.
					this.addFailure(PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS,
							"Functions with no Python literal arguments may benefit from hybridization.");

					if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects())
						this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
								"De-hybridizing a function with Python side-effects may alter semantics.");
				}

				// Here, we have a hybrid function with a tensor parameter.
				if (this.isRecursive() != null && this.isRecursive()) // if it's recursive.
					// issue a warning.
					this.addWarning("Recursive tf.functions are not supported by TensorFlow.");
			}

			// Warn if the function has side-effects.
			if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects())
				this.addWarning("This hybrid function potentially contains Python side-effects.");
		}
	}

	/**
	 * Discovers if this {@link Function} is hybrid. If so, populates this {@link Function}'s {@link HybridizationParameters}.
	 *
	 * @param monitor Progress monitor signaled while computing hybridization.
	 */
	public void computeHybridization(IProgressMonitor monitor) {
		// TODO: Consider mechanisms other than decorators (e.g., higher order functions; #3).
		monitor.beginTask("Computing hybridization ...", IProgressMonitor.UNKNOWN);

		// Reset cached state so a re-computation on the same instance starts clean. Without this, a previous hybrid
		// hit could leak past a subsequent run with no (or no hybrid) decorators and leave the function incorrectly
		// marked hybrid with stale `hybridizationParameters`. Function lifetime is currently per-refactoring-invocation
		// so re-computation isn't reachable today, but the reset is cheap and removes a reasoning hazard.
		this.hybridDecorator = null;
		this.hybridizationParameters = null;

		FunctionDefinition functionDefinition = this.getFunctionDefinition();
		decoratorsType[] decoratorArray = functionDefinition.getFunctionDef().decs;

		if (decoratorArray != null) {
			String containingModuleName = this.getContainingModuleName();
			File containingFile = this.getContainingFile();
			String containingFileName = containingFile.getName();
			IPythonNature nature = this.getNature();
			IProject project = this.getProject();

			// Iterate every decorator and remember the hybrid one (#118). The previous early-return-on-first
			// behaviour was correct for "is this function hybrid?" but forced HybridizationParameters to re-iterate
			// the decorators to recover the parameter source, running the expensive `isHybrid` probe a second time.
			// Now we run it once here, cache the hit, and let HybridizationParameters consume the cache. If a function
			// carries multiple hybrid decorators (unusual; stacking `@tf.function` is not semantically valid), the
			// last one in source order wins, matching legacy behaviour.
			for (decoratorsType decorator : decoratorArray) {
				String decoratorFunctionRepresentation = NodeUtils.getFullRepresentationString(decorator.func);
				LOG.info("Computing whether decorator: " + decoratorFunctionRepresentation + " is hybrid.");

				IDocument document = this.getContainingDocument();
				PySelection selection = null;

				// if this function is decorated with "tf.function."
				boolean hybrid = false;

				try {
					selection = Util.getSelection(decorator, document);
					hybrid = isHybrid(decorator, containingModuleName, containingFile, selection, nature, monitor);
				} catch (AmbiguousDeclaringModuleException | BadLocationException | NoDeclaringModuleException
						| NoTextSelectionException e) {
					String selectedText = null;
					try {
						selectedText = selection == null ? "(can't compute)" : selection.getSelectedText();
					} catch (BadLocationException e1) {
						// NOTE: No need to process; only for an error message.
						LOG.info("Can't get selected text.", e1);
					}

					if (Util.isGenerated(decorator))
						// Since tf.function isn't generated, skip generated decorators.
						LOG.info(String.format(
								"Encountered potentially generated decorator: %s in selection: %s, module: %s, file: %s, and project; %s.",
								decoratorFunctionRepresentation, selectedText, containingModuleName, containingFileName, project));
					else if (Util.isBuiltIn(decorator))
						// Since tf.function isn't built-in, skip built-in decorators.
						LOG.info(String.format(
								"Encountered potentially built-in decorator: %s in selection: %s, module: %s, file: %s, and project; %s.",
								decoratorFunctionRepresentation, selectedText, containingModuleName, containingFileName, project));
					else
						LOG.warn(String.format(
								"Can't determine if decorator: %s in selection: %s, module: %s, file: %s, and project; %s is hybrid.",
								decoratorFunctionRepresentation, selectedText, containingModuleName, containingFileName,
								nature.getProject()), e);
				}

				if (hybrid)
					this.hybridDecorator = decorator;
				monitor.worked(1);
			}
		}

		if (this.hybridDecorator != null) {
			this.setHybrid(TRUE);
			LOG.info(this + " is hybrid.");

			// Compute the hybridization parameters since we know now that this function is hybrid.
			LOG.info("Computing hybridization parameters.");
			this.hybridizationParameters = new HybridizationParameters();
			this.hybridizationParameters.computeParameters();
		} else {
			this.setHybrid(FALSE);
			LOG.info(this + " is not hybrid.");
		}
		monitor.done();
	}

	public void computeRecursion(CallGraph callGraph) throws CantComputeRecursionException, CoreException {
		// Get the nodes representing this function.
		Set<CGNode> nodes = this.getNodes(callGraph);

		if (nodes.isEmpty())
			throw new CantComputeRecursionException("Can't compute if " + this + " is recusive without a call graph node.");

		CGNode cgNode = nodes.iterator().next();

		if (Util.calls(cgNode, this.getMethodReference(), callGraph)) {
			// it's recursive.
			LOG.info(this + " is recursive.");
			this.setRecursive(true);
		} else {
			// not recursive.
			LOG.info(this + " is not recursive.");
			this.setRecursive(false);
		}
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if ((obj == null) || (this.getClass() != obj.getClass()))
			return false;
		Function other = (Function) obj;
		return Objects.equals(this.functionDefinition, other.functionDefinition);
	}

	private Set<PointerKey> filterSideEffects(Iterable<PointerKey> modSet, CallGraph callGraph,
			PointerAnalysis<InstanceKey> pointerAnalysis) throws CoreException {
		Set<PointerKey> ret = new HashSet<>();

		for (PointerKey pointerKey : modSet)
			if (pointerKey instanceof InstanceFieldPointerKey) {
				InstanceFieldPointerKey fieldPointerKey = (InstanceFieldPointerKey) pointerKey;
				InstanceKey instanceKey = fieldPointerKey.getInstanceKey();

				// Handle a special case where the instance is null.
				if (instanceKey instanceof ConstantKey) {
					ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
					if (constantKey.getValue() == null)
						continue; // filter this pointer out.
				}

				if (allCreationsWithinClosure(this.getMethodReference(), instanceKey, callGraph))
					continue; // filter this pointer out.

				ret.add(fieldPointerKey);
			} else if (pointerKey instanceof LocalPointerKey || pointerKey instanceof StaticFieldKey) {
				OrdinalSet<InstanceKey> pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);

				boolean skipPointerKey = true;

				for (InstanceKey ik : pointsToSet)
					skipPointerKey &= allCreationsWithinClosure(this.getMethodReference(), ik, callGraph);

				if (skipPointerKey && !pointsToSet.isEmpty())
					continue; // filter this pointer out.

				ret.add(pointerKey);
			} else if (pointerKey instanceof AstGlobalPointerKey) {
				AstGlobalPointerKey globalPointerKey = (AstGlobalPointerKey) pointerKey;

				if (!globalPointerKey.equals(GLOBAL_OUTPUT_STREAM_POINTER_KEY))
					throw new IllegalArgumentException("Not expecting global pointer key: " + globalPointerKey + ".");
				ret.add(globalPointerKey);
			} else
				throw new IllegalArgumentException("Not expecting pointer key: " + pointerKey + " of type: " + pointerKey.getClass() + ".");

		return ret;
	}

	/**
	 * Returns true iff we should use type hints regardless of a hybridization parameter.
	 *
	 * @return Whether we should use type hints regardless of what is specified in any hybridization parameters.
	 */
	public boolean getAlwaysFollowTypeHints() {
		return this.alwaysFollowTypeHints;
	}

	/**
	 * Returns true iff this {@link Function}'s tensor context should be considered.
	 *
	 * @return true iff this {@link Function}'s tensor context should be considered.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229">Issue 229</a>
	 */
	public boolean getUseSpeculativeAnalysis() {
		return useSpeculativeAnalysis;
	}

	public IDocument getContainingDocument() {
		return this.getFunctionDefinition().containingDocument;
	}

	/**
	 * Returns the {@link File} of where this {@link Function} is found.
	 *
	 * @return The {@link File} of where this {@link Function} is found.
	 */
	public File getContainingFile() {
		return this.getFunctionDefinition().containingFile;
	}

	/**
	 * Returns the {@link IFile} of where this {@link Function} is found.
	 *
	 * @return The {@link IFile} of where this {@link Function} is found.
	 */
	public IFile getContainingActualFile() {
		return this.getFunctionDefinition().containingActualFile;
	}

	/**
	 * Returns the Python module name of this {@link Function}.
	 *
	 * @return This {@link Function}'s Python module.
	 */
	public String getContainingModuleName() {
		return this.getFunctionDefinition().containingModuleName;
	}

	public TypeReference getDeclaringClass() throws CoreException {
		String filename = this.getDeclaringClassFilename().orElseThrow();
		String modifiedIdentifier = this.getIdentifier().replace('.', '/');
		String typeName = "Lscript " + filename + "/" + modifiedIdentifier;

		return TypeReference.findOrCreate(PythonTypes.pythonLoader, typeName);
	}

	protected Optional<String> getDeclaringClassFilename() throws CoreException {
		File containingFile = this.getContainingFile();
		List<File> pythonPath = getPythonPath(this.getProject());

		// If the PYTHONPATH isn't specified.
		if (pythonPath.isEmpty())
			// Revert to just the name.
			return Optional.of(containingFile.getName());

		for (File pathEntry : pythonPath) {
			String pathEntryAbsolutePath = pathEntry.getAbsoluteFile().getPath();
			String containingFileAbsolutePath = containingFile.getAbsolutePath();

			if (containingFileAbsolutePath.startsWith(pathEntryAbsolutePath)) {
				// Found it.
				Path pathEntryPath = Paths.get(pathEntryAbsolutePath);
				Path filePath = Paths.get(containingFileAbsolutePath);
				Path scriptRelativePath = pathEntryPath.relativize(filePath);
				return Optional.of(scriptRelativePath.toString());
			}
		}

		return Optional.empty(); // Not found.
	}

	public Set<String> getDecoratorNames(IProgressMonitor monitor) {
		SubMonitor progress = SubMonitor.convert(monitor);
		Set<String> ret = new HashSet<>();

		FunctionDefinition definition = this.getFunctionDefinition();
		FunctionDef def = definition.getFunctionDef();
		decoratorsType[] decs = def.decs;

		if (decs != null) {
			progress.setWorkRemaining(decs.length);

			for (decoratorsType decorator : decs)
				try {
					String decoratorFQN = getFQN(decorator, progress.split(1));
					ret.add(decoratorFQN);
				} catch (BadLocationException | AmbiguousDeclaringModuleException | NoDeclaringModuleException
						| NoTextSelectionException e) {
					// Best effort.
					LOG.info("Can't get name of decorator: " + decorator, e);
				}
		}

		return ret;
	}

	/**
	 * Converts the given {@link decoratorsType} to its corresponding qualified name as a {@link String}.
	 *
	 * @param decorator The decorator in question.
	 * @param monitor For progress monitoring.
	 * @return The corresponding decorator FQN.
	 * @throws NoTextSelectionException If a text selection over the decorator cannot be obtained.
	 * @throws BadLocationException If the decorator's location in the containing document is invalid.
	 * @throws AmbiguousDeclaringModuleException If the declaring module of the decorator cannot be resolved unambiguously.
	 * @throws NoDeclaringModuleException If the decorator has no resolvable declaring module.
	 */
	private String getFQN(decoratorsType decorator, IProgressMonitor monitor)
			throws NoTextSelectionException, BadLocationException, AmbiguousDeclaringModuleException, NoDeclaringModuleException {
		SubMonitor progress = SubMonitor.convert(monitor, 1);

		PySelection selection = Util.getSelection(decorator, this.getContainingDocument());

		return Util.getFullyQualifiedName(decorator, this.getContainingModuleName(), this.getContainingFile(), selection, this.getNature(),
				progress.split(1));
	}

	/**
	 * Returns the first {@link RefactoringStatusEntry} matching the given {@link PreconditionFailure}'s code in this {@link Function}'s
	 * {@link RefactoringStatus}.
	 *
	 * @param failure The {@link PreconditionFailure} whose {@link RefactoringStatusEntry} to find.
	 * @return The first {@link RefactoringStatusEntry} matching the given {@link PreconditionFailure}'s code in this {@link Function}'s
	 *         {@link RefactoringStatus}.
	 */
	public RefactoringStatusEntry getEntryMatchingFailure(PreconditionFailure failure) {
		return this.getStatus().getEntryMatchingCode(Function.PLUGIN_ID, failure.getCode());
	}

	public Set<RefactoringStatusEntry> getErrors() {
		return this.getRefactoringStatusEntries(RefactoringStatusEntry::isError);
	}

	/**
	 * This {@link Function}'s {@link FunctionDefinition}.
	 *
	 * @return The {@link FunctionDefinition} representing this {@link Function}.
	 */
	protected FunctionDefinition getFunctionDefinition() {
		return this.functionDefinition;
	}

	public Boolean getHasPythonSideEffects() {
		return this.hasPythonSideEffects;
	}

	/**
	 * This {@link Function}'s {@link HybridizationParameters}.
	 *
	 * @return This {@link Function}'s {@link HybridizationParameters}.
	 */
	public HybridizationParameters getHybridizationParameters() {
		return this.hybridizationParameters;
	}

	/**
	 * Returns the qualified name (QN) of this {@link Function}.
	 *
	 * @see <a href="https://peps.python.org/pep-3155">PEP 3155</a>
	 * @return This {@link Function}'s QN.
	 */
	public String getIdentifier() {
		FunctionDefinition functionDefinition = this.getFunctionDefinition();
		FunctionDef functionDef = functionDefinition.getFunctionDef();
		return Util.getQualifiedName(functionDef);
	}

	/**
	 * True iff booleans shouldn't be considered primitives.
	 *
	 * @return True iff boolean values shouldn't be considered primitives.
	 */
	protected boolean getIgnoreBooleans() {
		return this.ignoreBooleans;
	}

	/**
	 * True iff this {@link Function} is hybrid. Note that this only checks the decorator, i.e., whether all invocations of this
	 * {@link Function} are hybridized.
	 *
	 * @return True iff this {@link Function} is hybrid, i.e., whether it is decorated with tf.function.
	 */
	public Boolean isHybrid() {
		return this.hybrid;
	}

	public Boolean isRecursive() {
		return this.recursive;
	}

	/**
	 * Returns true iff this {@link Function} has at least one parameter that is likely a primitive.
	 *
	 * @return True iff this {@link Function} has at least one parameter that is likely a primitive.
	 */
	public Boolean getHasPrimitiveParameter() {
		return this.hasPrimitiveParameter;
	}

	/**
	 * True iff this {@link Function} likely has a tf.Tensor parameter. Since Python is dynamic, we may not be 100% sure.
	 *
	 * @return True iff this {@link Function} likely has a tf.Tensor parameter.
	 */
	public Boolean getHasTensorParameter() {
		return this.hasTensorParameter;
	}

	public MethodReference getMethodReference() throws CoreException {
		TypeReference typeReference = this.getDeclaringClass();
		return MethodReference.findOrCreate(typeReference, AstMethodReference.fnSelector);
	}

	/**
	 * Returns the {@link IPythonNature} for this {@link Function}.
	 *
	 * @return This {@link Function}'s {@link IPythonNature}.
	 */
	public IPythonNature getNature() {
		return this.getFunctionDefinition().getNature();
	}

	/**
	 * Get the {@link CallGraph} nodes corresponding to this {@link Function}.
	 *
	 * @param callGraph The {@link CallGraph} to search.
	 * @return The nodes in the {@link CallGraph} corresponding to this {@link Function}.
	 * @throws CoreException If resolving this function's {@link MethodReference} fails.
	 * @apiNote There can be multiple nodes for a single {@link Function} under the current representation.
	 */
	Set<CGNode> getNodes(CallGraph callGraph) throws CoreException {
		return getNodes(this.getMethodReference(), callGraph);
	}

	public int getNumberOfParameters() {
		return this.getFunctionDefinition().getFunctionDef().args.args.length;
	}

	/**
	 * Returns this {@link Function}'s positional parameters as {@link Parameter}s. The list is built once in the constructor and is
	 * immutable; empty if the function has no positional parameters.
	 *
	 * @return Unmodifiable list of {@link Parameter}s. Never {@code null}.
	 */
	public List<Parameter> getParameters() {
		return this.parameters;
	}

	public PreconditionSuccess getPassingPrecondition() {
		return this.passingPrecondition;
	}

	public IProject getProject() {
		return this.getFunctionDefinition().getProject();
	}

	public Refactoring getRefactoring() {
		return this.refactoring;
	}

	private Set<RefactoringStatusEntry> getRefactoringStatusEntries(Predicate<? super RefactoringStatusEntry> predicate) {
		return Arrays.stream(this.getStatus().getEntries()).filter(predicate).collect(Collectors.toSet());
	}

	public String getSimpleName() {
		return NodeUtils.getFullRepresentationString(this.getFunctionDefinition().getFunctionDef());
	}

	public RefactoringStatus getStatus() {
		return this.status;
	}

	public Set<Transformation> getTransformations() {
		return Collections.unmodifiableSet(this.transformations);
	}

	public Set<RefactoringStatusEntry> getWarnings() {
		return this.getRefactoringStatusEntries(RefactoringStatusEntry::isWarning);
	}

	@Override
	public int hashCode() {
		return Objects.hash(this.functionDefinition);
	}

	/**
	 * Returns true iff there is at most one {@link RefactoringStatusEntry} for a particular kind of failure.
	 *
	 * @apiNote This is to prevent counting a single kind of failure multiple times. Though that may be valid, I don't believe we have a
	 *          situation like this currently.
	 * @return True iff there is at most one failure per failure kind.
	 */
	public boolean hasOnlyOneFailurePerKind() {
		Map<Integer, List<RefactoringStatusEntry>> failureCodeToEntries = Arrays.stream(this.getStatus().getEntries())
				.filter(RefactoringStatusEntry::isError).collect(Collectors.groupingBy(RefactoringStatusEntry::getCode));

		for (Integer code : failureCodeToEntries.keySet()) {
			List<RefactoringStatusEntry> failuresForCode = failureCodeToEntries.get(code);
			if (failuresForCode.size() > 1)
				return false;
		}

		return true;
	}

	public void inferPrimitiveParameters(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis, IProgressMonitor monitor)
			throws CantInferPrimitiveParametersException, CoreException {
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Infering primitive parameters...", IProgressMonitor.UNKNOWN);
		Set<CGNode> nodes = this.getNodes(callGraph);

		if (nodes.isEmpty())
			throw new CantInferPrimitiveParametersException("Can't infer primitive parameters of " + this + " without a call graph node.");

		subMonitor.beginTask("Examining nodes...", nodes.size());

		for (CGNode nodeRepresentingThisFunction : nodes) {
			IR ir = nodeRepresentingThisFunction.getIR();

			subMonitor.beginTask("Examining explicit parameters (not self)...", ir.getNumberOfParameters() - 1);

			// Start at 1 or 2, depending on whether this is a method or not, because the first value is the function being invoked.
			// FIXME: Also consider kwargs and default args.
			// TODO: I wonder if ir.getParameterValueNumbers() returns kwargs as well.
			for (int paramInx = this.isMethod() ? 2 : 1; paramInx < ir.getNumberOfParameters(); paramInx++) {
				boolean allInstancesArePrimitive = true;

				int value = ir.getParameter(paramInx);
				PointerKey pointerKeyForLocal = pointerAnalysis.getHeapModel().getPointerKeyForLocal(nodeRepresentingThisFunction, value);
				OrdinalSet<InstanceKey> pointsToSet = pointerAnalysis.getPointsToSet(pointerKeyForLocal);

				subMonitor.beginTask("Examining instances...", pointsToSet.size());

				for (InstanceKey instanceKey : pointsToSet) {
					LOG.info("Parameter of: " + this + " with index: " + paramInx + " points to: " + instanceKey + ".");

					allInstancesArePrimitive &= containsPrimitive(instanceKey, this.getIgnoreBooleans(), pointerAnalysis,
							subMonitor.split(1));
					subMonitor.worked(1);
				}

				if (!pointsToSet.isEmpty() && allInstancesArePrimitive) {
					LOG.info(this + " likely has a primitive parameter.");
					this.hasPrimitiveParameter = TRUE;
					subMonitor.done();
					return;
				}

				subMonitor.worked(1);
			}

			subMonitor.worked(1);
		}

		LOG.info(this + " likely does not have a primitive parameter.");
		this.hasPrimitiveParameter = FALSE;
		subMonitor.done();
	}

	/**
	 * Infer Python side-effects potentially produced by executing this {@link Function}.
	 *
	 * @param mod The ModRef analysis result.
	 * @param callGraph The system {@link CallGraph}.
	 * @param pointerAnalysis The system {@link PointerAnalysis}.
	 * @throws UndeterminablePythonSideEffectsException If this {@link Function}'s representation isn't found in the given
	 *         {@link CallGraph}.
	 * @throws CoreException If resolving this function's {@link MethodReference} fails while looking up its call-graph nodes.
	 */
	public void inferPythonSideEffects(Map<CGNode, OrdinalSet<PointerKey>> mod, CallGraph callGraph,
			PointerAnalysis<InstanceKey> pointerAnalysis) throws UndeterminablePythonSideEffectsException, CoreException {
		// Get the nodes corresponding to this function's declaration. NOTE: There can be multiple nodes for a function declaration under
		// the current representation. It seems that there is a declaration node for each call to the function. Each node has a different
		// calling context.
		Set<CGNode> nodes = this.getNodes(callGraph);

		if (nodes.isEmpty())
			throw new UndeterminablePythonSideEffectsException(this.getMethodReference());

		// Only consider the first node. The calling context shouldn't matter for us right now.
		CGNode cgNode = nodes.iterator().next();

		// Get the locations (pointers) modified by this function.
		OrdinalSet<PointerKey> modSet = mod.get(cgNode);
		LOG.info("Found " + modSet.size() + " original modified location(s).");
		modSet.forEach(pk -> LOG.info("Original modified location: " + pk + "."));

		// Filter out the modified locations.
		Set<PointerKey> filteredModSet = this.filterSideEffects(modSet, callGraph, pointerAnalysis);
		LOG.info("Found " + filteredModSet.size() + " filtered modified location(s).");
		filteredModSet.forEach(pk -> LOG.info("Filtered modified location: " + pk + "."));

		// Log the locations we are removing.
		SetView<PointerKey> removed = Sets.difference(Sets.newHashSet(modSet), filteredModSet);
		LOG.info("Removed " + removed.size() + " locations.");
		removed.forEach(pk -> LOG.info("Removed modified location: " + pk + "."));

		if (!filteredModSet.isEmpty()) {
			this.setHasPythonSideEffects(TRUE);
			LOG.info(this + " has side-effects.");
			return;
		}

		this.setHasPythonSideEffects(FALSE);
		LOG.info(this + " does not have side-effects.");
	}

	/**
	 * Infer which parameters are likely tensor parameters.
	 *
	 * @param tensorAnalysis The tensor-type analysis result feeding the per-parameter classification.
	 * @param callGraph The system {@link CallGraph}.
	 * @param builder The call-graph builder, used to resolve definitions referenced by the analysis.
	 * @param monitor Progress monitor signaled while inferring tensor parameters.
	 * @throws Exception If the underlying call-graph, points-to, or AST lookup fails.
	 */
	public void inferTensorParameters(TensorTypeAnalysis tensorAnalysis, CallGraph callGraph, PythonSSAPropagationCallGraphBuilder builder,
			IProgressMonitor monitor) throws Exception {
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Infering tensor parameters...", IProgressMonitor.UNKNOWN);
		Set<CGNode> nodes = this.getNodes(callGraph);

		// True iff the function has a self parameter in the first position.
		boolean selfParam = false;

		List<Parameter> params = this.getParameters(); // FIXME: positional only (#108).
		subMonitor.setWorkRemaining(params.size());

		for (Parameter param : params) {
			if (param.isSelf()) {
				selfParam = true;
				subMonitor.worked(1);
				continue; // skip self parameters.
			}

			if (param.classifyAsTensor(tensorAnalysis, nodes, builder, subMonitor.split(IProgressMonitor.UNKNOWN))) {
				this.hasTensorParameter = TRUE;
				subMonitor.worked(1);
				continue; // next parameter.
			}

			subMonitor.worked(1);
		}

		// True if there is only one parameter that is self.
		final boolean onlySelfParam = params.size() == 1 && selfParam;

		// if we haven't yet determined if there's a tensor parameter and there's at least one parameter that's not only self.
		if (this.hasTensorParameter == null && !params.isEmpty() && !onlySelfParam)
			// check a special case where we consider context.
			if (this.getUseSpeculativeAnalysis() && this.hasTensorContext()) {
				this.hasTensorParameter = TRUE;
				LOG.info(this + " likely has a tensor parameter due to context.");
				this.addInfo(SPECULATIVE_ANALYSIS, "Used function context to infer parameter tensor types.");
			} else if (nodes.isEmpty())
				// if there are no nodes representing this function, then it most likely isn't called.
				throw new CantInferTensorParametersException("Can't infer tensor parameters for " + this + " without a call graph node.");

		if (this.hasTensorParameter == null) {
			this.hasTensorParameter = FALSE;
			LOG.info(this + " does not likely have a tensor parameter.");
		}

		subMonitor.done();
	}

	/**
	 * Infers the input signature of this function: an ordered tuple of {@link TensorType}s, one per non-{@code self} parameter the
	 * tensor-type analysis associated with at least one tensor type. Mirrors the no-argument pattern of {@link #getHasTensorParameter}: the
	 * values are computed during {@link #inferTensorParameters} (which caches per-parameter tensor types on each {@link Parameter}), and
	 * this method reads those cached values. For each non-{@code self} parameter, this method dispatches on {@link Parameter#isTensor()}
	 * into three categories:
	 * <ul>
	 * <li>Truly non-tensor ({@code isTensor() != TRUE}): drop the signature and emit a per-parameter INFO suggesting the source-side
	 * recovery (annotate as {@code tf.Tensor} and wrap call sites with {@code tf.constant(...)}). The tool does not synthesize a
	 * {@link TensorType} for the parameter because wrapping a Python primitive as a tensor changes AutoGraph's rewrite of Python control
	 * flow over the parameter.
	 * <li>Tensor-classified by type hint or container detection but no concrete shape/dtype evidence
	 * ({@code isTensor() == TRUE && getTensorTypes().isEmpty()}): drop the signature and emit a per-parameter INFO noting that the
	 * tool-side recovery (extending the {@link Parameter} API to expose this signal) is tracked at #509.
	 * <li>Phase-2 hit ({@code isTensor() == TRUE && !getTensorTypes().isEmpty()}): reduce the cached set via {@link #inferSpec} and add the
	 * reduced spec to the signature.
	 * </ul>
	 * Current scope: a single tensor type per parameter, with concrete dtype and concrete shape. Multi-context (#507) and other
	 * non-concrete cases (#494) return {@link Optional#empty} pending future PRs that extend {@link #inferSpec}.
	 *
	 * @return The inferred signature, or {@link Optional#empty} if any non-{@code self} parameter blocks inference or if {@link #inferSpec}
	 *         cannot reduce one of the per-parameter sets.
	 */
	public Optional<InputSignature> inferInputSignature() {
		List<TensorType> specs = new ArrayList<>();
		boolean blocked = false;

		for (Parameter param : this.getParameters()) {
			// `self` is excluded from the signature.
			if (param.isSelf())
				continue;

			Boolean classified = param.isTensor();
			if (classified == null || !classified) {
				// Category (a): truly non-tensor. The developer's source code is correct as-is; this is a design opportunity, not a
				// problem. Emit a source-side recovery suggestion. The tool does not synthesize a TensorType here because wrapping
				// a Python primitive as a tensor changes AutoGraph's rewrite of Python control flow over the parameter (`range(n)`
				// becomes problematic, `if n > 0` becomes `tf.cond`, etc.). See #508 for the design decision. Continue the loop so
				// all blocking parameters surface their INFOs in one pass instead of one per refactoring rerun.
				this.addInfo(INPUT_SIGNATURE_INFERENCE,
						"Parameter `" + param.getName() + "` of `" + this + "` is not classified as tensor-typed and prevents "
								+ "input-signature inference. Consider changing `" + param.getName() + "` to accept a `tf.Tensor` "
								+ "(annotate as `" + param.getName() + ": tf.Tensor` and pass `tf.constant(...)` at call sites). "
								+ "If the change is appropriate for this function's semantics, rerunning the refactoring will infer "
								+ "a complete input signature including `" + param.getName() + "`.");
				blocked = true;
				continue;
			}

			Set<TensorType> contexts = param.getTensorTypes();
			if (contexts.isEmpty()) {
				// Category (b): tensor-classified by Phase 1 (type hint) or Phase 3 (container) but no Phase 2 (Ariadne call-site)
				// shape/dtype evidence. Recovery is tool-side (extend the Parameter API to surface what Ariadne already knows for
				// containers, or extract dtype information from typed annotations), tracked at #509.
				this.addInfo(INPUT_SIGNATURE_INFERENCE,
						"Parameter `" + param.getName() + "` of `" + this + "` is classified as tensor-typed via type hint or "
								+ "container detection but has no concrete shape/dtype evidence; input-signature inference is "
								+ "dropped. Synthesizing a TensorSpec from this signal is tracked at #509.");
				blocked = true;
				continue;
			}

			Optional<TensorType> spec = inferSpec(contexts);
			if (spec.isEmpty()) {
				// Reduction returned bottom for this parameter; the whole signature collapses. Per-parameter INFO emission for the
				// `inferSpec`-side drops (multi-context, dtype-⊤, symbolic dim) is tracked at #510.
				blocked = true;
				continue;
			}

			specs.add(spec.get());
		}

		if (blocked || specs.isEmpty())
			return Optional.empty();

		return Optional.of(new InputSignature(specs));
	}

	/**
	 * Reduces the multi-context set of {@link TensorType}s seen for a single parameter to a single {@link TensorType}. Three steps:
	 * <ol>
	 * <li><b>Dtype consensus.</b> If the per-context dtypes don't agree on a single value, return {@link Optional#empty} (the
	 * {@code |D| ≠ 1 ⇒ ⊥} branch). If the agreed dtype is {@code UNKNOWN} (dtype-⊤), also drop—pending #494, since {@code tf.UNKNOWN} isn't
	 * a valid runtime dtype for {@code tf.function(input_signature=...)}.
	 * <li><b>Rank consensus or shape-⊤.</b> If any context has {@code dims == null} (unknown rank) or the ranks disagree across contexts,
	 * emit a coarse {@code TensorType(dtype, null)} (shape-⊤). This is a valid, runtime-accepted signature.
	 * <li><b>Per-position consensus or wildcard.</b> For each dimension position, if all contexts agree on a concrete value, keep it;
	 * otherwise emit a {@link SymbolicDim}({@code "?"}) wildcard. Any non-{@link NumericDim} context dim also yields a wildcard at that
	 * position.
	 * </ol>
	 *
	 * @param contexts The non-empty set of {@link TensorType}s Ariadne associated with the parameter across call contexts.
	 * @return The reduced single {@link TensorType}, or {@link Optional#empty} for the dtype-⊥ and dtype-⊤ branches.
	 */
	private static Optional<TensorType> inferSpec(Set<TensorType> contexts) {
		// Step 1: dtype consensus. Walk the contexts; any disagreement drops the signature.
		DType dtype = null;
		for (TensorType t : contexts) {
			DType d = t.getDType();
			if (dtype == null)
				dtype = d;
			else if (!dtype.equals(d))
				// Heterogeneous dtype across contexts: drop the signature (the `|D| ≠ 1 ⇒ ⊥` branch).
				return Optional.empty();
		}
		if (dtype == null || dtype == DType.UNKNOWN)
			// Empty contexts (filtered upstream by `inferInputSignature`'s `contexts.isEmpty()` check) or dtype-⊤. The latter is a
			// conservative drop because `tf.UNKNOWN` isn't a valid runtime dtype for `input_signature`. Pending #494.
			return Optional.empty();

		// Step 2: rank consensus or shape-⊤. If any context has shape = null or ranks disagree, emit `TensorType(dtype, null)`,
		// preserving the dtype axis even when the shape axis degrades.
		// `rank` uses -1 as a "not yet set" sentinel: dim list sizes are always non-negative, so the sentinel can't collide. A boxed
		// `Integer rank = null` would compile-fail under the bundle's strict null-analysis (-err:+nullAnalysis) on the auto-unboxing
		// sites below.
		int rank = -1;
		for (TensorType t : contexts) {
			List<Dimension<?>> dims = t.getDims();
			if (dims == null)
				return Optional.of(new TensorType(dtype, null));
			if (rank == -1)
				rank = dims.size();
			else if (rank != dims.size())
				return Optional.of(new TensorType(dtype, null));
		}

		// Step 3: per-dim consensus or wildcard. If all contexts agree on a concrete value at position j, keep it; else emit a
		// `SymbolicDim("?")` wildcard.
		// TODO(wala/ML#544, #524): once Ariadne ships a typed `RaggedDim` and this side emits `RaggedTensorSpec`, branch on
		// raggedness here rather than collapsing every non-`NumericDim` to a wildcard.
		List<Dimension<?>> shape = new ArrayList<>(rank);
		for (int j = 0; j < rank; j++) {
			Dimension<?> consensus = null;
			boolean disagreement = false;
			for (TensorType t : contexts) {
				Dimension<?> d = t.getDims().get(j);
				if (consensus == null)
					consensus = d;
				else if (!consensus.equals(d)) {
					disagreement = true;
					break;
				}
			}
			if (!disagreement && consensus instanceof NumericDim)
				shape.add(consensus);
			else
				shape.add(new SymbolicDim("?"));
		}

		return Optional.of(new TensorType(dtype, shape));
	}

	private boolean hasTensorContext() {
		String functionName = this.getSimpleName();
		boolean matches = functionName.matches(FUNCTION_NAME_CONTEXT_REGEX);

		// if we have a match and it's a functor.
		if (matches && (functionName.equals("call") || functionName.equals("__call__"))) {
			// check that we inherit from tf.keras.Model.
			FunctionDef functionDef = this.getFunctionDefinition().getFunctionDef();

			if (functionDef.parent instanceof ClassDef) {
				Set<String> parentNames = this.getAllClassParentNames(true);

				if (parentNames.stream().filter(pn -> pn.equals("Model")).findAny().isPresent())
					return true;
			}

			return false;
		}

		return matches;
	}

	private Set<String> getAllClassParentNames(boolean onlyLastSegment) {
		Set<String> ret = new HashSet<>();
		SimpleNode node = this.getFunctionDefinition().getFunctionDef().parent;

		if (node instanceof ClassDef) {
			ClassDef def = (ClassDef) node;

			PySelection selection = null;
			try {
				selection = Util.getSelection(def.name, getContainingDocument());
			} catch (NoTextSelectionException e) {
				LOG.info("Can't get class parent names for: " + this + " with enclosing class: " + def + " with name:" + def.name, e);
			}

			if (selection != null) {
				RefactoringRequest request = new RefactoringRequest(getContainingFile(), selection, getNature());
				IPyRefactoring2 refactoring = (Refactorer) AbstractPyRefactoring.getPyRefactoring();
				HierarchyNodeModel hierarchyNode = refactoring.findClassHierarchy(request, true);

				if (hierarchyNode != null)
					return getAllParentNames(hierarchyNode, onlyLastSegment);
			}

			// otherwise, just traverse the base in this AST node.
			ret.addAll(NodeUtils.getParentNames(def, onlyLastSegment));
		}

		return ret;
	}

	public boolean isHybridizationAvailable() {
		return RefactoringAvailabilityTester.isHybridizationAvailable(this.getFunctionDefinition().getFunctionDef());
	}

	/**
	 * Returns true iff this {@link Function} represents an instance method.
	 *
	 * @return True iff this {@link Function} is an instance method.
	 */
	public boolean isMethod() {
		List<Parameter> parameters = this.getParameters();
		return parameters.size() >= 1 && parameters.get(0).isSelf();
	}

	protected void setHasPythonSideEffects(Boolean hasPythonSideEffects) {
		assert this.hasPythonSideEffects == null : "Can only set side-effects once.";
		assert hasPythonSideEffects == null || this.getStatus().getEntryMatchingCode(PLUGIN_ID,
				PreconditionFailure.UNDETERMINABLE_SIDE_EFFECTS.getCode()) == null : "Can't set side-effects if they are undeterminable.";

		this.hasPythonSideEffects = hasPythonSideEffects;
	}

	protected void setHybrid(Boolean hybrid) {
		this.hybrid = hybrid;
	}

	protected void setRecursive(Boolean recursive) {
		this.recursive = recursive;
	}

	protected void setPassingPrecondition(PreconditionSuccess passingPrecondition) {
		this.passingPrecondition = passingPrecondition;
	}

	public void setRefactoring(Refactoring refactoring) {
		this.refactoring = refactoring;
	}

	@Override
	public String toString() {
		return this.getIdentifier() + "()";
	}

	public boolean willDehybridize() {
		return this.getTransformations().contains(CONVERT_TO_EAGER);
	}

	public List<TextEdit> transform() throws BadLocationException, MalformedTreeException, NoTextSelectionException,
			AmbiguousDeclaringModuleException, NoDeclaringModuleException {
		List<TextEdit> ret = new ArrayList<>();
		Set<Transformation> transformations = this.getTransformations();

		for (Transformation transformation : transformations) {
			switch (transformation) {
			case CONVERT_TO_HYBRID:
				ret.addAll(this.convertToHybrid());
				break;
			case CONVERT_TO_EAGER:
				ret.addAll(this.convertToEager());
				break;
			case RECONFIGURE:
				throw new UnsupportedOperationException();
			default:
				throw new IllegalStateException();
			}
		}

		return ret;
	}

	private List<TextEdit> convertToEager()
			throws NoTextSelectionException, BadLocationException, AmbiguousDeclaringModuleException, NoDeclaringModuleException {
		assert this.getDecoratorNames(null).contains(TF_FUNCTION_FQN) : "Already eager.";

		// there can be more than one.
		List<TextEdit> ret = new ArrayList<>();

		FunctionDefinition functionDefinition = this.getFunctionDefinition();
		FunctionDef functionDef = functionDefinition.getFunctionDef();

		for (decoratorsType decorator : functionDef.decs) {
			String fqn = this.getFQN(decorator, null);

			if (fqn.equals(TF_FUNCTION_FQN)) {
				IDocument doc = this.getContainingDocument();
				int offset = getOffset(doc, decorator);
				String fullRepresentationString = getFullRepresentationString(decorator.func);
				int length = fullRepresentationString.length() + 1;

				int newline = offset + length;
				char charAtEnd = doc.getChar(newline);

				// is the decorator on its own line?
				if (charAtEnd == '\n') {
					++length; // also remove the newline.

					// also remove the preceding text.
					int lineBeginOffset = offset - functionDef.beginColumn + 1;
					offset = lineBeginOffset;
					length += functionDef.beginColumn - 1;
				}

				TextEdit edit = new DeleteEdit(offset, length);
				MultiTextEdit mte = new MultiTextEdit();
				mte.addChild(edit);
				ret.add(mte);
			}
		}

		return ret;
	}

	private List<TextEdit> convertToHybrid() throws BadLocationException {
		assert !this.getDecoratorNames(null).contains(TF_FUNCTION_FQN) : "Already hybrid.";

		List<TextEdit> ret = new ArrayList<>();

		FunctionDefinition functionDefinition = this.getFunctionDefinition();
		FunctionDef functionDef = functionDefinition.getFunctionDef();

		IDocument doc = this.getContainingDocument();
		int offset = getOffset(doc, functionDef);
		int lineBeginOffset = offset - functionDef.beginColumn + 1;

		String precedingText = doc.get(lineBeginOffset, functionDef.beginColumn - 1);

		String prefix = getImportPrefix(doc);

		if (prefix == null) {
			// need to add an import if it doesn't already exist.
			File file = this.getContainingFile();

			if (!filesWithAddedImport.contains(file)) {
				int line = getLineToInsertImport(doc);
				int lineOffset = doc.getLineOffset(line);

				TextEdit edit = new InsertEdit(lineOffset, "from tensorflow import function\n");
				MultiTextEdit mte = new MultiTextEdit();
				mte.addChild(edit);
				ret.add(mte);
				filesWithAddedImport.add(file);
			}

			prefix = ""; // no prefix needed.
		}

		TextEdit edit = new InsertEdit(offset, "@" + prefix + "function\n" + precedingText);
		MultiTextEdit mte = new MultiTextEdit();
		mte.addChild(edit);
		ret.add(mte);

		return ret;
	}

	private static int getLineToInsertImport(IDocument doc) {
		PyImportsHandling handling = new PyImportsHandling(doc);
		int lastFoundImportLine = -1;

		for (Iterator<ImportHandle> it = handling.iterator(); it.hasNext();) {
			ImportHandle importHandle = it.next();
			lastFoundImportLine = importHandle.endFoundLine;
		}

		return lastFoundImportLine + 1;
	}

	private static String getImportPrefix(IDocument doc) {
		PyImportsHandling handling = new PyImportsHandling(doc);

		for (ImportHandle importHandle : handling)
			for (ImportHandleInfo importHandleInfo : importHandle.getImportInfo())
				for (String importStr : importHandleInfo.getImportedStr())
					if (importStr.equals("tensorflow"))
						return "tensorflow.";
					else if (importStr.startsWith("tensorflow as"))
						return importStr.substring("tensorflow as ".length(), importStr.length()) + ".";
					else {
						String fromImportStr = importHandleInfo.getFromImportStrWithoutUnwantedChars();
						if (fromImportStr != null && fromImportStr.equals("tensorflow"))
							switch (importStr) {
							case "*": // wild card import.
							case "function": // direct import.
								return ""; // no prefix needed.
							}
					}

		// not found.
		return null;
	}
}
