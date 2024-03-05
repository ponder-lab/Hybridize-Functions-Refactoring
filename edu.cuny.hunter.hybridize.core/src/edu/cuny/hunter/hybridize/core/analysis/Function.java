package edu.cuny.hunter.hybridize.core.analysis;

import static edu.cuny.hunter.hybridize.core.analysis.Information.TYPE_INFERENCING;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_PRIMITIVE_PARAMETERS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P1;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P2;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P3;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.OPTIMIZE_HYBRID_FUNCTION;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.CONVERT_TO_EAGER;
import static edu.cuny.hunter.hybridize.core.wala.ml.PythonModRefWithBuiltinFunctions.PythonModVisitorWithBuiltinFunctions.GLOBAL_OUTPUT_STREAM_POINTER_KEY;
import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;
import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.RefactoringStatusContext;
import org.eclipse.ltk.core.refactoring.RefactoringStatusEntry;
import org.osgi.framework.FrameworkUtil;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.VisitorBase;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.jython.ast.keywordType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.parser.visitors.TypeInfo;

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.ibm.wala.cast.ipa.callgraph.AstGlobalPointerKey;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.StaticFieldKey;
import com.ibm.wala.ipa.modref.ModRef;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;

import edu.cuny.hunter.hybridize.core.utils.RefactoringAvailabilityTester;
import edu.cuny.hunter.hybridize.core.wala.ml.PythonModRefWithBuiltinFunctions;

/**
 * A representation of a Python function.
 *
 * @author <a href="mailto:rk1424@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @author <a href="mailto:tcastrovelez@gradcenter.cuny.edu">Tatiana Castro Vélez</a>
 */
public class Function {

	public static final String PLUGIN_ID = FrameworkUtil.getBundle(Function.class).getSymbolicName();

	private final class FunctionStatusContext extends RefactoringStatusContext {
		@Override
		public Object getCorrespondingElement() {
			return Function.this;
		}
	}

	/**
	 * Parameters that may be passed to a tf.fuction decorator. Parameter descriptions found at:
	 * https://tensorflow.org/versions/r2.9/api_docs/python/tf/function Note: We are also parsing the deprecated parameters specified in the
	 * documentation. Users can still use these deprecated parameters. Therefore we need to be able to account for them. Please refer to
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki/tf.function-parameter's-version-information to see more
	 * information about the tf.function parameters according to the versions.
	 */
	public class HybridizationParameters {

		private static final String EXPERIMENTAL_FOLLOW_TYPE_HINTS = "experimental_follow_type_hints";

		private static final String EXPERIMENTAL_AUTOGRAPH_OPTIONS = "experimental_autograph_options";

		private static final String EXPERIMENTAL_IMPLEMENTS = "experimental_implements";

		private static final String REDUCE_RETRACING = "reduce_retracing";

		private static final String EXPERIMENTAL_RELAX_SHAPES = "experimental_relax_shapes";

		private static final String EXPERIMENTAL_COMPILE = "experimental_compile";

		private static final String JIT_COMPILE = "jit_compile";

		private static final String AUTOGRAPH = "autograph";

		private static final String INPUT_SIGNATURE = "input_signature";

		private static final String FUNC = "func";

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter autograph.
		 */
		private boolean autoGraphParamExists;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_follow_type_hints.
		 */
		private boolean experimentaFollowTypeHintsParamExists;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_autograph_options.
		 */
		private boolean experimentalAutographOptionsParamExists;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_implements.
		 */
		private boolean experimentalImplementsParamExists;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter func.
		 */
		private boolean funcParamExists;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter input_signature.
		 */
		private boolean inputSignatureParamExists;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter jit_compile.
		 */
		private boolean jitCompileParamExists;

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter reduce_retracing.
		 */
		private boolean reduceRetracingParamExists;

		private void computeParameterExistance(IProgressMonitor monitor) throws BadLocationException {
			FunctionDefinition functionDefinition = Function.this.getFunctionDefinition();
			decoratorsType[] decoratorArray = functionDefinition.getFunctionDef().decs;

			// Will contain the last tf.function decorator
			decoratorsType tfFunctionDecorator = null;

			// Iterate through the decorators of the function
			for (decoratorsType decorator : decoratorArray) {
				IDocument document = Function.this.getContainingDocument();

				// Save the hybrid decorator
				try {
					PySelection selection = Util.getSelection(decorator, document);
					if (Function.isHybrid(decorator, Function.this.getContainingModuleName(), Function.this.getContainingFile(), selection,
							Function.this.getNature(), monitor)) // TODO: Cache this from a previous call (#118).
						tfFunctionDecorator = decorator;
				} catch (AmbiguousDeclaringModuleException | NoDeclaringModuleException | NoTextSelectionException e) {
					throw new IllegalStateException("Can't determine whether decorator: " + decorator + " is hybrid.", e);
				}
			} // We expect to have the last tf.function decorator in tfFunctionDecorator

			if (tfFunctionDecorator != null) {
				// tfFunctionDecorator must be an instance of Call, because that's the only way we have parameters.
				if (tfFunctionDecorator.func instanceof Call) {
					Call callFunction = (Call) tfFunctionDecorator.func;
					// We only care about the actual keywords for now.
					// TODO: Parse positional arguments (#108).
					keywordType[] keywords = callFunction.keywords;
					for (keywordType keyword : keywords) {
						if (keyword.arg instanceof NameTok) {
							NameTok name = (NameTok) keyword.arg;
							if (name.id.equals(FUNC))
								// Found parameter func
								this.funcParamExists = true;
							else if (name.id.equals(INPUT_SIGNATURE))
								// Found parameter input_signature
								this.inputSignatureParamExists = true;
							else if (name.id.equals(AUTOGRAPH))
								// Found parameter autograph
								this.autoGraphParamExists = true;
							// The version of the API we are using allows
							// parameter names jit_compile and
							// deprecated name experimental_compile
							else if (name.id.equals(JIT_COMPILE) || name.id.equals(EXPERIMENTAL_COMPILE))
								// Found parameter jit_compile/experimental_compile
								this.jitCompileParamExists = true;
							// The version of the API we are using allows
							// parameter names reduce_retracing
							// and deprecated name experimental_relax_shapes
							else if (name.id.equals(REDUCE_RETRACING) || name.id.equals(EXPERIMENTAL_RELAX_SHAPES))
								// Found parameter reduce_retracing
								// or experimental_relax_shapes
								this.reduceRetracingParamExists = true;
							else if (name.id.equals(EXPERIMENTAL_IMPLEMENTS))
								// Found parameter experimental_implements
								this.experimentalImplementsParamExists = true;
							else if (name.id.equals(EXPERIMENTAL_AUTOGRAPH_OPTIONS))
								// Found parameter experimental_autograph_options
								this.experimentalAutographOptionsParamExists = true;
							else if (name.id.equals(EXPERIMENTAL_FOLLOW_TYPE_HINTS))
								// Found parameter experimental_follow_type_hints
								this.experimentaFollowTypeHintsParamExists = true;
						}
					}
				} // else, tf.function is used without parameters.
			} else
				throw new IllegalStateException("No decorator exists. Can't compute decorator parameter existance.");
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter autograph.
		 *
		 * @return True iff this {@link decoratorType} has parameter autograph.
		 */
		public boolean getAutoGraphParamExists() {
			return this.autoGraphParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_autograph_options.
		 *
		 * @return True iff this {@link decoratorType} has parameter experimental_autograph_options.
		 */
		public boolean getExperimentalAutographOptParamExists() {
			return this.experimentalAutographOptionsParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_implements.
		 *
		 * @return True iff this {@link decoratorType} has parameter experimental_implements.
		 */
		public boolean getExperimentalImplementsParamExists() {
			return this.experimentalImplementsParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_follow_type_hints.
		 *
		 * @return True iff this {@link decoratorType} has parameter experimental_follow_type_hints.
		 */
		public boolean getExperimentalFollowTypeHintsParamExists() {
			return this.experimentaFollowTypeHintsParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter has parameter func.
		 *
		 * @return True iff this {@link decoratorType} has parameter func.
		 */
		public boolean getFuncParamExists() {
			return this.funcParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter input_signature.
		 *
		 * @return True iff this {@link decoratorType} has parameter input_signature.
		 */
		public boolean getInputSignatureParamExists() {
			return this.inputSignatureParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter jit_compile.
		 *
		 * @return True iff this {@link decoratorType} has parameter jit_compile.
		 */
		public boolean getJitCompileParamExists() {
			return this.jitCompileParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter reduce_retracing.
		 *
		 * @return True iff this {@link Function} has parameter reduce_retracing.
		 */
		public boolean getReduceRetracingParamExists() {
			return this.reduceRetracingParamExists;
		}
	}

	private static final String TF_FUNCTION_FQN = "tensorflow.python.eager.def_function.function";

	private static final String TF_TENSOR_FQN = "tensorflow.python.framework.ops.Tensor";

	private static final String SELF_PARAMETER_NAME = "self";

	private static final ILog LOG = getLog(Function.class);

	/**
	 * True iff verbose output is desired.
	 */
	private static final boolean VERBOSE = false;

	/**
	 * This {@link Function}'s associated hybridization parameters.
	 */
	private Function.HybridizationParameters hybridizationParameters;

	/**
	 * The {@link FunctionDefinition} representing this {@link Function}.
	 */
	private FunctionDefinition functionDefinition;

	/**
	 * True iff this {@link Function} is decorated with tf.function. FIXME: This should be "hybrid?"
	 */
	private Boolean isHybrid;

	/**
	 * True iff this {@link Function} has at least one parameter that is a tf.Tensor (https://bit.ly/3vYG7iP).
	 */
	private Boolean likelyHasTensorParameter;

	/**
	 * True iff this {@link Function} has at least one parameter that is likely a primitive.
	 */
	private Boolean likelyHasPrimitiveParameters;

	/**
	 * True iff this {@link Function} has Python side-effects.
	 */
	private Boolean hasPythonSideEffects;

	/**
	 * TODO: Populate.
	 */
	private Set<Transformation> transformations = new HashSet<>();

	// private InstanceKey instanceKey;

	/**
	 * TODO: Populate.
	 */
	private PreconditionSuccess passingPrecondition;

	/**
	 * The refactoring that this {@link Function} qualifies for. There should be only one as the refactorings are mutually exclusive. TODO:
	 * Populate.
	 */
	private Refactoring refactoring;

	private RefactoringStatus status = new RefactoringStatus();

	private Boolean isRecursive;

	private boolean ignoreBooleans;

	private boolean alwaysFollowTypeHints;

	private static Map<MethodReference, Map<InstanceKey, Map<CallGraph, Boolean>>> creationsCache = Maps.newHashMap();

	public Function(FunctionDefinition fd, boolean ignoreBooleans, boolean alwaysFollowTypeHints) {
		this.functionDefinition = fd;
		this.ignoreBooleans = ignoreBooleans;
		this.alwaysFollowTypeHints = alwaysFollowTypeHints;
	}

	public void computeRecursion(CallGraph callGraph) throws CantComputeRecursionException {
		// Get the nodes representing this function.
		Set<CGNode> nodes = this.getNodes(callGraph);

		if (nodes.isEmpty())
			throw new CantComputeRecursionException("Can't compute if " + this + " is recusive without a call graph node.");

		CGNode cgNode = nodes.iterator().next();

		if (Util.calls(cgNode, this.getMethodReference(), callGraph)) {
			// it's recursive.
			LOG.info(this + " is recursive.");
			this.setIsRecursive(true);
		} else {
			// not recursive.
			LOG.info(this + " is not recursive.");
			this.setIsRecursive(false);
		}
	}

	public void inferPrimitiveParameters(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis, IProgressMonitor monitor)
			throws CantInferPrimitiveParametersException {
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Infering primitive parameters...", IProgressMonitor.UNKNOWN);
		Set<CGNode> nodes = this.getNodes(callGraph);

		if (nodes.isEmpty())
			throw new CantInferPrimitiveParametersException("Can't primitive paramaeters of " + this + " without a call graph node.");

		subMonitor.beginTask("Examining nodes...", nodes.size());

		for (CGNode nodeRepresentingThisFunction : nodes) {
			IR ir = nodeRepresentingThisFunction.getIR();

			subMonitor.beginTask("Examining explicit parameters (not self)...", ir.getNumberOfParameters() - 1);

			// Start at 1 or 2, depending on whether this is a method or not, because the first value is the function being invoked.
			// FIXME: Also consider kwargs and default args.
			// TODO: I wonder if ir.getParameterValueNumbers() returns kwargs as well.
			for (int paramInx = (this.isMethod() ? 2 : 1); paramInx < ir.getNumberOfParameters(); paramInx++) {
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
					this.likelyHasPrimitiveParameters = TRUE;
					subMonitor.done();
					return;
				}

				subMonitor.worked(1);
			}

			subMonitor.worked(1);
		}

		LOG.info(this + " likely does not have a primitive parameter.");
		this.likelyHasPrimitiveParameters = FALSE;
		subMonitor.done();
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
				if (!(foundBooleanValue && ignoreBooleans)) {
					// We have found a primitive.
					subMonitor.done();
					return true;
				}
			}
		} else if (instanceKey instanceof AllocationSiteInNode) {
			AllocationSiteInNode asin = (AllocationSiteInNode) instanceKey;
			IClass concreteType = asin.getConcreteType();
			Collection<IField> allInstanceFields = concreteType.getAllInstanceFields();

			subMonitor.beginTask("Examining fields...", allInstanceFields.size());

			for (IField field : allInstanceFields) {
				InstanceFieldPointerKey instanceFieldKey = (InstanceFieldPointerKey) pointerAnalysis.getHeapModel()
						.getPointerKeyForInstanceField(asin, field);
				OrdinalSet<InstanceKey> instanceFieldPointsToSet = pointerAnalysis.getPointsToSet(instanceFieldKey);

				subMonitor.beginTask("Examining instance field instances...", instanceFieldPointsToSet.size());

				for (InstanceKey key : instanceFieldPointsToSet)
					if (!seen.contains(key) && containsPrimitive(key, ignoreBooleans, pointerAnalysis, seen, subMonitor.split(1))) {
						subMonitor.done();
						return true;
					}

				subMonitor.worked(1);
			}
		}

		subMonitor.done();
		return false;
	}

	/**
	 * Infer Python side-effects potentially produced by executing this {@link Function}.
	 *
	 * @param callGraph The system {@link CallGraph}.
	 * @param pointerAnalysis The system {@link PointerAnalysis}.
	 * @throws UndeterminablePythonSideEffectsException If this {@link Function}'s representation isn't found in the given
	 *         {@link CallGraph}.
	 */
	public void inferPythonSideEffects(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis)
			throws UndeterminablePythonSideEffectsException {
		ModRef<InstanceKey> modRef = new PythonModRefWithBuiltinFunctions();
		Map<CGNode, OrdinalSet<PointerKey>> mod = modRef.computeMod(callGraph, pointerAnalysis);

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

	private Set<PointerKey> filterSideEffects(Iterable<PointerKey> modSet, CallGraph callGraph,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		Set<PointerKey> ret = new HashSet<>();

		for (PointerKey pointerKey : modSet) {
			if (pointerKey instanceof InstanceFieldPointerKey) {
				InstanceFieldPointerKey fieldPointerKey = (InstanceFieldPointerKey) pointerKey;
				InstanceKey instanceKey = fieldPointerKey.getInstanceKey();

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

				if (globalPointerKey.equals(GLOBAL_OUTPUT_STREAM_POINTER_KEY))
					ret.add(globalPointerKey);
				else
					throw new IllegalArgumentException("Not expecting global pointer key: " + globalPointerKey + ".");
			} else
				throw new IllegalArgumentException("Not expecting pointer key: " + pointerKey + " of type: " + pointerKey.getClass() + ".");
		}

		return ret;
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

		Boolean previous = cache3.put(callGraph, result);
		assert previous == null : "Should be a new key.";

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

	private static boolean allCreationsWithin(MethodReference methodReference, InstanceKey instanceKey, CallGraph callGraph) {
		int numCreations = 0;

		// for each creation site of the given instance.
		for (Iterator<Pair<CGNode, NewSiteReference>> it = instanceKey.getCreationSites(callGraph); it.hasNext();) {
			Pair<CGNode, NewSiteReference> creationSite = it.next();
			CGNode creationNode = creationSite.fst;
			NewSiteReference newSiteReference = creationSite.snd;

			// is this instance being created outside this function?
			if (!(creationNode.getMethod().getReference().equals(methodReference)
					|| newSiteReference.getDeclaredType().equals(methodReference.getDeclaringClass())))
				return false;

			++numCreations;
		}

		if (numCreations == 0) // if there are no creations.
			// then, they can't be within this method.
			return false;

		return true;
	}

	/**
	 * Get the {@link CallGraph} nodes corresponding to this {@link Function}.
	 *
	 * @param callGraph The {@link CallGraph} to search.
	 * @return The nodes in the {@link CallGraph} corresponding to this {@link Function}.
	 * @apiNote There can be multiple nodes for a single {@link Function} under the current representation.
	 */
	private Set<CGNode> getNodes(CallGraph callGraph) {
		return getNodes(this.getMethodReference(), callGraph);
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

			if (VERBOSE) {
				LOG.info("Method reference is: " + methodReference + ".");
				LOG.info("Call graph nodes:\n" + callGraph.stream().map(Objects::toString).collect(Collectors.joining("\n")));
			}
		}

		LOG.info("Found " + nodes.size() + " node(s) corresponding to: " + methodReference + ".");

		if (VERBOSE)
			LOG.info("Nodes:\n" + nodes.stream().map(Objects::toString).collect(Collectors.joining("\n")));

		return nodes;
	}

	public MethodReference getMethodReference() {
		TypeReference typeReference = getDeclaringClass();
		return MethodReference.findOrCreate(typeReference, AstMethodReference.fnSelector);
	}

	public TypeReference getDeclaringClass() {
		File containingFile = this.getContainingFile();
		String filename = containingFile.getName();
		String modifiedIdentifier = this.getIdentifier().replace('.', '/');
		String typeName = "Lscript " + filename + "/" + modifiedIdentifier;

		return TypeReference.findOrCreate(PythonTypes.pythonLoader, typeName);
	}

	/**
	 * Infer which parameters are likely tensor parameters.
	 */
	public void inferTensorTensorParameters(TensorTypeAnalysis analysis, CallGraph callGraph, IProgressMonitor monitor) throws Exception {
		monitor.beginTask("Analyzing whether function has a tensor parameter.", IProgressMonitor.UNKNOWN);

		// TODO: Use cast/assert statements?
		FunctionDef functionDef = this.getFunctionDefinition().getFunctionDef();
		argumentsType params = functionDef.args;

		if (params != null) {
			exprType[] actualParams = params.args; // FIXME: Looks like we are only considering position parameters here.

			if (actualParams != null) {
				for (int paramInx = 0; paramInx < actualParams.length; paramInx++) {
					exprType paramExpr = actualParams[paramInx];
					String paramName = NodeUtils.getRepresentationString(paramExpr);

					// don't consider `self` as a tensor.
					if (paramInx == 0 && paramName.equals(SELF_PARAMETER_NAME))
						continue; // next parameter.

					// check a special case where we consider type hints.
					boolean followTypeHints = this.getAlwaysFollowTypeHints() || this.getHybridizationParameters() != null
							// TODO: Actually get the value here (#111).
							&& this.getHybridizationParameters().getExperimentalFollowTypeHintsParamExists();

					// if we are considering type hints.
					if (followTypeHints) {
						LOG.info("Following type hints for: " + this + " and parameter: " + paramName + ".");

						// try to get its type from the AST.
						TypeInfo argTypeInfo = NodeUtils.getTypeForParameterFromAST(paramName, functionDef);

						if (argTypeInfo != null) {
							LOG.info("Found type for parameter " + paramName + " in " + this + ": " + argTypeInfo.getActTok() + ".");

							exprType node = argTypeInfo.getNode();
							Set<Attribute> allAttributes = getAllAttributes(node);

							if (attributesHaveTensorTypeHints(allAttributes, monitor.slice(IProgressMonitor.UNKNOWN))) {
								this.likelyHasTensorParameter = Boolean.TRUE;
								LOG.info(this + " likely has a tensor parameter: " + paramName + " due to a type hint.");
								monitor.worked(1);
								this.addInfo(TYPE_INFERENCING, "Used a type hint to infer tensor type for parameter: " + paramName
										+ " in function: " + this + ".");
								continue; // next parameter.
							}
						}
					}

					// Is this function in the call graph? FIXME: This is checked for **each** parameter.
					Set<CGNode> nodes = this.getNodes(callGraph);

					if (nodes.isEmpty())
						// if there are no nodes representing this function, then it most likely isn't called.
						throw new CantInferTensorParametersException(
								"Can't infer tensor parameters for " + this + " without a call graph node.");

					// Check the tensor type analysis. Check that the methods are the same, the parameters, and so on. If we match the
					// pointer key, then we know it's a tensor if the TensorType is not null.
					if (tensorAnalysisIncludesParameter(analysis, paramExpr, paramName, monitor.slice(IProgressMonitor.UNKNOWN))) {
						this.likelyHasTensorParameter = Boolean.TRUE;
						LOG.info(this + " likely has a tensor parameter: " + paramName + " due to tensor analysis.");
						monitor.worked(1);
						continue; // next parameter.
					}

					// Check for containers of tensors.
					if (tensorAnalysisIncludesParameterContainer(analysis, paramInx, callGraph, monitor.slice(IProgressMonitor.UNKNOWN))) {
						this.likelyHasTensorParameter = Boolean.TRUE;
						LOG.info(this + " likely has a tensor-like parameter: " + paramName + " due to tensor analysis.");
						monitor.worked(1);
						continue; // next parameter.
					}

					monitor.worked(1);
				}
			}
		}

		if (this.likelyHasTensorParameter == null) {
			this.likelyHasTensorParameter = Boolean.FALSE;
			LOG.info(this + " does not likely have a tensor parameter.");
		}

		monitor.done();
	}

	private boolean tensorAnalysisIncludesParameterContainer(TensorTypeAnalysis analysis, int paramInx, CallGraph callGraph,
			IProgressMonitor monitor) {
		Set<CGNode> nodes = this.getNodes(callGraph);
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Checking tensor analysis for containers of tensors sent as arguments.",
				nodes.size());

		for (CGNode nodeRepresentingThisFunction : nodes) {
			// Get the callers of this cgNode.
			for (Iterator<CGNode> predNodes = callGraph.getPredNodes(nodeRepresentingThisFunction); predNodes.hasNext();) {
				CGNode callerOfThisFunction = predNodes.next();
				for (Iterator<CallSiteReference> sites = callGraph.getPossibleSites(callerOfThisFunction,
						nodeRepresentingThisFunction); sites.hasNext();) {
					CallSiteReference callSiteReference = sites.next();
					SSAInstruction instruction = callerOfThisFunction.getIR().getInstructions()[callSiteReference.getProgramCounter()];

					if (instruction instanceof PythonInvokeInstruction) {
						PythonInvokeInstruction invokeInstruction = (PythonInvokeInstruction) instruction;
						// FIXME: Also consider kwargs and default args.
						int useNum = paramInx + 1; // The first use is the function being invoked.

						if (useNum < invokeInstruction.getNumberOfUses()) {
							int paramUse = invokeInstruction.getUse(useNum);
							DefUse du = callerOfThisFunction.getDU();

							Set<NewSiteReference> allNewSiteReferences = getAllNewSiteReferences(paramUse, du);

							for (Pair<PointerKey, TensorVariable> pair : analysis) {
								PointerKey pointerKey = pair.fst;

								if (pointerKey instanceof InstanceFieldPointerKey) {
									InstanceFieldPointerKey ifpk = (InstanceFieldPointerKey) pointerKey;
									InstanceKey instanceKey = ifpk.getInstanceKey();

									if (instanceKey instanceof AllocationSiteInNode) {
										AllocationSiteInNode asin = (AllocationSiteInNode) instanceKey;

										if (asin.getNode().equals(callerOfThisFunction) && allNewSiteReferences.contains(asin.getSite())
												&& Util.isContainerType(asin.getConcreteType().getReference())) {
											// We have a match.
											// check the existence of the tensor variable.
											assert pair.snd != null : "Tensor variable should be non-null if there is a PK.";
											subMonitor.done();
											return true;
										}
									}
								}
							}
						}
					}
				}
			}
			subMonitor.worked(1);
		}
		return false;
	}

	private boolean tensorAnalysisIncludesParameter(TensorTypeAnalysis analysis, exprType paramExpr, String paramName,
			IProgressMonitor monitor) {
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Checking tensor analysis for tensor parameters.", IProgressMonitor.UNKNOWN);

		for (Pair<PointerKey, TensorVariable> pair : analysis) {
			PointerKey pointerKey = pair.fst;

			if (pointerKey instanceof LocalPointerKey) {
				LocalPointerKey localPointerKey = (LocalPointerKey) pointerKey;

				if (localPointerKey.isParameter()) {
					// Does the pointer key match the parameter?
					if (matches(paramExpr, paramName, localPointerKey)) {
						LOG.info(paramExpr + " matches: " + localPointerKey + ".");

						// check the existence of the tensor variable.
						TensorVariable tensorVariable = pair.snd;

						if (tensorVariable != null) {
							subMonitor.done();
							return true;
						}

						throw new IllegalStateException("Tensor variable was null eventhough the PointerKey is present.");
					}
					LOG.info(paramExpr + " does not match: " + localPointerKey + ".");
				} else
					LOG.info(localPointerKey + " is not a parameter.");
			} else
				LOG.info("Encountered non-local pointer key in tensor analysis: " + pointerKey + ".");

			subMonitor.worked(1);
		}

		subMonitor.done();
		return false;
	}

	private boolean attributesHaveTensorTypeHints(Set<Attribute> attributes, IProgressMonitor monitor) throws BadLocationException {
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Examining type hints.", attributes.size() * 2);

		for (Attribute typeHintExpr : attributes) {
			// Look up the definition.
			IDocument document = this.getContainingDocument();

			String fqn;
			PySelection selection = null;
			try {
				selection = Util.getSelection(typeHintExpr.attr, document);
				fqn = Util.getFullyQualifiedName(typeHintExpr, this.getContainingModuleName(), this.getContainingFile(), selection,
						this.getNature(), subMonitor.split(1));
			} catch (AmbiguousDeclaringModuleException | NoDeclaringModuleException | NoTextSelectionException e) {
				LOG.warn(String.format(
						"Can't determine FQN of type hint expression: %s in selection: %s, module: %s, file: %s, and project: %s.",
						typeHintExpr, selection == null ? "null" : selection.getSelectedText(), getContainingModuleName(),
						this.getContainingFile().getName(), this.getProject()), e);

				subMonitor.worked(1);
				continue; // next attribute.
			}

			LOG.info("Found FQN: " + fqn + ".");

			if (fqn.equals(TF_TENSOR_FQN)) { // TODO: Also check for subtypes.
				subMonitor.done();
				return true;
			}

			subMonitor.worked(1);
		}

		subMonitor.done();
		return false;
	}

	private static Set<Attribute> getAllAttributes(exprType node) throws Exception {
		Set<Attribute> ret = Sets.newHashSet();

		if (node instanceof Attribute)
			ret.add((Attribute) node);

		node.traverse(new VisitorBase() {

			@Override
			public Object visitAttribute(Attribute node) throws Exception {
				ret.add(node);
				return super.visitAttribute(node);
			}

			@Override
			protected Object unhandled_node(SimpleNode node) throws Exception {
				return null;
			}

			@Override
			public void traverse(SimpleNode node) throws Exception {
				node.traverse(this);
			}
		});

		return ret;
	}

	private static Set<NewSiteReference> getAllNewSiteReferences(int use, DefUse du) {
		return getAllNewSiteReferences(use, du, new HashSet<>());
	}

	private static Set<NewSiteReference> getAllNewSiteReferences(int use, DefUse du, Set<PythonPropertyWrite> seen) {
		Set<NewSiteReference> ret = new HashSet<>();
		SSAInstruction def = du.getDef(use);

		if (def != null && def instanceof SSANewInstruction) {
			SSANewInstruction newInstruction = (SSANewInstruction) def;
			NewSiteReference newSite = newInstruction.getNewSite();
			ret.add(newSite);

			for (Iterator<SSAInstruction> uses = du.getUses(def.getDef()); uses.hasNext();) {
				SSAInstruction useInstruction = uses.next();

				if (useInstruction instanceof PythonPropertyWrite) {
					PythonPropertyWrite write = (PythonPropertyWrite) useInstruction;

					if (!seen.contains(write)) {
						seen.add(write);
						int value = write.getValue();
						ret.addAll(getAllNewSiteReferences(value, du, seen));
					}
				}
			}
		}
		return ret;
	}

	/**
	 * Returns true iff lhsParamExpr corresponds to rhsPointerKey.
	 *
	 * @param lhsParamExpr The "left hand side" expression to compare. Should represent a function parameter.
	 * @param lhsParamName The name of the parameter represented by lhsParamExpr.
	 * @param rhsPointerKey The rhsPointerKey representing the parameter.
	 * @return True iff lhsParamExpr corresponds to rhsPointerKey.
	 */
	private boolean matches(exprType lhsParamExpr, String lhsParamName, LocalPointerKey rhsPointerKey) {
		File containingFile = this.getContainingFile();
		CGNode node = rhsPointerKey.getNode();
		IMethod nodeMethod = node.getMethod();

		if (nodeMethod instanceof AstMethod) {
			AstMethod astMethod = (AstMethod) nodeMethod;
			String sourceFileName = astMethod.getDeclaringClass().getSourceFileName();

			// are they in the same file?
			if (containingFile.getAbsolutePath().equals(sourceFileName)) {
				// that also means that the module is the same according to https://bit.ly/3NcOvnz.

				// we know that rhsPointerKey is a parameter.
				assert rhsPointerKey.isParameter();

				// since we know that they are in the same file, it should suffice to know whether the source positions match.
				int lhsBeginColumn = lhsParamExpr.beginColumn;
				int lhsBeginLine = lhsParamExpr.beginLine;

				int paramIndex = rhsPointerKey.getValueNumber() - 1;
				Position parameterPosition = astMethod.getParameterPosition(paramIndex);
				LOG.info(rhsPointerKey + " position is: " + parameterPosition + ".");

				int rhsBeginColumn = parameterPosition.getFirstCol() + 1; // workaround https://github.com/jython/jython3/issues/48.
				int rhsBeginLine = parameterPosition.getFirstLine();

				// It should suffice to that the parameters have the same beginning column and the same beginning line. In other words, we
				// are not checking the parameters' expression length because Ariadne includes the type hint in the length while PyDev does
				// not.
				return lhsBeginColumn == rhsBeginColumn && lhsBeginLine == rhsBeginLine;
			}

			LOG.info(containingFile.getName() + " does not match: " + sourceFileName + ".");
		} else
			LOG.warn("Encountered non-AST method: " + nodeMethod + ".");

		return false;
	}

	/**
	 * Discovers if this {@link Function} is hybrid. If so, populated this {@link Function}'s {@link HybridizationParameters}.
	 */
	public void computeHybridization(IProgressMonitor monitor) throws BadLocationException {
		// TODO: Consider mechanisms other than decorators (e.g., higher order functions; #3).
		monitor.beginTask("Computing hybridization ...", IProgressMonitor.UNKNOWN);

		FunctionDefinition functionDefinition = this.getFunctionDefinition();
		decoratorsType[] decoratorArray = functionDefinition.getFunctionDef().decs;

		if (decoratorArray != null) {
			String containingModuleName = this.getContainingModuleName();
			File containingFile = this.getContainingFile();
			String containingFileName = containingFile.getName();
			IPythonNature nature = this.getNature();
			IProject project = this.getProject();

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
				} catch (AmbiguousDeclaringModuleException | BadLocationException | NoDeclaringModuleException | NoTextSelectionException
						| RuntimeException e) {
					String selectedText = null;
					try {
						selectedText = selection == null ? "(can't compute)" : selection.getSelectedText();
					} catch (BadLocationException e1) {
						// NOTE: No need to process; only for an error message.
						LOG.info("Can't get selected text.", e1);
					}

					if (Util.isGenerated(decorator)) {
						// Since tf.function isn't generated, skip generated decorators.
						LOG.info(String.format(
								"Encountered potentially generated decorator: %s in selection: %s, module: %s, file: %s, and project; %s.",
								decoratorFunctionRepresentation, selectedText, containingModuleName, containingFileName, project));
					} else if (Util.isBuiltIn(decorator)) {
						// Since tf.function isn't built-in, skip built-in decorators.
						LOG.info(String.format(
								"Encountered potentially built-in decorator: %s in selection: %s, module: %s, file: %s, and project; %s.",
								decoratorFunctionRepresentation, selectedText, containingModuleName, containingFileName, project));
					} else {
						LOG.warn(String.format(
								"Can't determine if decorator: %s in selection: %s, module: %s, file: %s, and project; %s is hybrid.",
								decoratorFunctionRepresentation, selectedText, containingModuleName, containingFileName,
								nature.getProject()), e);
					}
				}

				if (hybrid) {
					this.setIsHybrid(TRUE);
					LOG.info(this + " is hybrid.");

					// Compute the hybridization parameters since we know now that this function is hybrid.
					LOG.info("Computing hybridization parameters.");
					this.hybridizationParameters = new HybridizationParameters();
					this.hybridizationParameters.computeParameterExistance(monitor);

					monitor.done();
					return;
				}
				monitor.worked(1);
			}
		}

		this.setIsHybrid(FALSE);
		LOG.info(this + " is not hybrid.");
		monitor.done();
	}

	/**
	 * True iff the given decorator is a hybridization decorator.
	 *
	 * @param decorator The {@link decoratorsType} in question.
	 * @param containingModName The name of the module where the decorator is used.
	 * @param containingFile The {@link File} where the containingModName is defined.
	 * @param containingSelection The {@link PySelection} containing the decorator.
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

	public void addFailure(PreconditionFailure failure, String message) {
		// If is side-effects is filled, we can't set a precondition failure that we can't determine them.
		assert this.getHasPythonSideEffects() == null
				|| failure != PreconditionFailure.UNDETERMINABLE_SIDE_EFFECTS : "Can't both have side-effects filled and have tem undterminable.";

		this.addStatus(RefactoringStatus.ERROR, message, failure.getCode());
	}

	public void addWarning(String message) {
		addStatus(RefactoringStatus.WARNING, message, RefactoringStatusEntry.NO_CODE);
	}

	public void addInfo(String message) {
		this.addInfo(message, RefactoringStatusEntry.NO_CODE);
	}

	public void addInfo(Information information, String message) {
		this.addInfo(message, information.getCode());
	}

	private void addInfo(String message, int code) {
		addStatus(RefactoringStatus.INFO, message, code);
	}

	private void addStatus(int status, String message, int code) {
		RefactoringStatusContext context = new FunctionStatusContext();
		this.getStatus().addEntry(status, message, context, PLUGIN_ID, code, this);
	}

	/**
	 * Check refactoring preconditions.
	 */
	public void check() {
		if (!this.getIsHybrid()) { // Eager. Table 1.
			this.setRefactoring(CONVERT_EAGER_FUNCTION_TO_HYBRID);

			if (this.getLikelyHasTensorParameter() != null && this.getLikelyHasTensorParameter()) {
				this.addInfo("This eager function likely has a tensor parameter.");
				if (this.getLikelyHasPrimitiveParameters() != null && !this.getLikelyHasPrimitiveParameters()) {
					this.addInfo("This eager function likely does not have a primitive parameter.");
					if (this.getHasPythonSideEffects() != null && !this.getHasPythonSideEffects()) {
						this.addInfo("This eager function does not have Python side-effects.");
						if (this.getIsRecursive() != null && !this.getIsRecursive()) {
							this.addInfo("This eager function is not recursive.");
							this.addTransformation(Transformation.CONVERT_TO_HYBRID);
							this.setPassingPrecondition(P1);
						} else if (this.getIsRecursive() != null) // it's recursive.
							this.addFailure(PreconditionFailure.IS_RECURSIVE, "Can't hybridize a recursive function.");
					} else if (this.getHasPythonSideEffects() != null) { // it has side-effects.
						this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
								"Can't hybridize a function with Python side-effects.");

						if (this.getIsRecursive() != null && this.getIsRecursive())
							this.addFailure(PreconditionFailure.IS_RECURSIVE, "Can't hybridize a recursive function.");
					}
				} else if (this.getLikelyHasPrimitiveParameters() != null) { // it has primitive parameters.
					this.addFailure(HAS_PRIMITIVE_PARAMETERS, "Hybridizing a function with primitive parameters may induce retracing.");

					if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects())
						this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
								"Can't hybridize a function with Python side-effects.");

					if (this.getIsRecursive() != null && this.getIsRecursive())
						this.addFailure(PreconditionFailure.IS_RECURSIVE, "Can't hybridize a recursive function.");
				}
			} else if (this.getLikelyHasTensorParameter() != null) { // no tensor parameters.
				this.addFailure(PreconditionFailure.HAS_NO_TENSOR_PARAMETERS,
						"This function has no tensor parameters and may not benefit from hybridization.");

				if (this.getLikelyHasPrimitiveParameters() != null && this.getLikelyHasPrimitiveParameters())
					this.addFailure(HAS_PRIMITIVE_PARAMETERS, "Hybridizing a function with primitive parameters may induce retracing.");

				if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects())
					this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS, "Can't hybridize a function with Python side-effects.");

				if (this.getIsRecursive() != null && this.getIsRecursive())
					this.addFailure(PreconditionFailure.IS_RECURSIVE, "Can't hybridize a recursive function.");
			}
		} else { // Hybrid. Use table 2.
			this.setRefactoring(OPTIMIZE_HYBRID_FUNCTION);

			if (this.getLikelyHasTensorParameter() != null && !this.getLikelyHasTensorParameter()) {
				this.addInfo("This hybrid function does not likely have a tensor parameter.");
				if (this.getHasPythonSideEffects() != null && !this.getHasPythonSideEffects()) {
					this.addInfo("This hybrid function does not have Python side-effects.");
					this.addTransformation(CONVERT_TO_EAGER);
					this.setPassingPrecondition(P2);
				} else if (this.getHasPythonSideEffects() != null) // it has side-effects.
					this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
							"De-hybridizing a function with Python side-effects may alter semantics.");
			} else if (this.getLikelyHasTensorParameter() != null) { // it has a tensor parameter.
				this.addInfo("This hybrid function likely has a tensor parameter.");
				// if it has primitive parameters.
				if (this.getLikelyHasPrimitiveParameters() != null && this.getLikelyHasPrimitiveParameters()) {
					this.addInfo("This hybrid function likely has a primitive parameter.");
					// if it does not have side-effects.
					if (this.getHasPythonSideEffects() != null && !this.getHasPythonSideEffects()) {
						this.addInfo("This hybrid function does not have Python side-effects.");
						this.addTransformation(CONVERT_TO_EAGER);
						this.setPassingPrecondition(P3);
					} else if (this.getHasPythonSideEffects() != null) // it has side-effects.
						this.addFailure(HAS_PYTHON_SIDE_EFFECTS, "De-hybridizing a function with Python side-effects may alter semantics.");
				} else if (this.getLikelyHasPrimitiveParameters() != null) { // no primitive parameters.
					this.addFailure(PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS,
							"Functions with no Python literal arguments may benefit from hybridization.");

					if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects()) {
						this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
								"De-hybridizing a function with Python side-effects may alter semantics.");
					}
				}

				// Here, we have a hybrid function with a tensor parameter.
				if (this.getIsRecursive() != null && this.getIsRecursive()) // if it's recursive.
					// issue a warning.
					this.addWarning("Recursive tf.functions are not supported by TensorFlow.");
			}

			// Warn if the function has side-effects.
			if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects())
				this.addWarning("This hybrid function potentially contains Python side-effects.");
		}
	}

	public boolean willDehybridize() {
		return this.getTransformations().contains(CONVERT_TO_EAGER);
	}

	public IDocument getContainingDocument() {
		return this.getFunctionDefinition().containingDocument;
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
	 * Returns the {@link File} of where this {@link Function} is found.
	 *
	 * @return The {@link File} of where this {@link Function} is found.
	 */
	public File getContainingFile() {
		return this.getFunctionDefinition().containingFile;
	}

	/**
	 * Returns the Python module name of this {@link Function}.
	 *
	 * @return This {@link Function}'s Python module.
	 */
	public String getContainingModuleName() {
		return this.getFunctionDefinition().containingModuleName;
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
	 * This {@link Function}'s {@link FunctionDefinition}.
	 *
	 * @return The {@link FunctionDefinition} representing this {@link Function}.
	 */
	protected FunctionDefinition getFunctionDefinition() {
		return this.functionDefinition;
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
	 * True iff this {@link Function} is hybrid. Note that this only checks the decorator, i.e., whether all invocations of this
	 * {@link Function} are hybridized.
	 *
	 * @return True iff this {@link Function} is hybrid, i.e., whether it is decorated with tf.function.
	 */
	public Boolean getIsHybrid() {
		return this.isHybrid;
	}

	protected void setIsHybrid(Boolean isHybrid) {
		this.isHybrid = isHybrid;
	}

	/**
	 * True iff this {@link Function} likely has a tf.Tensor parameter. Since Python is dynamic, we may not be 100% sure.
	 *
	 * @return True iff this {@link Function} likely has a tf.Tensor parameter.
	 */
	public Boolean getLikelyHasTensorParameter() {
		return this.likelyHasTensorParameter;
	}

	@Override
	public String toString() {
		return this.getIdentifier() + "()";
	}

	@Override
	public int hashCode() {
		return Objects.hash(functionDefinition);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Function other = (Function) obj;
		return Objects.equals(functionDefinition, other.functionDefinition);
	}

	public String getSimpleName() {
		return NodeUtils.getFullRepresentationString(this.getFunctionDefinition().getFunctionDef());
	}

	public argumentsType getParameters() {
		return getFunctionDefinition().getFunctionDef().args;
	}

	public int getNumberOfParameters() {
		return getFunctionDefinition().getFunctionDef().args.args.length;
	}

	public RefactoringStatus getStatus() {
		return this.status;
	}

	public IProject getProject() {
		return this.getFunctionDefinition().getProject();
	}

	public boolean isHybridizationAvailable() {
		return RefactoringAvailabilityTester.isHybridizationAvailable(this.getFunctionDefinition().getFunctionDef());
	}

	public Set<Transformation> getTransformations() {
		return transformations;
	}

	protected void addTransformation(Transformation transformation) {
		this.transformations.add(transformation);
	}

	public PreconditionSuccess getPassingPrecondition() {
		return passingPrecondition;
	}

	protected void setPassingPrecondition(PreconditionSuccess passingPrecondition) {
		this.passingPrecondition = passingPrecondition;
	}

	public Refactoring getRefactoring() {
		return refactoring;
	}

	public void setRefactoring(Refactoring refactoring) {
		this.refactoring = refactoring;
	}

	public Boolean getHasPythonSideEffects() {
		return this.hasPythonSideEffects;
	}

	protected void setHasPythonSideEffects(Boolean hasPythonSideEffects) {
		assert this.hasPythonSideEffects == null : "Can only set side-effects once.";
		assert hasPythonSideEffects == null || this.getStatus().getEntryMatchingCode(PLUGIN_ID,
				PreconditionFailure.UNDETERMINABLE_SIDE_EFFECTS.getCode()) == null : "Can't set side-effects if they are undeterminable.";

		this.hasPythonSideEffects = hasPythonSideEffects;
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

	public static void clearCaches() {
		creationsCache.clear();
	}

	public Boolean getIsRecursive() {
		return this.isRecursive;
	}

	protected void setIsRecursive(Boolean isRecursive) {
		this.isRecursive = isRecursive;
	}

	private Set<RefactoringStatusEntry> getRefactoringStatusEntries(Predicate<? super RefactoringStatusEntry> predicate) {
		return Arrays.stream(this.getStatus().getEntries()).filter(predicate).collect(Collectors.toSet());
	}

	public Set<RefactoringStatusEntry> getWarnings() {
		return this.getRefactoringStatusEntries(RefactoringStatusEntry::isWarning);
	}

	public Set<RefactoringStatusEntry> getErrors() {
		return this.getRefactoringStatusEntries(RefactoringStatusEntry::isError);
	}

	/**
	 * Returns true iff this {@link Function} has at least one parameter that is likely a primitive.
	 *
	 * @return True iff this {@link Function} has at least one parameter that is likely a primitive.
	 */
	public Boolean getLikelyHasPrimitiveParameters() {
		return likelyHasPrimitiveParameters;
	}

	/**
	 * Returns true iff this {@link Function} represents an instance method.
	 *
	 * @return True iff this {@link Function} is an instance method.
	 */
	public boolean isMethod() {
		argumentsType parameters = this.getParameters();

		if (parameters.args.length > 1) {
			final exprType firstParam = parameters.args[0];
			final String firstParamName = NodeUtils.getRepresentationString(firstParam);

			if (firstParamName.equals(SELF_PARAMETER_NAME))
				return true;
		}

		return false;
	}

	/**
	 * True iff booleans shouldn't be considered primitives.
	 *
	 * @return True iff boolean values shouldn't be considered primitives.
	 */
	protected boolean getIgnoreBooleans() {
		return ignoreBooleans;
	}

	/**
	 * Returns true iff we should use type hints regardless of a hybridization parameter.
	 *
	 * @return Whether we should use type hints regardless of what is specified in any hybridization parameters.
	 */
	public boolean getAlwaysFollowTypeHints() {
		return alwaysFollowTypeHints;
	}
}
