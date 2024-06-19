package edu.cuny.hunter.hybridize.core.refactorings;

import static com.google.common.collect.Iterables.concat;
import static edu.cuny.hunter.hybridize.core.utils.Util.getPythonPath;
import static java.lang.Boolean.TRUE;
import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.core.runtime.Status;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.CompositeChange;
import org.eclipse.ltk.core.refactoring.NullChange;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.RefactoringStatusContext;
import org.eclipse.ltk.core.refactoring.TextChange;
import org.eclipse.ltk.core.refactoring.TextFileChange;
import org.eclipse.ltk.core.refactoring.participants.CheckConditionsContext;
import org.eclipse.ltk.core.refactoring.participants.RefactoringParticipant;
import org.eclipse.ltk.core.refactoring.participants.SharableParticipants;
import org.eclipse.text.edits.MalformedTreeException;
import org.eclipse.text.edits.TextEdit;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.preferences.InterpreterGeneralPreferences;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.ipa.callgraph.PytestEntrypointBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.PytesttEntrypoint;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.ide.util.ProgressMonitorDelegate;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.CallGraphBuilderCancelException;
import com.ibm.wala.ipa.callgraph.Entrypoint;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.modref.ModRef;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.intset.OrdinalSet;

import edu.cuny.citytech.refactoring.common.core.RefactoringProcessor;
import edu.cuny.citytech.refactoring.common.core.TimeCollector;
import edu.cuny.hunter.hybridize.core.analysis.AmbiguousDeclaringModuleException;
import edu.cuny.hunter.hybridize.core.analysis.CantComputeRecursionException;
import edu.cuny.hunter.hybridize.core.analysis.CantInferPrimitiveParametersException;
import edu.cuny.hunter.hybridize.core.analysis.CantInferTensorParametersException;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.analysis.NoDeclaringModuleException;
import edu.cuny.hunter.hybridize.core.analysis.NoTextSelectionException;
import edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure;
import edu.cuny.hunter.hybridize.core.analysis.UndeterminablePythonSideEffectsException;
import edu.cuny.hunter.hybridize.core.descriptors.HybridizeFunctionRefactoringDescriptor;
import edu.cuny.hunter.hybridize.core.messages.Messages;
import edu.cuny.hunter.hybridize.core.wala.ml.EclipsePythonProjectTensorAnalysisEngine;
import edu.cuny.hunter.hybridize.core.wala.ml.PythonModRefWithBuiltinFunctions;

@SuppressWarnings("unused")
public class HybridizeFunctionRefactoringProcessor extends RefactoringProcessor {

	private static final String DUMP_CALL_GRAPH_PROPERTY_KEY = "edu.cuny.hunter.hybridize.dumpCallGraph";

	private final class FunctionStatusContext extends RefactoringStatusContext {
		private final Function func;

		private FunctionStatusContext(Function func) {
			this.func = func;
		}

		@Override
		public Object getCorrespondingElement() {
			return func;
		}
	}

	private static final ILog LOG = getLog(HybridizeFunctionRefactoringProcessor.class);

	private static RefactoringStatus checkDecorators(Function func) {
		RefactoringStatus status = new RefactoringStatus();
		LOG.info("Checking decorators for: " + func + ".");
		// TODO: Is the function already decorated with tf.function? NOTE: May move to checkFinalConditions() as this
		// will be dependent on other things.
		return status;
	}

	private static RefactoringStatus checkParameters(Function func) {
		RefactoringStatus status = new RefactoringStatus();
		LOG.info("Checking parameters for: " + func + ".");
		// TODO: Does the function have a tensor parameter (#2)?
		// NOTE: Not sure if we will be checking everything individually here since we'll do the computation in the
		// Function class. Instead, we may just need to check everything in checkFinalConditions(), as it is likely that
		// the checking will depend on several things.
		return status;
	}

	private Set<Function> functions = new LinkedHashSet<>();

	private Map<IProject, PythonSSAPropagationCallGraphBuilder> projectToCallGraphBuilder = new HashMap<>();

	private Map<IProject, Map<CGNode, OrdinalSet<PointerKey>>> projectToMod = new HashMap<>();

	private Map<IProject, CallGraph> projectToCallGraph = new HashMap<>();

	private Map<IProject, TensorTypeAnalysis> projectToTensorTypeAnalysis = new HashMap<>();

	/**
	 * True iff the {@link CallGraph} should be displayed.
	 */
	private boolean dumpCallGraph = Boolean.getBoolean(DUMP_CALL_GRAPH_PROPERTY_KEY);

	private boolean alwaysCheckPythonSideEffects;

	private boolean alwaysCheckRecursion;

	private boolean ignoreBooleansInLiteralCheck = true;

	private boolean processFunctionsInParallel;

	/**
	 * True iff entry points corresponding to tests should be used in the {@link CallGraph} construction.
	 */
	private boolean useTestEntryPoints;

	/**
	 * True iff we should use type hints regardless of any hybridization arguments.
	 */
	private boolean alwaysFollowTypeHints;

	public HybridizeFunctionRefactoringProcessor() {
		// Force the use of typeshed. It's an experimental feature of PyDev.
		InterpreterGeneralPreferences.FORCE_USE_TYPESHED = TRUE;

		// Have WALA dump the call graph if appropriate.
		CAstCallGraphUtil.AVOID_DUMP = !this.dumpCallGraph;
	}

	public HybridizeFunctionRefactoringProcessor(boolean alwaysCheckPythonSideEffects) {
		this();
		this.alwaysCheckPythonSideEffects = alwaysCheckPythonSideEffects;
	}

	public HybridizeFunctionRefactoringProcessor(boolean alwaysCheckPythonSideEffects, boolean processFunctionsInParallel) {
		this(alwaysCheckPythonSideEffects);
		this.processFunctionsInParallel = processFunctionsInParallel;
	}

	public HybridizeFunctionRefactoringProcessor(boolean alwaysCheckPythonSideEffects, boolean processFunctionsInParallel,
			boolean alwaysCheckRecusion) {
		this(alwaysCheckPythonSideEffects, processFunctionsInParallel);
		this.alwaysCheckRecursion = alwaysCheckRecusion;
	}

	public HybridizeFunctionRefactoringProcessor(boolean alwaysCheckPythonSideEffects, boolean processFunctionsInParallel,
			boolean alwaysCheckRecusion, boolean ignoreBooleansInLiteralCheck) {
		this(alwaysCheckPythonSideEffects, processFunctionsInParallel, alwaysCheckRecusion);
		this.ignoreBooleansInLiteralCheck = ignoreBooleansInLiteralCheck;
	}

	public HybridizeFunctionRefactoringProcessor(boolean alwaysCheckPythonSideEffects, boolean processFunctionsInParallel,
			boolean alwaysCheckRecusion, boolean ignoreBooleansInLiteralCheck, boolean useTestEntryPoints) {
		this(alwaysCheckPythonSideEffects, processFunctionsInParallel, alwaysCheckRecusion, ignoreBooleansInLiteralCheck);
		this.useTestEntryPoints = useTestEntryPoints;
	}

	public HybridizeFunctionRefactoringProcessor(boolean alwaysCheckPythonSideEffects, boolean processFunctionsInParallel,
			boolean alwaysCheckRecusion, boolean ignoreBooleansInLiteralCheck, boolean useTestEntryPoints, boolean alwaysFollowTypeHints) {
		this(alwaysCheckPythonSideEffects, processFunctionsInParallel, alwaysCheckRecusion, ignoreBooleansInLiteralCheck,
				useTestEntryPoints);
		this.alwaysFollowTypeHints = alwaysFollowTypeHints;
	}

	public HybridizeFunctionRefactoringProcessor(Set<FunctionDefinition> functionDefinitionSet)
			throws TooManyMatchesException /* FIXME: This exception sounds too low-level. */ {
		this();

		// Convert the FunctionDefs to Functions.
		if (functions != null) {
			Set<Function> functionSet = this.getFunctions();

			for (FunctionDefinition fd : functionDefinitionSet) {
				Function function = new Function(fd, this.ignoreBooleansInLiteralCheck, this.alwaysFollowTypeHints);

				// Add the Function to the Function set.
				functionSet.add(function);
			}
		}
	}

	public HybridizeFunctionRefactoringProcessor(Set<FunctionDefinition> functionDefinitionSet, boolean alwaysCheckPythonSideEffects)
			throws TooManyMatchesException /* FIXME: This exception sounds too low-level. */ {
		this(functionDefinitionSet);
		this.alwaysCheckPythonSideEffects = alwaysCheckPythonSideEffects;
	}

	public HybridizeFunctionRefactoringProcessor(Set<FunctionDefinition> functionDefinitionSet, boolean alwaysCheckPythonSideEffects,
			boolean processFunctionsInParallel) throws TooManyMatchesException /* FIXME: This exception sounds too low-level. */ {
		this(functionDefinitionSet, alwaysCheckPythonSideEffects);
		this.processFunctionsInParallel = processFunctionsInParallel;
	}

	public HybridizeFunctionRefactoringProcessor(Set<FunctionDefinition> functionDefinitionSet, boolean alwaysCheckPythonSideEffects,
			boolean processFunctionsInParallel, boolean alwaysCheckRecursion)
			throws TooManyMatchesException /* FIXME: This exception sounds too low-level. */ {
		this(functionDefinitionSet, alwaysCheckPythonSideEffects, processFunctionsInParallel);
		this.alwaysCheckRecursion = alwaysCheckRecursion;
	}

	public HybridizeFunctionRefactoringProcessor(Set<FunctionDefinition> functionDefinitionSet, boolean alwaysCheckPythonSideEffects,
			boolean processFunctionsInParallel, boolean alwaysCheckRecursion, boolean useTestEntryPoints)
			throws TooManyMatchesException /* FIXME: This exception sounds too low-level. */ {
		this(functionDefinitionSet, alwaysCheckPythonSideEffects, processFunctionsInParallel, alwaysCheckRecursion);
		this.useTestEntryPoints = useTestEntryPoints;
	}

	public HybridizeFunctionRefactoringProcessor(Set<FunctionDefinition> functionDefinitionSet, boolean alwaysCheckPythonSideEffects,
			boolean processFunctionsInParallel, boolean alwaysCheckRecursion, boolean useTestEntryPoints, boolean alwaysFollowTypeHints)
			throws TooManyMatchesException /* FIXME: This exception sounds too low-level. */ {
		this();

		this.alwaysCheckPythonSideEffects = alwaysCheckPythonSideEffects;
		this.alwaysCheckRecursion = alwaysCheckRecursion;
		this.processFunctionsInParallel = processFunctionsInParallel;
		this.useTestEntryPoints = useTestEntryPoints;
		this.alwaysFollowTypeHints = alwaysFollowTypeHints;

		// Convert the FunctionDefs to Functions.
		if (functions != null) {
			Set<Function> functionSet = this.getFunctions();

			for (FunctionDefinition fd : functionDefinitionSet) {
				Function function = new Function(fd, this.ignoreBooleansInLiteralCheck, this.alwaysFollowTypeHints);

				// Add the Function to the Function set.
				functionSet.add(function);
			}
		}

	}

	@Override
	public RefactoringStatus checkFinalConditions(IProgressMonitor pm, CheckConditionsContext context)
			throws CoreException, OperationCanceledException {
		RefactoringStatus status = new RefactoringStatus();
		SubMonitor progress = SubMonitor.convert(pm, Messages.CheckingPreconditions, 100);
		// TODO: Adjust amount of work later.

		status.merge(this.checkFunctions(progress.split(1)));

		return status;
	}

	private RefactoringStatus checkFunctions(IProgressMonitor monitor) throws OperationCanceledException, CoreException {
		RefactoringStatus status = new RefactoringStatus();
		SubMonitor subMonitor = SubMonitor.convert(monitor, Messages.Analyzing, IProgressMonitor.UNKNOWN);
		TimeCollector timeCollector = this.getExcludedTimeCollector();

		Set<Function> allFunctions = this.getFunctions();

		// collect the projects to be analyzed.
		Map<IProject, Set<Function>> projectToFunctions = allFunctions.stream().filter(f -> f.getStatus().isOK())
				.collect(Collectors.groupingBy(Function::getProject, Collectors.toSet()));

		// process each project.
		subMonitor.beginTask("Processing projects ...", projectToFunctions.keySet().size());

		for (IProject project : projectToFunctions.keySet()) {
			// get the PYTHONPATH.
			List<File> pythonPath = getPythonPath(project);
			LOG.info("PYTHONPATH for " + project + " is: " + pythonPath + ".");
			assert pythonPath.stream().allMatch(File::exists) : "PYTHONPATH should exist.";

			// create the analysis engine for the project.
			EclipsePythonProjectTensorAnalysisEngine engine = new EclipsePythonProjectTensorAnalysisEngine(project, pythonPath);

			// build the call graph for the project.
			PythonSSAPropagationCallGraphBuilder builder;
			try {
				builder = computeCallGraphBuilder(engine);
			} catch (IOException e) {
				throw new CoreException(Status.error("Could not compute call graph builder for: " + project.getName(), e));
			}

			subMonitor.subTask("Building call graph.");
			CallGraph callGraph;
			try {
				callGraph = computeCallGraph(project, builder, subMonitor.split(IProgressMonitor.UNKNOWN, SubMonitor.SUPPRESS_NONE));
			} catch (CallGraphBuilderCancelException e) {
				throw new CoreException(Status.error("Could not build call graph for: " + project.getName(), e));
			}

			timeCollector.start();
			if (this.shouldDumpCallGraph()) {
				CAstCallGraphUtil.dumpCG(builder.getCFAContextInterpreter(), builder.getPointerAnalysis(), callGraph);
				// DotUtil.dotify(callGraph, null, PDFTypeHierarchy.DOT_FILE, "callgraph.pdf", "dot");
			}
			timeCollector.stop();

			TensorTypeAnalysis analysis;
			try {
				analysis = computeTensorTypeAnalysis(engine, builder);
			} catch (CancelException e) {
				throw new CoreException(Status.error("Could not analyze tensors for: " + project.getName(), e));
			}

			LOG.info("Tensor analysis: " + analysis.toString());

			subMonitor.checkCanceled();

			Set<Function> projectFunctions = projectToFunctions.get(project);

			// analyze Python functions.
			LOG.info("Analyzing " + projectFunctions.size() + " function" + (allFunctions.size() > 1 ? "s" : "") + ".");
			subMonitor.beginTask(Messages.AnalyzingFunctions, projectFunctions.size());

			// check preconditions.
			LOG.info("Checking " + projectFunctions.size() + " function" + (allFunctions.size() > 1 ? "s" : "") + ".");
			subMonitor.beginTask(Messages.CheckingFunctions, allFunctions.size());

			this.getStream(projectFunctions).forEach(func -> {
				LOG.info("Checking function: " + func + ".");

				// Find out if it's hybrid via the tf.function decorator.
				try {
					func.computeHybridization(subMonitor.split(IProgressMonitor.UNKNOWN));
				} catch (BadLocationException e) {
					throw new RuntimeException("Could not compute hybridization for: " + func + ".", e);
				}

				try {
					func.inferTensorTensorParameters(analysis, callGraph, builder, subMonitor.split(IProgressMonitor.UNKNOWN));
				} catch (CantInferTensorParametersException e) {
					LOG.warn("Unable to compute whether " + func + " has tensor parameters.", e);
					func.addFailure(PreconditionFailure.UNDETERMINABLE_TENSOR_PARAMETER,
							"Can't infer tensor parameters for this function.");
				} catch (Exception e) {
					throw new RuntimeException("Could not infer tensor parameters for: " + func + ".", e);
				}

				try {
					func.inferPrimitiveParameters(callGraph, builder.getPointerAnalysis(), subMonitor.split(IProgressMonitor.UNKNOWN));
				} catch (CantInferPrimitiveParametersException e) {
					LOG.warn("Unable to infer primitive parameters for: " + func + ".", e);
					func.addFailure(PreconditionFailure.UNDETERMINABLE_PRIMITIVE_PARAMETER,
							"Can't infer primitive parameters for this function.");
				} catch (CoreException e) {
					LOG.error("Can't infer primitive parameters.", e);
					throw new RuntimeException("Can't infer primitive parameters.", e);
				}

				// Check Python side-effects.
				try {
					if (this.shouldAlwaysCheckPythonSideEffects() || func.isHybrid()
							|| func.hasTensorParameter() != null && func.hasTensorParameter()) {
						Map<CGNode, OrdinalSet<PointerKey>> mod = this.computeMod(project, callGraph, builder.getPointerAnalysis());
						func.inferPythonSideEffects(mod, callGraph, builder.getPointerAnalysis());
					}
				} catch (UndeterminablePythonSideEffectsException e) {
					LOG.warn("Unable to infer side-effects of: " + func + ".", e);
					func.addFailure(PreconditionFailure.UNDETERMINABLE_SIDE_EFFECTS,
							"Can't infer side-effects, most likely due to a call graph issue caused by a decorator or a missing function call.");
				} catch (CoreException e) {
					LOG.error("Can't determine side-effects.", e);
					throw new RuntimeException("Can't determine side-effects.", e);
				}

				// Check recursion.
				try {
					// NOTE: Whether a hybrid function is recursive is irrelevant; if the function has no tensor parameter, de-hybridizing
					// it does not violate semantics preservation as potential retracing happens regardless. We do, however, issue a
					// refactoring warning when a hybrid function with a tensor parameter is recursive.
					if (this.shouldAlwaysCheckRecursion() || func.hasTensorParameter() != null && func.hasTensorParameter())
						func.computeRecursion(callGraph);
				} catch (CantComputeRecursionException e) {
					LOG.warn("Unable to compute whether " + func + " is recursive.", e);
					func.addFailure(PreconditionFailure.CANT_APPROXIMATE_RECURSION, "Can't compute whether this function is recursive.");
				} catch (CoreException e) {
					LOG.error("Can't compute recursion.", e);
					throw new RuntimeException("Can't compute recursion.", e);
				}

				// check the function preconditions.
				func.check();

				status.merge(checkParameters(func));
				subMonitor.checkCanceled();

				status.merge(checkDecorators(func));
				subMonitor.checkCanceled();

				status.merge(func.getStatus());
				subMonitor.worked(1);

				assert func.hasOnlyOneFailurePerKind() : "Count failures only once.";
			});
		}

		return status;
	}

	private Map<CGNode, OrdinalSet<PointerKey>> computeMod(IProject project, CallGraph callGraph,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		Map<IProject, Map<CGNode, OrdinalSet<PointerKey>>> projectToMod = this.getProjectToMod();

		if (!projectToMod.containsKey(project)) {
			ModRef<InstanceKey> modRef = new PythonModRefWithBuiltinFunctions();
			Map<CGNode, OrdinalSet<PointerKey>> mod = modRef.computeMod(callGraph, pointerAnalysis);
			projectToMod.put(project, mod);
		}

		return projectToMod.get(project);
	}

	/**
	 * Returns a {@link Stream} of {@link Function}s. Properties of the stream are dependent on the state of this
	 * {@link HybridizeFunctionRefactoringProcessor}.
	 *
	 * @param functions The {@link Set} of {@link Function}s from which to derive a {@link Stream}.
	 * @return A potentially parallel {@link Stream} of {@link Function}s.
	 */
	private Stream<Function> getStream(Set<Function> functions) {
		Stream<Function> stream = functions.stream();
		return this.shouldProcessFunctionsInParallel() ? stream.parallel() : stream;
	}

	private TensorTypeAnalysis computeTensorTypeAnalysis(EclipsePythonProjectTensorAnalysisEngine engine,
			PythonSSAPropagationCallGraphBuilder builder) throws CancelException {
		Map<IProject, TensorTypeAnalysis> projectToTensorTypeAnalysis = this.getProjectToTensorTypeAnalysis();
		IProject project = engine.getProject();

		if (!projectToTensorTypeAnalysis.containsKey(project)) {
			TensorTypeAnalysis analysis = engine.performAnalysis(builder);
			projectToTensorTypeAnalysis.put(project, analysis);
		}

		return projectToTensorTypeAnalysis.get(project);
	}

	private CallGraph computeCallGraph(IProject project, PythonSSAPropagationCallGraphBuilder builder, IProgressMonitor monitor)
			throws CallGraphBuilderCancelException {
		Map<IProject, CallGraph> projectToCallGraph = this.getProjectToCallGraph();

		if (!projectToCallGraph.containsKey(project)) {
			ProgressMonitorDelegate monitorDelegate = ProgressMonitorDelegate.createProgressMonitorDelegate(monitor);
			AnalysisOptions options = builder.getOptions();

			if (this.shouldUseTestEntryPoints()) {
				// Get the current entrypoints.
				Iterable<? extends Entrypoint> defaultEntrypoints = builder.getOptions().getEntrypoints();

				// Get the pytest entrypoints.
				Iterable<Entrypoint> pytestEntrypoints = new PytestEntrypointBuilder().createEntrypoints(builder.getClassHierarchy());

				// Add the pytest entrypoints.
				Iterable<Entrypoint> entrypoints = concat(defaultEntrypoints, pytestEntrypoints);

				// Set the new entrypoints.
				builder.getOptions().setEntrypoints(entrypoints);

				for (Entrypoint ep : builder.getOptions().getEntrypoints())
					if (ep instanceof PytesttEntrypoint)
						LOG.info("Using test entrypoint: " + ep.getMethod().getDeclaringClass().getName() + ".");
			}

			CallGraph callGraph = builder.makeCallGraph(options, monitorDelegate);
			projectToCallGraph.put(project, callGraph);
		}

		return projectToCallGraph.get(project);
	}

	private PythonSSAPropagationCallGraphBuilder computeCallGraphBuilder(EclipsePythonProjectTensorAnalysisEngine engine)
			throws IOException {
		Map<IProject, PythonSSAPropagationCallGraphBuilder> projectToCallGraphBuilder = this.getProjectToCallGraphBuilder();
		IProject project = engine.getProject();

		if (!projectToCallGraphBuilder.containsKey(project)) {
			PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
			projectToCallGraphBuilder.put(project, builder);
		}

		return projectToCallGraphBuilder.get(project);
	}

	@Override
	public RefactoringStatus checkInitialConditions(IProgressMonitor pm) throws CoreException, OperationCanceledException {
		return super.checkInitialConditions(pm);
	}

	@Override
	public void clearCaches() {
		this.getProjectToCallGraphBuilder().clear();
		this.getProjectToMod().clear();
		this.getProjectToCallGraph().clear();
		this.getProjectToTensorTypeAnalysis().clear();
		Function.clearCaches();
	}

	@Override
	public Change createChange(IProgressMonitor pm) throws CoreException, OperationCanceledException {
		Set<Function> optimizableFunctions = this.getOptimizableFunctions();

		if (optimizableFunctions.isEmpty())
			return new NullChange(Messages.NoFunctionsToOptimize);

		Map<IFile, Queue<TextEdit>> fileToEdits = new HashMap<>();

		// for each optimizable function (i.e., those with transformations.
		for (Function function : optimizableFunctions) {
			// get the containing file.
			IFile file = function.getContainingActualFile();

			// get the edits for that file.
			Queue<TextEdit> fileEdits = fileToEdits.get(file);

			// if we don't have one yet.
			if (fileEdits == null) {
				// create a new one.
				fileEdits = new PriorityQueue<>((x, y) -> x.getExclusiveEnd() - y.getExclusiveEnd());

				// store it in the map.
				fileToEdits.put(file, fileEdits);
			}

			// transform the function.
			List<TextEdit> funcEdits;
			try {
				funcEdits = function.transform();
			} catch (BadLocationException | MalformedTreeException | NoTextSelectionException | AmbiguousDeclaringModuleException
					| NoDeclaringModuleException e) {
				throw new CoreException(Status.error("Can't create change.", e));
			}

			// add the edit to the edits for the document.
			fileEdits.addAll(funcEdits);
		}

		CompositeChange ret = new CompositeChange("Hybridize");

		for (IFile file : fileToEdits.keySet()) {
			Queue<TextEdit> edits = fileToEdits.get(file);

			if (!edits.isEmpty()) {
				TextChange change = new TextFileChange(Messages.Name, file);
				change.setKeepPreviewEdits(true);
				change.setTextType("py");

				TextEdit rootEdit = edits.remove();
				change.setEdit(rootEdit);

				while (!edits.isEmpty()) {
					TextEdit edit = edits.remove();
					change.addEdit(edit);
				}

				ret.add(change);
			}
		}

		return ret;
	}

	@Override
	public Object[] getElements() {
		// TODO Auto-generated method stub
		return null;
	}

	public Set<Function> getFunctions() {
		return this.functions;
	}

	public Set<Function> getOptimizableFunctions() {
		return this.getFunctions().parallelStream().filter(f -> !f.getStatus().hasError()).collect(Collectors.toSet());
	}

	@Override
	public String getIdentifier() {
		return HybridizeFunctionRefactoringDescriptor.REFACTORING_ID;
	}

	@Override
	public String getProcessorName() {
		return Messages.Name;
	}

	private boolean shouldAlwaysCheckPythonSideEffects() {
		return this.alwaysCheckPythonSideEffects;
	}

	public boolean shouldAlwaysCheckRecursion() {
		return alwaysCheckRecursion;
	}

	@Override
	public boolean isApplicable() throws CoreException {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public RefactoringParticipant[] loadParticipants(RefactoringStatus status, SharableParticipants sharedParticipants)
			throws CoreException {
		// TODO Auto-generated method stub
		return new RefactoringParticipant[0];
	}

	protected Map<IProject, PythonSSAPropagationCallGraphBuilder> getProjectToCallGraphBuilder() {
		return projectToCallGraphBuilder;
	}

	protected Map<IProject, Map<CGNode, OrdinalSet<PointerKey>>> getProjectToMod() {
		return projectToMod;
	}

	protected Map<IProject, CallGraph> getProjectToCallGraph() {
		return projectToCallGraph;
	}

	protected boolean shouldDumpCallGraph() {
		return dumpCallGraph;
	}

	public Map<IProject, TensorTypeAnalysis> getProjectToTensorTypeAnalysis() {
		return projectToTensorTypeAnalysis;
	}

	/**
	 * True iff project functions should be processed in parallel. Otherwise, they are processed sequentially.
	 *
	 * @return True iff project functions should be processed in parallel.
	 */
	private boolean shouldProcessFunctionsInParallel() {
		return this.processFunctionsInParallel;
	}

	/**
	 * True iff we should implicitly consider test cases as entry points in the {@link CallGraph} construction.
	 *
	 * @return True iff entry points from tests are considered.
	 */
	protected boolean shouldUseTestEntryPoints() {
		return useTestEntryPoints;
	}

	/**
	 * Returns true iff we should follow type hints regardless of any hybridization arguments.
	 *
	 * @return True iff we should follow type hints regardless of any hybridization arguments.
	 */
	public boolean shouldAlwaysFollowTypeHints() {
		return alwaysFollowTypeHints;
	}
}
