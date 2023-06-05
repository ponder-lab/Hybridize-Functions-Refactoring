package edu.cuny.hunter.hybridize.core.refactorings;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.core.runtime.Status;
import org.eclipse.core.runtime.SubMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.ltk.core.refactoring.Change;
import org.eclipse.ltk.core.refactoring.NullChange;
import org.eclipse.ltk.core.refactoring.RefactoringStatus;
import org.eclipse.ltk.core.refactoring.participants.CheckConditionsContext;
import org.eclipse.ltk.core.refactoring.participants.RefactoringParticipant;
import org.eclipse.ltk.core.refactoring.participants.SharableParticipants;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.preferences.InterpreterGeneralPreferences;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.ide.util.ProgressMonitorDelegate;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.CallGraphBuilderCancelException;
import com.ibm.wala.util.CancelException;

import edu.cuny.citytech.refactoring.common.core.RefactoringProcessor;
import edu.cuny.citytech.refactoring.common.core.TimeCollector;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.descriptors.HybridizeFunctionRefactoringDescriptor;
import edu.cuny.hunter.hybridize.core.messages.Messages;
import edu.cuny.hunter.hybridize.core.wala.ml.EclipsePythonProjectTensorAnalysisEngine;

@SuppressWarnings("unused")
public class HybridizeFunctionRefactoringProcessor extends RefactoringProcessor {

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

	private Map<IProject, CallGraph> projectToCallGraph = new HashMap<>();

	private Map<IProject, TensorTypeAnalysis> projectToTensorTypeAnalysis = new HashMap<>();

	/**
	 * True iff the {@link CallGraph} should be displayed.
	 */
	private boolean dumpCallGraph = true;

	public HybridizeFunctionRefactoringProcessor() {
		// Force the use of typeshed. It's an experimental feature of PyDev.
		InterpreterGeneralPreferences.FORCE_USE_TYPESHED = Boolean.TRUE;

		// Have WALA dump the call graph if appropriate.
		CAstCallGraphUtil.AVOID_DUMP = !this.dumpCallGraph;
	}

	public HybridizeFunctionRefactoringProcessor(Set<FunctionDefinition> functionDefinitionSet) throws TooManyMatchesException {
		this();

		// Convert the FunctionDefs to Functions.
		if (functions != null) {
			Set<Function> functionSet = this.getFunctions();

			for (FunctionDefinition fd : functionDefinitionSet) {
				Function function = new Function(fd);

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
			// create the analysis engine for the project.
			EclipsePythonProjectTensorAnalysisEngine engine = new EclipsePythonProjectTensorAnalysisEngine(project);

			// build the call graph for the project.
			PythonSSAPropagationCallGraphBuilder builder;
			try {
				builder = computeCallGraphBuilder(engine);
			} catch (IOException e) {
				throw new CoreException(Status.error("Could not compute call graph builder for: " + project.getName(), e));
			}

			subMonitor.subTask("Building call graph.");
			SubMonitor splitMonitor = subMonitor.split(IProgressMonitor.UNKNOWN, SubMonitor.SUPPRESS_NONE);
			CallGraph callGraph;
			try {
				callGraph = computeCallGraph(project, builder, splitMonitor);
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

			subMonitor.checkCanceled();

			Set<Function> projectFunctions = projectToFunctions.get(project);

			// analyze Python functions.
			LOG.info("Analyzing" + projectFunctions.size() + " function" + (allFunctions.size() > 1 ? "s" : "") + ".");
			subMonitor.beginTask(Messages.AnalyzingFunctions, projectFunctions.size());

			// check preconditions.
			LOG.info("Checking " + projectFunctions.size() + " function" + (allFunctions.size() > 1 ? "s" : "") + ".");
			subMonitor.beginTask(Messages.CheckingFunctions, allFunctions.size());

			for (Function func : projectFunctions) {
				LOG.info("Checking function: " + func + ".");

				// Find out if it's hybrid via the tf.function decorator.
				try {
					func.computeHybridization(monitor);
				} catch (BadLocationException e) {
					throw new CoreException(Status.error("Could not compute hybridization for: : " + func, e));
				}

				// TODO: Whether a function has a tensor argument should probably be an initial
				// condition: functions w/o such arguments should not be candidates.
				try {
					func.inferTensorTensorParameters(analysis, monitor);
				} catch (BadLocationException e) {
					throw new CoreException(Status.error("Could not infer tensor parameters for: : " + func, e));
				}

				// TODO: It might be time to now go back to the paper to see how we can
				// formulate the preconditions. Have a look at the stream refactoring paper.

				status.merge(checkParameters(func));
				subMonitor.checkCanceled();

				status.merge(checkDecorators(func));
				subMonitor.checkCanceled();

				subMonitor.worked(1);
			}
		}

		return status;
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
	protected void clearCaches() {
		this.getProjectToCallGraphBuilder().clear();
		this.getProjectToCallGraph().clear();
		this.getProjectToTensorTypeAnalysis().clear();
	}

	@Override
	public Change createChange(IProgressMonitor pm) throws CoreException, OperationCanceledException {
		// TODO Auto-generated method stub
		return new NullChange();
	}

	@Override
	public Object[] getElements() {
		// TODO Auto-generated method stub
		return null;
	}

	public Set<Function> getFunctions() {
		return this.functions;
	}

	@Override
	public String getIdentifier() {
		return HybridizeFunctionRefactoringDescriptor.REFACTORING_ID;
	}

	@Override
	public String getProcessorName() {
		return Messages.Name;
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

	protected Map<IProject, CallGraph> getProjectToCallGraph() {
		return projectToCallGraph;
	}

	protected boolean shouldDumpCallGraph() {
		return dumpCallGraph;
	}

	public Map<IProject, TensorTypeAnalysis> getProjectToTensorTypeAnalysis() {
		return projectToTensorTypeAnalysis;
	}
}
