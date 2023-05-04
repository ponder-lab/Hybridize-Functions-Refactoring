package edu.cuny.hunter.hybridize.core.refactorings;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.IOException;
import java.util.HashSet;
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
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.client.AnalysisEngine;
import com.ibm.wala.ide.util.ProgressMonitorDelegate;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.util.CancelException;

import edu.cuny.citytech.refactoring.common.core.RefactoringProcessor;
import edu.cuny.citytech.refactoring.common.core.TimeCollector;
import edu.cuny.hunter.hybridize.core.analysis.Function;
import edu.cuny.hunter.hybridize.core.analysis.FunctionDefinition;
import edu.cuny.hunter.hybridize.core.descriptors.HybridizeFunctionRefactoringDescriptor;
import edu.cuny.hunter.hybridize.core.messages.Messages;
import edu.cuny.hunter.hybridize.core.wala.ml.EclipsePythonProjectTensorAnalysisEngine;

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

	/**
	 * {@link AnalysisEngine}s that already have their {@link CallGraph}s built.
	 */
	private Set<AnalysisEngine> enginesWithBuiltCallGraphs = new HashSet<>();

	/**
	 * True iff the {@link CallGraph} should be displayed.
	 */
	private boolean dumpCallGraph = true;

	public HybridizeFunctionRefactoringProcessor() {
	}

	public HybridizeFunctionRefactoringProcessor(Set<FunctionDefinition> functionDefinitionSet, IProgressMonitor monitor)
			throws TooManyMatchesException, BadLocationException {
		// Force the use of typeshed. It's an experimental feature of PyDev.
		InterpreterGeneralPreferences.FORCE_USE_TYPESHED = Boolean.TRUE;

		// Convert the FunctionDefs to Functions.
		if (functions != null) {
			Set<Function> functionSet = this.getFunctions();

			for (FunctionDefinition fd : functionDefinitionSet) {
				Function function = new Function(fd, monitor);

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
			try {
				this.buildCallGraph(engine, timeCollector, subMonitor.split(IProgressMonitor.UNKNOWN, SubMonitor.SUPPRESS_NONE));
			} catch (IOException | CancelException e) {
				throw new CoreException(Status.error("Could not build call graph for: " + project.getName(), e));
			}

			Set<Function> projectFunctions = projectToFunctions.get(project);

			subMonitor.checkCanceled();

			LOG.info("Checking " + projectFunctions.size() + " function" + (allFunctions.size() > 1 ? "s" : "") + ".");
			subMonitor = SubMonitor.convert(monitor, Messages.CheckingFunctions, allFunctions.size());

			for (Function func : projectFunctions) {
				LOG.info("Checking function: " + func + ".");

				// TODO: Whether a function has a tensor argument should probably be an initial
				// condition: functions w/o such arguments should not be candidates.
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

	/**
	 * Builds the call graph that is part of the given {@link AnalysisEngine}.
	 *
	 * @param engine The {@link AnalysisEngine} for which to build the call graph.
	 * @param timeCollector A {@link TimeCollector} to exclude any entry point finding.
	 * @param monitor An {@link IProgressMonitor} to track call graph building progress.
	 * @apiNote This method involves caching of call graphs.
	 */
	private void buildCallGraph(PythonTensorAnalysisEngine engine, TimeCollector timeCollector, IProgressMonitor monitor)
			throws IllegalArgumentException, IOException, CancelException {
		Set<AnalysisEngine> enginesWithBuiltCallGraphs = this.getEnginesWithBuiltCallGraphs();

		// if we haven't built the call graph yet.
		if (!enginesWithBuiltCallGraphs.contains(engine)) {
			// engine.getOptions();
			PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
			AnalysisOptions options = builder.getOptions();
			SubMonitor subMonitor = SubMonitor.convert(monitor, "Building call graph.", 1);
			ProgressMonitorDelegate monitorDelegate = ProgressMonitorDelegate.createProgressMonitorDelegate(subMonitor);
			CallGraph callGraph = builder.makeCallGraph(options, monitorDelegate);

			timeCollector.start();
			if (this.shouldDumpCallGraph()) {
				CAstCallGraphUtil.AVOID_DUMP = false;
				CAstCallGraphUtil.dumpCG(builder.getCFAContextInterpreter(), builder.getPointerAnalysis(), callGraph);
				// DotUtil.dotify(callGraph, null, PDFTypeHierarchy.DOT_FILE, "callgraph.pdf", "dot");
			}
			timeCollector.stop();

			enginesWithBuiltCallGraphs.add(engine);
		}
	}

	@Override
	public RefactoringStatus checkInitialConditions(IProgressMonitor pm) throws CoreException, OperationCanceledException {
		return super.checkInitialConditions(pm);
	}

	@Override
	protected void clearCaches() {
		// NOTE: Nothing right now.
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

	protected Set<AnalysisEngine> getEnginesWithBuiltCallGraphs() {
		return enginesWithBuiltCallGraphs;
	}

	protected boolean shouldDumpCallGraph() {
		return dumpCallGraph;
	}
}
