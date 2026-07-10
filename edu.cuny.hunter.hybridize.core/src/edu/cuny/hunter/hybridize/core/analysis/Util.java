package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.python.pydev.ast.codecompletion.revisited.visitors.Definition;
import org.python.pydev.ast.item_pointer.ItemPointer;
import org.python.pydev.ast.refactoring.AbstractPyRefactoring;
import org.python.pydev.ast.refactoring.HierarchyNodeModel;
import org.python.pydev.ast.refactoring.IPyRefactoring;
import org.python.pydev.ast.refactoring.RefactoringRequest;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IModule;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.shared_core.string.CoreTextSelection;

import com.google.common.collect.Sets;
import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Iterator2Iterable;
import com.ibm.wala.util.collections.Pair;

public class Util {

	private static final ILog LOG = getLog(Util.class);

	/**
	 * Get the name of the module defining the entity described in the given {@link PySelection}.
	 *
	 * @param selection The {@link PySelection} in question.
	 * @param containingModName The name of the module containing the {@link PySelection}.
	 * @param containingFile The {@link File} containing the module.
	 * @param nature The {@link IPythonNature} to use.
	 * @param monitor The IProgressMonitor to use.
	 * @return The name of the module defining the given {@link PySelection}.
	 * @throws AmbiguousDeclaringModuleException On ambiguous definitions found.
	 * @throws BadLocationException On a parsing error.
	 * @throws NoDeclaringModuleException When a declaring module can't be found.
	 */
	public static String getDeclaringModuleName(PySelection selection, String containingModName, File containingFile, IPythonNature nature,
			IProgressMonitor monitor) throws BadLocationException, AmbiguousDeclaringModuleException, NoDeclaringModuleException {
		monitor.beginTask("Getting declaring module name.", 1);

		LOG.info(String.format("Getting declaring module name for selection: %s in line: %s, module: %s, file: %s, and project: %s.",
				selection.getSelectedText(), selection.getLineWithoutCommentsOrLiterals().strip(), containingModName, containingFile,
				nature.getProject()));

		RefactoringRequest request = new RefactoringRequest(containingFile, selection, nature);

		request.acceptTypeshed = true;
		request.moduleName = containingModName;
		request.pushMonitor(monitor);

		IPyRefactoring pyRefactoring = AbstractPyRefactoring.getPyRefactoring();

		ItemPointer[] pointers;
		try {
			pointers = pyRefactoring.findDefinition(request);
		} catch (TooManyMatchesException e) {
			throw new AmbiguousDeclaringModuleException(selection, containingModName, containingFile, nature, e);
		}

		LOG.info("Found " + pointers.length + " \"pointer(s).\"");

		if (pointers.length == 0)
			throw new NoDeclaringModuleException(
					String.format("Can't find declaring module for selection: %s in line: %s, module: %s, file: %s, and project: %s.",
							selection.getSelectedText(), selection.getLineWithoutCommentsOrLiterals().strip(), containingModName,
							containingFile.getName(), nature.getProject()));

		// Collect the potential declaring module names.
		Set<String> potentialDeclaringModuleNames = new HashSet<>();

		// for each match.
		for (ItemPointer itemPointer : pointers) {
			Definition definition = itemPointer.definition;
			LOG.info("Found definition: " + definition + ".");

			IModule module = definition.module;
			LOG.info(String.format("Found module: %s.", module));

			String moduleName = module.getName();
			LOG.info(String.format("Found module name: %s.", moduleName));

			// add it to the set of found module names.
			potentialDeclaringModuleNames.add(moduleName);
		}

		// if we found a unique module name.
		if (potentialDeclaringModuleNames.size() == 1) {
			monitor.done();

			// return the first one.
			return potentialDeclaringModuleNames.iterator().next();
		}

		// otherwise, we have an ambiguous declaring module name.
		throw new AmbiguousDeclaringModuleException(selection, containingModName, containingFile, nature,
				potentialDeclaringModuleNames.size());
	}

	/**
	 * Get the FQN of the given decorator.
	 *
	 * @param decorator The {@link decoratorsType} in question.
	 * @param containingModName The name of the module where the decorator is used.
	 * @param containingFile The {@link File} where the containingModName is defined.
	 * @param containingSelection The {@link PySelection} containing the decorator.
	 * @param nature The {@link IPythonNature} to use.
	 * @param monitor The IProgressMonitor to use.
	 * @return The FQN of the given {@link decoratorsType}.
	 * @throws BadLocationException When the containing entities cannot be parsed.
	 * @throws AmbiguousDeclaringModuleException If the definition of the decorator is ambiguous.
	 * @throws NoDeclaringModuleException When a declaring module can't be found.:
	 */
	public static String getFullyQualifiedName(decoratorsType decorator, String containingModName, File containingFile,
			PySelection containingSelection, IPythonNature nature, IProgressMonitor monitor)
			throws BadLocationException, AmbiguousDeclaringModuleException, NoDeclaringModuleException {
		monitor.beginTask("Getting decorator FQN.", 3);

		exprType decoratorFunction = decorator.func;
		String fqn = getFullyQualifiedName(decoratorFunction, containingModName, containingFile, containingSelection, nature, monitor);

		monitor.done();
		return fqn;
	}

	// FIXME: `node` is only used for logging.
	public static String getFullyQualifiedName(SimpleNode node, String containingModName, File containingFile,
			PySelection containingSelection, IPythonNature nature, IProgressMonitor monitor)
			throws BadLocationException, AmbiguousDeclaringModuleException, NoDeclaringModuleException {
		monitor.subTask("Getting declaring module name.");
		LOG.info("Getting declaring module name for SimpleNode: " + node + ".");

		String declaringModuleName = getDeclaringModuleName(containingSelection, containingModName, containingFile, nature, monitor);
		LOG.info(String.format("Found declaring module: %s.", declaringModuleName));
		monitor.worked(1);

		String representationString = NodeUtils.getRepresentationString(node);
		LOG.info(String.format("\"Representation\" of %s: %s.", node, representationString));
		monitor.worked(1);

		String fqn = declaringModuleName + "." + representationString;
		LOG.info(String.format("FQN is: %s.", fqn));

		monitor.worked(1);
		return fqn;
	}

	private Util() {
	}

	/**
	 * Returns the qualified name corresponding to the given {@link FunctionDef}.
	 *
	 * @see <a href="https://peps.python.org/pep-3155">PEP 3155</a>
	 * @param functionDef The {@link FunctionDef} in question.
	 * @return The corresponding qualified name per PEP 3155.
	 */
	public static String getQualifiedName(FunctionDef functionDef) {
		String identifier = NodeUtils.getFullRepresentationString(functionDef);
		StringBuilder ret = new StringBuilder();
		SimpleNode parentNode = functionDef.parent;

		int count = 0;

		while (parentNode instanceof ClassDef || parentNode instanceof FunctionDef) {
			String identifierParent = NodeUtils.getFullRepresentationString(parentNode);

			if (count == 0) {
				ret.append(identifierParent);
				ret.append(".");
			} else {
				ret.insert(0, ".");
				ret.insert(0, identifierParent);
			}
			count++;

			parentNode = parentNode.parent;
		}

		ret.append(identifier);

		return ret.toString();
	}

	public static PySelection getSelection(decoratorsType decorator, IDocument document) throws NoTextSelectionException {
		exprType expression = getExpressionFromFunction(decorator);
		LOG.info("Getting PySelection for exprType: " + expression + ".");
		return getSelection(expression, document);
	}

	public static PySelection getSelection(SimpleNode node, IDocument document) throws NoTextSelectionException {
		CoreTextSelection coreTextSelection = getCoreTextSelection(document, node);
		return new PySelection(document, coreTextSelection);
	}

	public static CoreTextSelection getCoreTextSelection(IDocument document, SimpleNode expression) throws NoTextSelectionException {
		int offset;

		try {
			offset = NodeUtils.getOffset(document, expression);
		} catch (RuntimeException e) {
			throw new NoTextSelectionException(expression, e);
		}

		String representationString = NodeUtils.getRepresentationString(expression);

		if (representationString == null)
			throw new NoTextSelectionException(expression);

		CoreTextSelection coreTextSelection = new CoreTextSelection(document, offset, representationString.length());
		return coreTextSelection;
	}

	/**
	 * Returns the {@link exprType} associated with the given {@link decoratorsType}'s {@code function} attribute.
	 *
	 * @param decorator The {@link decoratorsType} for which to retrieve the associated {@link exprType} from its {@code function}
	 *        attribute.
	 * @return The {@link exprType} associated with the given {@link decoratorsType}'s {@code function} attribute.
	 */
	public static exprType getExpressionFromFunction(decoratorsType decorator) {
		exprType func = decorator.func;
		return getInnerExpression(func);
	}

	private static exprType getInnerExpression(exprType expr) {
		if (expr instanceof Attribute || expr instanceof Name)
			return expr;

		if (expr instanceof Call) {
			Call call = (Call) expr;
			exprType func = call.func;
			return getInnerExpression(func);
		}

		throw new IllegalArgumentException("Can't find attribute of: " + expr + ".");
	}

	/**
	 * Returns true iff the given {@link decoratorsType} corresponds to a Python generated decorator (e.g., "setter" for properties).
	 *
	 * @param decorator The {@link decoratorsType} in question.
	 * @return True iff the given {@link decoratorsType} is generated by the run-time (e.g., a property).
	 */
	public static boolean isGenerated(decoratorsType decorator) {
		String decoratorRepresentation = NodeUtils.getRepresentationString(decorator.func);
		return decoratorRepresentation.equals("setter");
	}

	public static boolean isBuiltIn(decoratorsType decorator) {
		String decoratorRepresentation = NodeUtils.getRepresentationString(decorator.func);
		return decoratorRepresentation.equals("property");
	}

	public static boolean calls(CGNode node, MethodReference methodReference, CallGraph callGraph) {
		return calls(node, methodReference, callGraph, Sets.newHashSet());
	}

	private static boolean calls(CGNode node, MethodReference methodReference, CallGraph callGraph, Set<MethodReference> seen) {
		seen.add(node.getMethod().getReference());

		// check the callees.
		for (Iterator<CGNode> succNodes = callGraph.getSuccNodes(node); succNodes.hasNext();) {
			CGNode next = succNodes.next();
			MethodReference reference = next.getMethod().getReference();

			if (methodReference.equals(reference))
				return true;

			// otherwise, check its callees.
			if (!seen.contains(reference) && calls(next, methodReference, callGraph, seen))
				return true;
		}

		return false;
	}

	/** The WALA type name of the TensorFlow module object (what {@code import tensorflow as tf} binds). */
	private static final String TENSORFLOW_MODULE_TYPE_NAME = "Ltensorflow";

	/** WALA type-name prefix for modeled TensorFlow operations, e.g. {@code Ltensorflow/functions/matmul}. */
	private static final String TENSORFLOW_FUNCTION_TYPE_NAME_PREFIX = "Ltensorflow/functions/";

	/** The TensorFlow module prefix used to recognize a call as a TensorFlow op from its fully-qualified name. */
	private static final String TENSORFLOW_FQN_PREFIX = "tensorflow.";

	/**
	 * Global-read names bound to the TensorFlow module by a module-level import ({@code import tensorflow [as tf]}), used to root an
	 * attribute chain at TensorFlow when the module global's points-to set is unavailable. See {@link #isTensorFlowModule}.
	 */
	private static final Set<String> TENSORFLOW_MODULE_GLOBAL_NAMES = Set.of("global tf", "global tensorflow");

	/**
	 * TensorFlow sub-namespaces that construct specs or protobufs rather than performing tensor computation, so a call into them does not
	 * count as a tensor op. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/709.
	 */
	private static final Set<String> NON_OP_TENSORFLOW_FQN_PREFIXES = Set.of("tensorflow.train.", "tensorflow.io.");

	/** The set of pointer keys the given {@link TensorTypeAnalysis} types as tensors. */
	public static Set<PointerKey> tensorTypedPointerKeys(TensorTypeAnalysis tensorAnalysis) {
		Set<PointerKey> keys = new HashSet<>();

		for (Pair<PointerKey, TensorVariable> pair : tensorAnalysis)
			keys.add(pair.fst);

		return keys;
	}

	/**
	 * True iff {@code node}, transitively over its call-graph successors, performs a TensorFlow tensor op. A body instruction counts as a
	 * tensor op when it either (a) defines a value the tensor-type analysis types as a tensor (which covers modeled ops, tensor operators,
	 * and layer calls, and correctly excludes proto and spec builders whose results are not tensors), or (b) invokes a {@code tensorflow.*}
	 * op recognized from the IR (which additionally covers ops not modeled by the tensor-type analysis). Only user-defined bodies are
	 * scanned—a TensorFlow library node's own body is skipped, since its ops are detected at the user call site—but the traversal walks
	 * through library nodes to their successors, so user callbacks passed to higher-order TensorFlow APIs ({@code strategy.run},
	 * {@code dataset.map}) are still reached. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/709.
	 *
	 * @param node The call-graph node to check.
	 * @param callGraph The call graph, used to follow callees transitively.
	 * @param pointerAnalysis The pointer analysis, used to resolve def, callee, and attribute-name pointer keys.
	 * @param tensorTypedKeys The pointer keys the tensor-type analysis types as tensors (see {@link #tensorTypedPointerKeys}).
	 * @return True iff a TensorFlow tensor op is reachable from {@code node}.
	 */
	public static boolean performsTensorFlowOp(CGNode node, CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			Set<PointerKey> tensorTypedKeys) {
		return performsTensorFlowOp(node, callGraph, pointerAnalysis, tensorTypedKeys, Sets.newHashSet());
	}

	private static boolean performsTensorFlowOp(CGNode node, CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			Set<PointerKey> tensorTypedKeys, Set<CGNode> seen) {
		if (!seen.add(node))
			return false;

		// Don't scan a TensorFlow library node's own body: its ops are detected at the user call site, and attributing its internal
		// IR to the analyzed function would misreport library internals as user computation.
		if (!isTensorFlowNode(node)) {
			IR ir = node.getIR();

			if (ir != null) {
				DefUse defUse = node.getDU();

				for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions())) {
					// (a) The instruction defines a tensor-typed value (operators, layer calls, modeled ops).
					if (definesTensor(node, instruction, pointerAnalysis, tensorTypedKeys))
						return true;

					// (b) The instruction invokes an unmodeled `tensorflow.*` op recognized from the IR.
					if (instruction instanceof PythonInvokeInstruction invoke && invokesTensorFlowOp(node, invoke, defUse, pointerAnalysis))
						return true;
				}
			}
		}

		// Transitively check callees, walking through TensorFlow library nodes rather than pruning them: a higher-order API
		// (`strategy.run`, `dataset.map`) invokes user callbacks whose tensor ops live behind the summary, and the successor edges
		// are per-node (context-sensitive), so the traversal reaches exactly the callbacks this function passes in.
		for (Iterator<CGNode> succNodes = callGraph.getSuccNodes(node); succNodes.hasNext();) {
			CGNode succNode = succNodes.next();

			if (performsTensorFlowOp(succNode, callGraph, pointerAnalysis, tensorTypedKeys, seen))
				return true;
		}

		return false;
	}

	/** True iff {@code node}'s declaring class is in the TensorFlow namespace, i.e. a modeled op or library node rather than user code. */
	private static boolean isTensorFlowNode(CGNode node) {
		String name = node.getMethod().getDeclaringClass().getReference().getName().toString();
		return name.startsWith(TENSORFLOW_MODULE_TYPE_NAME);
	}

	/** Method names whose invocation is only valid in eager execution (e.g. {@code Tensor.numpy()}). */
	private static final Set<String> EAGER_ONLY_METHOD_NAMES = Set.of("numpy");

	/** The {@code dtype} attribute: a trace-time-constant element type, so numpy over it is always safe (launders taint). */
	private static final String DTYPE_MEMBER_NAME = "dtype";

	/**
	 * The {@code shape} attribute: its read yields shape metadata (a {@link ShapeDescriptor} covering every dimension of the read tensor).
	 * numpy over shape metadata is safe iff the covered dimensions are statically known, which the scan decides per-dimension from the
	 * source tensor's {@link TensorType}. See {@link #scanForTaintedNumpySinks} and
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747.
	 */
	private static final String SHAPE_MEMBER_NAME = "shape";

	/** The built-in {@code slice} constructor, into which a Python {@code x[a:b:c]} subscript is lowered. */
	private static final String SLICE_BUILTIN_NAME = "slice";

	/**
	 * The unary-negation operator, matched by name: the operator enum ({@code IUnaryOpInstruction.Operator}) lives in a non-exported WALA
	 * package, so its type cannot be referenced here, but its {@code toString()} is stable.
	 */
	private static final String NEGATION_OPERATOR_NAME = "neg";

	/**
	 * True iff {@code node}, transitively over its call-graph successors, invokes an eager-only API (e.g. {@code Tensor.numpy()}), which
	 * raises under {@code tf.function} tracing. Detection is by callee attribute name rather than receiver typing: the receiver's tensor
	 * typing is frequently unavailable (e.g. the result of a user-defined callable), and missing a real {@code .numpy()} call would
	 * hybridize a function that crashes on first call, while over-matching only declines an optimization. Only user-defined bodies are
	 * scanned, and the traversal walks through TensorFlow library nodes to reach user callbacks, both mirroring
	 * {@link #performsTensorFlowOp(CGNode, CallGraph, PointerAnalysis, Set)}. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/363.
	 *
	 * @param node The call-graph node to check.
	 * @param callGraph The call graph, used to follow callees transitively.
	 * @param pointerAnalysis The pointer analysis, used to resolve the callee's attribute name.
	 * @return True iff an eager-only API call is reachable from {@code node}.
	 */
	public static boolean callsEagerOnlyApi(CGNode node, CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		return callsEagerOnlyApi(node, callGraph, pointerAnalysis, Sets.newHashSet());
	}

	private static boolean callsEagerOnlyApi(CGNode node, CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			Set<CGNode> seen) {
		if (!seen.add(node))
			return false;

		if (!isTensorFlowNode(node)) {
			IR ir = node.getIR();

			if (ir != null) {
				DefUse defUse = node.getDU();

				for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
					if (instruction instanceof PythonInvokeInstruction invoke && invokesEagerOnlyApi(node, invoke, defUse, pointerAnalysis))
						return true;
			}
		}

		for (Iterator<CGNode> succNodes = callGraph.getSuccNodes(node); succNodes.hasNext();) {
			CGNode succNode = succNodes.next();

			if (callsEagerOnlyApi(succNode, callGraph, pointerAnalysis, seen))
				return true;
		}

		return false;
	}

	/** True iff {@code invoke}'s callee is an attribute read whose member name is an eager-only method name (e.g. {@code numpy}). */
	private static boolean invokesEagerOnlyApi(CGNode node, PythonInvokeInstruction invoke, DefUse defUse,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		SSAInstruction def = defUse.getDef(invoke.getUse(0));

		if (def instanceof PythonPropertyRead read) {
			String member = resolveStringConstant(node, read.getMemberRef(), pointerAnalysis);
			return member != null && EAGER_ONLY_METHOD_NAMES.contains(member);
		}

		return false;
	}

	/**
	 * Global-read names identifying the numpy/scipy modules by import alias, mirroring {@link #TENSORFLOW_MODULE_GLOBAL_NAMES}. Unlike the
	 * TensorFlow fallback, whose over-recognition is incompleteness-safe (it only lets an eager function hybridize), over-recognition here
	 * over-blocks: a non-numpy global named {@code np} would fail the precondition. Accepted safety-first; the evaluation measures the
	 * cost. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/740.
	 */
	private static final Set<String> NUMPY_MODULE_GLOBAL_NAMES = Set.of("global numpy", "global np", "global scipy", "global sp");

	/** Type-name prefixes of the numpy/scipy module objects, for the points-to (precise) branch of the module test. */
	private static final Set<String> NUMPY_MODULE_TYPE_NAME_PREFIXES = Set.of("Lnumpy", "Lscipy");

	/**
	 * True iff {@code node} (transitively through user-defined callees) applies a numpy/scipy API to a value flowing from its parameters.
	 * Under {@code @tf.function}, parameter values become symbolic tensors during tracing, and numpy/scipy applied to them raises, so such
	 * a function crashes on first call when hybridized. The gate is an SSA def-use taint slice rather than a points-to intersection: a
	 * points-to-keyed gate silently passes a crasher under any modeling gap that empties a points-to set, and this precondition exists
	 * precisely to hold while upstream modeling is in motion. A {@code dtype} read launders taint (the element type is a trace-time
	 * constant); a {@code shape} read (or {@code tf.shape}/{@code get_shape_list}) yields shape metadata, over which numpy is declined only
	 * when a covered dimension is provably dynamic, decided per-dimension from the source tensor's inferred {@link TensorType} (the
	 * per-dimension shape-aware verdict of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747), with the shape's
	 * provenance propagated across call boundaries so staticness resolves interprocedurally
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756). Known narrowings, each documented on the issue:
	 * positional arguments only cross call sites, field-mediated and subscript-store flows are not tracked, and a method's receiver is
	 * never a source. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/740.
	 *
	 * @param node The call-graph node to check.
	 * @param method True iff the function is an instance method, in which case the receiver slot is not a taint source.
	 * @param callGraph The call graph, used to follow user-defined callees.
	 * @param pointerAnalysis The pointer analysis, used to resolve attribute names and module roots.
	 * @param tensorTypeIndex The (node, value number) index of the tensor-type analysis ({@link #indexTensorTypes}), consulted for the
	 *        per-dimension shape staticness at a numpy-over-shape sink.
	 * @return True iff a numpy/scipy API is applied to a parameter-flowing value reachable from {@code node}.
	 */
	public static boolean appliesNumpyToParameters(CGNode node, boolean method, CallGraph callGraph,
			PointerAnalysis<InstanceKey> pointerAnalysis, Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex) {
		IR ir = node.getIR();

		if (ir == null)
			return false;

		// Parameter value numbers: slot 0 is the function object itself; slot 1 is the receiver for an instance method. Neither is a
		// taint source—tainting the receiver would block any method mixing numpy over scalar fields with sanitized tensor use.
		int[] parameters = ir.getSymbolTable().getParameterValueNumbers();
		Set<Integer> sources = new HashSet<>();

		for (int i = method ? 2 : 1; i < parameters.length; i++)
			sources.add(parameters[i]);

		if (sources.isEmpty())
			return false;

		return scanForTaintedNumpySinks(node, sources, new HashSet<>(), new HashMap<>(), callGraph, pointerAnalysis, tensorTypeIndex,
				new HashMap<>()).sink();
	}

	/**
	 * Indexes {@code tensorTypeAnalysis}'s local pointer keys by (node, value number) once, so per-value lookups
	 * ({@link #lookupTensorTypes}) are constant-time rather than a full linear scan of the analysis per call. Build this once per function
	 * (or project) and reuse it across the function's call-graph nodes rather than rebuilding per node.
	 */
	static Map<CGNode, Map<Integer, Set<TensorType>>> indexTensorTypes(TensorTypeAnalysis tensorTypeAnalysis) {
		Map<CGNode, Map<Integer, Set<TensorType>>> index = new HashMap<>();

		for (Pair<PointerKey, TensorVariable> pair : tensorTypeAnalysis)
			if (pair.fst instanceof LocalPointerKey local && pair.snd != null)
				index.computeIfAbsent(local.getNode(), n -> new HashMap<>()).computeIfAbsent(local.getValueNumber(), v -> new HashSet<>())
						.addAll(pair.snd.getTypes());

		return index;
	}

	/**
	 * The {@link TensorType}s the tensor-type analysis associates with the local {@code valueNumber} in {@code node}, from the (node, value
	 * number) index built by {@link #indexTensorTypes}. Empty when the analysis has no tensor classification for the value (not analyzed,
	 * or classified not-a-tensor).
	 */
	private static Set<TensorType> lookupTensorTypes(CGNode node, int valueNumber,
			Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex) {
		return tensorTypeIndex.getOrDefault(node, Map.of()).getOrDefault(valueNumber, Set.of());
	}

	/**
	 * A shape-derived value tracked by the scan: the tensor whose shape it came from ({@code sourceTensor}, a value number in
	 * {@code sourceNode}) and which of that tensor's dimensions it covers ({@code dims}; {@code null} means all dimensions). A non-negative
	 * index is absolute; a negative index counts from the end (Python semantics), resolved against each {@link TensorType}'s own rank at
	 * the sink ({@link #numpyOverShapeStaticness}), so a suffix slice stays tracked even when the source's rank is statically unresolvable
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/761). numpy over such a value is safe iff the covered
	 * dimensions are statically known; the descriptor lets the sink consult the source tensor's per-dim {@link TensorType}. The source
	 * frame need not be the frame the sink sits in: the descriptor crosses call boundaries with the shape taint (argument seeding and
	 * return flow, https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756), and the (node, value number) lookup stays
	 * valid because the tensor-type index is global. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747.
	 */
	private record ShapeDescriptor(CGNode sourceNode, int sourceTensor, Set<Integer> dims) {
	}

	/** The staticness verdict for numpy over a shape-derived value: safe, a crash (dynamic dim), or unprovable (⊤/unresolved). */
	private enum ShapeStaticness {
		STATIC, DYNAMIC, TOP
	}

	/**
	 * The verdict for numpy applied to the shape-derived value described by {@code descriptor}. Consults the source tensor's per-dim
	 * {@link TensorType} across all its analysis contexts: {@link ShapeStaticness#DYNAMIC} if any covered dimension is non-numeric in any
	 * context (numpy would crash under tracing); {@link ShapeStaticness#STATIC} if every covered dimension is a {@link NumericDim} in every
	 * context (safe); {@link ShapeStaticness#TOP} if the source is untyped or its shape is ⊤ (staticness cannot be proven either way). A
	 * negative covered index counts from the end of each context's own rank, so a suffix slice resolves per-context without a
	 * statically-known rank (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/761); a covered index beyond a context's
	 * rank contributes no element there (Python clamps a slice to the vector's extent) and is ignored for that context.
	 */
	private static ShapeStaticness numpyOverShapeStaticness(ShapeDescriptor descriptor,
			Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex) {
		// An empty covered set (an empty slice such as `[:0]`) consumes no dimension values: vacuously static, even for an untyped
		// source.
		if (descriptor.dims() != null && descriptor.dims().isEmpty())
			return ShapeStaticness.STATIC;

		Set<TensorType> types = lookupTensorTypes(descriptor.sourceNode(), descriptor.sourceTensor(), tensorTypeIndex);

		if (types.isEmpty())
			return ShapeStaticness.TOP;

		boolean top = false;

		for (TensorType type : types) {
			List<Dimension<?>> typeDims = type.getDims();

			if (typeDims == null) {
				top = true;
				continue;
			}

			Set<Integer> covered = descriptor.dims();

			if (covered == null) {
				covered = new TreeSet<>();

				for (int i = 0; i < typeDims.size(); i++)
					covered.add(i);
			}

			for (int i : covered) {
				int index = i < 0 ? typeDims.size() + i : i;

				// Covered dimensions come from slices, and Python clamps a slice to the vector's extent: an index beyond this
				// context's rank contributes no element here, so it is ignored rather than degrading the verdict.
				if (index < 0 || index >= typeDims.size())
					continue;

				if (!(typeDims.get(index) instanceof NumericDim))
					return ShapeStaticness.DYNAMIC; // a non-numeric (dynamic/symbolic/ragged) covered dim crashes numpy.
			}
		}

		return top ? ShapeStaticness.TOP : ShapeStaticness.STATIC;
	}

	/**
	 * Resolves {@code value} in {@code node} to an integer constant, or {@code null} if it cannot be resolved. Handles a literal integer in
	 * the symbol table, a unary negation of a resolvable value, and an interprocedural constant surfaced by the pointer analysis (a
	 * {@link ConstantKey} in the value's points-to set), mirroring how {@link Function#inferPrimitiveParameters} recovers literal arguments
	 * across call sites.
	 */
	private static Integer resolveIntConstant(CGNode node, int value, DefUse defUse, PointerAnalysis<InstanceKey> pointerAnalysis) {
		SymbolTable symbolTable = node.getIR().getSymbolTable();

		if (symbolTable.isIntegerConstant(value))
			return symbolTable.getIntValue(value);

		SSAInstruction def = defUse.getDef(value);

		// A unary negation (the `-k` in a `[-k:]` slice); matched by name since the operator enum's type cannot be referenced here.
		if (def instanceof SSAUnaryOpInstruction unary && NEGATION_OPERATOR_NAME.equalsIgnoreCase(String.valueOf(unary.getOpcode()))) {
			Integer operand = resolveIntConstant(node, unary.getUse(0), defUse, pointerAnalysis);
			return operand == null ? null : -operand;
		}

		PointerKey pointerKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, value);
		Integer resolved = null;

		for (InstanceKey instanceKey : pointerAnalysis.getPointsToSet(pointerKey))
			if (instanceKey instanceof ConstantKey<?> constantKey && constantKey.getValue() instanceof Number number) {
				// Only an integral value in int range is a usable slice bound; a non-integral (e.g. float) constant would truncate under
				// intValue() and wrongly "prove" a bound, so treat it as unresolvable.
				long asLong = number.longValue();

				if (number.doubleValue() != asLong || asLong < Integer.MIN_VALUE || asLong > Integer.MAX_VALUE)
					return null;

				int candidate = (int) asLong;

				if (resolved != null && resolved != candidate)
					return null; // ambiguous across contexts; refuse to guess.

				resolved = candidate;
			}

		return resolved;
	}

	/**
	 * The dimension indices selected by a Python slice {@code [start:stop:step]} of a shape vector of the given {@code rank}, following
	 * Python's clamping and negative-index semantics, or {@code null} if the slice cannot be resolved to a concrete set (a non-unit or
	 * unknown step). {@code null} bounds default to the whole extent.
	 */
	private static Set<Integer> resolveSliceDims(Integer start, Integer stop, Integer step, int rank) {
		int stride = step == null ? 1 : step;

		if (stride != 1)
			return null; // only unit-stride slices are modeled precisely.

		int from = start == null ? 0 : start < 0 ? Math.max(0, rank + start) : Math.min(start, rank);
		int to = stop == null ? rank : stop < 0 ? Math.max(0, rank + stop) : Math.min(stop, rank);
		Set<Integer> dims = new TreeSet<>();

		for (int i = from; i < to; i++)
			dims.add(i);

		return dims;
	}

	/** True iff {@code invoke} calls the built-in {@code slice} constructor (how a Python {@code x[a:b:c]} subscript is modeled). */
	private static boolean invokesSliceBuiltin(PythonInvokeInstruction invoke, DefUse defUse) {
		SSAInstruction def = defUse.getDef(invoke.getUse(0));

		if (!(def instanceof AstLexicalRead lexical))
			return false;

		// Read the name via getName() rather than the variableName field, and guard the array: mirrors PythonModRefWithBuiltinFunctions
		// (WALA 1.8.0 encapsulated the field).
		Access[] accesses = lexical.getAccesses();
		return accesses.length > 0 && SLICE_BUILTIN_NAME.equals(accesses[0].getName().fst);
	}

	/**
	 * The result of a two-color taint scan: whether numpy hit a value-tainted argument, whether a value taint escaped this node, and the
	 * {@link ShapeDescriptor} the node's returns carry, so a caller keeps resolving staticness across the return boundary
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756). The descriptor is {@code null} when no shape-tainted
	 * value is returned <em>or</em> any returned shape lacks a descriptor (provenance is dropped rather than partially trusted);
	 * conflicting tracked returns merge to the ambiguous descriptor instead.
	 */
	private record NumpyScanResult(boolean sink, boolean valueEscapes, ShapeDescriptor returnDescriptor) {
	}

	/**
	 * Worklist taint propagation over {@code node}'s def-use chains, tracking two taint colors. A <em>value</em> taint marks the tensor
	 * value itself, over which numpy always raises under {@code tf.function} tracing (a sink, decline). A <em>shape</em> taint marks a
	 * value derived from a tensor's shape (via {@code .shape}/{@code .shape.as_list()}, {@code tf.shape}/{@code size}/{@code rank}, or a
	 * user-defined shape extractor such as {@code get_shape_list}); it carries a {@link ShapeDescriptor} identifying the source tensor and
	 * the dimensions it covers, narrowed as the shape vector is sliced ({@code [-k:]}, with the bound resolved via the pointer analysis).
	 * numpy over a shape-tainted argument is declined only when a covered dimension is provably dynamic in the source tensor's
	 * {@link TensorType} ({@link #numpyOverShapeStaticness}); a proven-static shape is safe, and only a provably-dynamic covered dimension
	 * is a sink. The descriptor crosses call boundaries with the taint
	 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/756): a shape-tainted argument seeds its descriptor onto the
	 * callee's parameter ({@code shapeDescriptorSeed}), and a callee returning a tracked shape carries its (possibly slice-narrowed)
	 * descriptor back to the call-site result, so staticness resolves in whichever frame the sink sits. An unprovable shape - a ⊤ type or
	 * untracked/ambiguous provenance - is permitted, favoring precision. The sound decline-unless-provably-static variant is tracked on
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/751; see
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747 for the per-dimension shape-aware verdict.
	 * <p>
	 * Call sites are tainted conservatively (a value argument taints the call-site result), which follows a comprehension's
	 * element-through-container flow (NLPGNN's {@code TUDataset.cat}, issue 745). The result is colored SHAPE only when the callee is a
	 * pure <em>shape extractor</em> — its value-tainted arguments are consumed solely by shape operations and never escape (e.g.
	 * {@code get_shape_list}), whose result then describes the tensor argument's shape — otherwise the result is value-tainted. A callee's
	 * {@code valueEscapes} bit records whether any value taint reached a non-shape use inside it, and is what distinguishes a shape
	 * extractor from a value carrier. A {@code dtype} read launders taint entirely. Memoized on (node, value seed, shape seed, shape
	 * descriptor seed) with an optimistic cycle guard. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/747.
	 */
	private static NumpyScanResult scanForTaintedNumpySinks(CGNode node, Set<Integer> valueSeed, Set<Integer> shapeSeed,
			Map<Integer, ShapeDescriptor> shapeDescriptorSeed, CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex, Map<String, NumpyScanResult> memo) {
		Map<Integer, String> descriptorKey = new TreeMap<>();
		shapeDescriptorSeed.forEach((value, descriptor) -> descriptorKey.put(value, renderDescriptorForMemoKey(descriptor, callGraph)));
		String key = callGraph.getNumber(node) + ":" + new TreeSet<>(valueSeed) + ":" + new TreeSet<>(shapeSeed) + ":" + descriptorKey;
		NumpyScanResult cached = memo.get(key);

		if (cached != null)
			return cached;

		// Optimistic cycle guard: a recursive revisit contributes nothing new.
		memo.put(key, new NumpyScanResult(false, false, null));

		IR ir = node.getIR();

		if (ir == null)
			return new NumpyScanResult(false, false, null);

		DefUse defUse = node.getDU();
		Set<Integer> valueTainted = new HashSet<>(valueSeed);
		Set<Integer> shapeTainted = new HashSet<>(shapeSeed);
		// For each shape-tainted value, the tensor+dimensions its shape came from (absent = shape-derived but provenance lost). Seeded
		// interprocedurally: a shape-tainted argument's descriptor arrives on the corresponding parameter (issue 756).
		Map<Integer, ShapeDescriptor> shapeDescriptors = new HashMap<>(shapeDescriptorSeed);
		Deque<Integer> worklist = new ArrayDeque<>();
		worklist.addAll(valueSeed);
		worklist.addAll(shapeSeed);
		boolean sink = false;
		boolean valueEscapes = false;
		// The descriptor this node's returns carry back to the caller: set while all tracked-shape returns agree, poisoned to
		// AMBIGUOUS_DESCRIPTOR on a conflict, and dropped entirely (null) when an untracked shape is returned.
		ShapeDescriptor returnDescriptor = null;
		boolean returnedUntrackedShape = false;

		while (!worklist.isEmpty()) {
			int valueNumber = worklist.pop();
			boolean valueColored = valueTainted.contains(valueNumber);

			for (Iterator<SSAInstruction> uses = defUse.getUses(valueNumber); uses.hasNext();) {
				SSAInstruction use = uses.next();

				// A `dtype` read is a trace-time constant and launders taint entirely; a `shape` read yields shape metadata (SHAPE color)
				// covering every dimension of the read tensor.
				if (use instanceof PythonPropertyRead read && read.getObjectRef() == valueNumber) {
					String member = resolveStringConstant(node, read.getMemberRef(), pointerAnalysis);

					if (DTYPE_MEMBER_NAME.equals(member))
						continue;

					if (SHAPE_MEMBER_NAME.equals(member)) {
						for (int d = 0; d < read.getNumberOfDefs(); d++)
							colorShapeFrom(read.getDef(d), new ShapeDescriptor(node, valueNumber, null), valueTainted, shapeTainted,
									shapeDescriptors, worklist);
						continue;
					}
				}

				if (use instanceof PythonInvokeInstruction invoke) {
					if (invokesNumpyApi(node, invoke, defUse, pointerAnalysis)) {
						// A value-tainted argument is a sink (a value escape). A shape-derived argument is a sink only on a
						// provably-dynamic covered dimension; an unprovable shape (⊤ or untracked/ambiguous provenance) is permitted,
						// favoring precision. The sound decline-unless-provably-static variant is tracked on
						// https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/751.
						if (valueColored) {
							sink = true;
							valueEscapes = true;
						} else {
							ShapeDescriptor descriptor = shapeDescriptors.get(valueNumber);

							if (descriptor != null && numpyOverShapeStaticness(descriptor, tensorTypeIndex) == ShapeStaticness.DYNAMIC)
								sink = true;
						}

						continue;
					}

					// A Python `x[a:b:c]` subscript is modeled as `slice(x, a, b, c)`. Only a slice of a shape-tainted vector stays shape
					// metadata (covering the sliced dimensions); slicing a tensor VALUE yields a sub-tensor value (e.g. `boxes1[..., :2]`),
					// which must remain value-tainted so numpy over it is still a sink. Guard on the receiver being shape-, not value-,
					// tainted.
					if (!valueColored && invokesSliceBuiltin(invoke, defUse) && invoke.getNumberOfUses() > 1
							&& invoke.getUse(1) == valueNumber) {
						ShapeDescriptor base = shapeDescriptors.get(valueNumber);
						ShapeDescriptor narrowed = base == null ? null
								: narrowBySlice(base, invoke, node, defUse, pointerAnalysis, tensorTypeIndex);

						for (int d = 0; d < invoke.getNumberOfDefs(); d++)
							if (narrowed != null)
								colorShapeFrom(invoke.getDef(d), narrowed, valueTainted, shapeTainted, shapeDescriptors, worklist);
							else
								colorShape(invoke.getDef(d), valueTainted, shapeTainted, worklist);
						continue;
					}

					// tf.shape/size/rank consume a tensor and return shape metadata covering every dimension of the argument tensor.
					if (invokesShapeMetadataOp(node, invoke, defUse, pointerAnalysis)) {
						ShapeDescriptor descriptor = invoke.getNumberOfUses() > 1 ? new ShapeDescriptor(node, invoke.getUse(1), null)
								: null;

						for (int d = 0; d < invoke.getNumberOfDefs(); d++)
							if (descriptor != null)
								colorShapeFrom(invoke.getDef(d), descriptor, valueTainted, shapeTainted, shapeDescriptors, worklist);
							else
								colorShape(invoke.getDef(d), valueTainted, shapeTainted, worklist);
						continue;
					}

					// Cross into user-defined callees, carrying each tainted argument's color to the corresponding parameter slot.
					Set<Integer> valueSlots = new TreeSet<>();
					Set<Integer> shapeSlots = new TreeSet<>();

					for (int j = 1; j < invoke.getNumberOfUses(); j++) {
						int arg = invoke.getUse(j);

						if (valueTainted.contains(arg))
							valueSlots.add(j);
						else if (shapeTainted.contains(arg))
							shapeSlots.add(j);
					}

					boolean calleeValueEscapes = false;
					boolean analyzedCallee = false;
					// The descriptor the callees' returns carry to this call site: all analyzed targets must agree on a tracked one; a
					// target returning an untracked shape (or none) drops it, and disagreeing targets poison it. Dropping forfeits only
					// the return-flow descriptor: a shape-extractor call site still falls back to the all-dimensions seed anchored to
					// the argument tensor (below), which over-approximates any shape metadata the extractor derives from that tensor.
					ShapeDescriptor calleeReturnDescriptor = null;
					boolean calleeReturnDescriptorDropped = false;

					if (!valueSlots.isEmpty() || !shapeSlots.isEmpty())
						for (CGNode target : callGraph.getPossibleTargets(node, invoke.getCallSite())) {
							if (isTensorFlowNode(target))
								continue;

							IR targetIr = target.getIR();

							if (targetIr == null)
								continue;

							int[] targetParameters = targetIr.getSymbolTable().getParameterValueNumbers();
							Set<Integer> targetValueSeed = new HashSet<>();
							Set<Integer> targetShapeSeed = new HashSet<>();
							Map<Integer, ShapeDescriptor> targetDescriptorSeed = new HashMap<>();

							for (int slot : valueSlots)
								if (slot < targetParameters.length)
									targetValueSeed.add(targetParameters[slot]);

							for (int slot : shapeSlots)
								if (slot < targetParameters.length) {
									targetShapeSeed.add(targetParameters[slot]);

									// Thread the argument's shape provenance into the callee's frame, so a numpy sink there still
									// resolves staticness against the source tensor (issue 756).
									ShapeDescriptor argumentDescriptor = shapeDescriptors.get(invoke.getUse(slot));

									if (argumentDescriptor != null)
										targetDescriptorSeed.put(targetParameters[slot], argumentDescriptor);
								}

							if (targetValueSeed.isEmpty() && targetShapeSeed.isEmpty())
								continue;

							analyzedCallee = true;

							NumpyScanResult r = scanForTaintedNumpySinks(target, targetValueSeed, targetShapeSeed, targetDescriptorSeed,
									callGraph, pointerAnalysis, tensorTypeIndex, memo);

							if (r.sink())
								sink = true;
							if (r.valueEscapes())
								calleeValueEscapes = true;

							if (r.returnDescriptor() == null)
								calleeReturnDescriptorDropped = true;
							else if (calleeReturnDescriptor == null)
								calleeReturnDescriptor = r.returnDescriptor();
							else if (!calleeReturnDescriptor.equals(r.returnDescriptor()))
								calleeReturnDescriptor = AMBIGUOUS_DESCRIPTOR;
						}

					if (calleeReturnDescriptorDropped)
						calleeReturnDescriptor = null;

					// A callee is a shape extractor iff it was analyzed and no value taint escaped inside it: then its result is shape
					// metadata (SHAPE). Otherwise the call-site result is conservatively value-tainted (a value argument reached a
					// non-shape use, directly or via a library callee), and that value has escaped this node.
					boolean hasValueArg = !valueSlots.isEmpty();
					boolean shapeExtractor = analyzedCallee && !calleeValueEscapes;
					boolean resultValue = hasValueArg && !shapeExtractor;
					boolean resultShape = !resultValue && (hasValueArg || !shapeSlots.isEmpty());

					if (resultValue)
						valueEscapes = true;

					// A shape-extractor result (e.g. `get_shape_list(t)`) is the shape vector of the tensor argument it consumed, covering
					// all of that tensor's dimensions: seed the descriptor from the (first) value-tainted argument. A descriptor carried
					// by the callee's return is preferred over that all-dimensions seed - it reflects any slice narrowing inside the
					// callee (issue 756).
					ShapeDescriptor extractorDescriptor = null;

					if (resultShape && shapeExtractor && !valueSlots.isEmpty())
						extractorDescriptor = new ShapeDescriptor(node, invoke.getUse(valueSlots.iterator().next()), null);

					ShapeDescriptor resultDescriptor = calleeReturnDescriptor != null ? calleeReturnDescriptor : extractorDescriptor;

					for (int d = 0; d < invoke.getNumberOfDefs(); d++) {
						int def = invoke.getDef(d);

						if (resultValue)
							colorValue(def, valueTainted, shapeTainted, worklist);
						else if (resultShape)
							if (resultDescriptor != null)
								colorShapeFrom(def, resultDescriptor, valueTainted, shapeTainted, shapeDescriptors, worklist);
							else
								colorShape(def, valueTainted, shapeTainted, worklist);
					}

					continue;
				}

				if (use instanceof SSAReturnInstruction) {
					if (valueColored)
						valueEscapes = true;
					else {
						// A returned shape-tainted value carries its descriptor back to the caller (issue 756). Conflicting descriptors
						// across returns (or a re-scan after poisoning) merge to AMBIGUOUS; an untracked shape return drops the
						// descriptor entirely.
						ShapeDescriptor descriptor = shapeDescriptors.get(valueNumber);

						if (descriptor == null)
							returnedUntrackedShape = true;
						else if (returnDescriptor == null)
							returnDescriptor = descriptor;
						else if (!returnDescriptor.equals(descriptor))
							returnDescriptor = AMBIGUOUS_DESCRIPTOR;
					}
					continue;
				}

				// Any other instruction is a non-shape use: a value-tainted operand escapes (its value flows onward), and the definitions
				// inherit the color (value dominates).
				if (valueColored)
					valueEscapes = true;

				for (int d = 0; d < use.getNumberOfDefs(); d++) {
					int def = use.getDef(d);

					if (valueColored)
						colorValue(def, valueTainted, shapeTainted, worklist);
					else
						colorShape(def, valueTainted, shapeTainted, worklist);
				}
			}
		}

		NumpyScanResult result = new NumpyScanResult(sink, valueEscapes, returnedUntrackedShape ? null : returnDescriptor);
		memo.put(key, result);
		return result;
	}

	/**
	 * A stable rendering of {@code descriptor} for the scan's memo key: the source node's call-graph number, the source tensor's value
	 * number, and the covered dimensions ({@code *} = all dimensions; {@code !} = ambiguous provenance).
	 */
	private static String renderDescriptorForMemoKey(ShapeDescriptor descriptor, CallGraph callGraph) {
		if (descriptor.sourceNode() == null)
			return "!";

		return callGraph.getNumber(descriptor.sourceNode()) + "@" + descriptor.sourceTensor()
				+ (descriptor.dims() == null ? ":*" : ":" + new TreeSet<>(descriptor.dims()));
	}

	/** Colors {@code value} with the value taint (which dominates any shape taint) and enqueues it when newly tainted. */
	private static void colorValue(int value, Set<Integer> valueTainted, Set<Integer> shapeTainted, Deque<Integer> worklist) {
		shapeTainted.remove(value);

		if (valueTainted.add(value))
			worklist.push(value);
	}

	/**
	 * Colors {@code value} with the shape taint and enqueues it when newly tainted, unless it is already value-tainted (value dominates).
	 */
	private static void colorShape(int value, Set<Integer> valueTainted, Set<Integer> shapeTainted, Deque<Integer> worklist) {
		if (valueTainted.contains(value))
			return;

		if (shapeTainted.add(value))
			worklist.push(value);
	}

	/**
	 * A poisoned shape descriptor: recorded when a value receives shape provenance from two conflicting sources (e.g. a phi merge), so the
	 * scan cannot rely on a single source's staticness. Its {@code null} source node makes {@link #numpyOverShapeStaticness} report
	 * {@link ShapeStaticness#TOP} (unprovable), the conservative outcome.
	 */
	private static final ShapeDescriptor AMBIGUOUS_DESCRIPTOR = new ShapeDescriptor(null, -1, null);

	/**
	 * Colors {@code value} with the shape taint and records the {@code descriptor} of the tensor dimensions it covers, so a downstream
	 * numpy sink can consult the source tensor's per-dim {@link TensorType}. Unless {@code value} is already value-tainted (value
	 * dominates). If {@code value} already carries a different descriptor, its provenance is ambiguous (two shape sources merged into it),
	 * so it is poisoned to {@link #AMBIGUOUS_DESCRIPTOR} rather than letting the last writer win.
	 */
	private static void colorShapeFrom(int value, ShapeDescriptor descriptor, Set<Integer> valueTainted, Set<Integer> shapeTainted,
			Map<Integer, ShapeDescriptor> shapeDescriptors, Deque<Integer> worklist) {
		if (valueTainted.contains(value))
			return;

		ShapeDescriptor existing = shapeDescriptors.get(value);

		// Poison on conflicting provenance (two shape sources merged into this value); an already-poisoned value stays poisoned.
		ShapeDescriptor updated = existing == AMBIGUOUS_DESCRIPTOR || existing != null && !existing.equals(descriptor)
				? AMBIGUOUS_DESCRIPTOR
				: descriptor;

		shapeDescriptors.put(value, updated);

		// Re-enqueue on a first taint OR a descriptor change, so a value already popped is re-scanned with the new (possibly poisoned)
		// descriptor rather than leaving a sink decided on the stale one. Descriptors are monotone (unset -> concrete -> ambiguous), so
		// this terminates.
		boolean newlyTainted = shapeTainted.add(value);

		if (newlyTainted || !updated.equals(existing))
			worklist.push(value);
	}

	/**
	 * Narrows {@code base} (a shape vector covering {@code base}'s dimensions) by the slice {@code slice(x, start, stop, step)} in
	 * {@code invoke}, resolving the bounds to integer constants and applying Python slice semantics against the source tensor's rank. When
	 * the rank is statically unresolvable (a ⊤ context or contexts of disagreeing rank), a pure prefix ({@code [:k]}) or suffix
	 * ({@code [-k:]}) slice is still narrowed rank-independently ({@link #resolveRankFreeSliceDims}), deferring resolution to the sink's
	 * per-context verdict - the recovery https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/761 needs, since the corpus
	 * {@code einsum_via_matmul} suffix slice otherwise drops the descriptor. Returns {@code null} when a bound cannot be resolved, the rank
	 * is needed but unresolvable, or {@code base} already covers a proper subset (nested slicing is not composed), so the caller falls back
	 * to an untracked shape taint.
	 */
	private static ShapeDescriptor narrowBySlice(ShapeDescriptor base, PythonInvokeInstruction invoke, CGNode node, DefUse defUse,
			PointerAnalysis<InstanceKey> pointerAnalysis, Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex) {
		if (base.dims() != null)
			return null; // only slices of a full shape vector are modeled; nested slicing drops the descriptor (unprovable, so permitted
							// under the current precision-favoring policy).

		Integer start = invoke.getNumberOfUses() > 2 ? resolveIntConstant(node, invoke.getUse(2), defUse, pointerAnalysis) : null;
		Integer stop = invoke.getNumberOfUses() > 3 ? resolveIntConstant(node, invoke.getUse(3), defUse, pointerAnalysis) : null;
		Integer step = invoke.getNumberOfUses() > 4 ? resolveIntConstant(node, invoke.getUse(4), defUse, pointerAnalysis) : null;

		// A no-op slice ([:], the Python copy idiom) covers every dimension regardless of rank: preserve the base descriptor.
		if ((start == null || start == 0) && stop == null && (step == null || step == 1))
			return base;

		int rank = sourceRank(base, tensorTypeIndex);
		Set<Integer> dims = rank < 0 ? resolveRankFreeSliceDims(start, stop, step) : resolveSliceDims(start, stop, step, rank);

		return dims == null ? null : new ShapeDescriptor(base.sourceNode(), base.sourceTensor(), dims);
	}

	/**
	 * The maximum number of dimensions a rank-free slice may cover. Tensor ranks are tiny in practice; without a rank to clamp against, a
	 * pathologically large resolved bound would otherwise enumerate an index per unit of the bound, so anything beyond this cap is treated
	 * as unresolvable instead.
	 */
	private static final int MAX_RANK_FREE_SLICE_EXTENT = 32;

	/**
	 * The dimension indices selected by a Python slice {@code [start:stop:step]} of a shape vector whose rank is statically unresolvable,
	 * or {@code null} if the slice's coverage depends on the rank (or exceeds {@link #MAX_RANK_FREE_SLICE_EXTENT}). Only two
	 * rank-independent forms exist: a pure prefix {@code [:k]} ({@code k >= 0}; {@code [:0]} covers nothing) covers the absolute indices
	 * {@code 0..k-1}, and a pure suffix {@code [-k:]} covers the last {@code k} dimensions, encoded as the negative indices {@code -k..-1}
	 * that {@link #numpyOverShapeStaticness} resolves against each context's own rank. An index beyond a context's rank contributes no
	 * element there and is ignored by the verdict, mirroring Python's slice clamping.
	 */
	private static Set<Integer> resolveRankFreeSliceDims(Integer start, Integer stop, Integer step) {
		if (step != null && step != 1)
			return null; // only unit-stride slices are modeled precisely.

		Set<Integer> dims = new TreeSet<>();

		// A pure prefix [:k]: absolute indices 0..k-1 (none for [:0]), no rank needed.
		if ((start == null || start == 0) && stop != null && stop >= 0 && stop <= MAX_RANK_FREE_SLICE_EXTENT) {
			for (int i = 0; i < stop; i++)
				dims.add(i);

			return dims;
		}

		// A pure suffix [-k:]: the last k dimensions, encoded as negative indices resolved per-context at the sink.
		if (start != null && start < 0 && start >= -MAX_RANK_FREE_SLICE_EXTENT && stop == null) {
			for (int i = start; i < 0; i++)
				dims.add(i);

			return dims;
		}

		return null;
	}

	/**
	 * The rank of the source tensor of {@code descriptor} if all its analysis contexts agree on a concrete (non-⊤) rank, else {@code -1}.
	 */
	private static int sourceRank(ShapeDescriptor descriptor, Map<CGNode, Map<Integer, Set<TensorType>>> tensorTypeIndex) {
		Set<TensorType> types = lookupTensorTypes(descriptor.sourceNode(), descriptor.sourceTensor(), tensorTypeIndex);
		int rank = -1;

		for (TensorType type : types) {
			if (type.getDims() == null)
				return -1;

			if (rank == -1)
				rank = type.getDims().size();
			else if (rank != type.getDims().size())
				return -1;
		}

		return rank;
	}

	/** True iff {@code invoke}'s callee resolves to the numpy/scipy namespace by attribute-chain walk or import-alias fallback. */
	private static boolean invokesNumpyApi(CGNode node, PythonInvokeInstruction invoke, DefUse defUse,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		return isNumpyRooted(node, invoke.getUse(0), defUse, pointerAnalysis);
	}

	/** Fully-qualified names of the TensorFlow shape-metadata ops whose results are shape (not value) tainted. */
	private static final Set<String> SHAPE_METADATA_FQNS = Set.of("tensorflow.shape", "tensorflow.size", "tensorflow.rank");

	/** True iff {@code invoke}'s callee is a TensorFlow shape-metadata op ({@code tf.shape}/{@code size}/{@code rank}). */
	private static boolean invokesShapeMetadataOp(CGNode node, PythonInvokeInstruction invoke, DefUse defUse,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		String fqn = resolveCalleeFullyQualifiedName(node, invoke.getUse(0), defUse, pointerAnalysis);
		return fqn != null && SHAPE_METADATA_FQNS.contains(fqn);
	}

	/** True iff {@code use}'s attribute chain roots at the numpy/scipy module (points-to preferred, import alias as fallback). */
	private static boolean isNumpyRooted(CGNode node, int use, DefUse defUse, PointerAnalysis<InstanceKey> pointerAnalysis) {
		if (isNumpyModule(node, use, defUse, pointerAnalysis))
			return true;

		SSAInstruction def = defUse.getDef(use);

		if (def instanceof PythonPropertyRead read)
			return isNumpyRooted(node, read.getObjectRef(), defUse, pointerAnalysis);

		return false;
	}

	/** True iff {@code use} refers to the numpy/scipy module. Prefers points-to; falls back to the import alias on a global read. */
	private static boolean isNumpyModule(CGNode node, int use, DefUse defUse, PointerAnalysis<InstanceKey> pointerAnalysis) {
		for (String prefix : NUMPY_MODULE_TYPE_NAME_PREFIXES)
			if (pointsToType(node, use, pointerAnalysis, prefix, false))
				return true;

		return defUse.getDef(use) instanceof AstGlobalRead global && NUMPY_MODULE_GLOBAL_NAMES.contains(global.getGlobalName());
	}

	/** True iff any value defined by {@code instruction} is typed as a tensor (its pointer key is in {@code tensorTypedKeys}). */
	private static boolean definesTensor(CGNode node, SSAInstruction instruction, PointerAnalysis<InstanceKey> pointerAnalysis,
			Set<PointerKey> tensorTypedKeys) {
		for (int i = 0; i < instruction.getNumberOfDefs(); i++) {
			PointerKey pointerKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, instruction.getDef(i));

			if (tensorTypedKeys.contains(pointerKey))
				return true;
		}

		return false;
	}

	private static boolean invokesTensorFlowOp(CGNode node, PythonInvokeInstruction invoke, DefUse defUse,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		int callee = invoke.getUse(0); // the invoked function (a reference to the callee).

		// Modeled op: the callee points to a TensorFlow function instance.
		if (pointsToType(node, callee, pointerAnalysis, TENSORFLOW_FUNCTION_TYPE_NAME_PREFIX, false))
			return true;

		// Unmodeled op: resolve the callee's fully-qualified name from the IR and test it against the TensorFlow namespace.
		String fqn = resolveCalleeFullyQualifiedName(node, callee, defUse, pointerAnalysis);

		return fqn != null && fqn.startsWith(TENSORFLOW_FQN_PREFIX) && NON_OP_TENSORFLOW_FQN_PREFIXES.stream().noneMatch(fqn::startsWith);
	}

	/**
	 * Resolves the fully-qualified name of the callee referenced by {@code use} (e.g. {@code tensorflow.train.Feature}) by walking the
	 * attribute (property-read) chain in {@code node}'s IR back to the TensorFlow module, or {@code null} when the chain does not root at
	 * the TensorFlow module.
	 */
	private static String resolveCalleeFullyQualifiedName(CGNode node, int use, DefUse defUse,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		SSAInstruction def = defUse.getDef(use);

		if (def instanceof PythonPropertyRead read) {
			String member = resolveStringConstant(node, read.getMemberRef(), pointerAnalysis);

			if (member == null)
				return null;

			int objectRef = read.getObjectRef();

			// If the receiver is the TensorFlow module, we have reached the root of the chain.
			if (isTensorFlowModule(node, objectRef, defUse, pointerAnalysis))
				return TENSORFLOW_FQN_PREFIX + member;

			// Otherwise, continue up the attribute chain, e.g. tf.train.Feature.
			String base = resolveCalleeFullyQualifiedName(node, objectRef, defUse, pointerAnalysis);

			return base == null ? null : base + "." + member;
		}

		return null;
	}

	/**
	 * True iff {@code use} refers to the TensorFlow module. Prefers points-to (precise), but falls back to the module import alias on a
	 * global read: a module-level {@code import tensorflow as tf} frequently leaves the {@code tf} global with an empty points-to set in a
	 * given call-graph context, which would otherwise strand every {@code tf.<op>} call as unresolvable and misreport the enclosing
	 * function as performing no tensor computation. The fallback keys off the global's name, so a non-TensorFlow global that happens to be
	 * named {@code tf} would match; that is incompleteness-safe here, since over-recognizing a tensor op only lets an eager function
	 * hybridize (the pre-benefit-signal default). See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/709.
	 */
	private static boolean isTensorFlowModule(CGNode node, int use, DefUse defUse, PointerAnalysis<InstanceKey> pointerAnalysis) {
		if (pointsToType(node, use, pointerAnalysis, TENSORFLOW_MODULE_TYPE_NAME, true))
			return true;

		return defUse.getDef(use) instanceof AstGlobalRead global && TENSORFLOW_MODULE_GLOBAL_NAMES.contains(global.getGlobalName());
	}

	/** The string value of a {@link ConstantKey} in {@code use}'s points-to set, or {@code null} if none. */
	private static String resolveStringConstant(CGNode node, int use, PointerAnalysis<InstanceKey> pointerAnalysis) {
		PointerKey pointerKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, use);

		for (InstanceKey instanceKey : pointerAnalysis.getPointsToSet(pointerKey))
			if (instanceKey instanceof ConstantKey<?> constantKey && constantKey.getValue() instanceof String value)
				return value;

		return null;
	}

	/**
	 * True iff any instance in {@code use}'s points-to set has a concrete type whose name equals (when {@code exact}) or starts with (when
	 * not {@code exact}) {@code typeName}.
	 */
	private static boolean pointsToType(CGNode node, int use, PointerAnalysis<InstanceKey> pointerAnalysis, String typeName,
			boolean exact) {
		PointerKey pointerKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, use);

		for (InstanceKey instanceKey : pointerAnalysis.getPointsToSet(pointerKey)) {
			String name = instanceKey.concreteType().getReference().getName().toString();

			if (exact ? name.equals(typeName) : name.startsWith(typeName))
				return true;
		}

		return false;
	}

	public static boolean isContainerType(TypeReference reference) {
		return reference.equals(PythonTypes.dict) || reference.equals(PythonTypes.enumerate) || reference.equals(PythonTypes.list)
				|| reference.equals(PythonTypes.set) || reference.equals(PythonTypes.tuple);
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static void addEntryPoints(Collection target, Iterable source) {
		for (Object entryPoint : source)
			if (target.add(entryPoint))
				LOG.info("Adding entrypoint: " + entryPoint);
	}

	public static Set<NewSiteReference> getAllNewSiteReferences(int use, DefUse du) {
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

	public static Set<String> getAllParentNames(HierarchyNodeModel hierarchyNode, boolean onlyLastSegment) {
		Set<String> ret = new HashSet<>();

		if (hierarchyNode != null) {
			if (hierarchyNode.ast != null)
				ret.addAll(NodeUtils.getParentNames(hierarchyNode.ast, onlyLastSegment));

			for (HierarchyNodeModel parenNode : hierarchyNode.parents)
				ret.addAll(getAllParentNames(parenNode, onlyLastSegment));
		}

		return ret;
	}
}
