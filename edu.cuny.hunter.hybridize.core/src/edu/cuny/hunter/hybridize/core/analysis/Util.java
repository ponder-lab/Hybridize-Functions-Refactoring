package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

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
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
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
	static boolean isTensorFlowNode(CGNode node) {
		String name = node.getMethod().getDeclaringClass().getReference().getName().toString();
		return name.startsWith(TENSORFLOW_MODULE_TYPE_NAME);
	}

	/** Method names whose invocation is only valid in eager execution (e.g. {@code Tensor.numpy()}). */
	private static final Set<String> EAGER_ONLY_METHOD_NAMES = Set.of("numpy");

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
	static String resolveCalleeFullyQualifiedName(CGNode node, int use, DefUse defUse, PointerAnalysis<InstanceKey> pointerAnalysis) {
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
	static String resolveStringConstant(CGNode node, int use, PointerAnalysis<InstanceKey> pointerAnalysis) {
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
	static boolean pointsToType(CGNode node, int use, PointerAnalysis<InstanceKey> pointerAnalysis, String typeName, boolean exact) {
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
