package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static edu.cuny.hunter.hybridize.core.analysis.Information.INPUT_SIGNATURE_INFERENCE;
import static edu.cuny.hunter.hybridize.core.analysis.Information.SPECULATIVE_ANALYSIS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_PRIMITIVE_PARAMETERS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P1;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P2;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P3;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P4;
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P6;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.CONVERT_EAGER_FUNCTION_TO_HYBRID;
import static edu.cuny.hunter.hybridize.core.analysis.Refactoring.OPTIMIZE_HYBRID_FUNCTION;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.CONVERT_TO_EAGER;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.CONVERT_TO_HYBRID;
import static edu.cuny.hunter.hybridize.core.analysis.Transformation.RECONFIGURE;
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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
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
import org.python.pydev.parser.jython.ast.Assign;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.Module;
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.Num;
import org.python.pydev.parser.jython.ast.Tuple;
import org.python.pydev.parser.jython.ast.VisitorBase;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.jython.ast.expr_contextType;
import org.python.pydev.parser.jython.ast.keywordType;
import org.python.pydev.parser.jython.ast.name_contextType;
import org.python.pydev.parser.jython.ast.stmtType;
import org.python.pydev.parser.visitors.NodeUtils;

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.ibm.wala.cast.ipa.callgraph.AstGlobalPointerKey;
import com.ibm.wala.cast.ipa.callgraph.ScopeMappingInstanceKeys.ScopeMappingInstanceKey;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.AppliedDTypeCoercion;
import com.ibm.wala.cast.python.ml.analysis.AppliedDTypeCoercion.Resolution;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
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
import com.ibm.wala.ipa.modref.DelegatingExtendedHeapModel;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.ipa.modref.ModRef;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Iterator2Iterable;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import com.python.pydev.analysis.refactoring.refactorer.Refactorer;

import edu.cuny.hunter.hybridize.core.analysis.InferenceResult.AbsenceReason;
import edu.cuny.hunter.hybridize.core.utils.RefactoringAvailabilityTester;
import edu.cuny.hunter.hybridize.core.wala.ml.PythonModRefWithBuiltinFunctions;

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
		 * The {@link InputSignature} parsed from an {@code input_signature=[tf.TensorSpec(...)]} argument supplied to this
		 * {@link Function}'s {@code @tf.function} decorator (in either keyword or positional form), or {@link Optional#empty} when none was
		 * supplied or its content could not be fully modeled. See {@link #getSuppliedInputSignature()} for the presence/parse contract.
		 */
		private Optional<InputSignature> suppliedInputSignature = Optional.empty();

		/**
		 * The AST expression node of the supplied {@code input_signature} argument's value—the {@code [tf.TensorSpec(...)]} list/tuple, or
		 * the bare name referencing one (#834)—whether supplied by keyword or by position, or {@code null} when none was supplied. Always
		 * the node at the decorator site, never a resolved referent. Retained for reporting and for the sanctioned future find-and-fix
		 * signature rewrite (issue 808); the refactoring itself never edits an existing signature. See
		 * {@link #getSuppliedInputSignatureNode()}.
		 */
		private exprType suppliedInputSignatureNode;

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
					for (int i = 0; i < limit; i++) {
						this.markParam(TF_FUNCTION_POSITIONAL_PARAMS[i]);

						// Parse the content of a positionally supplied `input_signature` (e.g. `@tf.function(None, [tf.TensorSpec(...)])`,
						// where index 1 binds to `input_signature`). Python forbids passing the same parameter both positionally and by
						// keyword, so this and the keyword branch below cannot both set the field for a well-formed decorator.
						if (INPUT_SIGNATURE.equals(TF_FUNCTION_POSITIONAL_PARAMS[i])) {
							this.suppliedInputSignatureNode = positionalArgs[i];
							this.suppliedInputSignature = this.modelSuppliedInputSignature(positionalArgs[i]);
						}
					}
				}

				// Process keyword arguments. Keyword args are unordered; each carries its parameter name
				// directly. A user can mix positional and keyword in the same call (e.g.
				// `@tf.function(my_func, autograph=False)`); both branches mark the same fields.
				keywordType[] keywords = callFunction.keywords;
				for (keywordType keyword : keywords)
					if (keyword.arg instanceof NameTok) {
						NameTok name = (NameTok) keyword.arg;
						this.markParam(name.id);

						// Parse the content of a keyword-form `input_signature=[tf.TensorSpec(...)]`.
						if (INPUT_SIGNATURE.equals(name.id)) {
							this.suppliedInputSignatureNode = keyword.value;
							this.suppliedInputSignature = this.modelSuppliedInputSignature(keyword.value);
						}
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
		 * The {@link InputSignature} parsed from an {@code input_signature=[tf.TensorSpec(...)]} argument supplied to this
		 * {@link Function}'s {@code @tf.function} decorator, in either keyword or positional form.
		 * <p>
		 * This getter and {@link #hasInputSignatureParam()} together carry a three-state contract that downstream reconfiguration must
		 * honor to avoid clobbering a user's signature:
		 * <ul>
		 * <li>{@code hasInputSignatureParam() == false}: no {@code input_signature} was supplied. Inference may synthesize one and write
		 * it.
		 * <li>{@code hasInputSignatureParam() == true} and the result is <em>present</em>: a signature was supplied <em>and</em> fully
		 * modeled. A replace-existing-signature decision can compare it against the inferred signature.
		 * <li>{@code hasInputSignatureParam() == true} and the result is <em>empty</em>: a signature was supplied but could not be fully
		 * modeled (an unsupported {@code TensorSpec} subtype such as {@code RaggedTensorSpec}/{@code SparseTensorSpec}—tracked by #524 and
		 * #533—or malformed content). It must be left as-is, never overwritten.
		 * </ul>
		 * An empty result therefore does <em>not</em> mean "no signature supplied"; callers must consult {@link #hasInputSignatureParam()}
		 * for that distinction. Both the keyword form {@code @tf.function(input_signature=[...])} and the positional form
		 * {@code @tf.function(None, [...])} are parsed.
		 *
		 * @return The parsed supplied input signature, or {@link Optional#empty} when none was supplied or its content could not be fully
		 *         modeled.
		 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/557">Issue 557</a>
		 */
		public Optional<InputSignature> getSuppliedInputSignature() {
			return this.suppliedInputSignature;
		}

		/**
		 * The AST expression node of the supplied {@code input_signature} value (the {@code [tf.TensorSpec(...)]} list/tuple, or the bare
		 * name referencing one; #834), or {@code null} when none was supplied. Always the node at the decorator site, never a resolved
		 * referent. Retained for reporting and for the sanctioned future find-and-fix signature rewrite (issue 808); the refactoring itself
		 * never edits an existing signature.
		 *
		 * @return The supplied {@code input_signature} value node, or {@code null}.
		 */
		public exprType getSuppliedInputSignatureNode() {
			return this.suppliedInputSignatureNode;
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

		/**
		 * Parse the value of an {@code input_signature=...} argument (keyword or positional) into an {@link InputSignature}. The value must
		 * be a list or tuple of {@code tf.TensorSpec(...)} calls; each element is reduced to a {@link TensorType} via
		 * {@link #parseTensorSpec}. The parse is all-or-nothing: if any element cannot be fully modeled (an unsupported subtype, a
		 * non-{@code TensorSpec} call, or malformed content), the whole signature is dropped to {@link Optional#empty} rather than
		 * producing a partial signature that downstream replace-existing-signature logic could not trust. A well-formed empty list/tuple
		 * ({@code input_signature=[]}, a no-arg function) is itself fully modeled and parses to a present, empty {@link InputSignature}.
		 *
		 * @param value The expression bound to {@code input_signature}, whether by keyword or by position.
		 * @return The parsed signature, or {@link Optional#empty} if the value is not a list/tuple of fully modeled {@code TensorSpec}s.
		 */
		private static Optional<InputSignature> parseSuppliedInputSignature(exprType value) {
			exprType[] elements;
			if (value instanceof org.python.pydev.parser.jython.ast.List)
				elements = ((org.python.pydev.parser.jython.ast.List) value).elts;
			else if (value instanceof Tuple)
				elements = ((Tuple) value).elts;
			else
				return Optional.empty();

			// A well-formed empty list/tuple is `input_signature=[]` (a no-arg function); it parses to an empty—but present—signature
			// rather than dropping to empty, which the contract reserves for a supplied signature that cannot be modeled.
			List<TensorType> parameterTypes = new ArrayList<>(elements == null ? 0 : elements.length);
			if (elements != null)
				for (exprType element : elements) {
					Optional<TensorType> tensorType = parseTensorSpec(element);
					if (tensorType.isEmpty())
						return Optional.empty();
					parameterTypes.add(tensorType.get());
				}

			return Optional.of(InputSignature.ofSingles(parameterTypes));
		}

		/**
		 * Model the value supplied to {@code input_signature}: parse it directly when it is a literal list/tuple, and otherwise, when it is
		 * a bare name, resolve the name to the module-level literal it references (#834) and parse that instead. Resolution is single-level
		 * (a name whose referent is itself a name is not chased) and inherits {@link #parseSuppliedInputSignature}'s all-or-nothing
		 * contract: a name that does not resolve to a fully modeled literal leaves the signature unmodeled, exactly as before #834.
		 *
		 * @param value The expression bound to {@code input_signature}, whether by keyword or by position.
		 * @return The parsed signature, or {@link Optional#empty} if neither the value nor its unambiguous module-level referent is a fully
		 *         modeled literal.
		 */
		private Optional<InputSignature> modelSuppliedInputSignature(exprType value) {
			Optional<InputSignature> parsed = parseSuppliedInputSignature(value);

			if (parsed.isPresent() || !(value instanceof Name name))
				return parsed;

			return this.resolveModuleLevelLiteral(name).flatMap(HybridizationParameters::parseSuppliedInputSignature);
		}

		/**
		 * Resolve a bare name supplied as the {@code input_signature} value to the module-level literal it references. The resolution is
		 * deliberately conservative: it succeeds only when the name's identifier is bound exactly once in the entire containing module, by
		 * a single-target module-level assignment. Under that sole-binding rule, Python's scoping cannot bind the decorator expression to
		 * any other value: a decorator evaluates in its enclosing (class-body or module) scope, and with no competing binding anywhere,
		 * both fall back to the module-level one. Attribute stores (e.g. {@code self.x = ...}) and call-site keyword names are not name
		 * bindings and do not compete; reassignments, shadowing definitions, imports, loop targets, deletions, and parameters all count as
		 * competing bindings and decline the resolution.
		 *
		 * @param reference The bare name supplied as the {@code input_signature} value.
		 * @return The value of the module-level assignment the name references, or {@link Optional#empty} when the name does not resolve
		 *         unambiguously.
		 */
		private Optional<exprType> resolveModuleLevelLiteral(Name reference) {
			SimpleNode root = Function.this.getFunctionDefinition().getContainingModule();

			if (!(root instanceof Module module) || module.body == null)
				return Optional.empty();

			String identifier = reference.id;
			int[] bindings = { 0 };

			try {
				module.accept(new VisitorBase() {

					@Override
					protected Object unhandled_node(SimpleNode node) {
						// `Name` bindings cover assignment targets, augmented/named (walrus) stores, deletions, loop and
						// comprehension targets, and parameters; `NameTok` bindings cover function, class, import, global/nonlocal,
						// and match-pattern names. Call-site keyword names and attribute names are the two `NameTok` roles that are
						// not bindings.
						if (node instanceof Name name && identifier.equals(name.id) && name.ctx != expr_contextType.Load
								&& name.ctx != expr_contextType.AugLoad)
							++bindings[0];
						else if (node instanceof NameTok tok && identifier.equals(tok.id) && tok.ctx != name_contextType.KeywordName
								&& tok.ctx != name_contextType.Attrib)
							++bindings[0];

						return null;
					}

					@Override
					public void traverse(SimpleNode node) throws Exception {
						node.traverse(this);
					}
				});
			} catch (Exception e) {
				LOG.error("Failed to traverse the module containing " + Function.this + " while resolving a name-referenced input"
						+ " signature.", e);
				return Optional.empty();
			}

			if (bindings[0] != 1)
				return Optional.empty();

			// The sole binding must be a single-target module-level assignment to the identifier; its value is the resolution. If the
			// one binding sits anywhere else (e.g., inside a function, where it cannot feed the decorator), no such statement exists
			// and the resolution declines.
			for (stmtType statement : module.body)
				if (statement instanceof Assign assign && assign.targets != null && assign.targets.length == 1
						&& assign.targets[0] instanceof Name target && identifier.equals(target.id))
					return Optional.ofNullable(assign.value);

			return Optional.empty();
		}

		/**
		 * Parse a single {@code tf.TensorSpec(shape, dtype)} call into a {@link TensorType}. The call's callee must name {@code TensorSpec}
		 * exactly; {@code RaggedTensorSpec}/{@code SparseTensorSpec} and any other callee return {@link Optional#empty} because the current
		 * {@link InputSignature} model cannot represent them (tracked by #524 and #533). Both the positional form
		 * {@code TensorSpec(shape, dtype)} and the keyword form {@code TensorSpec(shape=..., dtype=...)} are accepted.
		 *
		 * @param element A candidate {@code TensorSpec} expression from the supplied list/tuple.
		 * @return The reduced {@link TensorType}, or {@link Optional#empty} if {@code element} is not a fully modeled {@code TensorSpec}
		 *         call.
		 */
		private static Optional<TensorType> parseTensorSpec(exprType element) {
			if (!(element instanceof Call))
				return Optional.empty();

			Call call = (Call) element;
			// `getRepresentationString` returns the trailing attribute, so `tf.TensorSpec` and a bare `TensorSpec` both yield
			// "TensorSpec"; `RaggedTensorSpec`/`SparseTensorSpec` yield their own names and are rejected below.
			if (!"TensorSpec".equals(NodeUtils.getRepresentationString(call.func)))
				return Optional.empty();

			exprType shapeExpr = argumentValue(call, 0, "shape");
			exprType dtypeExpr = argumentValue(call, 1, "dtype");
			if (shapeExpr == null || dtypeExpr == null)
				return Optional.empty();

			Optional<DType> dtype = parseDType(dtypeExpr);
			if (dtype.isEmpty())
				return Optional.empty();

			return parseShape(shapeExpr).map(shape -> new TensorType(dtype.get(), shape.orElse(null)));
		}

		/**
		 * Resolve a {@code TensorSpec} argument by either positional index or keyword name. Positional arguments take precedence, matching
		 * Python's binding rules; if no positional argument occupies {@code position}, the keyword arguments are searched for {@code name}.
		 *
		 * @param call The {@code TensorSpec} call.
		 * @param position The positional index of the argument.
		 * @param name The keyword name of the argument.
		 * @return The bound expression, or {@code null} when neither form supplies the argument.
		 */
		private static exprType argumentValue(Call call, int position, String name) {
			if (call.args != null && position < call.args.length)
				return call.args[position];

			if (call.keywords != null)
				for (keywordType keyword : call.keywords)
					if (keyword.arg instanceof NameTok && name.equals(((NameTok) keyword.arg).id))
						return keyword.value;

			return null;
		}

		/**
		 * Parse a {@code TensorSpec} shape expression into a dimension list. A list/tuple yields one {@link Dimension} per element:
		 * {@code None} becomes {@link DynamicDim#INSTANCE} and an integer literal becomes a {@link NumericDim}; any other element fails the
		 * parse. A bare {@code None} (i.e., {@code shape=None}) yields {@link Optional#empty} dims, the shape-&#8868; encoding
		 * {@link Function#inferSpec} and {@link InputSignature#toTensorSpecList} already use for unknown rank.
		 *
		 * @param shapeExpr The expression bound to the {@code shape} argument.
		 * @return An {@link Optional} holding the dimension list (empty inner {@link Optional} for {@code shape=None}), or
		 *         {@link Optional#empty} (outer) when the shape cannot be modeled.
		 */
		private static Optional<Optional<List<Dimension<?>>>> parseShape(exprType shapeExpr) {
			if (shapeExpr instanceof Name && "None".equals(((Name) shapeExpr).id))
				// Shape-⊤: unknown rank. Represented as null dims downstream.
				return Optional.of(Optional.empty());

			exprType[] elements;
			if (shapeExpr instanceof org.python.pydev.parser.jython.ast.List)
				elements = ((org.python.pydev.parser.jython.ast.List) shapeExpr).elts;
			else if (shapeExpr instanceof Tuple)
				elements = ((Tuple) shapeExpr).elts;
			else
				return Optional.empty();

			List<Dimension<?>> dimensions = new ArrayList<>(elements == null ? 0 : elements.length);
			if (elements != null)
				for (exprType element : elements) {
					if (element instanceof Name && "None".equals(((Name) element).id)) {
						dimensions.add(DynamicDim.INSTANCE);
						continue;
					}
					if (element instanceof Num) {
						try {
							dimensions.add(new NumericDim(Integer.valueOf(((Num) element).num.trim())));
							continue;
						} catch (NumberFormatException _) {
							return Optional.empty();
						}
					}
					return Optional.empty();
				}

			return Optional.of(Optional.of(dimensions));
		}

		/**
		 * Parse a {@code TensorSpec} dtype expression (e.g., {@code tf.float32} or a bare {@code float32}) into a {@link DType}. The
		 * trailing attribute name is upper-cased and resolved against {@link DType#valueOf}; this inverts {@link InputSignature}'s
		 * lower-casing of {@link DType#name()}. An unrecognized name yields {@link Optional#empty}.
		 *
		 * @param dtypeExpr The expression bound to the {@code dtype} argument.
		 * @return The resolved {@link DType}, or {@link Optional#empty} when the name is not a modeled dtype.
		 */
		private static Optional<DType> parseDType(exprType dtypeExpr) {
			String name = NodeUtils.getRepresentationString(dtypeExpr);
			if (name == null)
				return Optional.empty();
			try {
				return Optional.of(DType.valueOf(name.toUpperCase(Locale.ROOT)));
			} catch (IllegalArgumentException _) {
				return Optional.empty();
			}
		}
	}

	private static Map<MethodReference, Map<InstanceKey, Map<CallGraph, Boolean>>> creationsCache = Maps.newHashMap();

	/**
	 * Per-node direct (non-transitive) mod sets, shared across {@link Function}s since the closure walks revisit the same nodes. Concurrent
	 * because functions may be processed in parallel.
	 */
	private static final Map<CGNode, Set<PointerKey>> directModCache = new ConcurrentHashMap<>();

	private static final ILog LOG = getLog(Function.class);

	public static final String PLUGIN_ID = FrameworkUtil.getBundle(Function.class).getSymbolicName();

	/**
	 * Containing {@link File}s that have had an import statement auto-injected during transformation, mapped to the bare TensorFlow names
	 * that injection brought into scope (always {@code function}, plus {@code TensorSpec} and the inferred signature's dtype constants when
	 * input-signature emission applies). The first hybridizable function in a file to need an injected import fixes the injected line;
	 * later functions in the same file reuse the recorded name set, so their emission gate ({@link #computeInputSignatureKeyword}) sees
	 * exactly what is in scope.
	 */
	private static Map<File, Set<String>> autoInjectedImportNames = new HashMap<>();

	/**
	 * Per containing {@link File}, the union of dtype constant names (e.g. {@code float32}, {@code int32}) required by the inferred input
	 * signatures of every function in the file that will be converted to hybrid. Computed by {@link #planAutoInjectedImports} before
	 * transformation so that {@link #convertToHybrid}'s auto-injected import line brings every such function's dtypes into scope, not just
	 * those of the first function processed in the file (#588).
	 */
	private static Map<File, Set<String>> fileInferredDTypeNames = new HashMap<>();

	/**
	 * Per containing {@link File}, the union of spec-type constructor names (e.g. {@code TensorSpec}, {@code RaggedTensorSpec}) required by
	 * the inferred input signatures of every function in the file that will be converted to hybrid. Computed by
	 * {@link #planAutoInjectedImports} alongside {@link #fileInferredDTypeNames} so that {@link #convertToHybrid}'s auto-injected import
	 * line brings every such function's spec types into scope: a ragged parameter needs {@code RaggedTensorSpec}, which {@code TensorSpec}
	 * being injected does not imply, so without this a ragged signature would be gated off emission and left with a bare {@code @function}
	 * (#524).
	 */
	private static Map<File, Set<String>> fileInferredSpecTypeNames = new HashMap<>();

	private static final String TF_FUNCTION_FQN = "tensorflow.python.eager.def_function.function";

	/**
	 * The TensorFlow module name as it appears in Python {@code import} statements, used by {@link #getImportContext(IDocument)} to detect
	 * the import shape (e.g. {@code import tensorflow}, {@code import tensorflow as tf}, {@code from tensorflow import ...}).
	 */
	private static final String TENSORFLOW_MODULE = "tensorflow";

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

		// Check the callees of every node of this reference. A shared synthetic summary (e.g., `tensorflow.data.map`) has one node
		// per callback context, each with a different callee, so a single node's successors miss the branch the analyzed function
		// actually reaches.
		for (CGNode node : cgNodes)
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
		directModCache.clear();
		Parameter.clearCaches();
		autoInjectedImportNames.clear();
		fileInferredDTypeNames.clear();
		fileInferredSpecTypeNames.clear();
	}

	/**
	 * Pre-computes, per file, the union of spec-type constructor names and dtype constants required by the inferred input signatures of the
	 * functions about to be converted to hybrid, so {@link #convertToHybrid} can auto-inject a single {@code from tensorflow import ...}
	 * line covering all of them. Without it, the first hybridizable function processed in an import-less file fixes the injected line to
	 * its own spec types and dtypes, and a later function needing a different dtype (#588) or a {@code RaggedTensorSpec} (#524) is gated
	 * off emission and left with a bare {@code @function} (follow-up to #574). Reads the memoized inferred signatures via
	 * {@link #getInferredInputSignature}; it never triggers inference, so it adds no per-parameter INFOs. Call it once before transforming
	 * a batch of functions.
	 *
	 * @param functions The functions about to be transformed.
	 */
	public static void planAutoInjectedImports(Collection<Function> functions) {
		for (Function function : functions) {
			if (!function.getTransformations().contains(CONVERT_TO_HYBRID) || !function.getInferInputSignatures())
				continue;

			function.getInferredInputSignature().ifPresent(sig -> {
				File file = function.getContainingFile();
				fileInferredDTypeNames.computeIfAbsent(file, k -> new TreeSet<>()).addAll(sig.requiredDTypeNames());
				fileInferredSpecTypeNames.computeIfAbsent(file, k -> new TreeSet<>()).addAll(sig.requiredSpecTypeNames());
			});
		}
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

			IClass concreteType = instanceKeyToProcess.concreteType();
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
	 * True iff the refactoring should emit an inferred {@code input_signature} keyword into the generated decorator.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	private boolean inferInputSignatures;

	/**
	 * True iff this {@link Function} declares a rest-keyword ({@code **kwargs}) parameter. Read off the declaration in the constructor,
	 * since an {@code input_signature} makes that slot inert and the reduction must decline rather than describe it (#902).
	 */
	private final boolean variableKeywordParameter;

	/**
	 * Memoizes {@link #inferInputSignature()}. {@code null} means "not yet computed"; once computed, holds the {@link InferenceResult} so
	 * the per-parameter INFOs the computation emits as a side effect are added at most once, regardless of how many call sites request the
	 * signature in a single pass (analysis, import injection, and the transform paths all ask for it).
	 */
	private InferenceResult inferredInputSignature;

	/**
	 * Per-parameter blocking reasons from the last {@link #computeInputSignature()} run, in parameter declaration order. Empty when
	 * inference produced a signature, was never run, or was blocked at the function level by
	 * {@link InferenceResult.AbsenceReason#SPECULATIVE_TENSOR_PARAMETER} (where no parameter is the blocker). Where
	 * {@link #getInferredInputSignatureAbsenceReason()} reports only the first blocking reason for the whole function, this retains every
	 * blocking parameter's reason for read-only per-parameter reporting (e.g. the evaluator); see {@link #getBlockingParameterReasons()}.
	 */
	private Map<Parameter, AbsenceReason> blockingParameterReasons = new LinkedHashMap<>();

	/**
	 * The per-parameter reduced spec entries from the last successful {@link #computeInputSignature()} run, in declaration order over the
	 * spec-contributing parameters. Empty until inference produces a signature. Retained so consumers can attribute the reduced spec's
	 * entries to their {@link Parameter}s (the unresolved statically-read-axis precondition; issue 811).
	 */
	private Map<Parameter, InputSignature.SpecEntry> inferredSpecByParameter = Map.of();

	/**
	 * True iff a parameter axis that this {@link Function}'s body (transitively) reads statically and consumes where a Python integer is
	 * required is left unresolved (wildcard) by the inferred input signature, so emitting that signature would break the function at trace
	 * time. {@code false} when no signature would be emitted (inference off or absent) or every such axis is concrete; {@code null} when it
	 * could not be determined (no call-graph node), in which case the precondition does not block. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/811.
	 */
	private Boolean hasUnresolvedStaticallyReadAxes;

	/**
	 * Whether this function is reached through {@code tf.distribute.Strategy.run} (issue 928). {@code null} until computed.
	 */
	private Boolean replicaInvoked;

	/**
	 * True iff this {@link Function}'s body snapshots a model's variable collection before the model's first invocation in the body and
	 * feeds the snapshot to an optimizer or gradient computation, which raises under {@code tf.function} tracing when slot creation lands
	 * on the variable-lifting re-trace. {@code null} when it could not be determined (no call-graph node), in which case the precondition
	 * does not block. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/822.
	 */
	private Boolean hasStaleVariableReads;

	/**
	 * True iff every known call path to this {@link Function} is dominated by a hybridized caller, so its computation is already traced and
	 * hybridizing it may add no benefit. Advisory-only (phase 1 of issue 767): consulted for an INFO on the P1 path and the evaluator's
	 * measurement column, never for a decision. {@code null} until the processor's project-wide caller-coverage pass assigns it. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/767.
	 */
	private Boolean callerCovered;

	/**
	 * True iff this {@link Function}'s body iterates a parameter-derived, tensor-typed value, which raises under {@code tf.function}
	 * tracing once the parameter is symbolic. {@code null} when it could not be determined (no call-graph node), in which case the
	 * precondition does not block. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/830.
	 */
	private Boolean hasTensorParameterIteration;

	/**
	 * This {@link Function}'s call-graph nodes reached only from expected-failure call sites (a call inside {@code assertRaises} or
	 * {@code pytest.raises}), whose evidence a specification must not be derived from (#888). Empty when none is, when the question was not
	 * asked, or when every node is guarded, in which case nothing is excluded: a function called only from expected-failure tests would
	 * otherwise lose its tensor-parameter classification, which moves preconditions rather than signatures and is out of scope here.
	 */
	private Set<CGNode> expectedFailureNodes = Set.of();

	/**
	 * True iff some call site of this {@link Function} passes a Keras symbolic tensor ({@code KerasTensor}), which {@code tf.function}
	 * refuses outright, so the decorator raises before anything is traced. {@code null} when it could not be determined (no call-graph
	 * node, or a caller whose arguments are invisible), in which case the precondition does not block. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/887.
	 */
	private Boolean hasKerasSymbolicArguments;

	/**
	 * True iff some parameter's direct consumers impose more than one concrete eager-effective dtype, so no single input signature
	 * reproduces eager coercion (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/861, Case 1). {@code null} when it
	 * could not be determined (no call-graph node), in which case the precondition does not block.
	 */
	private Boolean hasConflictingEagerDtypeCoercions;

	/**
	 * Per-{@link Parameter} eager-effective dtype pins: a parameter maps here when its direct consumers impose exactly one concrete dtype
	 * that differs from the parameter's own dtype evidence, and the emitted spec's dtype is substituted accordingly (the repair direction
	 * of https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/861, Case 1).
	 */
	private Map<Parameter, DType> eagerEffectiveDtypePins = new HashMap<>();

	/**
	 * The parameters whose applied dtype coercion resolves as {@link Resolution#CHANGED}, i.e. those the run feeds a dtype their consumers
	 * do not impose. Distinct from {@link #eagerEffectiveDtypePins}, which compares a parameter's REPORTED dtype against the imposition and
	 * is therefore empty for an operator spelling by construction: the analysis applies that coercion and reports its result, so the two
	 * agree and the divergence with the FED dtype stays invisible (wala/ML#838).
	 */
	private Set<Parameter> changedDtypeCoercions = new HashSet<>();

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
	 * True iff this {@link Function}'s body performs a (transitive) TensorFlow tensor computation. {@code null} when it could not be
	 * determined (e.g., no call-graph node), in which case the precondition does not block hybridization. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/709.
	 */
	private Boolean hasTensorComputation;

	/**
	 * True iff this {@link Function}'s body (transitively) invokes an eager-only API (e.g. {@code Tensor.numpy()}), which raises under
	 * {@code tf.function} tracing. {@code null} when it could not be determined (e.g., no call-graph node), in which case the precondition
	 * does not block hybridization. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/363.
	 */
	private Boolean hasEagerOnlyCalls;

	/**
	 * True iff this {@link Function}'s body (transitively) applies a numpy/scipy API to a value flowing from its parameters, which raises
	 * under {@code tf.function} tracing. {@code null} when it could not be determined (e.g., no call-graph node), in which case the
	 * precondition does not block hybridization. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/740.
	 */
	private Boolean hasNumpyCallsOnParameters;

	/**
	 * True iff this {@link Function}'s body passes a non-string constant where a TensorFlow API declares its {@code name} parameter (e.g.,
	 * {@code tf.sqrt(x, tf.float32)}), which raises under {@code tf.function} tracing. Unlike the call-graph-based safety checks above,
	 * this is computed by a syntactic scan of the body. {@code null} when it could not be determined (an AST traversal failure), in which
	 * case the precondition does not block hybridization. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/814.
	 */
	private Boolean hasInvalidNameArguments;

	/**
	 * True iff this {@link Function} has at least one parameter that is likely a primitive.
	 */
	private Boolean hasPrimitiveParameter;

	/**
	 * True iff this {@link Function} has at least one parameter that is a tf.Tensor (https://bit.ly/3vYG7iP).
	 */
	private Boolean hasTensorParameter;

	/**
	 * True iff {@link #hasTensorParameter} was set by {@link #inferTensorParameters}'s speculative context analysis rather than by any
	 * {@link Parameter}'s own classification. Speculation fires only when no parameter classified, so it leaves every {@link Parameter}'s
	 * {@link Parameter#isTensor()} at {@code FALSE} and its {@link Parameter#getTensorTypes()} empty; {@link #computeInputSignature()}
	 * reads this to report the honest blocking reason instead of dispatching per parameter and reporting each as non-tensor (#783).
	 */
	private boolean tensorParameterFromSpeculation;

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
		this(fd, ignoreBooleans, alwaysFollowTypeHints, useSpeculativeAnalysis, false);
	}

	public Function(FunctionDefinition fd, boolean ignoreBooleans, boolean alwaysFollowTypeHints, boolean useSpeculativeAnalysis,
			boolean inferInputSignatures) {
		this.functionDefinition = fd;
		this.ignoreBooleans = ignoreBooleans;
		this.alwaysFollowTypeHints = alwaysFollowTypeHints;
		this.useSpeculativeAnalysis = useSpeculativeAnalysis;
		this.inferInputSignatures = inferInputSignatures;

		// Jython's `argumentsType` is the whole parameter-list node; its `.args` field is the positional/positional-or-keyword name array.
		// `vararg`, `kwarg`, and `kwonlyargs` are sibling fields on the same node that we don't currently wrap.
		// Tracked at https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/465.
		argumentsType args = fd.getFunctionDef().args;
		List<Parameter> built = new ArrayList<>();
		if (args != null && args.args != null)
			for (int i = 0; i < args.args.length; i++)
				built.add(new Parameter(args, i, this));
		// Keyword-only parameters (declared after a bare `*`) are a sibling array; tensor parameters declared keyword-only (idiomatic in
		// Keras `call()` methods) would otherwise be missed (#607). `vararg`/`kwarg` remain unwrapped (#465).
		if (args != null && args.kwonlyargs != null)
			for (int i = 0; i < args.kwonlyargs.length; i++)
				built.add(new Parameter(args, i, this, true));
		this.parameters = Collections.unmodifiableList(built);

		// The rest-keyword slot is read off the declaration rather than wrapped, since what the signature reduction needs from it is only
		// whether it exists (#902). Wrapping it as a `Parameter` is #465.
		this.variableKeywordParameter = args != null && args.kwarg != null;
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

							if (this.getHasEagerOnlyCalls() != null && this.getHasEagerOnlyCalls())
								// Invokes an eager-only API, which raises under tf.function tracing (issue 363). A safety failure, so it
								// takes precedence over the barren benefit signal below.
								this.addFailure(PreconditionFailure.HAS_EAGER_ONLY_CALLS,
										"Can't hybridize a function that calls eager-only APIs like Tensor.numpy().");
							else if (this.getHasNumpyCallsOnParameters() != null && this.getHasNumpyCallsOnParameters())
								// Applies numpy to parameter-flowing values, which raises under tf.function tracing (issue 740). The
								// second safety failure in the family; also precedes the benefit signal.
								this.addFailure(PreconditionFailure.HAS_NUMPY_CALLS_ON_PARAMETERS,
										"Can't hybridize a function that applies numpy to values derived from its parameters.");
							else if (this.getHasInvalidNameArguments() != null && this.getHasInvalidNameArguments())
								// Passes a non-string constant where a TensorFlow API declares `name`, which only tracing validates
								// (issue 814). The third safety failure in the family; also precedes the benefit signal.
								this.addFailure(PreconditionFailure.HAS_INVALID_NAME_ARGUMENTS,
										"Can't hybridize a function that passes a non-string name argument to a TensorFlow API.");
							else if (this.getHasStaleVariableReads() != null && this.getHasStaleVariableReads())
								// Snapshots a model's variables before the model's first call, which raises under tracing when
								// optimizer slot creation lands on the lifting re-trace (issue 822). The fifth safety failure in the
								// family; also precedes the benefit signal.
								this.addFailure(PreconditionFailure.HAS_STALE_VARIABLE_READS,
										"Can't hybridize a function that reads a model's variables before calling the model in its body; "
												+ "tracing would create optimizer state after the first trace.");
							else if (this.getHasTensorParameterIteration() != null && this.getHasTensorParameterIteration())
								// Iterates a parameter-derived tensor, which raises under tracing once the parameter is symbolic
								// (issue 830). The sixth safety failure in the family; also precedes the benefit signal.
								this.addFailure(PreconditionFailure.HAS_TENSOR_PARAMETER_ITERATION,
										"Can't hybridize a function that iterates one of its tensor arguments with a Python loop; "
												+ "tracing makes the argument symbolic, which cannot be iterated.");
							else if (this.getHasConflictingEagerDtypeCoercions() != null && this.getHasConflictingEagerDtypeCoercions())
								// A parameter's direct consumers impose different eager-effective dtypes (issue 861, Case 1): eager
								// execution coerces the argument differently at each op, so no single input signature preserves
								// semantics and a bare decorator raises at the first mismatched op. The seventh safety failure in the
								// family; also precedes the benefit signal.
								this.addFailure(PreconditionFailure.HAS_CONFLICTING_EAGER_DTYPE_COERCIONS,
										"Can't hybridize a function whose parameter combines with tensors of different dtypes; "
												+ "eager execution coerces the argument per operation, so no single input signature "
												+ "preserves its semantics.");
							else if (this.requiresUnwritableEagerDtypePin())
								// A parameter is fed a dtype its consumers do not impose, and no specification will be written to
								// reproduce the conversion at the trace boundary. Eagerly the argument converts at the operation; under
								// tracing it materializes at the dtype it was fed and the operation raises (#861, Case 1). Unlike the
								// conflicting case above, a specification WOULD preserve this function exactly, naming the imposed dtype;
								// what is missing is the specification, not the answer.
								this.addFailure(PreconditionFailure.HAS_UNWRITABLE_EAGER_DTYPE_PIN,
										"Can't hybridize a function whose parameter is fed a dtype its consumers do not impose unless an "
												+ "input signature is written: eager execution converts the argument at the operation, and "
												+ "tracing without a signature carries the fed dtype in instead.");
							else if (this.getHasKerasSymbolicArguments() != null && this.getHasKerasSymbolicArguments())
								// A call site passes a Keras symbolic tensor (issue 887): tf.function is one of the APIs a KerasTensor
								// refuses, so the decorator raises on the first call before anything is traced, bare or with a
								// signature. The eighth safety failure in the family; also precedes the benefit signal.
								this.addFailure(PreconditionFailure.HAS_KERAS_SYMBOLIC_ARGUMENTS,
										"Can't hybridize a function called with a Keras symbolic tensor; tf.function refuses a "
												+ "KerasTensor, so the decorator raises before anything is traced.");
							else if (this.getHasTensorComputation() != null && !this.getHasTensorComputation())
								// Performs no tensor computation, so hybridization is unlikely to help (issue 709). Leaving it eager is
								// incompleteness-safe: it never violates semantics preservation.
								this.addFailure(PreconditionFailure.NO_TENSOR_COMPUTATION,
										"This function performs no tensor computation, so hybridization is unlikely to improve performance.");
							else if (TRUE.equals(this.getCallerCovered()))
								// Blocking as of issue 826, promoted from the phase-1 advisory (issue 767) on corpus evidence: a
								// covered function's computation is already traced on every executed path, so conversion adds only a
								// redundant nested trace boundary. Allow-on-unknown: only a determinate TRUE blocks.
								this.addFailure(PreconditionFailure.HAS_COVERED_CALLERS,
										"Every known call path to this function comes from hybridized code, so its computation "
												+ "is already traced; hybridizing it would add no benefit.");
							else {
								this.addTransformation(Transformation.CONVERT_TO_HYBRID);
								this.setPassingPrecondition(P1);

								/*
								 * The eager→hybrid conversion emits the inferred signature into the new decorator during the change
								 * (`convertToHybrid`). Compute it here too so the inferred signature is observable at analysis time
								 * (wizard, evaluator), mirroring how the reconfigure path computes it while checking preconditions. The
								 * result is memoized, so the change does not recompute it, and computing it has no bearing on the P1
								 * decision.
								 */
								if (this.getInferInputSignatures()) {
									this.inferInputSignature();

									// The third disposition of issue 864: the signature leaves unresolved an axis the body reads
									// statically (issue 811), so emitting it would break the function at trace time, while the bare
									// decorator is exactly what the tool ships with inference off and works there. Withhold the
									// signature, keep the conversion, and surface the withholding as the function-level absence so a
									// reader can tell "hybridized without a signature because one could not be written safely" from
									// both "hybridized with a signature" and "not hybridized". The reconfiguration path keeps
									// declining with the precondition failure: there the decoration already exists, and changing its
									// argument is the only action on the table.
									// Issue 928: the function is reached through `tf.distribute.Strategy.run`, which does not preserve
									// the declared argument structure. A signature accurate for a direct call therefore describes a
									// calling convention this function will not be called by, and it raises on arity before any
									// argument is examined. The bare decorator is what runs on that path, so withhold the signature
									// and keep the conversion, exactly as for the statically-read-axis case below.
									if (TRUE.equals(this.getReplicaInvoked()) && this.getInferredInputSignature().isPresent()) {
										this.inferredInputSignature = new InferenceResult.Absent(
												InferenceResult.AbsenceReason.WITHHELD_REPLICA_INVOKED);
										this.addInfo(INPUT_SIGNATURE_INFERENCE,
												"`" + this + "` is invoked through a distribution strategy, which does not preserve its "
														+ "declared argument structure, so the signature is withheld and the function is "
														+ "hybridized with a bare decorator instead.");
									}

									if (TRUE.equals(this.getHasUnresolvedStaticallyReadAxes())
											&& this.getInferredInputSignature().isPresent()) {
										this.inferredInputSignature = new InferenceResult.Absent(
												InferenceResult.AbsenceReason.WITHHELD_STATICALLY_READ_AXES);
										this.addInfo(INPUT_SIGNATURE_INFERENCE,
												"The inferred input signature of `" + this + "` leaves unspecified a tensor dimension "
														+ "its body reads statically, so the signature is withheld and the function is "
														+ "hybridized with a bare decorator instead.");
									}
								}
							}
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

			if (TRUE.equals(this.getCallerCovered()))
				// Advisory only, the measurement phase for de-hybridizing covered hybrid functions (issue 827), mirroring the
				// staging the conversion side went through before its blocking promotion (issue 826): a covered hybrid function's
				// decorator is redundant on every executed path, but removing it deletes an enforced boundary (any supplied
				// input_signature validation) and leans on the dead-caller semantics, so the transformation waits on corpus
				// evidence gathered through this INFO and the caller-covered column.
				this.addInfo(Information.CALLER_COVERAGE,
						"Every known call path to this hybrid function comes from hybridized code, so its decorator is redundant "
								+ "on every executed path; de-hybridizing it may be beneficial.");

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
					if (this.getHasTensorComputation() != null && !this.getHasTensorComputation()) {
						// Barren (issue 709): a hybrid function performing no tensor computation gains nothing from graph execution, only
						// tracing overhead, so de-hybridize it when semantics are preserved (no Python side-effects). This is the
						// hybrid-to-eager counterpart of the eager-to-hybrid NO_TENSOR_COMPUTATION precondition and a peer of P2/P3.
						this.addInfo("This hybrid function performs no tensor computation.");

						if (this.getHasPythonSideEffects() != null && !this.getHasPythonSideEffects()) {
							this.addInfo("This hybrid function does not have Python side-effects.");
							this.addTransformation(CONVERT_TO_EAGER);
							this.setPassingPrecondition(P6);
						} else if (this.getHasPythonSideEffects() != null) // it has side-effects.
							this.addFailure(HAS_PYTHON_SIDE_EFFECTS,
									"De-hybridizing a function with Python side-effects may alter semantics.");
					} else {
						/*
						 * This function is already correctly hybrid (tensor parameter, no primitive parameter). When input-signature
						 * inference is enabled, the function is side-effect-free and non-recursive, and a signature can be inferred and
						 * emitted, the decorator is reconfigured: if it carries no `input_signature` yet, add the inferred one (the add
						 * path); if it carries one that is more specific than, or incomparable with, the inferred one, overwrite it; if it
						 * carries one broader than the inferred one, preserve it (the broader signature may be intentional); if they agree,
						 * do nothing. A supplied signature whose content could not be modeled is left untouched. Gating on the flag keeps
						 * the default precondition matrix unchanged.
						 */
						// A signature the developer already supplied has already disabled a rest-keyword slot, so a caller passing a
						// keyword raises today, before this refactoring touches anything. Nothing here can repair that, and rewriting
						// the signature would not: the report is the action. Warned rather than failed, since the function's
						// hybridization is not what is wrong with it, and gated on inference like every other signature diagnostic
						// (#902).
						if (this.getInferInputSignatures() && this.hasVariableKeywordParameter()
								&& this.getHybridizationParameters().hasInputSignatureParam())
							this.addWarning("This hybrid function declares a `**kwargs` parameter alongside an input signature. The "
									+ "signature fixes the arguments the function accepts, so that parameter absorbs nothing and any "
									+ "caller passing a keyword argument raises; for a Keras layer that caller is Keras itself, which "
									+ "passes `training`. Removing the input signature, or the `**kwargs` parameter, resolves it.");

						// An unresolved statically-read axis blocks every reconfigure path: writing the inferred signature (adding or
						// overwriting) would break the function at trace time (issue 811).
						boolean unresolvedStaticallyReadAxes = this.getHasUnresolvedStaticallyReadAxes() != null
								&& this.getHasUnresolvedStaticallyReadAxes();

						// The reconfiguration gates minus the axis check, so the blocked case below can tell "the axis was the sole
						// blocker" (code 18 reports alone) from "reconfiguration was never otherwise viable" (the pre-inference
						// terminal applies); see issue 865.
						boolean reconfigureOtherwiseViable = this.getInferInputSignatures() && this.getHasPythonSideEffects() != null
								&& !this.getHasPythonSideEffects() && this.isRecursive() != null && !this.isRecursive()
								&& this.canEmitInferredInputSignature();

						boolean canReconfigure = reconfigureOtherwiseViable && !unresolvedStaticallyReadAxes;

						if (canReconfigure && !this.getHybridizationParameters().hasInputSignatureParam()) {
							// Add path: no existing `input_signature`.
							this.addInfo("This hybrid function has no input signature and will be reconfigured to add the inferred one.");
							this.addTransformation(RECONFIGURE);
							this.setPassingPrecondition(P4);
						} else if (canReconfigure && this.getHybridizationParameters().getSuppliedInputSignature().isPresent()
								&& this.inferInputSignature() instanceof InferenceResult.Inferred(InputSignature inferred)) {
							// Adjudication path: an existing, fully-modeled `input_signature` is present. Compare it against the
							// inferred one and REPORT; no relation rewrites the signature (issue 808). Since the inferred signature is
							// the join over the observed call sites, the tighter and incomparable relations can only arise when some
							// observed call violates the existing signature, i.e., the original program raises at that site; rewriting
							// the signature to admit those calls would repair rather than refactor, with zero retracing benefit (a
							// present signature already pins one trace). The rewrite is a sanctioned future find-and-fix
							// transformation, not this refactoring. `canReconfigure` implies inference succeeded (it gates on
							// `canEmitInferredInputSignature`), so the pattern always binds here; a hypothetical `Absent` falls
							// through to the no-primitive-parameter failure below.
							InputSignature supplied = this.getHybridizationParameters().getSuppliedInputSignature().get();

							switch (supplied.relate(inferred)) {
							case SUPPLIED_TIGHTER -> this
									.addWarning("This hybrid function's input signature is narrower than its call sites require; "
											+ "a nonconforming observed call raises at runtime. The signature is left unchanged: "
											+ "admitting those calls would change program behavior rather than preserve it.");
							case INCOMPARABLE -> this.addWarning("This hybrid function's input signature disagrees with its call sites; "
									+ "a nonconforming observed call raises at runtime. The signature is left unchanged: "
									+ "reconciling it would change the inputs the function accepts.");
							case SUPPLIED_BROADER -> this
									.addInfo("This hybrid function's input signature is broader than its call sites require; "
											+ "it is left unchanged in case the broader signature is intentional.");
							case AGREEMENT -> {
								// Nothing to report: the supplied signature matches the call-site evidence.
							}
							}

							// No transformation applies, so the already-optimal verdict reports (issue 865's model: a function the
							// refactoring cannot further improve "fails" it, benignly): staying hybrid is this function's best form,
							// and the relation entries above inform beside the verdict.
							this.addFailure(PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS,
									"Functions with no Python literal arguments may benefit from hybridization.");
						} else if (reconfigureOtherwiseViable && unresolvedStaticallyReadAxes) {
							// The one emission issue 865 removes: this function is NOT already optimal, since a signature
							// improvement existed and was withheld as unwritable, so the already-optimal verdict would be false
							// here. The unresolved axis is the operative failure and reports alone (issue 811).
							this.addFailure(PreconditionFailure.HAS_UNRESOLVED_STATICALLY_READ_AXES,
									"Can't reconfigure this function's input signature: "
											+ "its body reads a tensor dimension the inferred signature leaves unspecified.");
						} else {
							// The pre-inference terminal, unchanged: no signature flow resolved anything here, so the already-optimal
							// verdict reports as it always did.
							this.addFailure(PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS,
									"Functions with no Python literal arguments may benefit from hybridization.");

							if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects())
								this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
										"De-hybridizing a function with Python side-effects may alter semantics.");
						}
					}
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

	/**
	 * Determines whether this {@link Function}'s body performs a (transitive) TensorFlow tensor computation, storing the result in
	 * {@link #hasTensorComputation}. When there is no call-graph node for the function, the result is left {@code null} (undetermined), so
	 * the precondition does not block hybridization. See https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/709.
	 *
	 * @param callGraph The call graph.
	 * @param pointerAnalysis The pointer analysis.
	 * @param tensorTypedKeys The tensor-typed pointer keys mapped to their producing-library origins (see
	 *        {@link Util#computeTensorTypedOrigins}), used to identify tensor-op results in the body.
	 */
	public void computeTensorComputation(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			Map<PointerKey, Set<TensorOrigin>> tensorTypedKeys) {
		Set<CGNode> nodes;

		try {
			nodes = this.getNodes(callGraph);
		} catch (CoreException e) {
			// Undeterminable; leave null so the precondition does not block.
			LOG.warn("Can't determine whether " + this + " performs a tensor computation.", e);
			return;
		}

		if (nodes.isEmpty()) {
			// Undeterminable without a call-graph node; leave null so the precondition does not block.
			LOG.info("Can't determine whether " + this + " performs a tensor computation without a call graph node.");
			return;
		}

		// A function may have several call-graph nodes (context-sensitive copies, trampolines). It performs a tensor computation if any of
		// them does; sampling a single node can miss the op at an imprecise context.
		boolean performsTensorOp = nodes.stream()
				.anyMatch(cgNode -> Util.performsTensorFlowOp(cgNode, callGraph, pointerAnalysis, tensorTypedKeys));

		this.hasTensorComputation = performsTensorOp;

		LOG.info(this + (performsTensorOp ? " performs a tensor computation." : " performs no tensor computation."));
	}

	/**
	 * True iff this {@link Function}'s body performs a (transitive) TensorFlow tensor computation, {@code null} if undetermined.
	 *
	 * @return True iff this function performs a tensor computation, null if undetermined.
	 */
	public Boolean getHasTensorComputation() {
		return this.hasTensorComputation;
	}

	/**
	 * Computes whether this {@link Function}'s body (transitively) invokes an eager-only API (e.g. {@code Tensor.numpy()}), storing the
	 * result for {@link #getHasEagerOnlyCalls()}. Mirrors {@link #computeTensorComputation(CallGraph, PointerAnalysis, Map)}: when the
	 * function has no call-graph node, the result is left undetermined and the precondition neither blocks nor passes on it.
	 *
	 * @param callGraph The call graph.
	 * @param pointerAnalysis The pointer analysis.
	 */
	public void computeEagerOnlyCalls(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		Set<CGNode> nodes;

		try {
			nodes = this.getNodes(callGraph);
		} catch (CoreException e) {
			// Undeterminable; leave null so the precondition does not block.
			LOG.warn("Can't determine whether " + this + " calls an eager-only API.", e);
			return;
		}

		if (nodes.isEmpty()) {
			// Undeterminable without a call-graph node; leave null so the precondition does not block.
			LOG.info("Can't determine whether " + this + " calls an eager-only API without a call graph node.");
			return;
		}

		// A function may have several call-graph nodes (context-sensitive copies, trampolines). It calls an eager-only API if any of
		// them does; sampling a single node can miss the call at an imprecise context.
		EagerOnlyCallAnalysis analysis = new EagerOnlyCallAnalysis(callGraph, pointerAnalysis);
		boolean eagerOnly = nodes.stream().anyMatch(analysis::callsEagerOnlyApi);

		this.hasEagerOnlyCalls = eagerOnly;

		LOG.info(this + (eagerOnly ? " calls an eager-only API." : " calls no eager-only APIs."));
	}

	/**
	 * True iff this {@link Function}'s body (transitively) invokes an eager-only API, {@code null} if undetermined.
	 *
	 * @return True iff this function calls an eager-only API, null if undetermined.
	 */
	public Boolean getHasEagerOnlyCalls() {
		return this.hasEagerOnlyCalls;
	}

	/**
	 * Computes whether this {@link Function}'s body (transitively) applies a numpy/scipy API to a value flowing from its parameters,
	 * storing the result for {@link #getHasNumpyCallsOnParameters()}. Mirrors {@link #computeEagerOnlyCalls(CallGraph, PointerAnalysis)}:
	 * when the function has no call-graph node, the result is left undetermined and the precondition neither blocks nor passes on it.
	 *
	 * @param callGraph The call graph.
	 * @param pointerAnalysis The pointer analysis.
	 * @param tensorTypeAnalysis The tensor-type analysis, consulted for the per-dimension shape staticness of numpy over a tensor's shape.
	 */
	public void computeNumpyCallsOnParameters(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			TensorTypeAnalysis tensorTypeAnalysis) {
		Set<CGNode> nodes;

		try {
			nodes = this.getNodes(callGraph);
		} catch (CoreException e) {
			// Undeterminable; leave null so the precondition does not block.
			LOG.warn("Can't determine whether " + this + " applies numpy to its parameters.", e);
			return;
		}

		if (nodes.isEmpty()) {
			// Undeterminable without a call-graph node; leave null so the precondition does not block.
			LOG.info("Can't determine whether " + this + " applies numpy to its parameters without a call graph node.");
			return;
		}

		// A function may have several call-graph nodes (context-sensitive copies, trampolines). It applies numpy to its parameters if
		// any of them does; sampling a single node can miss the flow at an imprecise context. The analysis instance indexes the
		// tensor-type analysis once and shares its scan memo across the nodes rather than rebuilding either per node.
		NumpyParameterFlowAnalysis numpyParameterFlowAnalysis = new NumpyParameterFlowAnalysis(callGraph, pointerAnalysis,
				tensorTypeAnalysis);
		boolean numpyOnParameters = nodes.stream()
				.anyMatch(cgNode -> numpyParameterFlowAnalysis.appliesNumpyToParameters(cgNode, this.isMethod()));

		this.hasNumpyCallsOnParameters = numpyOnParameters;

		LOG.info(this + (numpyOnParameters ? " applies numpy to its parameters." : " applies no numpy to its parameters."));
	}

	/**
	 * True iff this {@link Function}'s body (transitively) applies a numpy/scipy API to a parameter-flowing value, {@code null} if
	 * undetermined.
	 *
	 * @return True iff this function applies numpy to its parameters, null if undetermined.
	 */
	public Boolean getHasNumpyCallsOnParameters() {
		return this.hasNumpyCallsOnParameters;
	}

	/**
	 * The declaring class of the synthetic node the engine uses to model {@code tf.distribute.Strategy.run}. A function invoked through
	 * that node receives the arguments {@code run} distributes rather than the ones its declaration names.
	 * <p>
	 * Only the plain spelling is matched. The {@code $}-prefixed form elsewhere in this package is the method trampoline of a
	 * <em>summarized</em> endpoint, and the dispatch is modeled as a synthetic node rather than a summary: no {@code $} variant of it
	 * appears in the call graph. Should that modeling change, this check stops firing rather than failing, which is why the coupling is
	 * stated on the issue rather than left implicit.
	 */
	private static final String DISTRIBUTE_RUN_CLASS_NAME = "Ltensorflow/distribute/run/run";

	/**
	 * Computes whether this {@link Function} is reached through {@code tf.distribute.Strategy.run}, storing the result for
	 * {@link #getReplicaInvoked()}. That call path does not preserve the declared argument structure: a single structured parameter arrives
	 * as separate positional arguments, so an emitted signature describes a calling convention that will not be used and the function
	 * raises on arity before any argument is examined. A direct call to the same function with the same signature succeeds, so this is a
	 * property of the caller rather than of the specification.
	 *
	 * @param callGraph The call graph, queried in the caller direction.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/928">Issue 928</a>
	 */
	public void computeReplicaInvoked(CallGraph callGraph) {
		// Without an emitted signature there is nothing to withhold, so the caller walk is not worth doing. The verdict is left UNSET
		// rather than set to FALSE: no caller was examined, and "not examined" is not "examined and found not to be the dispatch".
		// Setting FALSE here would report evidence that was never gathered, which is the distinction this check exists to preserve.
		if (!this.getInferInputSignatures())
			return;

		Set<CGNode> nodes;

		try {
			// Queried directly rather than through the shared `getNodes` helper, which logs at ERROR when the set is empty. Here an
			// empty set is the expected undetermined outcome for a function nothing calls, not a fault, and an error line for it
			// would be noise on a path this check takes routinely.
			nodes = callGraph.getNodes(this.getMethodReference());
		} catch (CoreException e) {
			// Undeterminable; leave the verdict unset so nothing is withheld on an unresolved reference.
			LOG.warn("Can't determine whether " + this + " is reached through the replica dispatch.", e);
			return;
		}

		if (nodes.isEmpty()) {
			// Undeterminable without a call-graph node: no caller is visible, so absence of evidence is not evidence of absence.
			LOG.info("Can't determine whether " + this + " is reached through the replica dispatch without a call graph node.");
			return;
		}

		for (CGNode node : nodes)
			for (CGNode predecessor : Iterator2Iterable.make(callGraph.getPredNodes(node)))
				if (DISTRIBUTE_RUN_CLASS_NAME.equals(predecessor.getMethod().getDeclaringClass().getName().toString())) {
					this.replicaInvoked = TRUE;
					return;
				}

		this.replicaInvoked = FALSE;
	}

	/**
	 * Whether this {@link Function} is reached through {@code tf.distribute.Strategy.run}.
	 *
	 * @return {@code TRUE} when some caller is the replica dispatch, {@code FALSE} when none is, or {@code null} when no caller could be
	 *         examined.
	 * @see #computeReplicaInvoked(CallGraph)
	 */
	public Boolean getReplicaInvoked() {
		return this.replicaInvoked;
	}

	/**
	 * Computes whether a parameter axis that this {@link Function}'s body (transitively) reads statically and consumes where a Python
	 * integer is required is left unresolved (wildcard) by the inferred input signature, storing the result for
	 * {@link #getHasUnresolvedStaticallyReadAxes()}. Under an emitted signature, a wildcard axis is {@code None} at trace time; a static
	 * read consumed at a weight shape, a reshape target, or integer arithmetic then raises (or silently misbehaves through a
	 * {@code [:None]} slice). A dynamic read ({@code tf.shape(x)[i]}) is safe and does not disqualify. When no signature would be emitted
	 * (inference disabled or blocked), the property holds trivially and the result is {@code false}; when the function has no call-graph
	 * node, the result is left undetermined, mirroring the sibling safety checks. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/811.
	 *
	 * @param callGraph The call graph.
	 * @param pointerAnalysis The pointer analysis.
	 */
	public void computeUnresolvedStaticallyReadAxes(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		// Without an emitted signature there is no wildcard to break on: inference off is determinately safe.
		if (!this.getInferInputSignatures()) {
			this.hasUnresolvedStaticallyReadAxes = FALSE;
			return;
		}

		InferenceResult result;

		try {
			result = this.inferInputSignature();
		} catch (IllegalStateException _) {
			// The degenerate no-spec state throws: inference's contract expects tensor-parameter-gated callers, which the
			// measurement override (alwaysCheckStaticShapeReads) is not. Nothing can be emitted there, so it is the same
			// determinate pass as a blocked inference.
			LOG.info("No inferable signature for " + this + "; the statically-read-axis check passes determinately.");
			this.hasUnresolvedStaticallyReadAxes = FALSE;
			return;
		}

		// A blocked inference emits nothing either: determinately safe.
		if (!(result instanceof InferenceResult.Inferred)) {
			this.hasUnresolvedStaticallyReadAxes = FALSE;
			return;
		}

		Set<CGNode> nodes;

		try {
			nodes = this.getNodes(callGraph);
		} catch (CoreException e) {
			// Undeterminable; leave null so the precondition does not block.
			LOG.warn("Can't determine whether " + this + " statically reads an unresolved axis.", e);
			return;
		}

		if (nodes.isEmpty()) {
			// Undeterminable without a call-graph node; leave null so the precondition does not block.
			LOG.info("Can't determine whether " + this + " statically reads an unresolved axis without a call graph node.");
			return;
		}

		// A function may have several call-graph nodes (context-sensitive copies, trampolines). Union the axis reads over all of them;
		// sampling a single node can miss a read at an imprecise context.
		StaticShapeReadAnalysis analysis = new StaticShapeReadAnalysis(callGraph, pointerAnalysis);
		Set<StaticShapeReadAnalysis.AxisRead> reads = new HashSet<>();
		Set<StaticShapeReadAnalysis.AxisRead> rankReads = new HashSet<>();

		for (CGNode node : nodes) {
			StaticShapeReadAnalysis.StaticShapeReads nodeReads = analysis.staticallyReadAxes(node, this.isMethod());
			reads.addAll(nodeReads.axisReads());
			rankReads.addAll(nodeReads.rankReads());
		}

		// An extent-sensitive read blocks on any non-concrete covered axis; a rank-sensitive-only read (`as_list`, `rank`/`ndims`,
		// `len`; #809) blocks only when the affected spec's rank itself is unknown (`shape=None`), which every such surface breaks on.
		boolean unresolved = reads.stream().anyMatch(this::isUnresolvedRead) || rankReads.stream().anyMatch(this::isRankUnresolvedRead);

		this.hasUnresolvedStaticallyReadAxes = unresolved;

		LOG.info(this + (unresolved ? " statically reads an axis its inferred signature leaves unresolved."
				: " statically reads no unresolved axis."));
	}

	/**
	 * True iff the given axis read is unresolved by the inferred signature: some read axis of some read parameter reduces to a non-concrete
	 * extent. Lost provenance widens to every spec-contributing parameter and lost coverage to every axis, so the check errs toward
	 * declining (the conservative default this precondition requires). A parameter without a spec entry was omitted from the signature (a
	 * defaulted, never-supplied parameter), so its axes come from its concrete default at trace time and are safe.
	 *
	 * @param read The axis read to resolve against the inferred signature.
	 * @return True iff the read consumes an axis the signature leaves unresolved.
	 */
	private boolean isUnresolvedRead(StaticShapeReadAnalysis.AxisRead read) {
		for (InputSignature.SpecEntry entry : this.affectedSpecEntries(read)) {
			// Deliberately still refuses a container outright, where the rank arm below attributes the read across the elements
			// instead. The two are asymmetric on purpose rather than by oversight. Relaxing a refusal is the direction that emits a
			// specification where one was withheld, so it wants a witness, and the rank arm has one: a container parameter whose
			// elements all answer the read, executed in both directions. No witness reaches this arm. A read is attributed to a
			// parameter rather than to a value derived from one, so an extent read on an unpacked element does not present here, and
			// three attempts to construct a case that does all resolved identically with the attribution removed. Whether this arm is
			// reachable at all is the open question; until someone answers it, refusing costs precision on a path nobody has
			// exhibited and risks nothing (#914).
			if (!(entry instanceof InputSignature.Single single))
				return true;

			List<Dimension<?>> dims = single.type().getDims();

			// Shape-⊤ renders `shape=None`: every axis is wild.
			if (dims == null)
				return true;

			if (read.axes() == null) {
				for (Dimension<?> dim : dims)
					if (!(dim instanceof NumericDim))
						return true;
			} else
				for (int axis : read.axes()) {
					int index = axis < 0 ? dims.size() + axis : axis;

					// An index beyond the spec's rank contributes no element at trace time (Python clamps a slice); a genuinely
					// out-of-range subscript raises regardless of the signature and is not this precondition's concern.
					if (index < 0 || index >= dims.size())
						continue;

					if (!(dims.get(index) instanceof NumericDim))
						return true;
				}
		}

		return false;
	}

	/**
	 * True iff the given rank-sensitive-only read ({@code as_list}, {@code rank}/{@code ndims}, {@code len}; #809) is unresolved by the
	 * inferred signature: some affected spec entry has an unknown rank ({@code shape=None}), which every rank-reading surface breaks on. A
	 * known rank satisfies the read even when axes are dynamic ({@code [None, 3].as_list()} succeeds; {@code rank} is a trace-time
	 * integer), which is exactly the precision the split from {@link #isUnresolvedRead} buys. A container entry stays conservative.
	 *
	 * @param read The rank-sensitive read to resolve against the inferred signature.
	 * @return True iff the read consumes a rank the signature leaves unknown.
	 */
	private boolean isRankUnresolvedRead(StaticShapeReadAnalysis.AxisRead read) {
		for (InputSignature.SpecEntry entry : this.affectedSpecEntries(read))
			for (TensorType covered : coveredTypes(entry))
				if (covered.getDims() == null)
					return true;

		return false;
	}

	/**
	 * The tensor types a spec entry covers: the one type of a single-tensor entry, or every element type of a container's.
	 * <p>
	 * A read's provenance names a parameter rather than a position inside it, so a read against a container cannot be attributed to one
	 * element. Requiring every element to satisfy the read is what that unattributability licenses, and it is strictly weaker than the
	 * refusal it replaces: a container whose elements all answer the read resolves it, where before no container ever did, and one whose
	 * elements do not still blocks (#914).
	 *
	 * @param entry The spec entry to expand.
	 * @return Every tensor type the entry covers.
	 */
	private static List<TensorType> coveredTypes(InputSignature.SpecEntry entry) {
		return entry instanceof InputSignature.Single single ? List.of(single.type()) : ((InputSignature.Sequence) entry).elementTypes();
	}

	/**
	 * The inferred-spec entries a read's provenance touches: every entry when provenance was lost ({@code null} ordinals), otherwise the
	 * entries of the named non-{@code self} parameters. A parameter without a spec entry was omitted from the signature (a defaulted,
	 * never-supplied parameter), so its axes come from its concrete default at trace time and contribute nothing.
	 *
	 * @param read The read whose affected entries to collect.
	 * @return The spec entries the read may constrain.
	 */
	private List<InputSignature.SpecEntry> affectedSpecEntries(StaticShapeReadAnalysis.AxisRead read) {
		List<InputSignature.SpecEntry> entries = new ArrayList<>();

		if (read.parameterOrdinals() == null)
			entries.addAll(this.inferredSpecByParameter.values());
		else {
			List<Parameter> nonSelfParameters = this.getParameters().stream().filter(p -> !p.isSelf()).toList();

			for (int ordinal : read.parameterOrdinals())
				if (ordinal < nonSelfParameters.size()) {
					InputSignature.SpecEntry entry = this.inferredSpecByParameter.get(nonSelfParameters.get(ordinal));

					if (entry != null)
						entries.add(entry);
				}
		}

		return entries;
	}

	/**
	 * True iff the inferred input signature leaves a statically-read parameter axis unresolved, {@code null} if undetermined.
	 *
	 * @return True iff a statically-read axis is unresolved, null if undetermined.
	 */
	public Boolean getHasUnresolvedStaticallyReadAxes() {
		return this.hasUnresolvedStaticallyReadAxes;
	}

	/**
	 * Computes whether this {@link Function}'s body snapshots a model's variable collection before the model's first invocation, directly
	 * in the body or transitively through a callee, and feeds the snapshot to an optimizer or gradient computation, storing the result for
	 * {@link #getHasStaleVariableReads()}. The hazard is trace-time only (the snapshot is silently empty eagerly), so the check is what
	 * keeps the refactoring from introducing it; reading the collection after the forward pass never fires. When the function has no
	 * call-graph node, the result is left undetermined, mirroring the sibling safety checks. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/822 and, for the transitive invocation scan,
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/861.
	 *
	 * @param callGraph The call graph.
	 * @param pointerAnalysis The pointer analysis.
	 */
	public void computeStaleVariableReads(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		Set<CGNode> nodes;

		try {
			nodes = this.getNodes(callGraph);
		} catch (CoreException e) {
			// Undeterminable; leave null so the precondition does not block.
			LOG.warn("Can't determine whether " + this + " snapshots stale variables.", e);
			return;
		}

		if (nodes.isEmpty()) {
			// Undeterminable without a call-graph node; leave null so the precondition does not block.
			LOG.info("Can't determine whether " + this + " snapshots stale variables without a call graph node.");
			return;
		}

		boolean stale = new StaleVariableReadAnalysis(callGraph, pointerAnalysis).hasStaleVariableRead(nodes);

		this.hasStaleVariableReads = stale;

		LOG.info(this + (stale ? " snapshots a model's variables before its first call." : " snapshots no stale variables."));
	}

	/**
	 * True iff this {@link Function}'s body feeds a pre-build variable-collection snapshot to an optimizer or gradient computation,
	 * {@code null} if undetermined.
	 *
	 * @return True iff a stale variable snapshot reaches a consumer, null if undetermined.
	 */
	public Boolean getHasStaleVariableReads() {
		return this.hasStaleVariableReads;
	}

	/**
	 * Computes whether this {@link Function}'s body iterates a parameter-derived, tensor-typed value, storing the result for
	 * {@link #getHasTensorParameterIteration()}. Eagerly such iteration works; under tracing the parameter is symbolic and iterating it
	 * raises, so the conversion must be declined (issue 830, the shield the barren verdict accidentally provided at earlier Ariadne
	 * versions). When the function has no call-graph node, the result is left undetermined, mirroring the sibling safety checks. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/830.
	 *
	 * @param callGraph The call graph.
	 * @param pointerAnalysis The pointer analysis, used to resolve the iterated value's producer for the {@code tf.range} exemption.
	 * @param tensorTypeAnalysis The tensor-type analysis, whose typing gates the iterated value.
	 */
	public void computeTensorParameterIteration(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis,
			TensorTypeAnalysis tensorTypeAnalysis) {
		Set<CGNode> nodes;

		try {
			nodes = this.getNodes(callGraph);
		} catch (CoreException e) {
			// Undeterminable; leave null so the precondition does not block.
			LOG.warn("Can't determine whether " + this + " iterates a tensor parameter.", e);
			return;
		}

		if (nodes.isEmpty()) {
			// Undeterminable without a call-graph node; leave null so the precondition does not block.
			LOG.info("Can't determine whether " + this + " iterates a tensor parameter without a call graph node.");
			return;
		}

		TensorIterationAnalysis analysis = new TensorIterationAnalysis(pointerAnalysis, tensorTypeAnalysis);
		boolean iterates = nodes.stream().anyMatch(node -> analysis.iteratesTensorParameter(node, this.isMethod()));

		this.hasTensorParameterIteration = iterates;

		LOG.info(this + (iterates ? " iterates a tensor parameter." : " iterates no tensor parameter."));
	}

	/**
	 * True iff this {@link Function}'s body iterates a parameter-derived, tensor-typed value, {@code null} if undetermined.
	 *
	 * @return True iff a tensor parameter is iterated, null if undetermined.
	 */
	/**
	 * True iff this {@link Function} declares a rest-keyword ({@code **kwargs}) parameter, which is what makes an {@code input_signature}
	 * unwritable for it (#902).
	 *
	 * @return True iff a {@code **kwargs} parameter is declared.
	 */
	public boolean hasVariableKeywordParameter() {
		return this.variableKeywordParameter;
	}

	public Boolean getHasTensorParameterIteration() {
		return this.hasTensorParameterIteration;
	}

	/**
	 * Computes this {@link Function}'s expected-failure nodes ({@link ExpectedFailureContextAnalysis}), the ones reached only from call
	 * sites the developer has declared must fail, whose evidence the signature reduction then leaves out (#888). Must run before
	 * {@link #inferTensorParameters}, which is where the per-node evidence is read. Excluding every node is treated as excluding none, so
	 * the change stays confined to what is emitted rather than to which precondition passes.
	 *
	 * @param callGraph The call graph, walked in the caller direction.
	 * @param pointerAnalysis The pointer analysis, used to resolve each guard call's member name.
	 */
	public void computeExpectedFailureNodes(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		// The exclusion is read in exactly one place, the signature reduction, so with inference off there is nothing for it to affect and
		// the walk is pure cost. Gating here rather than at the call site keeps that invariant next to the code that relies on it.
		if (!this.getInferInputSignatures())
			return;

		Set<CGNode> nodes;

		try {
			nodes = this.getNodes(callGraph);
		} catch (CoreException e) {
			LOG.warn("Can't determine expected-failure call sites for " + this + ".", e);
			return;
		}

		if (nodes.isEmpty())
			return;

		Set<CGNode> guarded = new ExpectedFailureContextAnalysis(callGraph, pointerAnalysis).guardedOnlyNodes(nodes);

		if (guarded.size() == nodes.size()) {
			LOG.info("Every call site of " + this + " is an expected failure; excluding none.");
			return;
		}

		this.expectedFailureNodes = guarded;

		if (!guarded.isEmpty())
			LOG.info(this + " has " + guarded.size() + " node(s) reached only from expected-failure call sites.");
	}

	/**
	 * This {@link Function}'s call-graph nodes reached only from expected-failure call sites, whose evidence the signature reduction leaves
	 * out.
	 *
	 * @return The excluded nodes; empty when none is excluded.
	 */
	Set<CGNode> getExpectedFailureNodes() {
		return this.expectedFailureNodes;
	}

	/**
	 * Computes whether some call site of this {@link Function} passes a Keras symbolic tensor ({@link KerasSymbolicArgumentAnalysis}),
	 * storing the result for {@link #getHasKerasSymbolicArguments()}. {@code tf.function} refuses a {@code KerasTensor} outright, so
	 * decorating such a function raises a {@code TypeError} on the first call, before anything is traced, and the conversion must be
	 * declined. When the function has no call-graph node, the result is left undetermined, mirroring the sibling safety checks. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/887.
	 *
	 * @param callGraph The call graph, walked in the caller direction.
	 * @param pointerAnalysis The pointer analysis, used to resolve each argument's producing API.
	 */
	public void computeKerasSymbolicArguments(CallGraph callGraph, PointerAnalysis<InstanceKey> pointerAnalysis) {
		Set<CGNode> nodes;

		try {
			nodes = this.getNodes(callGraph);
		} catch (CoreException e) {
			// Undeterminable; leave null so the precondition does not block.
			LOG.warn("Can't determine whether " + this + " is passed a Keras symbolic tensor.", e);
			return;
		}

		if (nodes.isEmpty()) {
			// Undeterminable without a call-graph node; leave null so the precondition does not block.
			LOG.info("Can't determine whether " + this + " is passed a Keras symbolic tensor without a call graph node.");
			return;
		}

		this.hasKerasSymbolicArguments = new KerasSymbolicArgumentAnalysis(pointerAnalysis).hasKerasSymbolicArgument(nodes, callGraph);

		LOG.info(this + " passed a Keras symbolic tensor: " + this.hasKerasSymbolicArguments + ".");
	}

	/**
	 * True iff some call site of this {@link Function} passes a Keras symbolic tensor, {@code null} if undetermined.
	 *
	 * @return True iff a call site passes a {@code KerasTensor}, null if undetermined.
	 */
	public Boolean getHasKerasSymbolicArguments() {
		return this.hasKerasSymbolicArguments;
	}

	/**
	 * Computes the eager-effective dtypes each parameter's direct consumers impose (see {@link EagerCoercionAnalysis}), storing a
	 * per-parameter pin when the set is a singleton that diverges from the parameter's own dtype evidence, and the conflicting verdict for
	 * {@link #getHasConflictingEagerDtypeCoercions()} when some parameter's set is plural. An indeterminate collection (a partner operand
	 * with ⊤ dtype) fires neither, the allowing direction. A pair of parameters combined directly with each other under single, divergent
	 * evidence also sets the conflicting verdict (#878): no single signature, and no bare decorator, reproduces eager coercion there. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/861, Case 1.
	 *
	 * @param callGraph The call graph.
	 * @param tensorTypeAnalysis The tensor-type analysis whose typing supplies partner-operand dtypes.
	 * @param pointerAnalysis The pointer analysis, used to resolve a coercing call's callee.
	 */
	public void computeEagerDtypeCoercions(CallGraph callGraph, TensorTypeAnalysis tensorTypeAnalysis,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		Set<CGNode> nodes;

		try {
			nodes = this.getNodes(callGraph);
		} catch (CoreException e) {
			// Undeterminable; leave null so the precondition does not block.
			LOG.warn("Can't determine eager-effective dtypes for " + this + ".", e);
			return;
		}

		if (nodes.isEmpty()) {
			// Undeterminable without a call-graph node; leave null so the precondition does not block.
			LOG.info("Can't determine eager-effective dtypes for " + this + " without a call graph node.");
			return;
		}

		EagerCoercionAnalysis analysis = new EagerCoercionAnalysis(tensorTypeAnalysis, pointerAnalysis);
		List<Parameter> parameters = this.getParameters();
		boolean conflicting = false;
		Map<Parameter, DType> pins = new HashMap<>();
		Set<Parameter> changed = new HashSet<>();

		for (int position = 0; position < parameters.size(); position++) {
			Parameter parameter = parameters.get(position);

			// Keyword-only parameters are excluded: their IR value-number ordering relative to the positional block is unproven,
			// and a misaligned lookup here could pin the wrong dtype. Positional parameters map to value slots by list position
			// (keyword-only entries trail the positional block in `getParameters()`, so skipping them keeps `position` aligned).
			if (parameter.isSelf() || parameter.isKeywordOnly())
				continue;

			// Merge outcomes across the function's nodes (contexts): dtypes union, indeterminate if any node is, parameter partners
			// mapped back to their declared parameters by value slot.
			Set<DType> dtypes = new HashSet<>();
			boolean indeterminate = false;
			Set<Parameter> parameterPartners = new HashSet<>();

			for (CGNode node : nodes) {
				IR ir = node.getIR();

				if (ir == null)
					continue;

				int[] parameterValues = ir.getSymbolTable().getParameterValueNumbers();

				// Value slot 0 is the callable itself; declared parameter `position` sits at `position + 1`.
				if (position + 1 >= parameterValues.length)
					continue;

				// The fed-versus-imposed question, which the pin below cannot ask (wala/ML#838). UNRESOLVED is deliberately not
				// folded in with CHANGED: it means the fed side could not be established, not that it diverges, and declining on it
				// would cost a working function its transformation. Recorded, never treated as safe, never silently dropped.
				PointerKey parameterKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, parameterValues[position + 1]);
				AppliedDTypeCoercion coercion = tensorTypeAnalysis.getAppliedDTypeCoercions().get(parameterKey);

				if (coercion != null && coercion.resolution() == Resolution.CHANGED) {
					changed.add(parameter);
					LOG.info("Parameter " + parameter.getName() + " of " + this + " is fed " + coercion.fed() + " where its consumers "
							+ "impose " + coercion.imposed() + "; the conversion needs reproducing at the trace boundary.");
				}

				EagerCoercionAnalysis.Outcome outcome = analysis.eagerEffectiveDtypes(node, parameterValues[position + 1]);
				dtypes.addAll(outcome.dtypes());
				indeterminate |= outcome.indeterminate();

				for (int partnerValue : outcome.parameterPartners())
					for (int slot = 1; slot < parameterValues.length; slot++)
						if (parameterValues[slot] == partnerValue && slot - 1 < parameters.size()) {
							Parameter partner = parameters.get(slot - 1);

							// A keyword-only partner's slot alignment is unproven (the same exclusion the outer loop applies).
							if (!partner.isSelf() && !partner.isKeywordOnly())
								parameterPartners.add(partner);

							break;
						}
			}

			// A partner that is another parameter imposes nothing (wala/ML#828's exclusion), but a definite evidence divergence
			// between the pair is a definite hazard (#878): the eager program's survival implies a weak operand whose identity the
			// pair alone cannot decide, either-orientation pin breaks one reading, a spec naming the fed dtypes raises at the op,
			// and the bare decorator materializes the weak argument at its own dtype and raises the same way. No transformation
			// survives, which is this precondition's plural-set meaning. ⊤ or plural evidence on either side stays allowing, the
			// established polarity.
			DType own = singleEvidenceDtype(parameter);

			if (own != null)
				for (Parameter partner : parameterPartners) {
					DType partnerDtype = singleEvidenceDtype(partner);

					if (partnerDtype != null && partnerDtype != own) {
						conflicting = true;
						LOG.info("Parameters " + parameter.getName() + " and " + partner.getName() + " of " + this
								+ " are combined with each other under divergent evidence (" + own + " vs. " + partnerDtype + ").");
					}
				}

			if (indeterminate) {
				LOG.info("Parameter " + parameter.getName() + " of " + this + " has an indeterminate eager-effective dtype set.");
				continue;
			}

			if (dtypes.size() > 1) {
				conflicting = true;
				LOG.info("Parameter " + parameter.getName() + " of " + this + " has conflicting eager-effective dtypes: " + dtypes + ".");
			} else if (dtypes.size() == 1) {
				DType eagerEffective = dtypes.iterator().next();
				Set<DType> parameterDtypes = parameter.getTensorTypes().stream().map(TensorType::getDType).collect(Collectors.toSet());

				// The pin exists only on a determinate divergence: the parameter's own evidence is a single concrete dtype that
				// differs from the eager-effective one. Matching dtypes need no repair; ⊤ or mixed parameter evidence already drops
				// the spec elsewhere.
				if (parameterDtypes.size() == 1) {
					DType parameterDtype = parameterDtypes.iterator().next();

					if (parameterDtype != DType.UNKNOWN && parameterDtype != eagerEffective) {
						pins.put(parameter, eagerEffective);
						LOG.info("Parameter " + parameter.getName() + " of " + this + " pins to the eager-effective dtype " + eagerEffective
								+ " (evidence " + parameterDtype + ").");
					}
				}
			}
		}

		this.hasConflictingEagerDtypeCoercions = conflicting;
		this.eagerEffectiveDtypePins = pins;
		this.changedDtypeCoercions = changed;
	}

	/**
	 * Reduces a container parameter's per-position element evidence into a nested specification entry, or records the reason a position
	 * could not be reduced. The sequence reduction of #781, shared by the two routes that reach it: a parameter whose own evidence is
	 * unusable, and one whose own evidence exists but is outranked by the elements it is the union of (#888).
	 *
	 * @param param The container parameter whose elements to reduce. Its element types must be non-null.
	 * @param specByParameter Receives the nested entry when every position reduces.
	 * @param blocking Receives the first position's drop reason otherwise.
	 */
	private void reduceContainerElements(Parameter param, Map<Parameter, InputSignature.SpecEntry> specByParameter,
			Map<Parameter, AbsenceReason> blocking) {
		List<Set<TensorType>> elements = param.getContainerElementTypes();
		List<TensorType> reduced = new ArrayList<>(elements.size());
		AbsenceReason elementReason = null;

		for (int j = 0; j < elements.size() && elementReason == null; j++) {
			Optional<TensorType> elementSpec = inferSpec(elements.get(j));

			if (elementSpec.isPresent())
				reduced.add(elementSpec.get());
			else
				elementReason = this.reportElementDrop(param, j, elements.get(j));
		}

		if (elementReason == null)
			specByParameter.put(param, new InputSignature.Sequence(reduced));
		else
			blocking.put(param, elementReason);
	}

	/**
	 * True iff some parameter is fed a dtype its consumers do not impose and no emitted specification will carry the conversion. The repair
	 * lives in the specification, so it is unavailable exactly when none is written: inference is off, or inference is on and the
	 * specification turns out to be absent.
	 *
	 * @return True iff a required boundary conversion cannot be written.
	 */
	private boolean requiresUnwritableEagerDtypePin() {
		if (this.changedDtypeCoercions.isEmpty())
			return false;

		if (!this.getInferInputSignatures())
			return true;

		return !(this.inferInputSignature() instanceof InferenceResult.Inferred);
	}

	/**
	 * The single concrete dtype of a parameter's own evidence, or {@code null} when the evidence is absent, plural, or ⊤.
	 *
	 * @param parameter The parameter whose evidence to reduce.
	 * @return The single concrete evidence dtype, or {@code null} if none.
	 */
	private static DType singleEvidenceDtype(Parameter parameter) {
		Set<DType> dtypes = parameter.getTensorTypes().stream().map(TensorType::getDType).collect(Collectors.toSet());

		if (dtypes.size() != 1)
			return null;

		DType only = dtypes.iterator().next();
		return only == null || only == DType.UNKNOWN ? null : only;
	}

	/**
	 * True iff some parameter's direct consumers impose more than one concrete eager-effective dtype, {@code null} if undetermined.
	 *
	 * @return True iff no single input signature reproduces eager coercion, null if undetermined.
	 */
	public Boolean getHasConflictingEagerDtypeCoercions() {
		return this.hasConflictingEagerDtypeCoercions;
	}

	/**
	 * Computes which of {@code functions} have every known call path dominated by a hybridized caller (issue 767, phase 1), delegating to
	 * the package-private {@link CallerCoverageAnalysis}. Hybridization must already be computed for every function.
	 *
	 * @param functions Every function under analysis in the project.
	 * @param callGraph The call graph.
	 * @return The covered subset.
	 */
	public static Set<Function> computeCallerCoverage(Set<Function> functions, CallGraph callGraph) {
		return CallerCoverageAnalysis.computeCovered(functions, callGraph);
	}

	/**
	 * Sets whether every known call path to this {@link Function} is dominated by a hybridized caller (the processor's project-wide
	 * caller-coverage pass; issue 767).
	 *
	 * @param callerCovered Whether this function is caller-covered.
	 */
	public void setCallerCovered(boolean callerCovered) {
		this.callerCovered = callerCovered;
	}

	/**
	 * True iff every known call path to this {@link Function} is dominated by a hybridized caller, {@code null} if not computed.
	 *
	 * @return True iff this function is caller-covered, null if not computed.
	 */
	public Boolean getCallerCovered() {
		return this.callerCovered;
	}

	/**
	 * TensorFlow ops whose public signature is exactly {@code (x, name=None)}, so a second positional argument binds to {@code name}. Keyed
	 * by the trailing attribute name; {@code tf.sqrt} and {@code tf.math.sqrt} resolve to the same op and both match. Generated by
	 * enumerating the {@code tf}, {@code tf.math}, {@code tf.linalg}, and {@code tf.nn} namespaces of TensorFlow 2.9.3 with
	 * {@code inspect.signature} and keeping the callables with exactly the positional-or-keyword parameters {@code x} and {@code name}.
	 */
	private static final Set<String> UNARY_TENSORFLOW_OP_NAMES = Set.of("abs", "acos", "acosh", "asin", "asinh", "atan", "atanh",
			"bessel_i0", "bessel_i0e", "bessel_i1", "bessel_i1e", "ceil", "conj", "cos", "cosh", "digamma", "erf", "erfc", "erfcinv",
			"erfinv", "exp", "expm1", "floor", "invert_permutation", "is_finite", "is_inf", "is_nan", "is_non_decreasing",
			"is_strictly_increasing", "lbeta", "lgamma", "log", "log1p", "log_sigmoid", "logical_not", "ndtri", "negative", "reciprocal",
			"reciprocal_no_nan", "rint", "round", "rsqrt", "sigmoid", "sign", "sin", "sinh", "sqrt", "square", "tan", "tanh", "trace");

	/**
	 * Module aliases under which the fixture and subject code reference TensorFlow. The syntactic scan only recognizes an attribute chain
	 * rooted at one of these names; other import forms (e.g., {@code from tensorflow import sqrt}) are conservatively skipped.
	 */
	private static final Set<String> TENSORFLOW_MODULE_ALIASES = Set.of("tf", "tensorflow");

	/**
	 * Computes whether this {@link Function}'s body passes a non-string constant where a TensorFlow API declares its {@code name}
	 * parameter, storing the result for {@link #getHasInvalidNameArguments()}. Eager execution never validates the name, but tracing opens
	 * a name scope with it and raises, so such a function must not be hybridized. Unlike its sibling safety checks, this is a purely
	 * syntactic scan of the body; no call graph is consulted. Mirrors the siblings' undetermined discipline: if the body cannot be scanned,
	 * the result is left {@code null} and the precondition does not block. See
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/814.
	 */
	public void computeInvalidNameArguments() {
		boolean[] found = { false };

		try {
			this.getFunctionDefinition().getFunctionDef().traverse(new VisitorBase() {

				@Override
				public void traverse(SimpleNode node) throws Exception {
					node.traverse(this);
				}

				@Override
				protected Object unhandled_node(SimpleNode node) throws Exception {
					return null;
				}

				@Override
				public Object visitCall(Call node) throws Exception {
					if (passesInvalidNameArgument(node))
						found[0] = true;
					// Keep descending; the motivating case nests the offending call inside other calls.
					node.traverse(this);
					return null;
				}
			});
		} catch (Exception e) {
			// Undeterminable; leave null so the precondition does not block, mirroring the sibling safety checks.
			LOG.warn("Can't determine whether " + this + " passes an invalid name argument.", e);
			return;
		}

		this.hasInvalidNameArguments = found[0];

		LOG.info(this + (found[0] ? " passes an invalid name argument." : " passes no invalid name arguments."));
	}

	/**
	 * True iff the given call passes a non-string constant where the callee declares its {@code name} parameter. A keyword {@code name=...}
	 * argument binds unambiguously for any TensorFlow callee; a positional argument is only recognized for the ops in
	 * {@link #UNARY_TENSORFLOW_OP_NAMES}, whose signatures declare {@code name} second.
	 *
	 * @param call A call in this {@link Function}'s body.
	 * @return True iff the call passes a non-string constant as its {@code name} argument.
	 */
	private static boolean passesInvalidNameArgument(Call call) {
		if (!isTensorFlowRooted(call.func))
			return false;

		if (call.keywords != null)
			for (keywordType keyword : call.keywords)
				if (keyword.arg instanceof NameTok && "name".equals(((NameTok) keyword.arg).id) && isNonStringConstant(keyword.value))
					return true;

		String calleeName = NodeUtils.getRepresentationString(call.func);
		return calleeName != null && UNARY_TENSORFLOW_OP_NAMES.contains(calleeName) && call.args != null && call.args.length >= 2
				&& isNonStringConstant(call.args[1]);
	}

	/**
	 * True iff the given expression is an attribute chain rooted at a TensorFlow module alias (e.g., {@code tf.sqrt},
	 * {@code tf.math.sqrt}).
	 *
	 * @param expression The expression to inspect.
	 * @return True iff the expression is rooted at a TensorFlow module alias.
	 */
	private static boolean isTensorFlowRooted(exprType expression) {
		exprType root = expression;

		while (root instanceof Attribute)
			root = ((Attribute) root).value;

		return root instanceof Name && TENSORFLOW_MODULE_ALIASES.contains(((Name) root).id);
	}

	/**
	 * True iff the given expression is a constant that is definitely not a string: a numeric literal, a boolean literal, or a TensorFlow
	 * dtype constant (e.g., {@code tf.float32}). {@code None} is not flagged—it is the declared default for {@code name}—and anything whose
	 * value is unknown (a variable, a call) is conservatively accepted.
	 *
	 * @param expression The expression bound to a {@code name} argument.
	 * @return True iff the expression is a non-string constant.
	 */
	private static boolean isNonStringConstant(exprType expression) {
		if (expression instanceof Num)
			return true;

		if (expression instanceof Name) {
			String id = ((Name) expression).id;
			return "True".equals(id) || "False".equals(id);
		}

		// A dtype constant like `tf.float32` (the word2vec shape). Requiring the TensorFlow root keeps a bare variable that happens to
		// share a dtype's name (e.g., `half`) from being misread as a constant.
		return expression instanceof Attribute && isTensorFlowRooted(expression)
				&& HybridizationParameters.parseDType(expression).isPresent();
	}

	/**
	 * True iff this {@link Function}'s body passes a non-string constant where a TensorFlow API declares its {@code name} parameter,
	 * {@code null} if undetermined.
	 *
	 * @return True iff this function passes an invalid name argument, null if undetermined.
	 */
	public Boolean getHasInvalidNameArguments() {
		return this.hasInvalidNameArguments;
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
	 * Returns the given mod set less the writes exclusive to Keras lazy-{@code build} protocol methods reachable from the given
	 * {@link CGNode}. A sublayer's {@code build}, reached through the {@code __call__} trampoline, creates the layer's weights once, on the
	 * first call—a framework-sanctioned initialization that {@code tf.function} supports on the first trace—so its writes (the
	 * {@code add_weight} stores and the {@code built} flag) are not the recurring Python side-effects the side-effect precondition guards
	 * against. Subtraction is at write-level granularity (issue 728): a pointer key is subtracted only when every reachable direct writer
	 * of it is a {@code build} method, so a key that {@code build} initializes and a recurring method re-assigns (NLPGNN's
	 * {@code cached_result}) survives and keeps blocking.
	 *
	 * @param modSet The transitive mod set of the function under analysis.
	 * @param node The {@link CGNode} of the function under analysis.
	 * @param callGraph The system {@link CallGraph}.
	 * @param pointerAnalysis The system {@link PointerAnalysis}, used to compute per-node direct mod sets.
	 * @return The pointer keys in {@code modSet} not exclusively written by reachable {@code build} methods.
	 * @throws CoreException If resolving this function's {@link MethodReference} fails.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/720">Issue 720</a>
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/728">Issue 728</a>
	 */
	private Set<PointerKey> subtractBuildProtocolContributions(OrdinalSet<PointerKey> modSet, CGNode node, CallGraph callGraph,
			PointerAnalysis<InstanceKey> pointerAnalysis) throws CoreException {
		MethodReference thisMethodReference = this.getMethodReference();
		ModRef<InstanceKey> modRef = new PythonModRefWithBuiltinFunctions();
		// The Python mod visitor requires the pipeline's own AstHeapModel; wrap only a heap model that isn't already extended.
		ExtendedHeapModel heapModel = pointerAnalysis.getHeapModel() instanceof ExtendedHeapModel ehm ? ehm
				: new DelegatingExtendedHeapModel(pointerAnalysis.getHeapModel());
		Set<PointerKey> buildDirect = new HashSet<>();
		Set<PointerKey> nonBuildDirect = new HashSet<>();
		Set<CGNode> seen = new HashSet<>();
		Deque<CGNode> worklist = new ArrayDeque<>();

		seen.add(node);
		worklist.add(node);

		while (!worklist.isEmpty()) {
			CGNode current = worklist.remove();
			MethodReference reference = current.getMethod().getReference();

			// The function under analysis may itself be a `build` method; its own writes go to the protecting side.
			boolean isBuild = !reference.equals(thisMethodReference)
					&& reference.getDeclaringClass().getName().toString().endsWith("/build");

			(isBuild ? buildDirect : nonBuildDirect).addAll(getDirectMod(current, modRef, heapModel, pointerAnalysis));

			for (Iterator<CGNode> succNodes = callGraph.getSuccNodes(current); succNodes.hasNext();) {
				CGNode next = succNodes.next();

				if (seen.add(next))
					worklist.add(next);
			}
		}

		Set<PointerKey> ret = new HashSet<>();

		for (PointerKey pointerKey : modSet)
			if (!buildDirect.contains(pointerKey) || nonBuildDirect.contains(pointerKey))
				ret.add(pointerKey);

		LOG.info("Subtracted " + (modSet.size() - ret.size()) + " lazy-`build` protocol modified location(s).");

		return ret;
	}

	/**
	 * Returns the direct (non-transitive) mod set of the given {@link CGNode}: the pointer keys its own instructions write, excluding
	 * callee contributions. Memoized in {@link #directModCache} since the per-function closure walks revisit the same nodes.
	 *
	 * @param node The {@link CGNode} whose direct mod set to compute.
	 * @param modRef The {@link ModRef} used to interpret heap-writing instructions.
	 * @param heapModel The {@link ExtendedHeapModel} for pointer-key construction.
	 * @param pointerAnalysis The system {@link PointerAnalysis}.
	 * @return The pointer keys directly written by {@code node}.
	 */
	private static Set<PointerKey> getDirectMod(CGNode node, ModRef<InstanceKey> modRef, ExtendedHeapModel heapModel,
			PointerAnalysis<InstanceKey> pointerAnalysis) {
		return directModCache.computeIfAbsent(node, n -> {
			Set<PointerKey> ret = new HashSet<>();
			IR ir = n.getIR();

			if (ir != null)
				for (SSAInstruction instruction : Iterator2Iterable.make(ir.iterateNormalInstructions()))
					ret.addAll(modRef.getMod(n, heapModel, pointerAnalysis, instruction, null));

			return ret;
		});
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

	/**
	 * Returns true iff the refactoring should emit an inferred {@code input_signature} keyword into the generated decorator.
	 *
	 * @return True iff the refactoring should emit an inferred {@code input_signature} keyword into the generated decorator.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	public boolean getInferInputSignatures() {
		return this.inferInputSignatures;
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

	/**
	 * The one-based line number on which this {@link Function}'s definition begins in its containing file. Distinguishes two bindings of
	 * one name in one module (Python legally rebinds a def), which the name-based identity columns alone cannot; see
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/860.
	 *
	 * @return The definition's beginning line number.
	 */
	public int getBeginningLineNumber() {
		return this.getFunctionDefinition().getFunctionDef().beginLine;
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

				if (!pointsToSet.isEmpty() && allInstancesArePrimitive && !this.isDefaultedUnsuppliedPrimitive(paramInx)) {
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
	 * Returns true iff the positional parameter at the given IR parameter index is a defaulted parameter that no call site supplies, and so
	 * is exempt from {@link PreconditionFailure#HAS_PRIMITIVE_PARAMETERS}. A defaulted parameter no caller supplies is always the default,
	 * a single constant value, so it induces no retracing and is not the retrace risk the precondition guards against. This is the
	 * supplied-parameter analysis of {@link #inferSuppliedParameters} (#788) applied on the primitive-parameter axis rather than the
	 * input-signature axis (#795). Conservative on ignorance: only {@link Parameter#isSuppliedAtCallSite()} {@code == FALSE} (definitely
	 * not supplied) exempts; {@code null} (undetermined) does not.
	 * <p>
	 * The IR parameter index maps to the positional {@link Parameter} at {@link Parameter#getIndex()} {@code == irParamIndex - 1} (IR slot
	 * 0 is the callable, so slot 1 is the first declared parameter, {@code self} for a method). Keyword-only parameters share the
	 * positional index space and are excluded via {@link Parameter#isKeywordOnly()}; a parameter that cannot be mapped is not exempt.
	 *
	 * @param irParamIndex The IR parameter index examined by {@link #inferPrimitiveParameters}.
	 * @return True iff that parameter is a defaulted, definitely-unsupplied positional parameter.
	 */
	private boolean isDefaultedUnsuppliedPrimitive(int irParamIndex) {
		int declarationIndex = irParamIndex - 1;
		List<Parameter> parameters = this.getParameters();

		if (declarationIndex < 0 || declarationIndex >= parameters.size())
			return false;

		Parameter parameter = parameters.get(declarationIndex);

		return !parameter.isKeywordOnly() && !parameter.isSelf() && parameter.getIndex() == declarationIndex && parameter.hasDefault()
				&& FALSE.equals(parameter.isSuppliedAtCallSite());
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

		// Subtract the Keras lazy-`build` protocol contributions.
		Set<PointerKey> modSetLessBuild = this.subtractBuildProtocolContributions(modSet, cgNode, callGraph, pointerAnalysis);

		// Filter out the modified locations.
		Set<PointerKey> filteredModSet = this.filterSideEffects(modSetLessBuild, callGraph, pointerAnalysis);
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
				this.tensorParameterFromSpeculation = true;
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
	 * Determines, for each defaulted parameter, whether any call site supplies an argument for it, caching the answer on the
	 * {@link Parameter} for {@link #computeInputSignature()} to read (#787).
	 * <p>
	 * A defaulted non-tensor parameter can be omitted from an inferred {@code input_signature}, because TensorFlow requires a
	 * {@code TensorSpec} per <em>required</em> argument rather than per argument. Omitting it changes the hybridized function's arity,
	 * though: the function then accepts exactly the arguments the signature names, so a call site that supplied the parameter would start
	 * raising a {@code TypeError}. The omission is therefore behavior-preserving only when no call site supplies it, which is a
	 * whole-program question rather than a syntactic one.
	 * <p>
	 * This and the Keras symbolic-argument analysis ({@link KerasSymbolicArgumentAnalysis}, #887) are the only analyses in the plug-in that
	 * walk the call graph in the caller direction; the others ask what a function's body reaches. Two wrinkles follow from that:
	 * <ul>
	 * <li><b>Trampolines.</b> Ariadne interposes a synthesized trampoline between a caller and an instance method to bind {@code self}, so
	 * a method's predecessors are trampolines rather than user code. The trampoline forwards the originating call's arguments verbatim, so
	 * the arity is preserved at the forwarded invoke and no second hop is needed.
	 * <li><b>Shared trampolines.</b> Through Ariadne 0.52.33, trampolines were cached per {@code (receiver, total argument count)}, summing
	 * positional and keyword arguments, so {@code f(x, False)} and {@code f(x, training=False)} could share one body shaped by whichever
	 * was seen first (wala/ML#740, fixed in 0.52.34 by keying on the receiver, the positional count, and the keyword-name set). The bundled
	 * version predates the fix, and the collision costs precision rather than correctness here either way: both colliding shapes mean an
	 * argument beyond the minimum was supplied, so the check reports "supplied" for both and can only over-block. The fix changes nothing
	 * this relies on; it makes the keyword names exact.
	 * </ul>
	 * Ignorance is recorded as {@code null} rather than {@code FALSE}: no call-graph node, an unresolvable predecessor, or a non-Python
	 * invoke all leave the parameter un-omittable, matching how every other axis of the reduction treats absent evidence.
	 *
	 * @param callGraph The call graph to query for this function's callers.
	 * @param monitor Progress monitor for the sub-work.
	 * @throws CoreException If the call-graph node lookup fails.
	 */
	public void inferSuppliedParameters(CallGraph callGraph, IProgressMonitor monitor) throws CoreException {
		List<Parameter> parameters = this.getParameters();
		SubMonitor subMonitor = SubMonitor.convert(monitor, "Inferring supplied parameters...", parameters.size());

		// Only a defaulted parameter can ever be omitted, so only it needs the question asked.
		List<Parameter> defaulted = parameters.stream().filter(p -> !p.isSelf()).filter(Parameter::hasDefault).toList();

		if (defaulted.isEmpty()) {
			subMonitor.done();
			return;
		}

		Set<CGNode> nodes = this.getNodes(callGraph);

		if (nodes.isEmpty()) {
			// No node means no observable call site, which is ignorance rather than evidence of absence. Leave the cache null.
			LOG.info("Can't determine supplied parameters for " + this + " without a call graph node.");
			subMonitor.done();
			return;
		}

		for (Parameter parameter : defaulted) {
			parameter.setSuppliedAtCallSite(this.isSuppliedAtAnyCallSite(parameter, nodes, callGraph));
			LOG.info("Parameter " + parameter + " supplied at a call site: " + parameter.isSuppliedAtCallSite() + ".");
			subMonitor.worked(1);
		}

		subMonitor.done();
	}

	/**
	 * Returns whether any call site of this function supplies an argument for the given parameter.
	 *
	 * @param parameter The parameter in question; assumed non-{@code self}.
	 * @param nodes The call-graph nodes of this function.
	 * @param callGraph The call graph.
	 * @return {@code TRUE} if some call site supplies it, {@code FALSE} if every reachable call site omits it, or {@code null} if any call
	 *         site could not be examined or an unpacked positional argument made its alignment undetermined (wala/ML#751). See
	 *         {@link #inferSuppliedParameters} for why ignorance is {@code null}.
	 */
	private Boolean isSuppliedAtAnyCallSite(Parameter parameter, Set<CGNode> nodes, CallGraph callGraph) {
		String name = parameter.getName();

		// The callee occupies positional slot 0 of the invoke, so the parameter at declaration index i (self at 0) is the invoke's
		// positional argument i + 1. This is the same off-by-one `tensorAnalysisIncludesParameterContainer` applies as `paramInx + 1`.
		int positionalSlot = parameter.getIndex() + 1;

		boolean sawCallSite = false;
		boolean sawIndeterminate = false;

		for (CGNode node : nodes)
			for (CGNode predecessor : Iterator2Iterable.make(callGraph.getPredNodes(node))) {
				IR ir = predecessor.getIR();

				if (ir == null) {
					LOG.warn("No IR for predecessor: " + predecessor + " of: " + this + ".");
					return null;
				}

				for (CallSiteReference site : Iterator2Iterable.make(callGraph.getPossibleSites(predecessor, node)))
					for (SSAAbstractInvokeInstruction instruction : ir.getCalls(site)) {
						if (!(instruction instanceof PythonInvokeInstruction invoke)) {
							LOG.warn("Not expecting a non-Python invoke: " + instruction + " calling: " + this + ".");
							return null;
						}

						sawCallSite = true;

						// A keyword argument supplies the parameter regardless of positional layout.
						if (invoke.getKeywords().contains(name))
							return TRUE;

						// A starred (unpacked) positional argument at or before the parameter's slot collapses a
						// statically-unknown number of arguments into one slot, so the positional alignment past it is unreliable:
						// the parameter may or may not be supplied by the unpack (wala/ML#751). Treat this call site as undetermined
						// rather than concluding it omits the parameter.
						if (hasStarredArgumentAtOrBefore(invoke, positionalSlot)) {
							sawIndeterminate = true;
							continue;
						}

						// No unpack precedes the slot, so the positional count is exact: the parameter is supplied iff a positional
						// argument occupies its slot.
						if (invoke.getNumberOfPositionalParameters() > positionalSlot)
							return TRUE;
					}
			}

		// TRUE (returned above) means some call site definitely supplies it. Absent that, a call site whose alignment an unpack made
		// unreliable leaves the answer undetermined; every reachable call site definitely omitting it is FALSE; and no call site at all
		// is ignorance, not absence.
		if (!sawCallSite || sawIndeterminate)
			return null;

		return FALSE;
	}

	/**
	 * Returns whether {@code invoke} has a starred (unpacked) positional argument at or before {@code positionalSlot}, past which the
	 * positional-to-parameter alignment is unreliable because an unpack collapses a statically-unknown number of arguments into one slot
	 * (wala/ML#751).
	 *
	 * @param invoke The call.
	 * @param positionalSlot The invoke positional slot of the parameter in question.
	 * @return True iff a starred positional argument occupies a slot at or before {@code positionalSlot}.
	 */
	private static boolean hasStarredArgumentAtOrBefore(PythonInvokeInstruction invoke, int positionalSlot) {
		for (int starredPosition : invoke.getStarredPositions())
			if (starredPosition <= positionalSlot)
				return true;

		return false;
	}

	/**
	 * Infers the input signature of this function: an ordered tuple of {@link TensorType}s, one per non-{@code self} parameter the
	 * tensor-type analysis associated with at least one tensor type. Mirrors the no-argument pattern of {@link #getHasTensorParameter}: the
	 * values are computed during {@link #inferTensorParameters} (which caches per-parameter tensor types on each {@link Parameter}), and
	 * this method reads those cached values.
	 * <p>
	 * A function whose tensor-parameter verdict came from speculative context analysis is blocked up front with
	 * {@link InferenceResult.AbsenceReason#SPECULATIVE_TENSOR_PARAMETER} and a single function-level INFO, before the per-parameter
	 * dispatch runs: context names no particular parameter and carries no shape or dtype evidence (#783).
	 * <p>
	 * Otherwise, for each non-{@code self} parameter, this method dispatches on {@link Parameter#isTensor()} into three categories:
	 * <ul>
	 * <li>Truly non-tensor ({@code isTensor() != TRUE}): drop the signature and emit a per-parameter INFO suggesting the source-side
	 * recovery (annotate as {@code tf.Tensor} and wrap call sites with {@code tf.constant(...)}). The tool does not synthesize a
	 * {@link TensorType} for the parameter because wrapping a Python primitive as a tensor changes AutoGraph's rewrite of Python control
	 * flow over the parameter.
	 * <li>Tensor-classified by type hint or container detection but no conforming Phase 2 entry
	 * ({@code isTensor() && getConformingTensorTypes().isEmpty()}, which a parameter whose every observed type came from an
	 * expected-failure call site also reaches): the two ways to land here now diverge (#781). A container
	 * ({@link Parameter#isTensorContainer()} {@code == TRUE}) whose element types were surfaced
	 * ({@link Parameter#getContainerElementTypes()}) reduces each position through {@link #inferSpec} and contributes a
	 * {@link InputSignature.Sequence} entry; a container of an unmodeled form blocks with
	 * {@link InferenceResult.AbsenceReason#TENSOR_CONTAINER_UNSUPPORTED}, and disagreeing sequence lengths block with
	 * {@link InferenceResult.AbsenceReason#HETEROGENEOUS_ARITY}. A type hint carries no dtype at all, and since an input signature admits
	 * no dtype-⊤ (#494), there is nothing to synthesize; it blocks with a per-parameter INFO and no follow-up to cite.
	 * <li>Phase-2 hit ({@code isTensor() && !getConformingTensorTypes().isEmpty()}): reduce the conforming set via {@link #inferSpec} and
	 * add the reduced spec to the signature.
	 * </ul>
	 * Current scope: a single tensor type per parameter, with concrete dtype and concrete shape. Multi-context (#507) and other
	 * non-concrete cases (#494) yield an {@link InferenceResult.Absent} carrying the blocking {@link InferenceResult.AbsenceReason} pending
	 * future PRs that extend {@link #inferSpec}.
	 * <p>
	 * The result is memoized: the per-parameter INFOs emitted as a side effect are added at most once even though several call sites
	 * (precondition checking, import injection, and the transform paths) request the signature within a single pass.
	 *
	 * @return An {@link InferenceResult.Inferred} carrying the signature, or an {@link InferenceResult.Absent} carrying the first blocking
	 *         {@link InferenceResult.AbsenceReason} when a parameter cannot be reduced to a concrete spec.
	 * @throws IllegalStateException If this function has no non-{@code self} parameter (it is parameterless or {@code self}-only). A
	 *         non-tensor parameter does not trigger this—it yields an {@link InferenceResult.Absent}. Every refactoring call site is gated
	 *         on {@link #getHasTensorParameter}, so the throw signals a direct, unguarded misuse rather than a normal "nothing to infer"
	 *         outcome.
	 */
	public InferenceResult inferInputSignature() {
		if (this.inferredInputSignature == null)
			this.inferredInputSignature = this.computeInputSignature();

		return this.inferredInputSignature;
	}

	/**
	 * Returns the memoized inferred input signature without triggering its computation. Unlike {@link #inferInputSignature()}, this never
	 * runs inference (and so never emits the per-parameter INFOs): it reports only what a prior call already computed during analysis or
	 * transformation. Returns {@link Optional#empty} both when inference was never requested for this function and when it was requested
	 * but blocked. Intended for read-only reporting (e.g. the evaluator) that must not perturb the function's status.
	 *
	 * @return The memoized inferred signature, or {@link Optional#empty} if it was not computed or did not reduce to one.
	 */
	public Optional<InputSignature> getInferredInputSignature() {
		return this.inferredInputSignature == null ? Optional.empty() : this.inferredInputSignature.signature();
	}

	/**
	 * Returns the reason a signature was not inferred, from the memoized result, without triggering inference. The side-effect-free
	 * counterpart of {@link #getInferredInputSignature()}: {@link Optional#empty} both when inference was never requested and when it
	 * succeeded; present only when a prior call computed an {@link InferenceResult.Absent}. Intended for read-only reporting (e.g. the
	 * evaluator) that must not perturb the function's status.
	 *
	 * @return The memoized absence reason, or {@link Optional#empty} if inference was not computed or did produce a signature.
	 */
	public Optional<AbsenceReason> getInferredInputSignatureAbsenceReason() {
		return this.inferredInputSignature == null ? Optional.empty() : this.inferredInputSignature.absenceReason();
	}

	/**
	 * Returns the blocking {@link InferenceResult.AbsenceReason} for each parameter that prevented input-signature inference, in parameter
	 * declaration order, from the memoized result without triggering inference. Where {@link #getInferredInputSignatureAbsenceReason()}
	 * collapses the function to its first blocking reason, this surfaces every blocking parameter so a consumer can report per-parameter
	 * attribution (#654). Empty when inference was never run, when it produced a signature, and when it was blocked at the function level
	 * by {@link InferenceResult.AbsenceReason#SPECULATIVE_TENSOR_PARAMETER}, since no parameter is the blocker in that case.
	 *
	 * @return An unmodifiable map from each blocking {@link Parameter} to its {@link InferenceResult.AbsenceReason}, in declaration order.
	 */
	public Map<Parameter, AbsenceReason> getBlockingParameterReasons() {
		return Collections.unmodifiableMap(this.blockingParameterReasons);
	}

	/**
	 * Computes the inferred input signature. Always recomputes; {@link #inferInputSignature()} memoizes the result. Emits the recovery
	 * INFOs as a side effect: one function-level INFO on the speculative short-circuit, otherwise one per blocking parameter. The
	 * {@link InferenceResult.Absent} result carries the <em>first</em> blocking {@link InferenceResult.AbsenceReason} encountered, but the
	 * loop still runs to completion so every blocking parameter surfaces its INFO in one pass.
	 *
	 * @return The {@link InferenceResult}. See {@link #inferInputSignature()} for the contract, including the no-non-self-parameter throw.
	 */
	private InferenceResult computeInputSignature() {
		/*
		 * The function-level tensor-parameter verdict came from context, not from any parameter. Speculation fires only when no parameter
		 * classified, so every parameter's `isTensor()` is FALSE with an empty `getTensorTypes()`, and the dispatch below would report
		 * NON_TENSOR_PARAMETER for each one. That contradicts the SPECULATIVE_ANALYSIS INFO this same pass emitted, and its recovery advice
		 * points at code the tool just judged tensor-typed by context. Speculation fires precisely because Ariadne could not type the
		 * parameter, so the likely cause is analysis incompleteness rather than a genuinely non-tensor parameter. Report the honest reason
		 * once instead. Verdict-neutral: the dispatch is already guaranteed to return `Absent` here (#783).
		 */
		/*
		 * An `input_signature` fixes what the function accepts, so a rest-keyword slot stops absorbing anything and a caller passing a
		 * keyword raises where it used to succeed. Keras is that caller for a layer, since it reads `call`'s declaration and passes
		 * `training` precisely because `**kwargs` says it may. Decided here on the declaration rather than per parameter: the slot names no
		 * parameter of its own, and the callers that make it fatal are outside the analyzed program, so no call-site evidence would settle
		 * it. The function still converts with a bare decorator; only the specification is withheld (#902).
		 */
		if (this.hasVariableKeywordParameter()) {
			this.addInfo(INPUT_SIGNATURE_INFERENCE,
					"`" + this + "` declares a `**kwargs` parameter. An input signature fixes the arguments the function accepts, so that "
							+ "parameter would stop absorbing keyword arguments and any caller passing one would fail; for a Keras layer "
							+ "that caller is Keras itself, which passes `training`. Input-signature inference is dropped and the function "
							+ "is hybridized with a bare decorator.");
			return new InferenceResult.Absent(AbsenceReason.VARIABLE_KEYWORD_PARAMETER);
		}

		if (this.tensorParameterFromSpeculation) {
			this.addInfo(INPUT_SIGNATURE_INFERENCE,
					"`" + this + "` is classified as having a tensor parameter from its context rather than from any particular parameter, "
							+ "so no shape or dtype evidence is available and input-signature inference is dropped.");
			return new InferenceResult.Absent(AbsenceReason.SPECULATIVE_TENSOR_PARAMETER);
		}

		List<Parameter> nonSelfParameters = this.getParameters().stream().filter(p -> !p.isSelf()).toList();

		// Keyed by parameter rather than a bare list: the suffix rule below needs each spec's declaration position.
		Map<Parameter, InputSignature.SpecEntry> specByParameter = new LinkedHashMap<>();
		Map<Parameter, AbsenceReason> blocking = new LinkedHashMap<>();

		// Defaulted, non-tensor, and supplied by no call site, so omittable from the signature entirely. Provisional until the suffix rule
		// runs: a parameter is only genuinely omittable if nothing after it needs a spec.
		Set<Parameter> omittable = new LinkedHashSet<>();

		for (Parameter param : nonSelfParameters) {
			Boolean classified = param.isTensor();
			if (classified == null || !classified) {
				/*
				 * Category (a): not tensor-typed. TensorFlow requires a `TensorSpec` per *required* argument rather than per argument, so a
				 * parameter declaring a default may be omittable instead of blocking (#787). Ask that first; only a required parameter is
				 * unconditionally fatal.
				 */
				if (param.hasDefault()) {
					// Keras supplies `training`/`mask` to a `call` override through `Layer.__call__` on every invocation, from
					// outside the analyzed program, so a call-site examination cannot see the supplier: omitting the parameter cuts
					// the hybridized function's arity below what the dispatch passes, and the first call raises (#881). The gate is
					// local to this reduction rather than folded into `isSuppliedAtCallSite`, whose other consumer (the #795
					// star-arg exemption) must keep treating a source-unsupplied parameter as bound to its default.
					if (KERAS_FRAMEWORK_SUPPLIED_PARAMETER_NAMES.contains(param.getName()) && this.isKerasCallOverride()) {
						this.addInfo(INPUT_SIGNATURE_INFERENCE,
								"Parameter `" + param.getName() + "` of `" + this + "` is supplied by Keras itself on every "
										+ "invocation, so it cannot be left out of an input signature, and being non-tensor it has "
										+ "no spec; input-signature inference is dropped.");
						blocking.put(param, AbsenceReason.DEFAULTED_PARAMETER_SUPPLIED);
						continue;
					}

					Boolean supplied = param.isSuppliedAtCallSite();

					if (supplied != null && !supplied) {
						// Provisionally omittable. No INFO yet: the suffix rule may still block it, and only one of the two outcomes
						// should reach the user.
						omittable.add(param);
						continue;
					}

					// Supplied somewhere, or the call sites could not be examined. Omitting it would cut the hybridized function's arity
					// below what a caller passes, so it must be covered, and no spec exists for it. `null` lands here deliberately:
					// ignorance is not evidence of absence.
					this.addInfo(INPUT_SIGNATURE_INFERENCE,
							"Parameter `" + param.getName() + "` of `" + this + "` has a default value but is passed explicitly by at "
									+ "least one caller, so it cannot be left out of an input signature, and being non-tensor it has no "
									+ "spec; input-signature inference is dropped. Removing the argument from those call sites, so the "
									+ "default always applies, would let the parameter be omitted.");
					blocking.put(param, AbsenceReason.DEFAULTED_PARAMETER_SUPPLIED);
					continue;
				}

				// Required and non-tensor. The developer's source code is correct as-is; this is a design opportunity, not a
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
				blocking.put(param, AbsenceReason.NON_TENSOR_PARAMETER);
				continue;
			}

			// The reduction reads the conforming evidence, not everything observed: a call the callee is specified to reject is not
			// evidence of what it accepts (#888). Classification above deliberately still reads every node.
			Set<TensorType> contexts = param.getConformingTensorTypes();

			// Everything observed for this parameter came from a declared expected failure, so nothing of its own is left to reduce. What
			// the conforming callers pass may still be a container, whose element structure classification surfaced for exactly this case,
			// so the container branch below runs first and only its failure reports the exclusion (#888).
			boolean expectedFailureEvidenceOnly = contexts.isEmpty() && !param.getTensorTypes().isEmpty();

			// A container's element evidence outranks the parameter's own typing, and does so however much of the latter there is. A tuple
			// of tensors reaching a parameter is reported as the union of its elements' types, so the flat typing here is those very
			// elements flattened: reducing it writes one specification whose disagreeing members collapse to a wildcard, and binding a
			// pair to that specification makes the body's unpacking raise. The elements are what the callers actually pass, position by
			// position, so the nested structure is both the more precise answer and the only executable one (#888).
			if (param.isTensorContainer() != null && param.isTensorContainer() && param.getContainerElementTypes() != null
					&& !contexts.isEmpty()) {
				this.reduceContainerElements(param, specByParameter, blocking);
				continue;
			}

			if (contexts.isEmpty()) {
				/*
				 * Category (b): tensor-classified without conforming Phase 2 (Ariadne call-site) shape/dtype evidence. The ways to land
				 * here have different evidence situations, so each names its own disposition rather than sharing one tracker (#782). Phase
				 * 3 (container) leaves `isTensorContainer()` TRUE; Phase 1 (type hint) returns before the container question is asked,
				 * leaving it null. FALSE reaches here only on the expected-failure route (#888), where a Phase 2 hit whose every type was
				 * excluded asked the container question and got no for an answer; on the Phase 3 route a FALSE verdict falls through to
				 * `tensor = FALSE`, i.e. category (a).
				 */
				AbsenceReason reason;

				if (param.isTensorContainer() != null && param.isTensorContainer()) {
					// The sequence reduction (#781): a list or tuple of tensors with a modeled element structure reduces each element
					// position independently through `inferSpec`, and the parameter contributes a nested entry rather than blocking.
					if (param.getContainerElementTypes() != null) {
						this.reduceContainerElements(param, specByParameter, blocking);
						continue;
					}

					if (param.getContainerAritiesDisagree() != null && param.getContainerAritiesDisagree()) {
						// TensorFlow rejects a sequence of a different length than the signature declares, and no wildcard length
						// exists, so no single signature admits call sites passing lists of disagreeing lengths.
						this.addInfo(INPUT_SIGNATURE_INFERENCE,
								"Parameter `" + param.getName() + "` of `" + this + "` receives lists of tensors whose lengths disagree "
										+ "across call sites; an input signature fixes the length, so no single signature admits them "
										+ "all and input-signature inference is dropped.");
						reason = AbsenceReason.HETEROGENEOUS_ARITY;
					} else {
						// A container form the reduction does not model: a dict or set, an element structure the analysis cannot
						// enumerate, a nested container, a position without tensor evidence, or container and non-container values
						// mixed at the call sites. The wizard text deliberately cites no tracker (#782).
						this.addInfo(INPUT_SIGNATURE_INFERENCE,
								"Parameter `" + param.getName() + "` of `" + this + "` is classified as a container of tensors of a "
										+ "form not currently reduced to an input signature; the signature is dropped.");
						reason = AbsenceReason.TENSOR_CONTAINER_UNSUPPORTED;
					}
				} else if (expectedFailureEvidenceOnly) {
					// Tensor-typed, but only by calls the tests declare must fail, and the conforming callers passed no container either.
					// Nothing a specification may be derived from is left, so the function is hybridized with a bare decorator.
					this.addInfo(INPUT_SIGNATURE_INFERENCE,
							"Every tensor type observed for parameter `" + param.getName() + "` of `" + this + "` comes from a call site "
									+ "the tests declare must fail, so it describes an input the function rejects rather than one it "
									+ "accepts, and the conforming call sites carry no container evidence in its place; input-signature "
									+ "inference is dropped and the function is hybridized with a bare decorator.");
					reason = AbsenceReason.EXPECTED_FAILURE_EVIDENCE_ONLY;
				} else {
					// A bare `x: tf.Tensor` annotation carries no dtype, and `tf.function(input_signature=...)` admits no dtype-⊤ (#494),
					// so there is no valid `TensorSpec` to synthesize from this signal and no follow-up to point at.
					this.addInfo(INPUT_SIGNATURE_INFERENCE,
							"Parameter `" + param.getName() + "` of `" + this + "` is classified as tensor-typed via its type hint, but a "
									+ "type hint carries no dtype and an input signature requires a concrete one; input-signature "
									+ "inference is dropped. Passing `tf.constant(...)` at the call sites would supply the missing "
									+ "shape and dtype evidence.");
					reason = AbsenceReason.TYPE_HINT_WITHOUT_DTYPE;
				}

				blocking.put(param, reason);
				continue;
			}

			DType pin = this.eagerEffectiveDtypePins.get(param);

			if (pin != null) {
				// The repair direction of issue 861, Case 1: the parameter's direct consumers all impose one concrete dtype that
				// diverges from the argument evidence, so the spec pins the eager-effective dtype. The boundary cast then reproduces
				// exactly what eager execution did at each op (runtime-verified on the pinned TF 2.9.3), where emitting the observed
				// dtype would carry the mismatch into the trace.
				Optional<TensorType> observed = inferSpec(contexts);

				if (observed.isPresent()) {
					TensorType pinned = new TensorType(pin, observed.get().getDims());
					this.addInfo(INPUT_SIGNATURE_INFERENCE,
							"Parameter `" + param.getName() + "` of `" + this + "` receives arguments observed as "
									+ observed.get().getDType() + " but combines them with " + pin + " tensors; eager execution coerces "
									+ "at each operation, so the spec pins the eager-effective dtype " + pin + ".");
					specByParameter.put(param, new InputSignature.Single(observed.get().isSparse() ? pinned.asSparse() : pinned));
					continue;
				}
			}

			Optional<TensorType> spec = inferSpec(contexts);
			if (spec.isEmpty()) {
				/*
				 * `inferSpec` reduced to bottom. With the per-context reduction
				 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/480) it drops for three reasons: heterogeneous
				 * dtype (|D| ≠ 1), dtype-⊤ (a single agreed `UNKNOWN`), or mixed sparse/dense
				 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/642). Shape-⊤ and symbolic-dim no longer drop
				 * here—it emits a coarse `TensorType(dtype, null)` or a `SymbolicDim` wildcard instead. Classify in `inferSpec`'s own
				 * precedence order (dtype before sparseness) so the reason is exact, and emit a per-parameter INFO naming it; see
				 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/510.
				 */
				boolean heterogeneousDtype = contexts.stream().map(TensorType::getDType).distinct().count() > 1;
				boolean unknownDtype = !heterogeneousDtype && contexts.stream().anyMatch(t -> t.getDType() == DType.UNKNOWN);
				AbsenceReason reason;
				if (heterogeneousDtype) {
					this.addInfo(INPUT_SIGNATURE_INFERENCE, "Parameter `" + param.getName() + "` of `" + this
							+ "` receives tensors with conflicting dtypes across call sites, so a single input signature cannot be inferred; it is dropped.");
					reason = AbsenceReason.HETEROGENEOUS_DTYPE;
				} else if (unknownDtype) {
					this.addInfo(INPUT_SIGNATURE_INFERENCE, "Parameter `" + param.getName() + "` of `" + this
							+ "` receives a tensor whose dtype cannot be determined, so a single input signature cannot be inferred; it is dropped.");
					reason = AbsenceReason.UNKNOWN_DTYPE;
				} else {
					this.addInfo(INPUT_SIGNATURE_INFERENCE, "Parameter `" + param.getName() + "` of `" + this
							+ "` is sparse at some call sites and dense at others, so a single input signature cannot be inferred; it is dropped.");
					reason = AbsenceReason.HETEROGENEOUS_SPARSITY;
				}
				blocking.put(param, reason);
				continue;
			}

			specByParameter.put(param, new InputSignature.Single(spec.get()));
		}

		/*
		 * Suffix rule. `input_signature` covers a prefix of the parameter list positionally, so an omittable parameter can only actually be
		 * omitted when nothing after it contributes a spec: dropping one that precedes a spec would bind that spec to the dropped
		 * parameter's position. Python's grammar already forces defaults to trail the non-defaulted parameters, but it permits a defaulted
		 * tensor parameter after a defaulted non-tensor one, which is the `def call(self, x, training=False, mask=None)` shape (#787).
		 */
		int lastSpecPosition = -1;
		for (int i = 0; i < nonSelfParameters.size(); i++)
			if (specByParameter.containsKey(nonSelfParameters.get(i)))
				lastSpecPosition = i;

		for (int i = 0; i < lastSpecPosition; i++) {
			Parameter param = nonSelfParameters.get(i);

			if (omittable.remove(param)) {
				this.addInfo(INPUT_SIGNATURE_INFERENCE,
						"Parameter `" + param.getName() + "` of `" + this + "` has a default value and no caller passes it, but a later "
								+ "parameter needs a spec, and an input signature covers parameters by position; leaving it out would "
								+ "apply the later parameter's spec to it. Input-signature inference is dropped.");
				blocking.put(param, AbsenceReason.DEFAULTED_PARAMETER_PRECEDES_SPEC);
			}
		}

		// Report per-parameter reasons in declaration order. The suffix rule can block a parameter that precedes one already blocked in
		// the first pass, so insertion order is not declaration order.
		Map<Parameter, AbsenceReason> orderedBlocking = new LinkedHashMap<>();
		for (Parameter param : nonSelfParameters)
			if (blocking.containsKey(param))
				orderedBlocking.put(param, blocking.get(param));

		this.blockingParameterReasons = orderedBlocking;

		// A signature must be total over the parameters: any blocking reason makes the whole result Absent, even if some parameters
		// reduced. The reason carried is the first in declaration order.
		if (!orderedBlocking.isEmpty())
			return new InferenceResult.Absent(orderedBlocking.values().iterator().next());

		// Degenerate case: nothing blocked, yet no parameter contributed a spec, so there is nothing to describe. Either the function has
		// no non-`self` parameter at all, or every one of them was omittable, which means none is a tensor. Both are unreachable from the
		// refactoring, whose call sites are gated on `getHasTensorParameter()`; reaching here signals a direct, unguarded call.
		if (specByParameter.isEmpty())
			throw new IllegalStateException("Cannot infer an input signature for `" + this + "`: no non-self parameter contributes a spec ("
					+ nonSelfParameters.size() + " non-self parameter(s), " + omittable.size()
					+ " omittable). Refactoring call sites are gated on `getHasTensorParameter()`.");

		// Retain the per-parameter mapping for consumers needing parameter-level attribution of the reduced spec (the unresolved
		// statically-read-axis precondition resolves its parameter ordinals through it; issue 811).
		this.inferredSpecByParameter = Collections.unmodifiableMap(specByParameter);

		return new InferenceResult.Inferred(new InputSignature(new ArrayList<>(specByParameter.values())));
	}

	/**
	 * Classifies and reports one element position of a sequence-container parameter whose {@link #inferSpec} reduced to bottom, mirroring
	 * the flat parameter's drop classification (dtype disagreement, dtype-⊤, then mixed sparseness, in {@link #inferSpec}'s own precedence
	 * order) with the element position named in the diagnostic (#781).
	 *
	 * @param param The sequence-container parameter.
	 * @param element The zero-based element position that did not reduce.
	 * @param contexts The element position's {@link TensorType}s across call contexts.
	 * @return The blocking {@link AbsenceReason} for the parameter.
	 */
	private AbsenceReason reportElementDrop(Parameter param, int element, Set<TensorType> contexts) {
		boolean heterogeneousDtype = contexts.stream().map(TensorType::getDType).distinct().count() > 1;
		boolean unknownDtype = !heterogeneousDtype && contexts.stream().anyMatch(t -> t.getDType() == DType.UNKNOWN);

		if (heterogeneousDtype) {
			this.addInfo(INPUT_SIGNATURE_INFERENCE, "Element " + element + " of parameter `" + param.getName() + "` of `" + this
					+ "` receives tensors with conflicting dtypes across call sites, so a single input signature cannot be inferred; it is dropped.");
			return AbsenceReason.HETEROGENEOUS_DTYPE;
		}

		if (unknownDtype) {
			this.addInfo(INPUT_SIGNATURE_INFERENCE, "Element " + element + " of parameter `" + param.getName() + "` of `" + this
					+ "` receives a tensor whose dtype cannot be determined, so a single input signature cannot be inferred; it is dropped.");
			return AbsenceReason.UNKNOWN_DTYPE;
		}

		this.addInfo(INPUT_SIGNATURE_INFERENCE, "Element " + element + " of parameter `" + param.getName() + "` of `" + this
				+ "` is sparse at some call sites and dense at others, so a single input signature cannot be inferred; it is dropped.");
		return AbsenceReason.HETEROGENEOUS_SPARSITY;
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
	 * otherwise emit a {@link SymbolicDim}({@code "?"}) wildcard. A consensus {@link RaggedDim} is preserved (it drives
	 * {@link InputSignature#toTensorSpecList} to emit a {@code RaggedTensorSpec}); any other non-{@link NumericDim} context dim yields a
	 * wildcard at that position.
	 * </ol>
	 * <p>
	 * Visible (rather than {@code private}) so the reduction can be exercised directly with a hand-built context set, isolating algorithm
	 * behavior from upstream tensor-type precision. This is the only seam for branches that upstream cannot currently produce as a fixture
	 * (e.g., the dtype-⊤ singleton, <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/494">#494</a>);
	 * {@link #inferInputSignature()} cannot stand in because it requires a fully classified {@link Function}.
	 *
	 * @param contexts The non-empty set of {@link TensorType}s Ariadne associated with the parameter across call contexts.
	 * @return The reduced single {@link TensorType}, or {@link Optional#empty} for the dtype-⊥ and dtype-⊤ branches.
	 */
	public static Optional<TensorType> inferSpec(Set<TensorType> contexts) {
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

		// Sparseness consensus: a parameter must be uniformly sparse or uniformly dense across contexts. A `SparseTensorSpec` admits only
		// sparse tensors and a dense `TensorSpec` admits only dense tensors (#533), so a parameter that is sparse at some call sites and
		// dense at others has no single spec that accepts both layouts. Emitting either would reject traffic the function accepts, so the
		// conservative reduction is bottom—the same `|sparseness| ≠ 1 ⇒ ⊥` discipline the dtype axis uses above (#642).
		boolean anySparse = contexts.stream().anyMatch(TensorType::isSparse);
		boolean allSparse = contexts.stream().allMatch(TensorType::isSparse);
		if (anySparse && !allSparse)
			return Optional.empty();
		boolean sparse = allSparse;

		// Step 2: rank consensus or shape-⊤. If any context has shape = null or ranks disagree, emit `TensorType(dtype, null)`,
		// preserving the dtype axis even when the shape axis degrades.
		// `rank` uses -1 as a "not yet set" sentinel: dim list sizes are always non-negative, so the sentinel can't collide. A boxed
		// `Integer rank = null` would compile-fail under the bundle's strict null-analysis (-err:+nullAnalysis) on the auto-unboxing
		// sites below.
		int rank = -1;
		for (TensorType t : contexts) {
			List<Dimension<?>> dims = t.getDims();
			if (dims == null)
				return Optional.of(withSparseness(new TensorType(dtype, null), sparse));
			if (rank == -1)
				rank = dims.size();
			else if (rank != dims.size())
				return Optional.of(withSparseness(new TensorType(dtype, null), sparse));
		}

		// Step 3: per-dim consensus or wildcard. If all contexts agree on a concrete value at position j, keep it; else emit a
		// `SymbolicDim("?")` wildcard. `DynamicDim` and `RaggedDim` (typed sentinels shipped in Ariadne 0.45.0 per
		// https://github.com/wala/ML/issues/545 and https://github.com/ponder-lab/ML/issues/320) get explicit branches so future precision
		// improvements can refine each case independently—https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/524 routes
		// the `RaggedDim` branch to `RaggedTensorSpec` emission.
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
			if (disagreement)
				shape.add(new SymbolicDim("?"));
			else if (consensus instanceof NumericDim)
				shape.add(consensus);
			else if (consensus instanceof DynamicDim)
				shape.add(new SymbolicDim("?"));
			else if (consensus instanceof RaggedDim)
				/*
				 * Preserve the ragged marker so the emission can produce a `RaggedTensorSpec` rather than a dense `TensorSpec` (#524). The
				 * position renders as `None` on the spec surface either way; the marker drives the spec-type choice in
				 * `InputSignature.toTensorSpecList`.
				 */
				shape.add(consensus);
			else
				shape.add(new SymbolicDim("?"));
		}

		return Optional.of(withSparseness(new TensorType(dtype, shape), sparse));
	}

	/**
	 * Returns the sparse view of the given {@link TensorType} when {@code sparse} is true, otherwise the type unchanged. Used by
	 * {@link #inferSpec} to carry a sparse-layout consensus onto the reduced type so {@link InputSignature#toTensorSpecList} emits a
	 * {@code SparseTensorSpec} (#533).
	 *
	 * @param type The reduced {@link TensorType}.
	 * @param sparse True iff the parameter's contexts agreed it is sparse.
	 * @return The sparse view when {@code sparse}, otherwise {@code type}.
	 */
	private static TensorType withSparseness(TensorType type, boolean sparse) {
		return sparse ? type.asSparse() : type;
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

	/**
	 * Names of the parameters Keras itself supplies to a {@code call} override through {@code Layer.__call__} on every invocation
	 * ({@code training}, {@code mask}): the framework contract on {@code Layer.call}. No source-visible call site passes them, so call-site
	 * examination cannot establish they are supplied (#881).
	 */
	private static final Set<String> KERAS_FRAMEWORK_SUPPLIED_PARAMETER_NAMES = Set.of("training", "mask");

	/** The last segments of the Keras base-class names whose {@code call} overrides Keras dispatches to. */
	private static final Set<String> KERAS_CALLABLE_BASE_LAST_SEGMENTS = Set.of("Model", "Layer");

	/**
	 * True iff this function is a {@code call} override of a Keras model or layer subclass: a method named {@code call} whose enclosing
	 * class transitively extends a base whose last segment is {@code Model} or {@code Layer}, resolved through the PyDev class hierarchy
	 * with an AST-base fallback (the same resolution {@link #hasTensorContext()} uses for candidate recognition). Matching by last segment
	 * over-recognizes a non-Keras base of the same name; that direction only withholds an unwritable signature (#881), the conservative
	 * default.
	 *
	 * @return True iff this function is a {@code call} override of a Keras model or layer subclass.
	 */
	private boolean isKerasCallOverride() {
		if (!"call".equals(this.getSimpleName()) || !(this.getFunctionDefinition().getFunctionDef().parent instanceof ClassDef))
			return false;

		return this.getAllClassParentNames(true).stream().anyMatch(KERAS_CALLABLE_BASE_LAST_SEGMENTS::contains);
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
				ret.addAll(this.reconfigure());
				break;
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

				// A called decorator (`@tf.function(...)`) carries an argument list past its name; `getFullRepresentationString`
				// yields only the dotted name, so extend the span through the matching close bracket. Otherwise the delete strips
				// just `@tf.function` and orphans the arguments as an invalid bare `(...)` (issue #681). A bare `@tf.function` has no
				// `(` here, so its span is unchanged.
				int afterName = offset + length;
				if (afterName < doc.getLength() && doc.getChar(afterName) == '(') {
					int depth = 0;
					int end = afterName;
					for (; end < doc.getLength(); end++) {
						char c = doc.getChar(end);
						if (c == '(' || c == '[' || c == '{')
							++depth;
						else if (c == ')' || c == ']' || c == '}') {
							--depth;
							if (depth == 0)
								break;
						}
					}
					length = end - offset + 1;
				}

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

		ImportContext ctx = getImportContext(doc);

		if (ctx == null) {
			// No TensorFlow import in scope: auto-inject one. The first hybridizable function in the file fixes the injected line and
			// records which names it brings into scope; later functions in the same file reuse that record (#574).
			File file = this.getContainingFile();
			Set<String> injectedNames = autoInjectedImportNames.get(file);

			if (injectedNames == null) {
				// `function` is always needed for the decorator. When input-signature emission applies, also bring the signature's
				// spec-type
				// constructors (`TensorSpec`, and `RaggedTensorSpec` for a ragged parameter) and dtype constants into scope so the emission
				// proceeds unqualified rather than being skipped. The names are sorted for deterministic emission; `function` leads.
				Set<String> names = new LinkedHashSet<>();
				names.add("function");

				if (this.getInferInputSignatures()) {
					/*
					 * Union this function's spec-type and dtype names with those of every other to-be-hybridized function in the file
					 * (pre-computed by `planAutoInjectedImports`), so the single injected import line brings every function's names into
					 * scope rather than only the first-processed function's (#588). The spec-type names are the signature's own
					 * `requiredSpecTypeNames` rather than a hardcoded `TensorSpec`, so a ragged parameter brings `RaggedTensorSpec` into
					 * scope and its signature emits unqualified rather than being gated off (#524). Falls back to this function's own names
					 * when no plan was computed (e.g. a direct `transform()` without the processor's pre-pass).
					 */
					SortedSet<String> specTypeNames = new TreeSet<>();
					SortedSet<String> dtypeNames = new TreeSet<>();
					this.inferInputSignature().signature().ifPresent(sig -> {
						specTypeNames.addAll(sig.requiredSpecTypeNames());
						dtypeNames.addAll(sig.requiredDTypeNames());
					});

					Set<String> plannedSpecTypeNames = fileInferredSpecTypeNames.get(file);
					if (plannedSpecTypeNames != null)
						specTypeNames.addAll(plannedSpecTypeNames);

					Set<String> plannedDTypeNames = fileInferredDTypeNames.get(file);
					if (plannedDTypeNames != null)
						dtypeNames.addAll(plannedDTypeNames);

					if (!dtypeNames.isEmpty()) {
						names.addAll(specTypeNames);
						names.addAll(dtypeNames);
					}
				}

				int line = getLineToInsertImport(doc);
				int lineOffset = doc.getLineOffset(line);

				TextEdit edit = new InsertEdit(lineOffset, "from tensorflow import " + String.join(", ", names) + "\n");
				MultiTextEdit mte = new MultiTextEdit();
				mte.addChild(edit);
				ret.add(mte);
				autoInjectedImportNames.put(file, names);
				injectedNames = names;
			}

			// Emission is reachable iff this function's required names are among those the injected line brought into scope; the
			// `computeInputSignatureKeyword` gate enforces that, so a later function needing a dtype the first did not inject is
			// safely skipped rather than emitting a `NameError`-raising decorator.
			ctx = new ImportContext("", false, injectedNames);
		}

		// Compose the whole decorator into one InsertEdit rather than three same-offset ones, so correctness doesn't depend on Eclipse
		// sequencing zero-length same-offset edits by add-order (#575). Wrap it in a MultiTextEdit (a container) so every element of
		// `ret` is a container: the processor builds the per-file TextChange by making the first edit the root and adding the rest as
		// children, which requires the root to accept children.
		String decorator = "@" + ctx.prefix() + "function" + this.addInputSignature(ctx).orElse("") + "\n" + precedingText;
		MultiTextEdit mte = new MultiTextEdit();
		mte.addChild(new InsertEdit(offset, decorator));
		ret.add(mte);

		return ret;
	}

	/**
	 * Reconfigures this already-hybrid function's {@code @tf.function} decorator to carry the inferred {@code input_signature} (the
	 * {@code RECONFIGURE} transformation). Only the add path exists: {@link #check()} selects {@code RECONFIGURE} solely for a decorator
	 * with no {@code input_signature}; an existing signature is adjudicated report-only and never rewritten, since the rewrite would repair
	 * a nonconforming observed call rather than preserve behavior (issue 808; the sanctioned rewrite is a future find-and-fix
	 * transformation, not this refactoring). Reuses the existing import-shape resolution ({@link #getImportContext(IDocument)}) and
	 * emission gate ({@link #computeInputSignatureKeyword(ImportContext)} / {@link #addInputSignature(ImportContext)}); a hybrid function
	 * necessarily imports TensorFlow (the decorator references it), so {@code getImportContext} is expected to resolve; the {@code null}
	 * check below is defensive and yields no edits rather than failing. When the signature's names are not reachable under the file's
	 * import shape (e.g. {@code from tensorflow import function} without {@code TensorSpec}), the gate yields no keyword and no edit is
	 * produced, matching {@link #convertToHybrid()}'s silent skip.
	 *
	 * @return The edits adding {@code input_signature=[...]} to the decorator, or an empty list when emission is gated out.
	 * @throws BadLocationException If a document offset cannot be resolved.
	 */
	private List<TextEdit> reconfigure() throws BadLocationException {
		assert this.getDecoratorNames(null).contains(TF_FUNCTION_FQN) : "Not hybrid.";
		assert this.getHybridizationParameters() == null || !this.getHybridizationParameters()
				.hasInputSignatureParam() : "RECONFIGURE is selected only for a decorator without an input_signature (issue 808).";

		List<TextEdit> ret = new ArrayList<>();

		IDocument doc = this.getContainingDocument();
		ImportContext ctx = getImportContext(doc);

		if (ctx == null)
			return ret;

		decoratorsType decorator = this.hybridDecorator;

		// Offset just past the decorator name (e.g. just past `function` in `@tf.function`). Mirrors `convertToEager`'s proven offset
		// computation: the `decoratorsType` node begins at `@`, and `getFullRepresentationString(decorator.func)` yields the dotted name
		// without arguments for both the bare and called forms (the trailing `+ 1` accounts for the leading `@`). The inner `func` expr's
		// own position is unreliable for decorators, so it is not used directly.
		int afterName = getOffset(doc, decorator) + getFullRepresentationString(decorator.func).length() + 1;

		MultiTextEdit mte = new MultiTextEdit();

		if (decorator.func instanceof Call) {
			// `@tf.function(...)`: append `input_signature=...` at the END of the existing argument list, just before the matching close
			// parenthesis. A trailing keyword argument is always valid Python, whereas front-insertion would place the keyword before any
			// existing positional argument (e.g. `@tf.function(None)`), producing a syntax error. Handles the empty-parentheses
			// (`@tf.function()`) and non-empty (`@tf.function(reduce_retracing=True)`) forms uniformly.
			Call call = (Call) decorator.func;

			// Find the open parenthesis, tolerating any whitespace between the callee and `(`.
			int parenOffset = afterName;
			while (parenOffset < doc.getLength() && doc.getChar(parenOffset) != '(')
				++parenOffset;

			// Find the matching close parenthesis by tracking bracket nesting from the open parenthesis (assumes no `)` inside a string
			// argument, which `@tf.function` decorators do not use in practice).
			int depth = 0;
			int closeOffset = parenOffset;
			for (; closeOffset < doc.getLength(); closeOffset++) {
				char c = doc.getChar(closeOffset);
				if (c == '(' || c == '[' || c == '{')
					++depth;
				else if (c == ')' || c == ']' || c == '}') {
					--depth;
					if (depth == 0)
						break;
				}
			}

			boolean hasArguments = (call.args != null && call.args.length > 0) || (call.keywords != null && call.keywords.length > 0)
					|| call.starargs != null || call.kwargs != null;

			// Insertion point is just before the matching close parenthesis; captured as effectively final for the lambda below.
			final int insertOffset = closeOffset;

			this.computeInputSignatureKeyword(ctx)
					.ifPresent(kw -> mte.addChild(new InsertEdit(insertOffset, hasArguments ? ", " + kw : kw)));
		} else
			// Bare `@tf.function` (no parentheses): append a parenthesized argument list right after the decorator name. This is the
			// argless-existing-decorator sub-case `addInputSignature` is documented to serve.
			this.addInputSignature(ctx).ifPresent(s -> mte.addChild(new InsertEdit(afterName, s)));

		if (mte.hasChildren())
			ret.add(mte);

		return ret;
	}

	/**
	 * The TensorFlow import shape observed in a Python source file, carrying the prefix to use when referring to TensorFlow names, whether
	 * {@code TensorSpec} is reachable in the file's namespace, and which other names are reachable under the prefix. The two empty-prefix
	 * shapes ({@code from tensorflow import *} and {@code from tensorflow import function}) differ on the latter two: the wildcard form
	 * pulls all public names into scope (including {@code TensorSpec} and every dtype constant), while the named-import form brings only
	 * the explicitly listed names.
	 *
	 * @param prefix The TensorFlow module prefix (e.g., {@code "tf."}, {@code "tensorflow."}, or {@code ""}).
	 * @param allNamesReachable True iff every TensorFlow name (including all dtype constants) is reachable under the {@code prefix} without
	 *        an additional import—the case for qualified ({@code import tensorflow [as X]}) and wildcard ({@code from tensorflow import *})
	 *        shapes. False for the named-import shape, where only {@code namedImports} are in scope.
	 * @param namedImports The bare names brought into scope by a {@code from tensorflow import ...} statement; consulted only when
	 *        {@code allNamesReachable} is false.
	 */
	private record ImportContext(String prefix, boolean allNamesReachable, Set<String> namedImports) {

		/**
		 * Whether the bare TensorFlow name {@code name} (e.g., a dtype constant like {@code "float32"}) can be referenced as
		 * {@code prefix + name} without an additional import. Qualified and wildcard shapes ({@code allNamesReachable}) bring every name
		 * into scope; a named {@code from tensorflow import ...} brings only the explicitly listed {@link #namedImports}.
		 *
		 * @param name The bare TensorFlow name to test.
		 * @return True iff {@code name} is reachable under this import shape.
		 */
		boolean nameReachable(String name) {
			return this.allNamesReachable() || this.namedImports().contains(name);
		}
	}

	/**
	 * Returns the {@code input_signature=[tfPrefix + "TensorSpec(...)", ...]} keyword argument when the flag is on and the inference
	 * produces a signature whose names are all reachable under the import context. Returns {@link Optional#empty} otherwise (flag off, no
	 * signature, a required spec-type constructor not reachable, or a required dtype constant not reachable). The keyword text only;
	 * callers handle the surrounding syntax (parenthesization via {@link #addInputSignature(ImportContext)}, or a leading {@code ", "} when
	 * injecting into an existing arg list).
	 * <p>
	 * The reachability checks guard the {@code from tensorflow import ...} named-import path: {@code TensorSpec} being in scope does not
	 * imply the signature's dtype constants (e.g. {@code float32}) are too, nor that {@code RaggedTensorSpec} is in scope for a ragged
	 * parameter ({@link InputSignature#requiredSpecTypeNames}), so emitting unconditionally would produce a {@code NameError}-raising
	 * decorator. When any required name is out of scope, emission is skipped rather than qualified—the named-import shape has no module
	 * prefix to qualify with.
	 *
	 * @param ctx The import context for the containing file.
	 * @return The {@code input_signature=...} keyword argument, or empty.
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/585">Issue 585</a>
	 */
	private Optional<String> computeInputSignatureKeyword(ImportContext ctx) {
		if (!this.getInferInputSignatures())
			return Optional.empty();
		/*
		 * The signature's own spec-type names (`TensorSpec` and/or `RaggedTensorSpec`) are authoritative for reachability. A separate
		 * upfront `TensorSpec`-reachable gate would be redundant for a dense signature and would wrongly block a ragged-only signature when
		 * `RaggedTensorSpec` is imported but `TensorSpec` is not.
		 */
		return this.inferInputSignature().signature().filter(sig -> sig.requiredSpecTypeNames().stream().allMatch(ctx::nameReachable))
				.filter(sig -> sig.requiredDTypeNames().stream().allMatch(ctx::nameReachable))
				.map(sig -> "input_signature=" + sig.toTensorSpecList(ctx.prefix()));
	}

	/**
	 * Whether an inferred input signature can actually be emitted into this function's decorator under the containing file's import shape.
	 * Gates {@code RECONFIGURE} selection in {@link #check()} so a passing precondition is never reported for a no-op transformation: a
	 * hybrid function always imports TensorFlow (its decorator references it), but the named-import shape ({@code from tensorflow import
	 * function}) can leave {@code TensorSpec} or a dtype constant out of scope, in which case
	 * {@link #computeInputSignatureKeyword(ImportContext)} yields nothing and {@link #reconfigure()} would produce no edit. True implies
	 * both that a signature was inferred and that all its names are reachable, so a selected reconfiguration always rewrites the decorator.
	 *
	 * @return True iff the inferred input signature is emittable under this file's import shape.
	 */
	private boolean canEmitInferredInputSignature() {
		ImportContext ctx = getImportContext(this.getContainingDocument());
		return ctx != null && this.computeInputSignatureKeyword(ctx).isPresent();
	}

	/**
	 * Returns the parenthesized {@code (input_signature=[tf.TensorSpec(...)])} argument-list text, or empty if
	 * {@link #computeInputSignatureKeyword(ImportContext)}'s gate fails. Used for the fresh-decorator and argless-existing-decorator cases
	 * (the latter is a Phase 3 {@code RECONFIGURE} sub-case). Callers compose this into their surrounding text (a single {@code InsertEdit}
	 * in {@link #convertToHybrid()}) or wrap it in an {@code InsertEdit} at an AST-derived offset ({@link #reconfigure()}). For injecting
	 * into an existing non-empty argument list, use {@link #computeInputSignatureKeyword(ImportContext)} directly with a leading
	 * {@code ", "}.
	 *
	 * @param ctx The import context for the containing file.
	 * @return The parenthesized {@code (input_signature=...)} text, or empty if the gate fails.
	 */
	private Optional<String> addInputSignature(ImportContext ctx) {
		return this.computeInputSignatureKeyword(ctx).map(kw -> "(" + kw + ")");
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

	private static ImportContext getImportContext(IDocument doc) {
		PyImportsHandling handling = new PyImportsHandling(doc);

		// Full pass over every import, then decide — a single `from tensorflow import function` must not short-circuit the scan and
		// miss a later `import tensorflow as tf` (or a `TensorSpec` in the same statement) that does make the signature emittable (#578).
		String qualifiedPrefix = null;
		boolean wildcard = false;
		Set<String> namedImports = new HashSet<>();

		for (ImportHandle importHandle : handling)
			for (ImportHandleInfo importHandleInfo : importHandle.getImportInfo()) {
				String fromImportStr = importHandleInfo.getFromImportStrWithoutUnwantedChars();
				boolean fromTensorflow = fromImportStr != null && fromImportStr.equals(TENSORFLOW_MODULE);

				for (String importStr : importHandleInfo.getImportedStr())
					if (importStr.equals(TENSORFLOW_MODULE))
						qualifiedPrefix = TENSORFLOW_MODULE + ".";
					else if (importStr.startsWith(TENSORFLOW_MODULE + " as"))
						qualifiedPrefix = importStr.substring((TENSORFLOW_MODULE + " as ").length(), importStr.length()) + ".";
					else if (fromTensorflow)
						if (importStr.equals("*")) // wildcard: TensorSpec and the dtype constants are reachable unqualified.
							wildcard = true;
						else
							// Every explicitly named symbol, so the gate can check `TensorSpec` and the signature's dtype constants.
							namedImports.add(importStr);
			}

		// Precedence: a qualified `import tensorflow [as X]` qualifies `function`, `TensorSpec`, and the dtype constants under one
		// prefix, so it wins over a named `from`-import that may bring only a subset into scope. A wildcard brings everything
		// unqualified. Otherwise a named `from tensorflow import ...` makes `function` reachable unqualified, and `TensorSpec` and the
		// dtype constants only if they too were named.
		if (qualifiedPrefix != null)
			return new ImportContext(qualifiedPrefix, true, Collections.emptySet());
		if (wildcard)
			return new ImportContext("", true, Collections.emptySet());
		if (namedImports.contains("function"))
			return new ImportContext("", false, namedImports);

		// not found.
		return null;
	}
}
