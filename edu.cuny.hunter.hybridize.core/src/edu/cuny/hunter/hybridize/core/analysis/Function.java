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
import static edu.cuny.hunter.hybridize.core.analysis.PreconditionSuccess.P5;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
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
import org.eclipse.text.edits.ReplaceEdit;
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
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.Num;
import org.python.pydev.parser.jython.ast.Tuple;
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
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
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

import edu.cuny.hunter.hybridize.core.analysis.InferenceResult.AbsenceReason;
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
		 * The {@link InputSignature} parsed from an {@code input_signature=[tf.TensorSpec(...)]} argument supplied to this
		 * {@link Function}'s {@code @tf.function} decorator (in either keyword or positional form), or {@link Optional#empty} when none was
		 * supplied or its content could not be fully modeled. See {@link #getSuppliedInputSignature()} for the presence/parse contract.
		 */
		private Optional<InputSignature> suppliedInputSignature = Optional.empty();

		/**
		 * The AST expression node of the supplied {@code input_signature} argument's value (the {@code [tf.TensorSpec(...)]} list/tuple),
		 * whether supplied by keyword or by position, or {@code null} when none was supplied. Retained so {@link #reconfigure()} can locate
		 * the existing value's source span to overwrite it. See {@link #getSuppliedInputSignatureNode()}.
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
							this.suppliedInputSignature = parseSuppliedInputSignature(positionalArgs[i]);
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
							this.suppliedInputSignature = parseSuppliedInputSignature(keyword.value);
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
		 * modeled. A validate-then-overwrite decision can compare it against the inferred signature.
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
		 * The AST expression node of the supplied {@code input_signature} value (the {@code [tf.TensorSpec(...)]} list/tuple), or
		 * {@code null} when none was supplied. {@link #reconfigure()} uses it to locate the existing value's source span when overwriting.
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
		 * producing a partial signature that downstream validate-then-overwrite logic could not trust. A well-formed empty list/tuple
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

			return Optional.of(new InputSignature(parameterTypes));
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
		autoInjectedImportNames.clear();
		fileInferredDTypeNames.clear();
	}

	/**
	 * Pre-computes, per file, the union of dtype constants required by the inferred input signatures of the functions about to be converted
	 * to hybrid, so {@link #convertToHybrid} can auto-inject a single {@code from tensorflow import ...} line covering all of them. Without
	 * it, the first hybridizable function processed in an import-less file fixes the injected line to its own dtypes, and a later function
	 * needing a different dtype is gated off emission and left with a bare {@code @function} (#588, follow-up to #574). Reads the memoized
	 * inferred signatures via {@link #getInferredInputSignature}; it never triggers inference, so it adds no per-parameter INFOs. Call it
	 * once before transforming a batch of functions.
	 *
	 * @param functions The functions about to be transformed.
	 */
	public static void planAutoInjectedImports(Collection<Function> functions) {
		for (Function function : functions) {
			if (!function.getTransformations().contains(CONVERT_TO_HYBRID) || !function.getInferInputSignatures())
				continue;

			function.getInferredInputSignature().ifPresent(sig -> fileInferredDTypeNames
					.computeIfAbsent(function.getContainingFile(), k -> new TreeSet<>()).addAll(sig.requiredDTypeNames()));
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
	 * True iff the refactoring should emit an inferred {@code input_signature} keyword into the generated decorator.
	 *
	 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/563">Issue 563</a>
	 */
	private boolean inferInputSignatures;

	/**
	 * Memoizes {@link #inferInputSignature()}. {@code null} means "not yet computed"; once computed, holds the {@link InferenceResult} so
	 * the per-parameter INFOs the computation emits as a side effect are added at most once, regardless of how many call sites request the
	 * signature in a single pass (analysis, import injection, and the transform paths all ask for it).
	 */
	private InferenceResult inferredInputSignature;

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

							/*
							 * The eager→hybrid conversion emits the inferred signature into the new decorator during the change
							 * (`convertToHybrid`). Compute it here too so the inferred signature is observable at analysis time (wizard,
							 * evaluator), mirroring how the reconfigure path computes it while checking preconditions. The result is
							 * memoized, so the change does not recompute it, and computing it has no bearing on the P1 decision.
							 */
							if (this.getInferInputSignatures())
								this.inferInputSignature();
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
					/*
					 * This function is already correctly hybrid (tensor parameter, no primitive parameter). When input-signature inference
					 * is enabled, the function is side-effect-free and non-recursive, and a signature can be inferred and emitted, the
					 * decorator is reconfigured: if it carries no `input_signature` yet, add the inferred one (the add path); if it carries
					 * one that is more specific than, or incomparable with, the inferred one, overwrite it; if it carries one broader than
					 * the inferred one, preserve it (the broader signature may be intentional); if they agree, do nothing. A supplied
					 * signature whose content could not be modeled is left untouched. Gating on the flag keeps the default precondition
					 * matrix unchanged.
					 */
					boolean canReconfigure = this.getInferInputSignatures() && this.getHasPythonSideEffects() != null
							&& !this.getHasPythonSideEffects() && this.isRecursive() != null && !this.isRecursive()
							&& this.canEmitInferredInputSignature();

					if (canReconfigure && !this.getHybridizationParameters().hasInputSignatureParam()) {
						// Add path: no existing `input_signature`.
						this.addInfo("This hybrid function has no input signature and can be reconfigured to add the inferred one.");
						this.addTransformation(RECONFIGURE);
						this.setPassingPrecondition(P4);
					} else if (canReconfigure && this.getHybridizationParameters().getSuppliedInputSignature().isPresent()
							&& this.inferInputSignature() instanceof InferenceResult.Inferred(InputSignature inferred)) {
						// Modify path: an existing, fully-modeled `input_signature` is present. Compare it against the inferred one.
						// `canReconfigure` implies inference succeeded (it gates on `canEmitInferredInputSignature`), so the pattern always
						// binds here; a hypothetical `Absent` falls through to the no-primitive-parameter failure below.
						InputSignature supplied = this.getHybridizationParameters().getSuppliedInputSignature().get();

						switch (supplied.relate(inferred)) {
						case SUPPLIED_TIGHTER -> {
							this.addInfo("This hybrid function's input signature is narrower than its call sites require; "
									+ "it can be reconfigured to admit the observed inputs.");
							this.addTransformation(RECONFIGURE);
							this.setPassingPrecondition(P5);
						}
						case INCOMPARABLE -> {
							this.addWarning("This hybrid function's input signature disagrees with its call sites; "
									+ "reconfiguring it will change the inputs the function accepts.");
							this.addTransformation(RECONFIGURE);
							this.setPassingPrecondition(P5);
						}
						case SUPPLIED_BROADER -> {
							this.addInfo("This hybrid function's input signature is broader than its call sites require; "
									+ "it is left unchanged in case the broader signature is intentional.");
							this.addFailure(PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS,
									"Functions with no Python literal arguments may benefit from hybridization.");
						}
						case AGREEMENT -> this.addFailure(PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS,
								"Functions with no Python literal arguments may benefit from hybridization.");
						}
					} else {
						this.addFailure(PreconditionFailure.HAS_NO_PRIMITIVE_PARAMETERS,
								"Functions with no Python literal arguments may benefit from hybridization.");

						if (this.getHasPythonSideEffects() != null && this.getHasPythonSideEffects())
							this.addFailure(PreconditionFailure.HAS_PYTHON_SIDE_EFFECTS,
									"De-hybridizing a function with Python side-effects may alter semantics.");
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
	 * Computes the inferred input signature. Always recomputes; {@link #inferInputSignature()} memoizes the result. Emits the per-parameter
	 * recovery INFOs as a side effect. The {@link InferenceResult.Absent} result carries the <em>first</em> blocking
	 * {@link InferenceResult.AbsenceReason} encountered, but the loop still runs to completion so every blocking parameter surfaces its
	 * INFO in one pass.
	 *
	 * @return The {@link InferenceResult}. See {@link #inferInputSignature()} for the contract, including the no-non-self-parameter throw.
	 */
	private InferenceResult computeInputSignature() {
		List<TensorType> specs = new ArrayList<>();
		AbsenceReason firstReason = null;

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
				if (firstReason == null)
					firstReason = AbsenceReason.NON_TENSOR_PARAMETER;
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
				if (firstReason == null)
					firstReason = AbsenceReason.NO_SHAPE_OR_DTYPE_EVIDENCE;
				continue;
			}

			Optional<TensorType> spec = inferSpec(contexts);
			if (spec.isEmpty()) {
				/*
				 * `inferSpec` reduced to bottom. With the per-context reduction
				 * (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/480) it now drops for only two reasons:
				 * heterogeneous dtype (|D| ≠ 1) or dtype-⊤ (a single agreed `UNKNOWN`). Shape-⊤ and symbolic-dim no longer drop here—it
				 * emits a coarse `TensorType(dtype, null)` or a `SymbolicDim` wildcard instead. Emit a per-parameter INFO naming the
				 * reason; see https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/510.
				 */
				boolean heterogeneous = contexts.stream().map(TensorType::getDType).distinct().count() > 1;
				if (heterogeneous)
					this.addInfo(INPUT_SIGNATURE_INFERENCE, "Parameter `" + param.getName() + "` of `" + this
							+ "` receives tensors with conflicting dtypes across call sites, so a single input signature cannot be inferred; it is dropped.");
				else
					this.addInfo(INPUT_SIGNATURE_INFERENCE, "Parameter `" + param.getName() + "` of `" + this
							+ "` receives a tensor whose dtype cannot be determined, so a single input signature cannot be inferred; it is dropped.");
				if (firstReason == null)
					firstReason = heterogeneous ? AbsenceReason.HETEROGENEOUS_DTYPE : AbsenceReason.UNKNOWN_DTYPE;
				continue;
			}

			specs.add(spec.get());
		}

		// A signature must be total over the parameters: any blocking reason makes the whole result Absent, even if some parameters
		// reduced.
		if (firstReason != null)
			return new InferenceResult.Absent(firstReason);

		// Degenerate case: no non-`self` parameter contributed a spec and none blocked, i.e. there are no non-`self` parameters at all.
		// Every refactoring call site is gated on `getHasTensorParameter()`, so this cannot arise there; it signals a direct, unguarded
		// call on a parameterless (or `self`-only) function, which is a programmer error.
		if (specs.isEmpty())
			throw new IllegalStateException("Cannot infer an input signature for `" + this
					+ "`: it has no non-self parameters. Refactoring call sites are gated on `getHasTensorParameter()`.");

		return new InferenceResult.Inferred(new InputSignature(specs));
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
				// `function` is always needed for the decorator. When input-signature emission applies, also bring `TensorSpec` and the
				// signature's dtype constants into scope so the emission proceeds unqualified rather than being skipped. The dtype
				// constants are sorted for deterministic emission; `function` and `TensorSpec` lead to match the conventional spelling.
				Set<String> names = new LinkedHashSet<>();
				names.add("function");

				if (this.getInferInputSignatures()) {
					/*
					 * Union this function's dtype constants with those of every other to-be-hybridized function in the file (pre-computed
					 * by `planAutoInjectedImports`), so the single injected import line brings every function's dtypes into scope rather
					 * than only the first-processed function's (#588). Falls back to this function's own dtypes when no plan was computed
					 * (e.g. a direct `transform()` without the processor's pre-pass).
					 */
					SortedSet<String> dtypeNames = new TreeSet<>();
					this.inferInputSignature().signature().ifPresent(sig -> dtypeNames.addAll(sig.requiredDTypeNames()));

					Set<String> plannedDTypeNames = fileInferredDTypeNames.get(file);
					if (plannedDTypeNames != null)
						dtypeNames.addAll(plannedDTypeNames);

					if (!dtypeNames.isEmpty()) {
						names.add("TensorSpec");
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
	 * {@code RECONFIGURE} transformation). When the decorator has no {@code input_signature}, the inferred one is added; when it already
	 * has one that {@link #check()} determined should be overwritten, the existing value is replaced in place. Reuses the existing
	 * import-shape resolution ({@link #getImportContext(IDocument)}) and emission gate
	 * ({@link #computeInputSignatureKeyword(ImportContext)} / {@link #addInputSignature(ImportContext)}); a hybrid function necessarily
	 * imports TensorFlow (the decorator references it), so {@code getImportContext} is non-null. When the signature's names are not
	 * reachable under the file's import shape (e.g. {@code from tensorflow import function} without {@code TensorSpec}), the gate yields no
	 * keyword and no edit is produced, matching {@link #convertToHybrid()}'s silent skip.
	 *
	 * @return The edits adding or replacing {@code input_signature=[...]} on the decorator, or an empty list when emission is gated out.
	 * @throws BadLocationException If a document offset cannot be resolved.
	 */
	private List<TextEdit> reconfigure() throws BadLocationException {
		assert this.getDecoratorNames(null).contains(TF_FUNCTION_FQN) : "Not hybrid.";

		List<TextEdit> ret = new ArrayList<>();

		IDocument doc = this.getContainingDocument();
		ImportContext ctx = getImportContext(doc);

		if (ctx == null)
			return ret;

		decoratorsType decorator = this.hybridDecorator;

		// Overwrite path: an existing `input_signature` value is present (its node was retained during parameter parsing). Replace its
		// bracketed list/tuple in place with the inferred one. `check()` selects this only when the inferred signature is emittable, so
		// `inferInputSignature` is present here.
		exprType existingValue = this.getHybridizationParameters() == null ? null
				: this.getHybridizationParameters().getSuppliedInputSignatureNode();

		if (existingValue != null) {
			// Span the existing value's bracketed list/tuple: from its first opening bracket to the matching close, tracking nesting.
			int bracket = getOffset(doc, existingValue);
			while (bracket < doc.getLength() && doc.getChar(bracket) != '[' && doc.getChar(bracket) != '(')
				++bracket;

			int depth = 0;
			int end = bracket;
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

			final int valueOffset = bracket;
			final int valueLength = end - bracket + 1;
			MultiTextEdit replacement = new MultiTextEdit();
			this.inferInputSignature().signature()
					.ifPresent(sig -> replacement.addChild(new ReplaceEdit(valueOffset, valueLength, sig.toTensorSpecList(ctx.prefix()))));

			if (replacement.hasChildren())
				ret.add(replacement);

			return ret;
		}

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
