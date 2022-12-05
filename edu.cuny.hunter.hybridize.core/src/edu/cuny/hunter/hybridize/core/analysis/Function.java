package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.util.Objects;

import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.NameTokType;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.jython.ast.keywordType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.parser.visitors.TypeInfo;
import org.python.pydev.shared_core.string.CoreTextSelection;

import edu.cuny.citytech.refactoring.common.core.RefactorableProgramEntity;

/**
 * A representation of a Python function.
 *
 * @author <a href="mailto:rk1424@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @author <a href="mailto:tcastrovelez@gradcenter.cuny.edu">Tatiana Castro Vélez</a>
 */
public class Function extends RefactorableProgramEntity {

	/**
	 * Parameters that may be passed to a tf.fuction decorator. Parameter descriptions found at:
	 * https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function
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

		public HybridizationParameters(IProgressMonitor monitor) throws TooManyMatchesException, BadLocationException {
			FunctionDefinition functionDefinition = Function.this.getFunctionDefinition();
			decoratorsType[] decoratorArray = functionDefinition.getFunctionDef().decs;

			// Will contain the last tf.function decorator
			decoratorsType tfFunctionDecorator = null;

			// Iterate through the decorators of the function
			for (decoratorsType decorator : decoratorArray) {
				IDocument document = Function.this.getContainingDocument();
				PySelection selection = getSelection(decorator, document);

				// Save the hybrid decorator
				if (Function.isHybrid(decorator, Function.this.containingModuleName, Function.this.containingFile, selection,
						Function.this.nature, monitor))
					tfFunctionDecorator = decorator;
			} // We expect to have the last tf.function decorator in tfFunctionDecorator

			if (tfFunctionDecorator != null)
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
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter autograph.
		 *
		 * @return True iff this {@link decoratorType} has parameter autograph.
		 */
		public boolean hasAutoGraphParam() {
			return this.autoGraphParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_autograph_options.
		 *
		 * @return True iff this {@link decoratorType} has parameter experimental_autograph_options.
		 */
		public boolean hasExperimentalAutographOptParam() {
			return this.experimentalAutographOptionsParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_implements.
		 *
		 * @return True iff this {@link decoratorType} has parameter experimental_implements.
		 */
		public boolean hasExperimentalImplementsParam() {
			return this.experimentalImplementsParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_follow_type_hints.
		 *
		 * @return True iff this {@link decoratorType} has parameter experimental_follow_type_hints.
		 */
		public boolean hasExperimentalTypeHintsParam() {
			return this.experimentaFollowTypeHintsParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter has parameter func.
		 *
		 * @return True iff this {@link decoratorType} has parameter func.
		 */
		public boolean hasFuncParam() {
			return this.funcParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter input_signature.
		 *
		 * @return True iff this {@link decoratorType} has parameter input_signature.
		 */
		public boolean hasInputSignatureParam() {
			return this.inputSignatureParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter jit_compile.
		 *
		 * @return True iff this {@link decoratorType} has parameter jit_compile.
		 */
		public boolean hasJitCompileParam() {
			return this.jitCompileParamExists;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter reduce_retracing.
		 *
		 * @return True iff this {@link Function} has parameter reduce_retracing.
		 */
		public boolean hasReduceRetracingParam() {
			return this.reduceRetracingParamExists;
		}
	}

	private static final String TF_FUNCTION_FQN = "tensorflow.python.eager.def_function.function";

	private static final ILog LOG = getLog(Function.class);

	/**
	 * Information about this {@link Function} tf.function's parameters.
	 */
	private Function.HybridizationParameters hybridizationParameters;

	/**
	 * The {@link FunctionDefinition} representing this {@link Function}.
	 */
	private FunctionDefinition functionDefinition;

	/**
	 * True iff this {@link Function} is decorated with tf.function.
	 */
	private boolean isHybrid;

	/**
	 * True iff this {@link Function} has at least one parameter that is a tf.Tensor (https://bit.ly/3vYG7iP).
	 */
	private boolean likelyHasTensorParameter;

	/**
	 * Module name of {@link FunctionDefinition}.
	 */
	private String containingModuleName;

	/**
	 * File of {@link FunctionDefinition}.
	 */
	private File containingFile;

	/**
	 * Nature of {@link FunctionDefinition}.
	 */
	private IPythonNature nature;

	public Function(FunctionDefinition fd, IProgressMonitor monitor) throws TooManyMatchesException, BadLocationException {
		this.functionDefinition = fd;

		// Find out if it's hybrid via the tf.function decorator.
		this.computeIsHybrid(monitor);

		// If function is hybrid, then parse the existence of the parameters.
		if (this.isHybrid()) {
			LOG.info("Checking the hybridization parameters ...");
			this.hybridizationParameters = this.new HybridizationParameters(monitor);
		}

		this.computeHasTensorParameter(monitor);
	}

	private void computeHasTensorParameter(IProgressMonitor monitor) throws TooManyMatchesException, BadLocationException {
		// TODO: Use type info API. If that gets info from type hints, then we'll need another field indicating whether
		// type hints are used.
		// TODO: What if there are no current calls to the function? How will we determine its type? Maybe from type hints? Or docstring?
		// TODO: Use cast/assert statements?
		FunctionDef functionDef = this.getFunctionDefinition().getFunctionDef();
		argumentsType params = functionDef.args;

		if (params != null) {
			exprType[] actualParams = params.args;

			if (actualParams != null) {
				// for each parameter.
				for (exprType paramExpr : actualParams) {
					String paramName = NodeUtils.getRepresentationString(paramExpr);

					// if hybridization parameters are specified.
					if (this.getHybridizationParameters() != null) {
						// if we are considering type hints.
						// TODO: Actually get the value here (#111).
						if (this.getHybridizationParameters().hasExperimentalTypeHintsParam()) {
							// try to get its type from the AST.
							TypeInfo argTypeInfo = NodeUtils.getTypeForParameterFromAST(paramName, functionDef);

							if (argTypeInfo != null) {
								exprType node = argTypeInfo.getNode();
								Attribute typeHint = (Attribute) node;
								NameTokType attr = typeHint.attr;

								// Look up the definition.
								IDocument document = this.getContainingDocument();
								PySelection selection = getSelection(attr, document);

								String fqn = Util.getFullyQualifiedName(attr, this.containingModuleName, containingFile, selection,
										this.nature, monitor);
								System.out.println(fqn);
								// tensorflow.python.framework.ops.Tensor

								// TODO: If it's a Tensor or tf.Variable, then check for experimental_type_hints.
								// if that's set, then, set likelyHasTensorParameter to true.
								// this is a special case.
							}
						}
					}
				}
			}
		}

		this.likelyHasTensorParameter = false;
		LOG.info(this + " does not likely have a tensor parameter.");
	}

	private void computeIsHybrid(IProgressMonitor monitor) throws TooManyMatchesException, BadLocationException {
		// TODO: Consider mechanisms other than decorators (e.g., higher order functions; #3).
		monitor.setTaskName("Computing hybridization ...");

		FunctionDefinition functionDefinition = this.getFunctionDefinition();
		decoratorsType[] decoratorArray = functionDefinition.getFunctionDef().decs;

		if (decoratorArray != null) {
			this.containingModuleName = this.getContainingModuleName();
			this.containingFile = this.getContainingFile();
			this.nature = this.getNature();

			for (decoratorsType decorator : decoratorArray) {
				IDocument document = this.getContainingDocument();
				PySelection selection = getSelection(decorator, document);

				// if this function is decorated with "tf.function."
				if (isHybrid(decorator, this.containingModuleName, this.containingFile, selection, this.nature, monitor)) {
					this.isHybrid = true;
					LOG.info(this + " is hybrid.");
					return;
				}
			}
		}

		this.isHybrid = false;
		LOG.info(this + " is not hybrid.");
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
	 * @throws TooManyMatchesException If the definition of the decorator is ambiguous.
	 * @throws BadLocationException When the containing entities cannot be parsed.
	 */
	private static boolean isHybrid(decoratorsType decorator, String containingModuleName, File containingFile, PySelection selection,
			IPythonNature nature, IProgressMonitor monitor) throws TooManyMatchesException, BadLocationException {
		String decoratorFQN = Util.getFullyQualifiedName(decorator, containingModuleName, containingFile, selection, nature, monitor);

		LOG.info("Found decorator: " + decoratorFQN + ".");

		// if this function is decorated with "tf.function."
		if (decoratorFQN.equals(TF_FUNCTION_FQN))
			return true;

		LOG.info(decoratorFQN + " does not equal " + TF_FUNCTION_FQN + ".");
		return false;
	}

	public IDocument getContainingDocument() {
		return this.getFunctionDefinition().containingDocument;
	}

	public IPythonNature getNature() {
		return this.functionDefinition.nature;
	}

	private static PySelection getSelection(decoratorsType decorator, IDocument document) {
		exprType decoratorFunction = decorator.func;
		return getSelection(decoratorFunction, document);
	}

	private static PySelection getSelection(SimpleNode node, IDocument document) {
		CoreTextSelection coreTextSelection = Util.getCoreTextSelection(document, node);
		return new PySelection(document, coreTextSelection);
	}

	public File getContainingFile() {
		return this.getFunctionDefinition().containingFile;
	}

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
	public String getIdentifer() {
		FunctionDefinition functionDefinition = this.getFunctionDefinition();
		FunctionDef functionDef = functionDefinition.getFunctionDef();
		return Util.getQualifiedName(functionDef);
	}

	/**
	 * Accessor for private member variable isHybrid.
	 *
	 * @return Boolean that states if this {@link Function} is decorated with tf.function.
	 */
	public boolean isHybrid() {
		return this.isHybrid;
	}

	/**
	 * True iff this {@link Function} likely has a tf.Tensor parameter. Since Python is dynamic, we may not be 100% sure.
	 *
	 * @return True iff this {@link Function} likely has a tf.Tensor parameter.
	 */
	public boolean likelyHasTensorParameter() {
		return this.likelyHasTensorParameter;
	}

	@Override
	public String toString() {
		return this.getIdentifer();
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
}
