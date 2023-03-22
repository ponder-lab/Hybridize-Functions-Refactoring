package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Set;

import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.python.pydev.ast.codecompletion.revisited.visitors.Definition;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.jython.ast.keywordType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.parser.visitors.TypeInfo;

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
	 * https://tensorflow.org/versions/r2.9/api_docs/python/tf/function Note: We are also parsing the deprecated parameters specified in
	 * the documentation. Users can still use these deprecated parameters. Therefore we need to be able to account for them. Please refer to
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/wiki/tf.function-parameter's-version-information to see more
	 * information about the tf.function parameters according to the versions.
	 */
	public class HybridizationParameters {

		/**
		 * Available in TF version [2.4,2.11].
		 */
		private static final String EXPERIMENTAL_FOLLOW_TYPE_HINTS = "experimental_follow_type_hints";

		/**
		 * Available in TF version [2.0,2.11].
		 */
		private static final String EXPERIMENTAL_AUTOGRAPH_OPTIONS = "experimental_autograph_options";

		/**
		 * Available in TF version [2.1,2.11].
		 */
		private static final String EXPERIMENTAL_IMPLEMENTS = "experimental_implements";

		/**
		 * Available in TF version [2.9,2.11]. Previously, experimental_relax_shapes which is available TF version [2.0,2.11].
		 */
		private static final String REDUCE_RETRACING = "reduce_retracing";

		/**
		 * Available in TF version [2.0,2.11].
		 */
		private static final String EXPERIMENTAL_RELAX_SHAPES = "experimental_relax_shapes";

		/**
		 * Available in TF version [2.1,2.11].
		 */
		private static final String EXPERIMENTAL_COMPILE = "experimental_compile";

		/**
		 * Available in TF version [2.5,2.11]. Previously, experimental_compile which is available TF version [2.1,2.11].
		 */
		private static final String JIT_COMPILE = "jit_compile";

		/**
		 * Available in TF version [2.0,2.11].
		 */
		private static final String AUTOGRAPH = "autograph";

		/**
		 * Available in TF version [2.0,2.11].
		 */
		private static final String INPUT_SIGNATURE = "input_signature";

		/**
		 * Available in TF version [2.0,2.11].
		 */
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

		public HybridizationParameters(IProgressMonitor monitor) throws BadLocationException {
			FunctionDefinition functionDefinition = Function.this.getFunctionDefinition();
			decoratorsType[] decoratorArray = functionDefinition.getFunctionDef().decs;

			// Will contain the last tf.function decorator
			decoratorsType tfFunctionDecorator = null;

			// Declaring definitions of the decorator, if it contains multiple definitions there might be more than one in this set. Since
			// we are dealing with tf.function, we check we it has one definition.
			Definition declaringDefinition = null;

			// Iterate through the decorators of the function
			for (decoratorsType decorator : decoratorArray) {
				IDocument document = Function.this.getContainingDocument();
				PySelection selection = Util.getSelection(decorator, document);

				// Save the hybrid decorator
				try {
					if (Function.isHybrid(decorator, Function.this.containingModuleName, Function.this.containingFile, selection,
							Function.this.nature, monitor)) { // TODO: Cache this from a previous call (#118).
						tfFunctionDecorator = decorator;
						// Returns the set of potential declaring definitions of the selection.
						Set<Definition> potentialDeclaringDefitinion = Util.getDeclaringDefinition(selection,
								Function.this.containingModuleName, Function.this.containingFile, Function.this.nature, monitor);

						// If it has an element, we store the definition in a variable.
						if (potentialDeclaringDefitinion.iterator().hasNext())
							declaringDefinition = potentialDeclaringDefitinion.iterator().next();
						else
							throw new IllegalStateException("Can't determine tf.function decorator defintion.");
					}
				} catch (AmbiguousDeclaringModuleException e) {
					throw new IllegalStateException("Can't determine whether decorator: " + decorator + " is hybrid.", e);
				}
			} // We expect to have the last tf.function decorator in tfFunctionDecorator

			// Getting tf.functions Python definition arguments.
			ArrayList<String> argumentIdDeclaringDefintion = getTfFunctionPythonDefinitionArguments(declaringDefinition);

			if (tfFunctionDecorator != null)
				// tfFunctionDecorator must be an instance of Call, because that's the only way we have parameters.
				if (tfFunctionDecorator.func instanceof Call) {
					Call callFunction = (Call) tfFunctionDecorator.func;

					// We iterate over the tf.function's parameters positions.
					for (int i = 0; i < callFunction.args.length; i++) {
						// From the position i, we use the tf.function's definition to verify which parameter we are analyzing.
						String evaluatedArgument = argumentIdDeclaringDefintion.get(i);

						// Matching the arguments from the definition and the arguments from the code being analyzed.
						if (evaluatedArgument.equals(FUNC))
							this.funcParamExists = true;
						else if (evaluatedArgument.equals(INPUT_SIGNATURE))
							this.inputSignatureParamExists = true;
						else if (evaluatedArgument.equals(AUTOGRAPH))
							this.autoGraphParamExists = true;
						// In our accepted interval version ([2.0,2.11]) of the API allows parameter names jit_compile and
						// deprecated name experimental_compile.
						else if (evaluatedArgument.equals(JIT_COMPILE) || evaluatedArgument.equals(EXPERIMENTAL_COMPILE))
							this.jitCompileParamExists = true;
						// In our accepted interval version ([2.0,2.11]) of the API allows parameter names reduce_retracing
						// and deprecated name experimental_relax_shapes.
						else if (evaluatedArgument.equals(REDUCE_RETRACING))
							this.reduceRetracingParamExists = true;
						else if (evaluatedArgument.equals(EXPERIMENTAL_RELAX_SHAPES))
							this.reduceRetracingParamExists = true;
						else if (evaluatedArgument.equals(EXPERIMENTAL_IMPLEMENTS))
							this.experimentalImplementsParamExists = true;
						else if (evaluatedArgument.equals(EXPERIMENTAL_AUTOGRAPH_OPTIONS))
							this.experimentalAutographOptionsParamExists = true;
						else if (evaluatedArgument.equals(EXPERIMENTAL_FOLLOW_TYPE_HINTS))
							this.experimentaFollowTypeHintsParamExists = true;
						else
							throw new IllegalArgumentException("Unable to process tf.function argument.");
					}

					// Processing keywords arguments
					// If we have keyword parameter, afterwards, we cannot have positional parameters because it would result in invalid
					// Python code. This is why we check the keywords last.
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
							// In our accepted interval version ([2.0,2.11]) of the API allows parameter names jit_compile and
							// deprecated name experimental_compile.
							else if (name.id.equals(JIT_COMPILE) || name.id.equals(EXPERIMENTAL_COMPILE))
								// Found parameter jit_compile/experimental_compile
								this.jitCompileParamExists = true;
							// In our accepted interval version ([2.0,2.11]) of the API allows parameter names reduce_retracing
							// and deprecated name experimental_relax_shapes.
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
		 * Get the tf.function parameter names from the {@link Definition}.
		 *
		 * @param declaringDefinition The Definition to use.
		 * @return An array with the names of the arguments given by {@link Definition}.
		 */
		private ArrayList<String> getTfFunctionPythonDefinitionArguments(Definition declaringDefinition) {
			// Python source arguments from the declaring definition
			exprType[] declaringArguments = null;

			// Getting the arguments from TensorFlow source
			if (declaringDefinition != null) {
				if (declaringDefinition.ast instanceof FunctionDef) {
					FunctionDef declaringFunctionDefinition = (FunctionDef) declaringDefinition.ast;
					argumentsType declaringArgumentTypes = declaringFunctionDefinition.args;
					declaringArguments = declaringArgumentTypes.args;
				}
			}

			// Python source arguments from the declaring definition
			ArrayList<String> argumentIdDeclaringDefintion = new ArrayList<>();

			// Getting the arguments from the definition
			if (declaringArguments != null) {
				for (exprType declaredArgument : declaringArguments) {
					if (declaredArgument instanceof Name) {
						Name argumentName = (Name) declaredArgument;
						argumentIdDeclaringDefintion.add(argumentName.id);
					}
				}
			}

			return argumentIdDeclaringDefintion;
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
		public boolean hasExperimentalFollowTypeHintsParam() {
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

	private static final String TF_TENSOR_FQN = "tensorflow.python.framework.ops.Tensor";

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

	public Function(FunctionDefinition fd, IProgressMonitor monitor) throws BadLocationException {
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

	private void computeHasTensorParameter(IProgressMonitor monitor) throws BadLocationException {
		monitor.beginTask("Analyzing whether function has a tensor parameter.", IProgressMonitor.UNKNOWN);
		// TODO: What if there are no current calls to the function? How will we determine its type?
		// TODO: Use cast/assert statements?
		FunctionDef functionDef = this.getFunctionDefinition().getFunctionDef();
		argumentsType params = functionDef.args;

		if (params != null) {
			exprType[] actualParams = params.args;

			if (actualParams != null) {
				// for each parameter.
				for (exprType paramExpr : actualParams) {
					String paramName = NodeUtils.getRepresentationString(paramExpr);

					// check a special case where we consider type hints.

					// if hybridization parameters are specified.
					if (this.getHybridizationParameters() != null) {
						// if we are considering type hints.
						// TODO: Actually get the value here (#111).
						if (this.getHybridizationParameters().hasExperimentalFollowTypeHintsParam()) {
							LOG.info("Following type hints for: " + this + ".");

							// try to get its type from the AST.
							TypeInfo argTypeInfo = NodeUtils.getTypeForParameterFromAST(paramName, functionDef);

							if (argTypeInfo != null) {
								LOG.info("Found type for parameter " + paramName + " in " + this + ": " + argTypeInfo.getActTok() + ".");

								exprType node = argTypeInfo.getNode();
								Attribute typeHintExpr = (Attribute) node;

								// Look up the definition.
								IDocument document = this.getContainingDocument();
								PySelection selection = Util.getSelection(typeHintExpr.attr, document);

								String fqn;
								try {
									fqn = Util.getFullyQualifiedName(typeHintExpr, this.containingModuleName, containingFile, selection,
											this.nature, monitor);
								} catch (AmbiguousDeclaringModuleException e) {
									LOG.warn(String.format(
											"Can't determine FQN of type hint expression: %s in selection: %s, module: %s, file: %s, and project: %s.",
											typeHintExpr, selection.getSelectedText(), containingModuleName, containingFile.getName(),
											nature.getProject()), e);

									monitor.worked(1);
									continue; // next parameter.
								}

								LOG.info("Found FQN: " + fqn + ".");

								if (fqn.equals(TF_TENSOR_FQN)) { // TODO: Also check for subtypes.
									this.likelyHasTensorParameter = true;
									LOG.info(this + " likely has a tensor parameter.");
									monitor.done();
									return;
								}
							}
						}
					}
					monitor.worked(1);
				}
			}
		}

		this.likelyHasTensorParameter = false;
		LOG.info(this + " does not likely have a tensor parameter.");
		monitor.done();
	}

	private void computeIsHybrid(IProgressMonitor monitor) {
		// TODO: Consider mechanisms other than decorators (e.g., higher order functions; #3).
		monitor.beginTask("Computing hybridization ...", IProgressMonitor.UNKNOWN);

		FunctionDefinition functionDefinition = this.getFunctionDefinition();
		decoratorsType[] decoratorArray = functionDefinition.getFunctionDef().decs;

		if (decoratorArray != null) {
			this.containingModuleName = this.getContainingModuleName();
			this.containingFile = this.getContainingFile();
			this.nature = this.getNature();

			for (decoratorsType decorator : decoratorArray) {
				String decoratorFunctionRepresentation = NodeUtils.getFullRepresentationString(decorator.func);
				LOG.info("Computing whether decorator: " + decoratorFunctionRepresentation + " is hybrid.");

				IDocument document = this.getContainingDocument();
				PySelection selection = null;

				// if this function is decorated with "tf.function."
				boolean hybrid = false;

				try {
					selection = Util.getSelection(decorator, document);
					hybrid = isHybrid(decorator, this.containingModuleName, this.containingFile, selection, this.nature, monitor);
				} catch (AmbiguousDeclaringModuleException | BadLocationException | RuntimeException e) {
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
								decoratorFunctionRepresentation, selectedText, this.containingModuleName, this.containingFile.getName(),
								this.nature.getProject()));
						// TODO: Add info status here (#120).
					} else if (Util.isBuiltIn(decorator)) {
						// Since tf.function isn't built-in, skip built-in decorators.
						LOG.info(String.format(
								"Encountered potentially built-in decorator: %s in selection: %s, module: %s, file: %s, and project; %s.",
								decoratorFunctionRepresentation, selectedText, this.containingModuleName, this.containingFile.getName(),
								this.nature.getProject()));
						// TODO: Add info status here (#120).
					} else {
						LOG.warn(String.format(
								"Can't determine if decorator: %s in selection: %s, module: %s, file: %s, and project; %s is hybrid.",
								decoratorFunctionRepresentation, selectedText, this.containingModuleName, this.containingFile.getName(),
								this.nature.getProject()), e);

						// TODO: Add a failure status here? (#120). It could just be that we're taking the last defined one. A failure
						// status entry would fail the entire function.
					}
				}

				if (hybrid) {
					this.isHybrid = true;
					LOG.info(this + " is hybrid.");
					monitor.done();
					return;
				}
				monitor.worked(1);
			}
		}

		this.isHybrid = false;
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
	 */
	private static boolean isHybrid(decoratorsType decorator, String containingModuleName, File containingFile, PySelection selection,
			IPythonNature nature, IProgressMonitor monitor) throws BadLocationException, AmbiguousDeclaringModuleException {
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
		return this.getIdentifer() + "()";
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
