package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;
import java.util.ArrayList;
import java.util.Objects;

import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.List;
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.Num;
import org.python.pydev.parser.jython.ast.Str;
import org.python.pydev.parser.jython.ast.Tuple;
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
	 * https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function Note: We are also parsing the deprecated parameters specified in
	 * the documentation. Users can still use these deprecated parameters. Therefore we need to be able to account for them. Please refer to
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
		 * Value of this {@link Function}'s {@link decoratorsType} parameter autograph. The values could be True or False.
		 */
		private String autoGraphParamValue;

		/**
		 * Value of this {@link Function}'s {@link decoratorsType} parameter experimental_follow_type_hints. The values could be None, False
		 * or True.
		 */
		private String experimentaFollowTypeHintsParamValue;

		/**
		 * Value of this {@link Function}'s {@link decoratorsType} parameter experimental_autograph_options. The values could be an optional
		 * tuple or value of tf.autograph.experimental.Feature values or None.
		 */
		private String experimentalAutographOptionsParamValue;

		/**
		 * Value of this {@link Function}'s {@link decoratorsType} parameter experimental_implements. The value could be None or a name of a
		 * "known" function this implements.
		 */
		private String experimentalImplementsParamValue;

		/**
		 * Value of this {@link Function}'s {@link decoratorsType} parameter func. The value could be None, or the function name to be
		 * compiled.
		 */
		private String funcParamValue;

		/**
		 * Value of this {@link Function}'s {@link decoratorsType} parameter input_signature. The value could be None, or a possibly nested
		 * sequence of tf.TensorSpec objects specifying the shapes and dtypes of the Tensors that will be supplied to this function
		 */
		private String inputSignatureParamValue;

		/**
		 * Value of this {@link Function}'s {@link decoratorsType} parameter jit_compile. The values could be None, False or True.
		 */
		private String jitCompileParamValue;

		/**
		 * Value of this {@link Function}'s {@link decoratorsType} has parameter reduce_retracing. The values could be False or True.
		 */
		private String reduceRetracingParamValue;

		public HybridizationParameters(IProgressMonitor monitor) throws BadLocationException {
			FunctionDefinition functionDefinition = Function.this.getFunctionDefinition();
			decoratorsType[] decoratorArray = functionDefinition.getFunctionDef().decs;

			// Will contain the last tf.function decorator
			decoratorsType tfFunctionDecorator = null;

			// Iterate through the decorators of the function
			for (decoratorsType decorator : decoratorArray) {
				IDocument document = Function.this.getContainingDocument();
				PySelection selection = Util.getSelection(decorator, document);

				// Save the hybrid decorator
				try {
					if (Function.isHybrid(decorator, Function.this.containingModuleName, Function.this.containingFile, selection,
							Function.this.nature, monitor)) // TODO: Cache this from a previous call (#118).
						tfFunctionDecorator = decorator;
				} catch (AmbiguousDeclaringModuleException e) {
					throw new IllegalStateException("Can't determine whether decorator: " + decorator + " is hybrid.", e);
				}
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
							if (name.id.equals(FUNC)) {
								// Found parameter func
								// Example of value: Name of function or None
								if (keyword.value instanceof Name) {
									Name value = (Name) keyword.value;
									if (value.id == "None") // Checking only literals
										this.funcParamValue = value.id;
									else
										throw new IllegalArgumentException("Unable to process " + FUNC + " argument.");
								} else {
									throw new IllegalArgumentException("Unable to process " + FUNC + " argument.");
								}
							} else if (name.id.equals(INPUT_SIGNATURE)) {
								// Found parameter input_signature
								// Example of value: None
								if (keyword.value instanceof Name) {
									Name value = (Name) keyword.value;
									if (value.id == "None") // Checking only literals
										this.inputSignatureParamValue = value.id;
									else
										throw new IllegalArgumentException("Unable to process " + INPUT_SIGNATURE + " argument.");
									// Example: (tf.TensorSpec(shape=[None], dtype=tf.float32),)
								} else if (keyword.value instanceof Tuple) {
									Tuple value = (Tuple) keyword.value;
									exprType[] valueElements = value.elts;
									ArrayList<TensorSpec> tensorSpecList = processTensorSpecs(valueElements);
									this.inputSignatureParamValue = "(" + createTupleOrListOfTensorSpec(tensorSpecList);
									if (value.endsWithComma)
										this.inputSignatureParamValue += ",)";
									else
										this.inputSignatureParamValue += ")";
									// Example: [tf.TensorSpec(shape=[None], dtype=tf.float32)]
								} else if (keyword.value instanceof List) {
									List value = (List) keyword.value;
									exprType[] valueElements = value.elts;
									ArrayList<TensorSpec> tensorSpecList = processTensorSpecs(valueElements);
									this.inputSignatureParamValue = "[" + createTupleOrListOfTensorSpec(tensorSpecList) + "]";
								} else {
									throw new IllegalArgumentException("Unable to process " + INPUT_SIGNATURE + " argument.");
								}
							} else if (name.id.equals(AUTOGRAPH)) {
								// Found parameter autograph
								// Example of value: True, False
								if (keyword.value instanceof Name) {
									Name value = (Name) keyword.value;
									if (value.id == "True" || value.id == "False") // Checking only literals
										this.autoGraphParamValue = value.id;
									else
										throw new IllegalArgumentException("Unable to process " + AUTOGRAPH + " argument.");
								} else {
									throw new IllegalArgumentException("Unable to process " + AUTOGRAPH + " argument.");
								}
								// The version of the API we are using allows
								// parameter names jit_compile and
								// deprecated name experimental_compile
							} else if (name.id.equals(JIT_COMPILE) || name.id.equals(EXPERIMENTAL_COMPILE)) {
								// Found parameter jit_compile or experimental_compile
								// Example of value: True, False, None
								if (keyword.value instanceof Name) {
									Name value = (Name) keyword.value;
									if (value.id == "True" || value.id == "False" || value.id == "None") // Checking only literals
										this.jitCompileParamValue = value.id;
									else
										throw new IllegalArgumentException(
												"Unable to process " + JIT_COMPILE + "/" + EXPERIMENTAL_COMPILE + " argument.");
								} else {
									throw new IllegalArgumentException(
											"Unable to process " + JIT_COMPILE + "/" + EXPERIMENTAL_COMPILE + " argument.");
								}
								// The version of the API we are using allows
								// parameter names reduce_retracing
								// and deprecated name experimental_relax_shapes
							} else if (name.id.equals(REDUCE_RETRACING) || name.id.equals(EXPERIMENTAL_RELAX_SHAPES)) {
								// Found parameter reduce_retracing or experimental_relax_shapes
								// Example of value: True, False
								if (keyword.value instanceof Name) {
									Name value = (Name) keyword.value;
									if (value.id == "True" || value.id == "False") // Checking only literals
										this.reduceRetracingParamValue = value.id;
									else
										throw new IllegalArgumentException(
												"Unable to process " + REDUCE_RETRACING + "/" + EXPERIMENTAL_RELAX_SHAPES + " argument.");
								} else {
									throw new IllegalArgumentException(
											"Unable to process " + REDUCE_RETRACING + "/" + EXPERIMENTAL_RELAX_SHAPES + " argument.");
								}
							} else if (name.id.equals(EXPERIMENTAL_IMPLEMENTS)) {
								// Found parameter experimental_implements
								// Example of value: "google.matmul_low_rank_matrix"
								if (keyword.value instanceof Str) {
									Str value = (Str) keyword.value;
									this.experimentalImplementsParamValue = value.s;
									// Example of value: None
								} else if (keyword.value instanceof Name) {
									Name value = (Name) keyword.value;
									if (value.id == "None") // Checking only literals
										this.experimentalImplementsParamValue = value.id;
									else
										throw new IllegalArgumentException("Unable to process " + EXPERIMENTAL_IMPLEMENTS + " argument.");
								} else {
									throw new IllegalArgumentException("Unable to process " + EXPERIMENTAL_IMPLEMENTS + " argument.");
								}
							} else if (name.id.equals(EXPERIMENTAL_AUTOGRAPH_OPTIONS)) {
								// Found parameter experimental_autograph_options
								// Example of value: tf.autograph.experimental.Feature.EQUALITY_OPERATORS
								if (keyword.value instanceof Attribute) {
									Attribute keywordAttribute = (Attribute) keyword.value;
									this.experimentalAutographOptionsParamValue = processAttributeForAutographOptions(keywordAttribute);
									// Example of value: (tf.autograph.experimental.Feature.EQUALITY_OPERATORS,
									// tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS)
								} else if (keyword.value instanceof Tuple) {
									Tuple keywordTuple = (Tuple) keyword.value;
									exprType[] keywordExpr = keywordTuple.elts;
									String finalTuple = "";
									int count = 0;
									for (exprType expr : keywordExpr) {
										if (expr instanceof Attribute) {
											Attribute keywordAttribute = (Attribute) expr;
											if (count == 0)
												finalTuple += processAttributeForAutographOptions(keywordAttribute);
											else
												finalTuple += ", " + processAttributeForAutographOptions(keywordAttribute);
										}
										count++;
									}
									this.experimentalAutographOptionsParamValue = "(" + finalTuple + ")";
									// Example of value: None
								} else if (keyword.value instanceof Name) {
									Name value = (Name) keyword.value;
									if (value.id == "None") // Checking only literals
										this.experimentalAutographOptionsParamValue = value.id;
									else
										throw new IllegalArgumentException(
												"Unable to process " + EXPERIMENTAL_AUTOGRAPH_OPTIONS + " argument.");
								} else {
									throw new IllegalArgumentException(
											"Unable to process " + EXPERIMENTAL_AUTOGRAPH_OPTIONS + " arguments");
								}
							} else if (name.id.equals(EXPERIMENTAL_FOLLOW_TYPE_HINTS)) {
								// Found parameter experimental_follow_type_hints
								// Example of value: True, False, None
								if (keyword.value instanceof Name) {
									Name value = (Name) keyword.value;
									if (value.id == "None" || value.id == "True" || value.id == "False") // Checking only literals
										this.experimentaFollowTypeHintsParamValue = value.id;
									else
										throw new IllegalArgumentException(
												"Unable to process " + EXPERIMENTAL_FOLLOW_TYPE_HINTS + " argument.");
								} else {
									throw new IllegalArgumentException(
											"Unable to process " + EXPERIMENTAL_FOLLOW_TYPE_HINTS + " arguments");
								}
							}
						}
					}
				} // else, tf.function is used without parameters.
		}

		/**
		 * Parses expressions to return a string of the shape of a TensorSpec for input signature.
		 *
		 * @return String of TensorSpec shape.
		 */
		private String processTupleOrListForShape(exprType[] exprTupleOrList) {
			int count = 0;
			String tempString = "";

			for (exprType expr : exprTupleOrList) {
				if (expr instanceof Num) {
					if (count == 0)
						tempString += ((Num) expr).num;
					else
						tempString += ", " + ((Num) expr).num;
					count++;
				} else if (expr instanceof Name) {
					if (((Name) expr).id == "None") // Checking only literals
						tempString = ((Name) expr).id;
					else
						throw new IllegalArgumentException("Unable to process " + INPUT_SIGNATURE + " argument.");
				} else
					throw new IllegalArgumentException("Unable to process " + INPUT_SIGNATURE + " argument.");
			}

			return tempString;

		}

		/**
		 * Parses attributes to return a string of the autograph options.
		 *
		 * @return String of autograph options that contains various attributes.
		 */
		private String processAttributeForAutographOptions(Attribute keywordAttribute) {
			StringBuilder argument = new StringBuilder();
			Attribute tempAttr = keywordAttribute;

			while (tempAttr.value instanceof Attribute) {
				NameTok valueAttribute = (NameTok) tempAttr.attr;
				argument.insert(0, valueAttribute.id);
				argument.insert(0, ".");
				if (tempAttr.value instanceof Attribute)
					tempAttr = (Attribute) tempAttr.value;
				else
					throw new IllegalArgumentException("Unable to process " + EXPERIMENTAL_AUTOGRAPH_OPTIONS + " argument.");
			}

			return ((Name) tempAttr.value).id + "." + ((NameTok) tempAttr.attr).id + argument.toString();
		}

		/**
		 * Parses expressions to retrieve information about the TensorSpecs for input signature.
		 *
		 * @return Array of TensorSpecs with the parsed information.
		 */
		private ArrayList<TensorSpec> processTensorSpecs(exprType[] valueElements) {
			ArrayList<TensorSpec> tensorSpecList = new ArrayList<>();
			for (exprType expr : valueElements) {
				if (expr instanceof Call) {
					Call callTuple = (Call) expr;
					TensorSpec tensor = new TensorSpec();

					// Positional arguments for TensorSpecs
					exprType[] tensorArgs = callTuple.args;
					for (exprType tensorArg : tensorArgs) {
						if (tensorArg instanceof Tuple) {
							tensor.setShape(processTupleOrListForShape(((Tuple) tensorArg).elts));
							tensor.setShapeKeyword(false);
						} else if (tensorArg instanceof List) {
							tensor.setShape(processTupleOrListForShape(((List) tensorArg).elts));
							tensor.setShapeKeyword(false);
						} else if (tensorArg instanceof Attribute) {
							Attribute attrValue = (Attribute) tensorArg;
							tensor.setDType(((Name) attrValue.value).id + "." + ((NameTok) attrValue.attr).id);
							tensor.setDTypeKeyword(false);
						} else {
							throw new IllegalArgumentException("Unable to process " + INPUT_SIGNATURE + " argument.");
						}

					}
					// Keyword Arguments for TensorSpecs
					keywordType[] keywordsCall = callTuple.keywords;
					for (keywordType keyword : keywordsCall) {
						if (keyword.value instanceof Tuple)
							tensor.setShape(processTupleOrListForShape(((Tuple) keyword.value).elts));
						else if (keyword.value instanceof List)
							tensor.setShape(processTupleOrListForShape(((List) keyword.value).elts));
						else if (keyword.value instanceof Attribute) {
							Attribute attrValue = (Attribute) keyword.value;
							tensor.setDType(((Name) attrValue.value).id + "." + ((NameTok) attrValue.attr).id);
						} else {
							throw new IllegalArgumentException("Unable to process " + INPUT_SIGNATURE + " argument.");
						}
					}
					tensorSpecList.add(tensor);
				} else {
					throw new IllegalArgumentException("Unable to process " + INPUT_SIGNATURE + " argument.");
				}
			}
			return tensorSpecList;
		}

		/**
		 * Gets the array of Tensorspecs and returns the tuple or list of them, if necessary.
		 *
		 * @return String of nested TensorSpecs.
		 */
		private String createTupleOrListOfTensorSpec(ArrayList<TensorSpec> tensorSpecList) {
			String tempString = "";

			int count = 0;
			for (TensorSpec tensor : tensorSpecList) {
				if (count == 0)
					tempString += tensor.toString();
				else
					tempString += ", " + tensor.toString();
				count++;
			}

			return tempString;
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter autograph.
		 *
		 * @return True iff this {@link decoratorType} has parameter autograph.
		 */
		public boolean hasAutoGraphParam() {
			return (this.autoGraphParamValue != null);

		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_autograph_options.
		 *
		 * @return True iff this {@link decoratorType} has parameter experimental_autograph_options.
		 */
		public boolean hasExperimentalAutographOptParam() {
			return (this.experimentalAutographOptionsParamValue != null);
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_implements.
		 *
		 * @return True iff this {@link decoratorType} has parameter experimental_implements.
		 */
		public boolean hasExperimentalImplementsParam() {
			return (this.experimentalImplementsParamValue != null);
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter experimental_follow_type_hints.
		 *
		 * @return True iff this {@link decoratorType} has parameter experimental_follow_type_hints.
		 */
		public boolean hasExperimentalFollowTypeHintsParam() {
			return (this.experimentaFollowTypeHintsParamValue != null);
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter has parameter func.
		 *
		 * @return True iff this {@link decoratorType} has parameter func.
		 */
		public boolean hasFuncParam() {
			return (this.funcParamValue != null);
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter input_signature.
		 *
		 * @return True iff this {@link decoratorType} has parameter input_signature.
		 */
		public boolean hasInputSignatureParam() {
			return (this.inputSignatureParamValue != null);
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter jit_compile.
		 *
		 * @return True iff this {@link decoratorType} has parameter jit_compile.
		 */
		public boolean hasJitCompileParam() {
			return (this.jitCompileParamValue != null);
		}

		/**
		 * True iff this {@link Function}'s {@link decoratorsType} has parameter reduce_retracing.
		 *
		 * @return True iff this {@link Function} has parameter reduce_retracing.
		 */
		public boolean hasReduceRetracingParam() {
			return (this.reduceRetracingParamValue != null);
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter autograph.
		 *
		 * @return String of this {@link decoratorType} parameter autograph.
		 */
		public String getAutoGraphArg() {
			return this.autoGraphParamValue;
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter experimental_autograph_options.
		 *
		 * @return String of this {@link decoratorType} parameter experimental_autograph_options.
		 */
		public String getExperimentalAutographOptArg() {
			return this.experimentalAutographOptionsParamValue;
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter experimental_implements.
		 *
		 * @return String of this {@link decoratorType} parameter experimental_implements.
		 */
		public String getExperimentalImplementsArg() {
			return this.experimentalImplementsParamValue;
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter experimental_follow_type_hints.
		 *
		 * @return String of this {@link decoratorType} parameter experimental_follow_type_hints.
		 */
		public String getExperimentalFollowTypeHintsArg() {
			return this.experimentaFollowTypeHintsParamValue;
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter has parameter func.
		 *
		 * @return String of this {@link decoratorType} parameter func.
		 */
		public String getFuncArg() {
			return this.funcParamValue;
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter input_signature.
		 *
		 * @return String of this {@link decoratorType} parameter input_signature.
		 */
		public String getInputSignatureArg() {
			return this.inputSignatureParamValue;
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter jit_compile.
		 *
		 * @return String of this {@link decoratorType} parameter jit_compile.
		 */
		public String getJitCompileArg() {
			return this.jitCompileParamValue;
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter reduce_retracing.
		 *
		 * @return String of this {@link Function} parameter reduce_retracing.
		 */
		public String getReduceRetracingArg() {
			return this.reduceRetracingParamValue;
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter experimental_compile.
		 *
		 * @return String of this {@link decoratorType} parameter experimental_compile.
		 */
		public String getExperimentalCompileArg() {
			return this.jitCompileParamValue;
		}

		/**
		 * Value of {@link Function}'s {@link decoratorsType} parameter experimental_relax_shapes.
		 *
		 * @return String of this {@link Function} parameter experimental_relax_shapes.
		 */
		public String getExperimentalRelaxShapeArg() {
			return this.reduceRetracingParamValue;
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
