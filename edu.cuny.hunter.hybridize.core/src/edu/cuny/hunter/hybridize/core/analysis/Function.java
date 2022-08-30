package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import org.eclipse.core.runtime.ILog;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.jython.ast.keywordType;
import org.python.pydev.parser.visitors.NodeUtils;

import edu.cuny.citytech.refactoring.common.core.RefactorableProgramEntity;

/**
 * A representation of a Python function.
 *
 * @author <a href="mailto:rk1424@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @author <a href="mailto:tcastrovelez@gradcenter.cuny.edu">Tatiana Castro Vélez</a>
 */
public class Function extends RefactorableProgramEntity {

	private static final ILog LOG = getLog(Function.class);

	/**
	 * True iff this {@link Function} has argument autograph.
	 */
	private boolean autograph;

	/**
	 * True iff this {@link Function} has argument
	 * experimental_autograph_options.
	 */
	private boolean experimental_autograph_options;

	/**
	 * True iff this {@link Function} has argument
	 * experimental_follow_type_hints.
	 */
	private boolean experimental_follow_type_hints;

	/**
	 * True iff this {@link Function} has argument experimental_implements.
	 */
	private boolean experimental_implements;

	/**
	 * True iff this {@link Function} has argument func.
	 */
	private boolean func;

	/**
	 * The {@link FunctionDef} representing this {@link Function}.
	 */
	private FunctionDef functionDef;

	/**
	 * True iff this {@link Function} has argument input_signature.
	 */
	private boolean input_signature;

	/**
	 * True iff this {@link Function} is decorated with tf.function.
	 */
	private boolean isHybrid;

	/**
	 * True iff this {@link Function} has argument jit_compile.
	 */
	private boolean jit_compile;

	/**
	 * True iff this {@link Function} has argument reduce_retracing.
	 */
	private boolean reduce_retracing;
  
	/**
	 * True iff this {@link Function} has at least one parameter that is a tf.Tensor (https://bit.ly/3vYG7iP).
	 */
	private boolean likelyHasTensorParameter;

	public Function(FunctionDef functionDef) {
		this.functionDef = functionDef;

		// Find out if it's hybrid via the tf.function decorator.
		this.computeIsHybrid();
		this.computeHasTensorParameter();
    
		// Parse the tf.function parameters
		this.computeExistenceOfParameters();
	}

	private void computeHasTensorParameter() {
		// TODO: Use type info API. If that gets info from type hints, then we'll need another field indicating whether
		// type hints are used.
	}

	private void computeIsHybrid() {
		// FIXME: This is fragile. What we really want to know is whether the decorator is
		// tensorflow.python.eager.def_function.function, which is "exported" as "function." See https://bit.ly/3O5xpFH
		// (#47).
		// TODO: Consider mechanisms other than decorators (e.g., higher order functions). See #3.
		decoratorsType[] decoratorArray = this.functionDef.decs;

		if (decoratorArray != null)
			for (decoratorsType decorator : decoratorArray)
				// If it is not an attribute then we cannot access it this way,
				// therefore we need the if statement
				if (decorator.func instanceof Attribute) { // e.g., tf.function
					Attribute decoratorFunction = (Attribute) decorator.func;
					if (decoratorFunction.value instanceof Name) {
						Name decoratorName = (Name) decoratorFunction.value;
						// We have a viable prefix. Get the attribute.
						if (decoratorName.id.equals("tf") && decoratorFunction.attr instanceof NameTok) {
							NameTok decoratorAttribute = (NameTok) decoratorFunction.attr;
							if (decoratorAttribute.id.equals("function")) {
								// Found "tf.function."
								this.isHybrid = true;
								LOG.info(this + " is hybrid.");
							}
						}
					}
				} else if (decorator.func instanceof Call) { // e.g., tf.function(...)
					Call decoratorFunction = (Call) decorator.func;
					if (decoratorFunction.func instanceof Attribute) {
						Attribute callFunction = (Attribute) decoratorFunction.func;
						if (callFunction.value instanceof Name) {
							Name decoratorName = (Name) callFunction.value;
							// We have a viable prefix. Get the attribute.
							if (decoratorName.id.equals("tf") && callFunction.attr instanceof NameTok) {
								NameTok decoratorAttribute = (NameTok) callFunction.attr;
								if (decoratorAttribute.id.equals("function")) {
									// Found tf.function(...)
									this.isHybrid = true;
									LOG.info(this + " is hybrid.");
								}
							}
						}
					}
				}
	}

	/**
	 * Accessor for private member variable functionDef.
	 *
	 * @return The {@link FunctionDef} representing this {@link Function}
	 */
	public FunctionDef getFunctionDef() {
		return this.functionDef;
	}

	private void computeExistenceOfParameters() {
		decoratorsType[] decoratorArray = functionDef.decs;

		if (decoratorArray != null)
			for (decoratorsType decorator : decoratorArray)
				if (decorator.func instanceof Call) {
					// If tf.function has parameters it will be of instance Call
					Call decoratorFunction = (Call) decorator.func;
					// Get the keywords that will contain the parameters
					keywordType[] keywordArray = decoratorFunction.keywords;
					if (keywordArray != null)
						// Traverse through the keywords
						for (keywordType keyword : keywordArray)
							if (keyword.arg instanceof NameTok) {
								NameTok decoratorArg = (NameTok) keyword.arg;
								if (decoratorArg.id.equals("func"))
									// Found parameter func
									this.func = true;
								else if (decoratorArg.id.equals("input_signature"))
									// Found parameter input_signature
									this.input_signature = true;
								else if (decoratorArg.id.equals("autograph"))
									// Found parameter autograph
									this.autograph = true;
								else if (decoratorArg.id.equals("jit_compile")
										|| decoratorArg.id.equals("experimental_compile"))
									// Found parameter jit_compile/experimental_compile
									this.jit_compile = true;
								else if (decoratorArg.id.equals("reduce_retracing")
										|| decoratorArg.id.equals("experimental_relax_shapes"))
									// Found parameter reduce_retracing/experimental_relax_shapes
									this.reduce_retracing = true;
								else if (decoratorArg.id.equals("experimental_implements"))
									// Found parameter experimental_implements
									this.experimental_implements = true;
								else if (decoratorArg.id.equals("experimental_autograph_options"))
									// Found parameter experimental_autograph_options
									this.experimental_autograph_options = true;
								else if (decoratorArg.id.equals("experimental_follow_type_hints"))
									// Found parameter experimental_follow_type_hints
									this.experimental_follow_type_hints = true;
							}
				}
	}

	/**
	 * Accessor for private member variable autograph
	 *
	 * @return Boolean that states if this {@link Function} has parameter
	 *         autograph
	 */
	public boolean getAutographParam() {
		return autograph;
	}

	/**
	 * Accessor for private member variable experimental_autograph_options
	 *
	 * @return Boolean that states if this {@link Function} has parameter
	 *         experimental_autograph_options
	 */
	public boolean getExpAutographOptParam() {
		return experimental_autograph_options;
	}

	/**
	 * Accessor for private member variable experimental_implements
	 *
	 * @return Boolean that states if this {@link Function} has parameter
	 *         experimental_implements
	 */
	public boolean getExpImplementsParam() {
		return experimental_implements;
	}

	/**
	 * Accessor for private member variable experimental_follow_type_hints
	 *
	 * @return Boolean that states if this {@link Function} has parameter
	 *         experimental_follow_type_hints
	 */
	public boolean getExpTypeHintsParam() {
		return experimental_follow_type_hints;
	}

	/**
	 * Accessor for private member variable func
	 *
	 * @return Boolean that states if this {@link Function} has parameter func
	 */
	public boolean getFuncParam() {
		return func;
	}

	/**
	 * Accessor for private member variable functionDef
	 *
	 * @return The {@link FunctionDef} representing this {@link Function}
	 */
	public FunctionDef getFunctionDef() {
		return this.functionDef;
	}

	/**
	 * Returns the FQN of this {@link Function}.
	 *
	 * @see <a href="https://peps.python.org/pep-3155">PEP 3155</a>
	 * @return This {@link Function}'s FQN.
	 */
	public String getIdentifer() {
		String identifier = NodeUtils.getFullRepresentationString(this.functionDef);
		StringBuilder ret = new StringBuilder();
		SimpleNode parentNode = this.functionDef.parent;

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

	/**
	 * Accessor for private member variable isHybrid.
	 *
	 * @return Boolean that states if this {@link Function} is decorated with tf.function.
	 */
	public boolean isHybrid() {
		return this.isHybrid;
	}
	
	/**
	 * Accessor for private member variable input_signature
	 *
	 * @return Boolean that states if this {@link Function} has parameter
	 *         input_signature
	 */
	public boolean getInputSignatureParam() {
		return input_signature;
	}

	/**
	 * Accessor for private member variable jit_compile
	 *
	 * @return Boolean that states if this {@link Function} has parameter
	 *         jit_compile
	 */
	public boolean getJitCompileParam() {
		return jit_compile;
	}

	/**
	 * Accessor for private member variable reduce_retracing
	 *
	 * @return Boolean that states if this {@link Function} has parameter
	 *         reduce_retracing
	 */
	public boolean getReduceRetracingParam() {
		return reduce_retracing;
	}

	/**
	 * True iff this {@link Function} likely has a tf.Tensor parameter. Since Python is dynamic, we may not be 100%
	 * sure.
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
}
