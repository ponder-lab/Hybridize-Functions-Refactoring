package edu.cuny.hunter.hybridize.core.analysis;

import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.visitors.NodeUtils;
import org.python.pydev.parser.jython.SimpleNode;

import edu.cuny.citytech.refactoring.common.core.RefactorableProgramEntity;

/**
 * A representation of a Python function.
 *
 * @author <a href="mailto:rk1424@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Function extends RefactorableProgramEntity {

	/**
	 * The {@link FunctionDef} representing this {@link Function}.
	 */
	private FunctionDef functionDef;

	/**
	 * True iff this {@link Function} is decorated with tf.function.
	 */
	private boolean isHybrid;

	public Function(FunctionDef functionDef) {
		this.functionDef = functionDef;

		// Find out if it's hybrid via the tf.function decorator.
		this.computeIsHybrid();
	}

	private void computeIsHybrid() {
		// FIXME: This is fragile. What we really want to know is whether the decorator
		// is tensorflow.python.eager.def_function.function, which is "exported" as
		// "function." See https://bit.ly/3O5xpFH.
		// TODO: Consider mechanisms other than decorators (e.g., higher order
		// functions).
		decoratorsType[] decoratorArray = functionDef.decs;

		if (decoratorArray != null)
			for (decoratorsType decorator : decoratorArray) {
				// If it is not an attribute then we cannot access it this way, therefore we need the if statement
				if (decorator.func instanceof Attribute) {
					System.out.println(decorator);
					Attribute decoratorFunction = (Attribute) decorator.func;
					System.out.println(decoratorFunction);
					Name decoratorName = (Name) decoratorFunction.value;
					System.out.println(decoratorName);
					if (decoratorName.id.equals("tf")) {
						// We have a viable prefix. Get the attribute.
						NameTok decoratorAttribute = (NameTok) decoratorFunction.attr;
						if (decoratorAttribute.id.equals("function")) {
							// Found "tf.function."
							this.isHybrid = true;
						}
					}
				}
			}
	}

	@Override
	public String toString() {
		return this.getIdentifer();
	}

	/**
	 * Returns the FQN of this {@link Function}.
	 *
	 * @see https://peps.python.org/pep-3155
	 * @return This {@link Function}'s FQN.
	 */
	public String getIdentifer() {
		String identifier = NodeUtils.getFullRepresentationString(this.functionDef);
		StringBuilder ret = new StringBuilder();
		SimpleNode parentNode = this.functionDef.parent;

		int count = 0;

		while(parentNode instanceof ClassDef || parentNode instanceof FunctionDef) {
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
	 * Accessor for private member variable isHybrid
	 *
	 * @return Boolean that states if this {@link Function} is decorated with tf.function.
	 */
	public boolean isHybrid() {
		return isHybrid;
	}
	
	/**
	 * Accessor for private member variable functionDef
	 *
	 * @return The {@link FunctionDef} representing this {@link Function}
	 */
	public FunctionDef getFunctionDef() {
		return functionDef;
	}
}
