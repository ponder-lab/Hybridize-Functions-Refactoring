package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.io.File;

import org.eclipse.core.runtime.ILog;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jface.text.BadLocationException;
import org.python.pydev.ast.refactoring.TooManyMatchesException;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.decoratorsType;
import org.python.pydev.parser.visitors.NodeUtils;

import edu.cuny.citytech.refactoring.common.core.RefactorableProgramEntity;

/**
 * A representation of a Python function.
 *
 * @author <a href="mailto:rk1424@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @author <a href="mailto:tcastrovelez@gradcenter.cuny.edu">Tatiana Castro Vélez</a>
 */
public class Function extends RefactorableProgramEntity {

	private static final String TF_FUNCTION_FQN = "tensorflow.python.eager.def_function.function";

	private static final ILog LOG = getLog(Function.class);

	/**
	 * The {@link FunctionDef} representing this {@link Function}.
	 */
	private FunctionDef functionDef;

	/**
	 * True iff this {@link Function} is decorated with tf.function.
	 */
	private boolean isHybrid;

	/**
	 * True iff this {@link Function} has at least one parameter that is a tf.Tensor (https://bit.ly/3vYG7iP).
	 */
	private boolean likelyHasTensorParameter;

	private String containingModuleName;

	private File containingFile;

	private IPythonNature nature;

	public Function(FunctionDef functionDef, String containingModuleName, File containingFile, IPythonNature nature,
			IProgressMonitor monitor) throws TooManyMatchesException, BadLocationException {
		this.functionDef = functionDef;
		this.containingModuleName = containingModuleName;
		this.containingFile = containingFile;
		this.nature = nature;

		// Find out if it's hybrid via the tf.function decorator.
		this.computeIsHybrid(monitor);
		this.computeHasTensorParameter();
	}

	private void computeHasTensorParameter() {
		// TODO: Use type info API. If that gets info from type hints, then we'll need another field indicating whether
		// type hints are used.
	}

	private void computeIsHybrid(IProgressMonitor monitor) throws TooManyMatchesException, BadLocationException {
		// TODO: Consider mechanisms other than decorators (e.g., higher order functions; #3).
		monitor.setTaskName("Computing hybridization ...");

		FunctionDef functionDef = this.getFunctionDef();
		decoratorsType[] decoratorArray = functionDef.decs;

		if (decoratorArray != null) {
			String containingModuleName = this.getContainingModuleName();
			File containingFile = this.getContainingFile();
			IPythonNature nature = this.getNature();

			for (decoratorsType decorator : decoratorArray) {
				PySelection selection = this.getSelection(decorator);

				String decoratorFQN = Util.getFullyQualifiedName(decorator, containingModuleName, containingFile, selection, nature,
						monitor);

//				// if this function is decorated with "tf.function."
				if (decoratorFQN.equals(TF_FUNCTION_FQN)) {
					this.isHybrid = true;
					LOG.info(this + " is hybrid.");
					return;
				}
			}
		} else
			this.isHybrid = false;
	}

	private IPythonNature getNature() {
		return this.nature;
	}

	private PySelection getSelection(decoratorsType decorator) {
		// TODO Auto-generated method stub
		return null;
	}

	private File getContainingFile() {
		return this.containingFile;
	}

	private String getContainingModuleName() {
		return this.containingModuleName;
	}

	/**
	 * Accessor for private member variable functionDef.
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
		FunctionDef functionDef = this.getFunctionDef();
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

	/**
	 * Accessor for private member variable isHybrid.
	 *
	 * @return Boolean that states if this {@link Function} is decorated with tf.function.
	 */
	public boolean isHybrid() {
		return this.isHybrid;
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
