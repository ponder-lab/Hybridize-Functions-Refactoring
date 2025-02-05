package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayList;
import java.util.Collection;

import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.VisitorBase;

/**
 * Extracts function definitions from a given AST node.
 *
 * @author <a href="mailto:rk1424@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class FunctionExtractor extends VisitorBase {

	private Collection<FunctionDef> definitions = new ArrayList<>();

	public Collection<FunctionDef> getDefinitions() {
		return this.definitions;
	}

	@Override
	public void traverse(SimpleNode node) throws Exception {
		node.traverse(this);
	}

	@Override
	protected Object unhandled_node(SimpleNode node) throws Exception {
		return null;
	}

	@Override
	public Object visitFunctionDef(FunctionDef node) throws Exception {
		this.getDefinitions().add(node);
		return super.visitFunctionDef(node);
	}
}
