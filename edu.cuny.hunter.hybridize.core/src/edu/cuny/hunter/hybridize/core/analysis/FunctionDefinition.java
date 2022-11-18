package edu.cuny.hunter.hybridize.core.analysis;

import java.io.File;
import java.util.Objects;

import org.eclipse.jface.text.IDocument;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.visitors.NodeUtils;

// TODO: This class needs documentation.
public final class FunctionDefinition {

	FunctionDef functionDef;

	String containingModuleName;

	File containingFile;

	IDocument containingDocument;

	IPythonNature nature;

	public FunctionDefinition(FunctionDef functionDef, String containingModuleName, File containingFile, IDocument containingDocument,
			IPythonNature nature) {
		this.functionDef = functionDef;
		this.containingModuleName = containingModuleName;
		this.containingFile = containingFile;
		this.containingDocument = containingDocument;
		this.nature = nature;
	}

	@Override
	public int hashCode() {
		return getFunctionDef().hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;

		if (obj == null)
			return false;

		if (getClass() != obj.getClass())
			return false;

		FunctionDefinition other = (FunctionDefinition) obj;

		// FIXME: I would think we need other members.
		return Objects.equals(getFunctionDef(), other.getFunctionDef());
	}

	@Override
	public String toString() {
		String fullRepresentationString = NodeUtils.getFullRepresentationString(functionDef);
		return fullRepresentationString + "() in " + containingModuleName;
	}

	public FunctionDef getFunctionDef() {
		return functionDef;
	}
}
