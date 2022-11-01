package edu.cuny.hunter.hybridize.core.analysis;

import java.io.File;
import java.util.Objects;

import org.eclipse.jface.text.IDocument;
import org.python.pydev.parser.jython.ast.FunctionDef;

public class FunctionDefinition {

	FunctionDef functionDef;

	String containingModuleName;

	File containingFile;

	IDocument containingDocument;

	public FunctionDefinition(FunctionDef functionDef, String containingModuleName, File containingFile, IDocument containingDocument) {
		this.functionDef = functionDef;
		this.containingModuleName = containingModuleName;
		this.containingFile = containingFile;
		this.containingDocument = containingDocument;
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

		return Objects.equals(getFunctionDef(), other.getFunctionDef());
	}

	public FunctionDef getFunctionDef() {
		return functionDef;
	}
}
