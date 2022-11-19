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
		FunctionDef functionDef = this.getFunctionDef();

		return Objects.hash(functionDef, Util.getQualifiedName(functionDef), this.containingModuleName, this.containingFile,
				this.containingDocument, functionDef.beginColumn, functionDef.beginLine, functionDef.getId());
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

		// first, check if the FunctionDefs are the same.
		FunctionDef lhsFunctionDef = getFunctionDef();
		FunctionDef rhsFunctionDef = other.getFunctionDef();

		boolean functionDefsEqual = Objects.equals(lhsFunctionDef, rhsFunctionDef);

		if (functionDefsEqual) {
			// now, check their qualified names.
			String lhsQualifiedName = Util.getQualifiedName(lhsFunctionDef);
			String rhsQualifiedName = Util.getQualifiedName(rhsFunctionDef);

			boolean qualifiedNamesEqual = lhsQualifiedName.equals(rhsQualifiedName);

			// if the qualified names equal.
			if (qualifiedNamesEqual) {
				// check other attributes.
				return Objects.equals(this.containingModuleName, other.containingModuleName)
						&& Objects.equals(this.containingFile, other.containingFile)
						&& Objects.equals(this.containingDocument, other.containingDocument)
						&& Objects.equals(lhsFunctionDef.beginColumn, rhsFunctionDef.beginColumn)
						&& Objects.equals(lhsFunctionDef.beginLine, rhsFunctionDef.beginColumn)
						&& Objects.equals(lhsFunctionDef.getId(), rhsFunctionDef.getId());
			}
		}

		return false;
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
