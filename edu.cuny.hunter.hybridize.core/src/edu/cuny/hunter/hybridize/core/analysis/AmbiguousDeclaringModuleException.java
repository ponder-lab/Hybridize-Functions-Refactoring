package edu.cuny.hunter.hybridize.core.analysis;

import java.io.File;

import org.eclipse.jface.text.BadLocationException;
import org.python.pydev.core.IPythonNature;
import org.python.pydev.core.docutils.PySelection;

public class AmbiguousDeclaringModuleException extends Exception {

	private static final long serialVersionUID = 4364177931172130140L;

	public AmbiguousDeclaringModuleException(PySelection selection, String containingModName, File containingFile, IPythonNature nature,
			int matchesFound) throws BadLocationException {
		super(String.format("Ambigious definitions (%d) found for selection: %s in line: %s, module: %s, file: %s, and project: %s.",
				Integer.valueOf(matchesFound), selection.getSelectedText(), selection.getLineWithoutCommentsOrLiterals().strip(),
				containingModName, containingFile.getName(), nature.getProject()));
	}
}
