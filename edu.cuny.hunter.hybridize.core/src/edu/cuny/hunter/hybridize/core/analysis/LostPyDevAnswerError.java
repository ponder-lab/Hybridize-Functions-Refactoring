package edu.cuny.hunter.hybridize.core.analysis;

import java.io.File;

import org.eclipse.core.resources.IProject;

/**
 * Thrown when one of the analysis's PyDev lookups lost its answer to a dead PyDev completion server: a request the lookup made to the
 * server on the calling thread got no answer (issue 1121). Such a lookup comes back incomplete, often empty, as if nothing were found, and
 * an analysis continuing from it reports different results without saying so. So it stops the run.
 * <p>
 * It's an {@link Error} rather than an {@link Exception}, so that the analysis's many catches of {@link Exception}, which turn a failed
 * step into a per-function result, let it through. Where a catch of {@link Throwable} would isolate it, it's rethrown explicitly.
 *
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1121">Issue 1121</a>
 */
public class LostPyDevAnswerError extends Error {

	private static final long serialVersionUID = 2877349184385470129L;

	/**
	 * Creates an error for the given lookup, whose answer was lost.
	 *
	 * @param lookup The kind of lookup whose answer was lost.
	 * @param selectedText The text the lookup was made for.
	 * @param containingModName The name of the module containing the text.
	 * @param containingFile The {@link File} containing the module.
	 * @param project The project containing the module.
	 */
	public LostPyDevAnswerError(String lookup, String selectedText, String containingModName, File containingFile, IProject project) {
		super(String.format(
				"PyDev's %s lookup lost its answer to its completion server for selection: %s, module: %s, file: %s, and project: %s. The analysis is stopped, as its results would be incomplete.",
				lookup, selectedText, containingModName, containingFile, project));
	}
}
