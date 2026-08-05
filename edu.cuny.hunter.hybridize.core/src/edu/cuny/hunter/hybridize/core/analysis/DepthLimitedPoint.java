package edu.cuny.hunter.hybridize.core.analysis;

/**
 * A points-to result the tensor analysis abandoned because the targeted k-CFA depth was too short to reach the relevant context (Ariadne's
 * depth-limited signal, surfaced per project so an operator can raise the per-project {@code targetedCfaDepth} accordingly;
 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/670). Rendered WALA-free so consumers outside the core bundle (the
 * evaluator) need no visibility into the embedded WALA packages.
 *
 * @param method The function containing the abandoned points-to result, identified by its CAst declaring-class name: the Python front end
 *        models each function as its own synthetic class (e.g., {@code Lscript train.py/sanity_check}), so the class name names the
 *        function, the same identification the body-local scans use.
 * @param valueNumber The SSA value number of the abandoned result within {@link #method()}.
 * @param callStringLength The call-string length at which the analysis stopped following the context.
 */
public record DepthLimitedPoint(String method, int valueNumber, int callStringLength) {
}
