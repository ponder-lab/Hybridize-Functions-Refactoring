package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Arrays;

public enum Information {

	/**
	 * Information related to (tensor) type inferencing.
	 */
	TYPE_INFERENCING(1),

	/**
	 * Context being used.
	 */
	SPECULATIVE_ANALYSIS(2),

	/**
	 * Information related to {@link Function#inferInputSignature} — diagnostics emitted when input-signature inference is dropped because a
	 * parameter blocks it (e.g., not classified as tensor-typed, or classified by type hint / container detection without concrete
	 * shape/dtype evidence).
	 */
	INPUT_SIGNATURE_INFERENCE(3),

	/**
	 * The caller-coverage advisory (issue 767): every known call path is dominated by a hybridized caller, so the function's computation is
	 * already traced and hybridizing it may add no benefit. Advisory-only in phase 1; see
	 * https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/767.
	 */
	CALLER_COVERAGE(4);

	static {
		// check that the codes are unique.
		assert Arrays.stream(Information.values()).map(Information::getCode).distinct()
				.count() == Information.values().length : "Codes must be unique.";
	}

	private int code;

	private Information(int code) {
		this.code = code;
	}

	public int getCode() {
		return this.code;
	}

}
