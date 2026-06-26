package edu.cuny.hunter.hybridize.eval.config;

import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * The evaluator's command-line and system-property configuration options, single-sourced so the argument parser
 * ({@code EvaluateHybridizeFunctionRefactoringApplication}) and the readers ({@code EvaluateHybridizeFunctionRefactoringHandler}) cannot
 * drift: each option contributes both its camelCase name (the canonical form of its {@code --kebab-case} flag) and its full system-property
 * key. Adding an option is one entry here.
 * <p>
 * The per-project {@code targetedCfaDepth} is deliberately not an option: it is an integer read from {@code eval.properties} (with a global
 * system-property fallback), not a {@code --kebab-case} boolean flag. Its key still shares {@link #PREFIX}.
 */
public enum EvaluationOption {

	PERFORM_ANALYSIS("performAnalysis"), PERFORM_CHANGE("performChange"), ALWAYS_CHECK_PYTHON_SIDE_EFFECTS("alwaysCheckPythonSideEffects"),
	ALWAYS_CHECK_RECURSION("alwaysCheckRecursion"), PROCESS_FUNCTIONS_IN_PARALLEL("processFunctionsInParallel"),
	USE_TEST_ENTRYPOINTS("useTestEntrypoints"), ALWAYS_FOLLOW_TYPE_HINTS("alwaysFollowTypeHints"),
	USE_SPECULATIVE_ANALYSIS("useSpeculativeAnalysis"), INFER_INPUT_SIGNATURES("inferInputSignatures"), OUTPUT_CALLS("outputCalls"),
	PROJECTS("projects");

	/** Common prefix of the evaluator's configuration system properties. */
	public static final String PREFIX = "edu.cuny.hunter.hybridize.eval.";

	private final String propertyName;

	EvaluationOption(String propertyName) {
		this.propertyName = propertyName;
	}

	/**
	 * Returns this option's camelCase configuration name, the canonical form of its {@code --kebab-case} flag and the suffix of its
	 * system-property key.
	 *
	 * @return This option's camelCase configuration name.
	 */
	public String propertyName() {
		return this.propertyName;
	}

	/**
	 * Returns this option's full system-property key: {@link #PREFIX} followed by {@link #propertyName()}.
	 *
	 * @return This option's full system-property key.
	 */
	public String key() {
		return PREFIX + this.propertyName;
	}

	/**
	 * Returns the camelCase names of all options, for validating the configuration names given on the command line.
	 *
	 * @return The camelCase names of all options.
	 */
	public static Set<String> propertyNames() {
		return Arrays.stream(values()).map(EvaluationOption::propertyName).collect(Collectors.toUnmodifiableSet());
	}
}
