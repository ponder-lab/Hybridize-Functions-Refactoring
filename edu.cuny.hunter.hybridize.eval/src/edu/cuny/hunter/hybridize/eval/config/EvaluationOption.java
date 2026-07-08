package edu.cuny.hunter.hybridize.eval.config;

import java.util.Arrays;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * The evaluator's command-line and system-property configuration options, single-sourced so the argument parser
 * ({@code EvaluateHybridizeFunctionRefactoringApplication}) and the readers ({@code EvaluateHybridizeFunctionRefactoringHandler}) cannot
 * drift: each constant yields both its camelCase name (the canonical form of its {@code --kebab-case} flag) and its full system-property
 * key, both derived from the constant itself. Adding an option is one entry here.
 * <p>
 * The per-project {@code targetedCfaDepth} is deliberately not an option: it is an integer read from {@code eval.properties} (with a global
 * system-property fallback), not a {@code --kebab-case} boolean flag. Its key still shares {@link #PREFIX}.
 */
public enum EvaluationOption {

	PERFORM_ANALYSIS, PERFORM_CHANGE, ALWAYS_CHECK_PYTHON_SIDE_EFFECTS, ALWAYS_CHECK_RECURSION, ALWAYS_CHECK_TENSOR_COMPUTATION,
	ALWAYS_CHECK_EAGER_ONLY_CALLS, ALWAYS_CHECK_NUMPY_CALLS, PROCESS_FUNCTIONS_IN_PARALLEL, USE_TEST_ENTRYPOINTS, ALWAYS_FOLLOW_TYPE_HINTS,
	USE_SPECULATIVE_ANALYSIS, INFER_INPUT_SIGNATURES, OUTPUT_CALLS, PROJECTS;

	/** Common prefix of the evaluator's configuration system properties. */
	public static final String PREFIX = "edu.cuny.hunter.hybridize.eval.";

	/**
	 * Returns this option's camelCase configuration name, the canonical form of its {@code --kebab-case} flag and the suffix of its
	 * system-property key, derived from the constant name (e.g. {@code PERFORM_ANALYSIS} yields {@code performAnalysis}).
	 *
	 * @return This option's camelCase configuration name.
	 */
	public String propertyName() {
		String[] words = name().toLowerCase(Locale.ROOT).split("_");
		StringBuilder camelCase = new StringBuilder(words[0]);

		for (int word = 1; word < words.length; word++)
			camelCase.append(Character.toUpperCase(words[word].charAt(0))).append(words[word].substring(1));

		return camelCase.toString();
	}

	/**
	 * Returns this option's full system-property key: {@link #PREFIX} followed by {@link #propertyName()}.
	 *
	 * @return This option's full system-property key.
	 */
	public String key() {
		return PREFIX + propertyName();
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
