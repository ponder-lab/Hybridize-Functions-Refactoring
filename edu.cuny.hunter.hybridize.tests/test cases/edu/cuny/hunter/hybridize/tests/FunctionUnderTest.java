package edu.cuny.hunter.hybridize.tests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A specification of a Python function being tested.
 *
 * @author <a href="mailto:rk1424@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class FunctionUnderTest {

	/**
	 * The simple name of this function under test.
	 */
	private String name;

	/**
	 * The names of the parameteres of this function under test.
	 */
	private List<String> parameters = new ArrayList<>();

	/**
	 * Whether this function under test should be a hybrid function.
	 */
	private boolean hybrid;

	public FunctionUnderTest(String name) {
		this.name = name;
	}

	public FunctionUnderTest(String name, boolean hybrid) {
		this(name);
		this.hybrid = hybrid;
	}

	public boolean addParameters(String... parameter) {
		return this.parameters.addAll(Arrays.asList(parameter));
	}

	public String getName() {
		return name;
	}

	public List<String> getParameters() {
		return Collections.unmodifiableList(parameters);
	}

	public boolean isHybrid() {
		return hybrid;
	}

	@Override
	public int hashCode() {
		return Objects.hash(name, parameters, hybrid);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		FunctionUnderTest other = (FunctionUnderTest) obj;
		return Objects.equals(name, other.name) && Objects.equals(parameters, other.parameters) && Objects.equals(hybrid, other.hybrid);
	}
}
