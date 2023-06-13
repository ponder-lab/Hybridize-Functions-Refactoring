package edu.cuny.hunter.hybridize.tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import org.python.pydev.parser.jython.ast.argumentsType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.visitors.NodeUtils;

import edu.cuny.hunter.hybridize.core.analysis.Function;

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

	public FunctionUnderTest(String name, String... parameters) {
		this.name = name;
		this.addParameters(parameters);
	}

	public boolean addParameters(String... parameters) {
		return this.parameters.addAll(Arrays.asList(parameters));
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

	/**
	 * Tests that the given {@link Function} matches the one we expect to test, i.e., this {@link FunctionUnderTest}.
	 *
	 * @param function The actual {@link Function}.
	 */
	public void compareTo(Function function) {
		assertNotNull(function);
		assertEquals(this.isHybrid(), function.isHybrid());

		argumentsType params = function.getParameters();

		exprType[] actualParams = params.args;
		List<String> expectedParameters = this.getParameters();
		assertEquals(expectedParameters.size(), actualParams.length);

		for (int i = 0; i < actualParams.length; i++) {
			exprType actualParameter = actualParams[i];
			assertNotNull(actualParameter);

			String paramName = NodeUtils.getRepresentationString(actualParameter);
			assertEquals(expectedParameters.get(i), paramName);
		}
	}
}
