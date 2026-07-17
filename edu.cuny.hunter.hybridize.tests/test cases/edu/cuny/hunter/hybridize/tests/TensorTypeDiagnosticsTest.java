package edu.cuny.hunter.hybridize.tests;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.UNKNOWN;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Set;

import org.junit.Test;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;

import edu.cuny.hunter.hybridize.core.analysis.Parameter;
import edu.cuny.hunter.hybridize.core.analysis.Parameter.TensorTypeDimensionRow;

/**
 * Synthesized-input tests for {@link Parameter#computeTensorTypeDiagnostics(Set, List)} (#780), the per-dimension tensor-lattice
 * diagnostic. Inputs are hand-built {@link TensorType} sets so the classification is exercised in isolation from upstream precision,
 * mirroring {@link InferSpecTest}.
 *
 * @see <a href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/780">Issue 780</a>
 */
@SuppressWarnings("static-method")
public class TensorTypeDiagnosticsTest {

	/**
	 * Each raw {@code Dimension} subtype renders to its own class, distinct from the wildcard {@code inferSpec} would collapse them to: a
	 * numeric dimension is a constant, a symbolic one carries its name, and unresolved/dynamic/ragged are separable.
	 */
	@Test
	public void testPerDimensionClasses() {
		TensorType type = new TensorType(FLOAT32,
				List.of(new NumericDim(3), new SymbolicDim("batch"), UnresolvedDim.INSTANCE, DynamicDim.INSTANCE, RaggedDim.INSTANCE));
		List<TensorTypeDimensionRow> rows = Parameter.computeTensorTypeDiagnostics(Set.of(type), null);

		assertEquals("One row per dimension.", 5, rows.size());
		assertEquals("Constant,3", rows.get(0).dimClass());
		assertEquals("Symbolic,batch", rows.get(1).dimClass());
		assertEquals("Unresolved", rows.get(2).dimClass());
		assertEquals("Dynamic", rows.get(3).dimClass());
		assertEquals("Ragged", rows.get(4).dimClass());

		for (int i = 0; i < rows.size(); i++) {
			assertEquals("Rank is the dimension count.", "5", rows.get(i).rank());
			assertEquals(Integer.valueOf(i), rows.get(i).dimIndex());
			assertEquals("FLOAT32", rows.get(i).dtype());
			assertFalse("A concrete dtype is not top.", rows.get(i).dtypeTop());
			assertNull("A direct type carries no container position.", rows.get(i).containerPosition());
		}
	}

	/** A shape-top type (unknown rank, {@code getDims() == null}) contributes a single summary row with no dimension index. */
	@Test
	public void testShapeTop() {
		List<TensorTypeDimensionRow> rows = Parameter.computeTensorTypeDiagnostics(Set.of(new TensorType(FLOAT32, null)), null);

		assertEquals(1, rows.size());
		assertEquals("TOP", rows.get(0).rank());
		assertEquals("TOP", rows.get(0).dimClass());
		assertNull("A shape-top type has no dimension index.", rows.get(0).dimIndex());
	}

	/** A scalar (rank zero, empty dimension list) contributes a single summary row, distinct from shape-top. */
	@Test
	public void testScalar() {
		List<TensorTypeDimensionRow> rows = Parameter.computeTensorTypeDiagnostics(Set.of(new TensorType(FLOAT32, List.of())), null);

		assertEquals(1, rows.size());
		assertEquals("0", rows.get(0).rank());
		assertEquals("scalar", rows.get(0).dimClass());
		assertNull("A scalar has no dimension index.", rows.get(0).dimIndex());
	}

	/** A dtype-top ({@code UNKNOWN}) type flags {@code dtypeTop}, orthogonal to the shape axis. */
	@Test
	public void testDtypeTop() {
		List<TensorTypeDimensionRow> rows = Parameter
				.computeTensorTypeDiagnostics(Set.of(new TensorType(UNKNOWN, List.of(new NumericDim(3)))), null);

		assertEquals(1, rows.size());
		assertTrue("An UNKNOWN dtype is top.", rows.get(0).dtypeTop());
		assertEquals("UNKNOWN", rows.get(0).dtype());
		assertEquals("The shape axis is unaffected by a top dtype.", "Constant,3", rows.get(0).dimClass());
	}

	/** Container element types are emitted per position, each carrying its element position rather than a null one. */
	@Test
	public void testContainerPositions() {
		Set<TensorType> position0 = Set.of(new TensorType(FLOAT32, List.of(new NumericDim(2))));
		Set<TensorType> position1 = Set.of(new TensorType(FLOAT32, List.of(new NumericDim(4))));
		List<TensorTypeDimensionRow> rows = Parameter.computeTensorTypeDiagnostics(null, List.of(position0, position1));

		assertEquals(2, rows.size());
		assertEquals(Integer.valueOf(0), rows.get(0).containerPosition());
		assertEquals("Constant,2", rows.get(0).dimClass());
		assertEquals(Integer.valueOf(1), rows.get(1).containerPosition());
		assertEquals("Constant,4", rows.get(1).dimClass());
	}

	/** A parameter with no inferred tensor type produces no rows. */
	@Test
	public void testNoTypesNoRows() {
		assertTrue("Null direct types yield no rows.", Parameter.computeTensorTypeDiagnostics(null, null).isEmpty());
		assertTrue("An empty type set yields no rows.", Parameter.computeTensorTypeDiagnostics(Set.of(), null).isEmpty());
	}
}
