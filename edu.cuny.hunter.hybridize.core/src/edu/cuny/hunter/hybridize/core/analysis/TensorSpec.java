package edu.cuny.hunter.hybridize.core.analysis;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A representation of a tf.Tensorspec which describes a tf.Tensor
 */
public class TensorSpec {

	public enum Dtype {
		float16, float32, float64, int32, int64, uint8, uint16, uint32, uint64, int16, int8, complex64, complex128, string, bool, qint8,
		quint8, qint16, quint16, qint32, bfloat16, half, resource, variant
	}

	/**
	 * Shape of the tensor being described by {@link TensorSpec}.
	 */
	private List<Integer> shape;

	/**
	 * Type of the tensor being described by {@link TensorSpec}.
	 */
	private Dtype dtype;

	public TensorSpec() {
		// Initialize to empty list
		this.shape = new ArrayList<>();
		this.dtype = Dtype.float32;
	}

	public TensorSpec(List<Integer> s, Dtype d) {
		this.shape = s;
		this.dtype = d;
	}

	/**
	 * Shape of {@link TensorSpec}.
	 *
	 * @return List of dimensions (null if the shape is unspecified) of this {@link TensorSpec} shape.
	 */
	public List<Integer> getShape() {
		return this.shape;
	}

	/**
	 * Dtype of {@link TensorSpec}.
	 *
	 * @return Dtype of this {@link TensorSpec} dtype.
	 */
	public Dtype getDType() {
		return this.dtype;
	}

	/**
	 * Set shape of {@link TensorSpec}.
	 */
	public void setShape(List<Integer> s) {
		this.shape = s;
	}

	/**
	 * Set dtype of {@link TensorSpec}.
	 */
	public void setDType(Dtype d) {
		this.dtype = d;
	}

	@Override
	public int hashCode() {
		return Objects.hash(shape, dtype);
	}

	@Override
	public boolean equals(Object tensorObject) {

		if (tensorObject == this) {
			return true;
		}

		if (!(tensorObject instanceof TensorSpec)) {
			return false;
		}

		TensorSpec tensor = (TensorSpec) tensorObject;

		return shape.equals(tensor.shape) && dtype.equals(tensor.dtype);
	}
}
