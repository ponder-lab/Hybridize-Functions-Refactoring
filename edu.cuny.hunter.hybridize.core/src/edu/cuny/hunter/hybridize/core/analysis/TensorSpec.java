package edu.cuny.hunter.hybridize.core.analysis;

import java.util.Objects;

/**
 * A representation of a tf.Tensorspec which describes a tf.Tensor
 */
public class TensorSpec {

	/**
	 * Shape of the tensor being described by {@link TensorSpec}.
	 */
	private String shape;

	/**
	 * Type of the tensor being described by {@link TensorSpec}.
	 */
	private String dtype;

	public TensorSpec() {
		this.shape = "";
		this.dtype = "";
	}

	public TensorSpec(String s, String d) {
		this.shape = s;
		this.dtype = d;
	}

	/**
	 * Shape of {@link TensorSpec}.
	 *
	 * @return String of this {@link TensorSpec} shape.
	 */
	public String getShape() {
		return this.shape;
	}

	/**
	 * Dtype of {@link TensorSpec}.
	 *
	 * @return String of this {@link TensorSpec} dtype.
	 */
	public String getDType() {
		return this.dtype;
	}

	/**
	 * Set shape of {@link TensorSpec}.
	 */
	public void setShape(String s) {
		this.shape = s;
	}

	/**
	 * Set dtype of {@link TensorSpec}.
	 */
	public void setDType(String d) {
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