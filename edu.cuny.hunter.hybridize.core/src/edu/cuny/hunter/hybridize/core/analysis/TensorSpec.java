package edu.cuny.hunter.hybridize.core.analysis;

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

}
