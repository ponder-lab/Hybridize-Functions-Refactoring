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

	/**
	 * True if the {@link TensorSpec} is using keyword arguments for the shape.
	 */
	private boolean shapeKeyword;

	/**
	 * True if the {@link TensorSpec} is using keyword arguments for the type.
	 */
	private boolean dtypeKeyword;

	public TensorSpec() {
		this.shape = "";
		this.dtype = "";
		this.shapeKeyword = true;
		this.dtypeKeyword = true;

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

	/**
	 * Set if the type of the argument for shape of {@link TensorSpec} is a keyword.
	 */
	public void setShapeKeyword(boolean s) {
		this.shapeKeyword = s;
	}

	/**
	 * Set if the type of the argument for dtype of {@link TensorSpec} is a keyword.
	 */
	public void setDTypeKeyword(boolean d) {
		this.dtypeKeyword = d;
	}

	@Override
	public String toString() {
		if (this.dtype.isEmpty() && this.shape.isEmpty())
			return "tf.TensorSpec([])";
		if (!this.shapeKeyword && !this.dtypeKeyword)
			return "tf.TensorSpec([" + this.shape + "]" + ", " + this.dtype + ")";
		if (!this.shapeKeyword && this.dtypeKeyword)
			return "tf.TensorSpec([" + this.shape + "]" + ", dtype=" + this.dtype + ")";
		return "tf.TensorSpec(shape=[" + this.shape + "]" + ", dtype=" + this.dtype + ")";
	}
}
