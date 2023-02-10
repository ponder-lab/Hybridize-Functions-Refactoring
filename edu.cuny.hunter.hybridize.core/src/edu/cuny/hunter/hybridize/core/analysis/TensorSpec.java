package edu.cuny.hunter.hybridize.core.analysis;

public class TensorSpec {
	private String shape;
	private String dtype;
	private boolean shapeKeyword;
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

	public String getShape() {
		return this.shape;
	}

	public String getDType() {
		return this.dtype;
	}

	public void setShape(String s) {
		this.shape = s;
	}

	public void setDType(String d) {
		this.dtype = d;
	}

	public void setShapeKeyword(boolean s) {
		this.shapeKeyword = s;
	}

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
