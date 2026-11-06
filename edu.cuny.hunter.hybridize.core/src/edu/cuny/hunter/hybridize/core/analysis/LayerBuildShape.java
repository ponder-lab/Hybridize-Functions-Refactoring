package edu.cuny.hunter.hybridize.core.analysis;

/**
 * The shapes of layer building that change a method's behavior under {@code tf.function}, each a reason for
 * {@link PreconditionFailure#MAY_BUILD_LAYERS} (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1224). A method that
 * only calls sublayers constructed in {@code __init__} or {@code build}, on every call, has none of them: Keras builds each sublayer under
 * {@code init_scope} during the first trace, and a retrace builds nothing. See {@link LayerBuildAnalysis}.
 */
public enum LayerBuildShape {

	/**
	 * The traced body constructs a layer, such as {@code tf.keras.layers.Dense(2)(x)} in a {@code call}. Each trace creates new variables,
	 * and the second trace raises "tf.function only supports singleton tf.Variables created on the first call".
	 */
	CONSTRUCTED_IN_BODY("constructs a layer in its traced body, so each trace would create new variables"),

	/**
	 * A call that may build a layer runs under a branch or loop whose condition can differ between traces, such as
	 * {@code if flag: self.b(x)} for a parameter {@code flag}. The first trace doesn't build the layer, and a later trace does, which
	 * raises as {@link #CONSTRUCTED_IN_BODY} does.
	 */
	FIRST_CALL_ON_RETRACE(
			"may first call a layer under a condition that can differ between traces, so a retrace would create its variables"),

	/**
	 * Two sublayers reachable from the method's receiver may carry the same constant name, or one carries a name computed at run time.
	 * Decorating the method names the variables built in its trace without the enclosing layers' scopes, so the names may collide, and a
	 * save keyed by name, such as to HDF5, fails or mixes weights.
	 */
	COLLIDING_NAMES("reaches sublayers whose names may collide once the enclosing layers' name scopes are dropped");

	/** The reason this shape refuses, for a status message. */
	private final String reason;

	/**
	 * Creates the shape.
	 *
	 * @param reason The reason the shape refuses, for a status message.
	 */
	LayerBuildShape(String reason) {
		this.reason = reason;
	}

	/**
	 * The reason this shape refuses, for a status message.
	 *
	 * @return The reason, completing "Can't hybridize a function that".
	 */
	public String getReason() {
		return this.reason;
	}
}
