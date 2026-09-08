package edu.cuny.hunter.hybridize.tests;

import static com.ibm.wala.cast.python.ml.types.TensorOrigin.ANNOTATION;
import static com.ibm.wala.cast.python.ml.types.TensorOrigin.NUMPY;
import static com.ibm.wala.cast.python.ml.types.TensorOrigin.PARAMETER;
import static com.ibm.wala.cast.python.ml.types.TensorOrigin.TENSORFLOW;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import java.util.Set;

import org.junit.Test;

import com.ibm.wala.cast.python.ml.types.TensorOrigin;

import edu.cuny.hunter.hybridize.core.analysis.Util;

/**
 * Synthesized-input tests for {@link Util#producerOrigins(Set)}, the rule separating markers that say what produced a value from markers
 * that say where its type came from. The numpy-only test in {@code definesTensor} compares against {@code {NUMPY}}, and comparing a raw
 * origin set by equality is a claim that the origin vocabulary is closed over producers. That claim stopped holding when
 * {@link TensorOrigin#ANNOTATION} joined the vocabulary.
 */
@SuppressWarnings("static-method")
public class ProducerOriginsTest {

	/**
	 * The case that was wrong in the field: a numpy value whose type came from a user annotation. Before the producer filter, the raw set
	 * {@code {NUMPY, ANNOTATION}} was unequal to {@code {NUMPY}}, so the def counted as a TensorFlow computation on the strength of having
	 * been annotated. Observed live on a whole-project run, on a module-level {@code np.array(...)} binding named by a sidecar entry.
	 */
	@Test
	public void testAnnotatedNumpyValueStaysNumpyOnly() {
		assertEquals("An annotated numpy value is still produced by numpy alone.", Set.of(NUMPY),
				Util.producerOrigins(Set.of(NUMPY, ANNOTATION)));
	}

	/**
	 * The witness for the test above: a genuinely mixed producer set is not reduced. Without this, the filter would be equally consistent
	 * with discarding every marker beside the first, and the assertion above would pin nothing about {@code ANNOTATION} specifically.
	 */
	@Test
	public void testMixedProducersSurvive() {
		assertEquals("A mixed op dispatches to TensorFlow, so both producers are kept.", Set.of(NUMPY, TENSORFLOW),
				Util.producerOrigins(Set.of(NUMPY, TENSORFLOW)));
	}

	/**
	 * A parameter origin is a producer for this question and must survive: a tensor parameter is a symbolic tensor under tracing regardless
	 * of what its eager feeds were produced by (wala/ML#726), so a numpy-fed parameter is not numpy-only.
	 */
	@Test
	public void testParameterOriginSurvives() {
		assertEquals("A parameter origin is not a type-provenance marker.", Set.of(NUMPY, PARAMETER),
				Util.producerOrigins(Set.of(NUMPY, PARAMETER)));
	}

	/**
	 * A set carrying nothing to remove is returned as-is rather than copied, which is the overwhelmingly common case and the reason the
	 * filter is cheap enough to sit in a per-def loop.
	 */
	@Test
	public void testUnaffectedSetIsNotCopied() {
		Set<TensorOrigin> origins = Set.of(NUMPY);
		assertSame("Nothing to remove, so the argument itself comes back.", origins, Util.producerOrigins(origins));
	}
}
