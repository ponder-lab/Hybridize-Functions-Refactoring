package edu.cuny.hunter.hybridize.core.analysis;

import static java.util.Collections.unmodifiableList;
import static java.util.Collections.unmodifiableMap;
import static java.util.Collections.unmodifiableSet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Maps;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.collections.Pair;

/**
 * The tensor-type analysis indexed once for the lookups Hybridize makes after it, so that each function and parameter reads the entries it
 * needs instead of rescanning the whole analysis (#1027). Each index is built on first use and cached per analysis.
 */
final class TensorTypeIndex {

	/** The index of each analysis, built on first use. */
	private static final Map<TensorTypeAnalysis, TensorTypeIndex> CACHE = Maps.newConcurrentMap();

	/**
	 * Returns the index of the given analysis, building it on first use.
	 *
	 * @param analysis The {@link TensorTypeAnalysis} to index.
	 * @return The analysis's index.
	 */
	static TensorTypeIndex of(TensorTypeAnalysis analysis) {
		return CACHE.computeIfAbsent(analysis, TensorTypeIndex::new);
	}

	/**
	 * Clears the cached indices.
	 */
	static void clearCaches() {
		CACHE.clear();
	}

	/** The tensor types of each local, by node and value number, leaving out {@code None} members. */
	private final Map<CGNode, Map<Integer, Set<TensorType>>> localTypes;

	/** The analysis's entries for parameter locals of program methods, by the source file declaring the method. */
	private final Map<String, List<Pair<LocalPointerKey, TensorVariable>>> parameterEntries;

	private TensorTypeIndex(TensorTypeAnalysis analysis) {
		Map<CGNode, Map<Integer, Set<TensorType>>> types = new HashMap<>();
		Map<String, List<Pair<LocalPointerKey, TensorVariable>>> parameters = new HashMap<>();

		for (Pair<PointerKey, TensorVariable> pair : analysis)
			if (pair.fst instanceof LocalPointerKey local) {
				if (pair.snd != null)
					// A None member is no tensor here; it matters only to the signature (wala/ML#1040).
					pair.snd.getTypes().stream().filter(type -> !type.isNone())
							.forEach(type -> types.computeIfAbsent(local.getNode(), n -> new HashMap<>())
									.computeIfAbsent(local.getValueNumber(), v -> new HashSet<>()).add(type));

				if (local.isParameter() && local.getNode().getMethod() instanceof AstMethod method)
					parameters.computeIfAbsent(method.getDeclaringClass().getSourceFileName(), f -> new ArrayList<>())
							.add(Pair.make(local, pair.snd));
			}

		Map<CGNode, Map<Integer, Set<TensorType>>> frozenTypes = new HashMap<>();
		types.forEach((node, byValue) -> {
			Map<Integer, Set<TensorType>> frozen = new HashMap<>();
			byValue.forEach((value, set) -> frozen.put(value, unmodifiableSet(set)));
			frozenTypes.put(node, unmodifiableMap(frozen));
		});

		Map<String, List<Pair<LocalPointerKey, TensorVariable>>> frozenParameters = new HashMap<>();
		parameters.forEach((file, entries) -> frozenParameters.put(file, unmodifiableList(entries)));

		this.localTypes = unmodifiableMap(frozenTypes);
		this.parameterEntries = unmodifiableMap(frozenParameters);
	}

	/**
	 * The tensor types the analysis gives the local {@code valueNumber} in {@code node}, leaving out {@code None} members.
	 *
	 * @param node A call-graph node.
	 * @param valueNumber A value number in {@code node}.
	 * @return The local's tensor types; empty when the analysis gives it none.
	 */
	Set<TensorType> typesOf(CGNode node, int valueNumber) {
		return this.localTypes.getOrDefault(node, Map.of()).getOrDefault(valueNumber, Set.of());
	}

	/**
	 * The analysis's entries for parameter locals of methods declared in the given source file, in the analysis's order.
	 *
	 * @param sourceFileName The absolute path of a source file.
	 * @return The entries; empty when the file declares no method with a typed parameter.
	 */
	List<Pair<LocalPointerKey, TensorVariable>> parameterEntries(String sourceFileName) {
		return this.parameterEntries.getOrDefault(sourceFileName, List.of());
	}
}
