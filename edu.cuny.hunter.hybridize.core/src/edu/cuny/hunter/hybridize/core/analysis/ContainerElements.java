package edu.cuny.hunter.hybridize.core.analysis;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.dict;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.set;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.parseIntOrNull;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeSet;
import java.util.WeakHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstGlobalWrite;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder.ElementMirrorKey;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.TensorGenerator;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.IntSet;
import com.ibm.wala.util.intset.IntSetUtil;
import com.ibm.wala.util.intset.MutableIntSet;
import com.ibm.wala.util.intset.OrdinalSet;

/**
 * The elements of a container instance, as the analysis models them: a container's length, the key of each element, whether an element may
 * be {@code None}, and the tensor types the analysis gives them, read off the container's object catalog and the element channels
 * ({@link PythonSSAPropagationCallGraphBuilder#LIST_APPEND_CONTENTS_FIELD},
 * {@link PythonSSAPropagationCallGraphBuilder#LIST_OPERATION_CONTENTS_FIELD}). This is HFR's copy of Ariadne's class of the same name,
 * which Ariadne keeps for its own reads of a container (https://github.com/wala/ML/issues/1190); it reads only Ariadne's and WALA's public
 * API. The element tensor types are read from the solved {@link TensorTypeAnalysis}'s typed keys, through its
 * {@link TensorTypeAnalysis#iterator()}.
 * <p>
 * A container here means a container in the analysis's element model: a list, tuple, set, dict or {@code enumerate} object, the same set
 * {@code PythonTypes.isContainer} names. It does not mean {@code collections.abc.Container}: an {@code enumerate} object, for one, is a
 * one-pass iterator with no length, exhausted after one pass. Any other instance has no element keys here beyond those its catalog records.
 * <p>
 * The length is reported only when it is known: a {@link com.ibm.wala.cast.python.types.PythonTypes#list} or
 * {@link com.ibm.wala.cast.python.types.PythonTypes#tuple} display that program code allocates, whose elements all sit at constant indices
 * forming the run {@code 0..n-1}, with no element whose position is unknown (one appended, built by a comprehension, unpacked from a
 * starred element, or produced by list concatenation or repetition), and, for a list, that nothing in the program may mutate unrecorded
 * (see {@link #mutations}). A list or tuple a library summary or an operator model allocates reports none: its catalog holds the model's
 * placeholder slots, not its run-time elements. Every other container reports no length.
 */
final class ContainerElements {

	private static final Logger LOGGER = Logger.getLogger(ContainerElements.class.getName());

	/**
	 * The binary operators whose augmented assignments mutate a list or a set in place, {@code l += m}, {@code l *= n} and {@code s |= t},
	 * by name, since the engine bundle doesn't export WALA's operator type.
	 */
	private static final Set<String> IN_PLACE_OPERATOR_NAMES = Set.of("add", "mul", "or");

	/**
	 * Per solved analysis, the types of each key it gives a type. Weak, so an analysis is released with its project; synchronized, since
	 * functions may be processed in parallel.
	 */
	private static final Map<TensorTypeAnalysis, Map<PointerKey, Set<TensorType>>> TYPED = Collections.synchronizedMap(new WeakHashMap<>());

	/**
	 * The list methods that may change a list's length or the order of its elements, whose effect the analysis does not record among the
	 * list's positioned elements. {@code append} is among them: the analysis records an appended element under its own channel, but not
	 * that the length grew.
	 */
	private static final Set<String> REPOSITIONING_LIST_METHODS = Set.of("append", "extend", "insert", "pop", "remove", "clear", "sort",
			"reverse", "__setitem__", "__delitem__", "__iadd__", "__imul__");

	/** The type name of the builtin a {@code del} statement is lowered to a call of. */
	private static final String DELETE_BUILTIN = "Lwala/builtin/__delete__";

	/** The type name of the builtin a slice, {@code l[i:j]}, is lowered to a call of. */
	private static final String SLICE_BUILTIN = "Lwala/builtin/slice";

	/** The name a {@code del} statement's lowering reads the deletion builtin by. */
	private static final String DELETE_NAME = "__delete__";

	/**
	 * The type-name prefixes of the libraries whose functions do not resize a Python list passed to them, so an unresolved call of one
	 * marks nothing; see {@link #mutations}.
	 */
	private static final List<String> NON_RESIZING_LIBRARIES = List.of("Ltensorflow", "Lnumpy");

	/**
	 * The library functions named like a method that may add an element ({@link #ADDING_METHODS}) that resize no container, by the library
	 * module's type name and the attributes read off it: {@code tf.keras.backend.update} assigns a variable (wala/ML#1182).
	 */
	private static final Set<String> NON_RESIZING_LIBRARY_FUNCTIONS = Set.of("Ltensorflow/keras/backend/update");

	/** The builtins whose call builds a fresh container (wala/ML#1182). */
	private static final Set<String> CONTAINER_BUILTINS = Set.of("list", "dict", "set");

	/**
	 * The methods that may add an element to a list, dict or set without the analysis recording it among the container's element keys, a
	 * set's {@code add} among them. A list's {@code append} is not: the analysis records an appended element under its own channel.
	 */
	private static final Set<String> ADDING_METHODS = Set.of("extend", "insert", "__setitem__", "__iadd__", "__imul__", "add", "update",
			"setdefault", "__ior__");

	/**
	 * What the program may do to containers that the analysis does not record.
	 *
	 * @param lists The instance keys, by their index in the pointer analysis's instance key mapping, of the lists whose positioned elements
	 *        it may so change.
	 * @param allLists Whether it may so change any list's at all.
	 * @param open The instance keys, by the same index, of the lists, dicts and sets it may add an element to under a key the analysis does
	 *        not record.
	 * @param allOpen Whether it may so add an element to any list, dict or set at all.
	 */
	private record Mutations(IntSet lists, boolean allLists, IntSet open, boolean allOpen) {
	}

	/** The mutations {@link #mutations} is collecting. */
	private static final class Marks {

		/** See {@link Mutations#lists}. */
		final MutableIntSet lists = IntSetUtil.make();

		/** See {@link Mutations#open}. */
		final MutableIntSet open = IntSetUtil.make();

		/** See {@link Mutations#allLists}. */
		boolean allLists;

		/** See {@link Mutations#allOpen}. */
		boolean allOpen;

		/**
		 * Marks the containers a local may point to.
		 *
		 * @param pa The pointer analysis.
		 * @param node The local's node.
		 * @param vn The local.
		 * @param adds Whether the mutation may also add an element the analysis does not record.
		 */
		void mark(PointerAnalysis<InstanceKey> pa, CGNode node, int vn, boolean adds) {
			for (InstanceKey ik : pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, vn))) {
				TypeReference type = ik.concreteType().getReference();
				int index = pa.getInstanceKeyMapping().getMappedIndex(ik);
				if (type.equals(list))
					this.lists.add(index);
				if (adds && isMutable(type))
					this.open.add(index);
			}
		}

		/**
		 * Marks what a value names for a store under a member the analysis cannot name, when the same body also stores that value in the
		 * object's append contents: a list or set then holds no element the analysis does not see, so only a list's positions are lost,
		 * while a dict's keys still are (wala/ML#1178).
		 *
		 * @param pa The pointer analysis.
		 * @param node The node of the store.
		 * @param vn The object's value number.
		 */
		void markMirrored(PointerAnalysis<InstanceKey> pa, CGNode node, int vn) {
			for (InstanceKey ik : pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, vn))) {
				TypeReference type = ik.concreteType().getReference();
				int index = pa.getInstanceKeyMapping().getMappedIndex(ik);
				if (type.equals(list))
					this.lists.add(index);
				if (type.equals(dict))
					this.open.add(index);
			}
		}

		/**
		 * Marks every list and, when {@code adds}, every list, dict and set.
		 *
		 * @param adds Whether the mutation may also add an element the analysis does not record.
		 */
		void markAll(boolean adds) {
			this.allLists = true;
			if (adds)
				this.allOpen = true;
		}
	}

	/**
	 * Whether a type is a container the program can add an element to: a list, dict or set.
	 *
	 * @param type The type.
	 * @return {@code true} iff it is such a container.
	 */
	private static boolean isMutable(TypeReference type) {
		return type.equals(list) || type.equals(dict) || type.equals(set);
	}

	/**
	 * Per builder, its {@link Mutations}. They hold indices, not keys, so the cache pins nothing the builder does not.
	 */
	private static final Map<PropagationCallGraphBuilder, Mutations> MUTATIONS = new WeakHashMap<>();

	/** The length, or -1 when it is not known. */
	private final int arity;

	/** The element key at each position, in order; empty when the length is not known. */
	private final List<PointerKey> positions;

	/** Every key that can hold an element. */
	private final Set<PointerKey> allElements;

	/** Whether {@link #allElements} holds every key that can hold an element. */
	private final boolean complete;

	/** The builder whose pointer analysis holds the keys. */
	private final PropagationCallGraphBuilder builder;

	private ContainerElements(int arity, List<PointerKey> positions, Set<PointerKey> allElements, boolean complete,
			PropagationCallGraphBuilder builder) {
		this.arity = arity;
		this.positions = Collections.unmodifiableList(positions);
		this.allElements = Collections.unmodifiableSet(allElements);
		this.complete = complete;
		this.builder = builder;
	}

	/**
	 * Reads the elements of a container instance.
	 *
	 * @param builder The {@link PropagationCallGraphBuilder} whose pointer analysis models the container.
	 * @param container The container instance.
	 * @return Its elements.
	 */
	static ContainerElements of(PropagationCallGraphBuilder builder, InstanceKey container) {
		OrdinalSet<InstanceKey> catalog = builder.getPointerAnalysis()
				.getPointsToSet(((AstPointerKeyFactory) builder.getPointerKeyFactory()).getPointerKeyForObjectCatalog(container));
		TypeReference type = container.concreteType().getReference();
		boolean keyed = type.equals(dict);

		TreeSet<Integer> indices = new TreeSet<>();
		Set<String> fieldNames = new LinkedHashSet<>();
		boolean unpositioned = false;
		boolean complete = true;

		if (catalog != null)
			for (InstanceKey name : catalog) {
				if (!(name instanceof ConstantKey<?> constant)) {
					// A key the analysis could not name: an element may sit anywhere, under any key.
					unpositioned = true;
					complete = false;
					continue;
				}
				Object value = constant.getValue();
				if (PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD.equals(value)
						|| PythonSSAPropagationCallGraphBuilder.LIST_OPERATION_CONTENTS_FIELD.equals(value)) {
					unpositioned = true;
					fieldNames.add((String) value);
					continue;
				}
				Integer index = index(value);
				if (index != null) {
					indices.add(index);
					fieldNames.add(index.toString());
				} else if (keyed)
					// Any constant keys a dict: a float, a bool, None or a string, the field named by its
					// text. An attribute field of the dict counts too, which only adds a key.
					fieldNames.add(String.valueOf(value));
				// Otherwise an attribute name the catalog also records, such as a method read off the
				// instance, which is no element (wala/ML#603).
			}

		// Only a display program code allocates has its run-time elements in its catalog.
		if (!isProgramDisplay(container))
			unpositioned = true;

		// A container the program may mutate in a way the analysis does not record in its elements.
		if (isMutable(type)) {
			Mutations mutations = mutations(builder);
			int index = mappedIndex(builder, container);
			if (type.equals(list) && (mutations.allLists() || mutations.lists().contains(index)))
				unpositioned = true;
			if (mutations.allOpen() || mutations.open().contains(index))
				complete = false;
		}

		// The channels that hold elements whose position is unknown are element keys of every
		// positional or set-like container, whether or not its catalog names them: a list built by
		// concatenation or repetition keeps its elements there without recording the name.
		if (!keyed) {
			fieldNames.add(PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD);
			fieldNames.add(PythonSSAPropagationCallGraphBuilder.LIST_OPERATION_CONTENTS_FIELD);
		}

		Set<PointerKey> allElements = new LinkedHashSet<>();
		for (String fieldName : fieldNames) {
			PointerKey key = fieldKey(builder, container, fieldName);
			if (key != null)
				allElements.add(key);
		}

		boolean sequence = type.equals(list) || type.equals(tuple);
		boolean contiguous = indices.isEmpty() || (indices.first() == 0 && indices.last() == indices.size() - 1);
		if (!sequence || unpositioned || !contiguous)
			return new ContainerElements(-1, List.of(), allElements, complete, builder);

		List<PointerKey> positions = new ArrayList<>(indices.size());
		for (int i = 0; i < indices.size(); i++) {
			PointerKey key = fieldKey(builder, container, Integer.toString(i));
			// An index the class hierarchy cannot resolve to a field leaves its element unreadable, and
			// with it the length's correspondence to the keys.
			if (key == null)
				return new ContainerElements(-1, List.of(), allElements, complete, builder);
			positions.add(key);
		}
		return new ContainerElements(indices.size(), positions, allElements, complete, builder);
	}

	/**
	 * The container's length, when the analysis knows it.
	 *
	 * @return The number of elements, or empty when the length is not known.
	 */
	OptionalInt arity() {
		return this.arity < 0 ? OptionalInt.empty() : OptionalInt.of(this.arity);
	}

	/**
	 * The key of the element at each position.
	 *
	 * @return The keys of positions {@code 0..n-1}, where {@code n} is {@link #arity()}; empty when the length is not known.
	 */
	List<PointerKey> positions() {
		return this.positions;
	}

	/**
	 * Whether {@link #allElements()} holds every key that can hold an element of the container. It does not when the analysis saw an
	 * element stored under a key it could not name, such as {@code
	 * d[k] = t} with {@code k} unresolved, or the program may add one by a means it does not record, such as {@code l.extend(m)} or an
	 * unresolved call the container is passed to. The element keys, and the types read off them, are then a subset of the run-time ones.
	 *
	 * @return {@code true} iff the element keys are all of them.
	 */
	boolean isComplete() {
		return this.complete;
	}

	/**
	 * Every key that can hold an element of the container, including those of elements whose position is not known. Exact only when
	 * {@link #isComplete()}; otherwise a subset.
	 *
	 * @return The element keys.
	 */
	Set<PointerKey> allElements() {
		return this.allElements;
	}

	/**
	 * Whether an element of the container may be {@code None}, over every element key ({@link #allElements()}), read with
	 * {@link TensorGenerator#anyNullConstant(OrdinalSet)}. When the keys are not all of them ({@link #isComplete()}), an element may be
	 * anything, {@code None} too.
	 *
	 * @return {@code true} iff some element key may hold the {@code None} constant, or the keys are not all of them.
	 */
	boolean mayHoldNone() {
		if (!this.complete)
			return true;
		for (PointerKey key : this.allElements)
			if (TensorGenerator.anyNullConstant(this.builder.getPointerAnalysis().getPointsToSet(key)))
				return true;
		return false;
	}

	/**
	 * The tensor types the analysis gives the element at each position.
	 *
	 * @param analysis The solved {@link TensorTypeAnalysis}.
	 * @return Per position of {@link #positions()}, the element's tensor types, an empty set when the analysis gives it none; empty when
	 *         the length is not known. Exact only when {@link #isComplete()}.
	 */
	List<Set<TensorType>> positionTypes(TensorTypeAnalysis analysis) {
		List<Set<TensorType>> ret = new ArrayList<>(this.positions.size());
		for (PointerKey key : this.positions)
			ret.add(typesOf(analysis, key));
		return Collections.unmodifiableList(ret);
	}

	/**
	 * The tensor types the analysis gives the container's elements, whatever their positions, over every element key
	 * ({@link #allElements()}), those of elements whose position is unknown included.
	 *
	 * @param analysis The solved {@link TensorTypeAnalysis}.
	 * @return The union of the elements' tensor types; empty when no element has tensor evidence. Exact only when {@link #isComplete()};
	 *         otherwise the types of the elements the keys name, and the client chooses the conservative reading for its own question.
	 */
	Set<TensorType> elementTypes(TensorTypeAnalysis analysis) {
		Set<TensorType> ret = new HashSet<>();
		for (PointerKey key : this.allElements)
			ret.addAll(typesOf(analysis, key));
		return Collections.unmodifiableSet(ret);
	}

	/**
	 * The tensor types a solved analysis gives a key.
	 *
	 * @param analysis The solved analysis.
	 * @param key The key.
	 * @return Its types; empty for a key the analysis gives none.
	 */
	private static Set<TensorType> typesOf(TensorTypeAnalysis analysis, PointerKey key) {
		return TYPED.computeIfAbsent(analysis, ContainerElements::typed).getOrDefault(key, Set.of());
	}

	/**
	 * The keys a solved analysis gives a tensor type, each with its types, as the analysis's {@link TensorTypeAnalysis#iterator()} yields
	 * them.
	 *
	 * @param analysis The analysis.
	 * @return Each key's types.
	 */
	private static Map<PointerKey, Set<TensorType>> typed(TensorTypeAnalysis analysis) {
		Map<PointerKey, Set<TensorType>> ret = new HashMap<>();
		for (Iterator<Pair<PointerKey, TensorVariable>> it = analysis.iterator(); it.hasNext();) {
			Pair<PointerKey, TensorVariable> pair = it.next();
			ret.computeIfAbsent(pair.fst, k -> new HashSet<>()).addAll(pair.snd.getTypes());
		}
		ret.replaceAll((k, types) -> Collections.unmodifiableSet(types));
		return ret;
	}

	/**
	 * The call graph of {@code builder}, as WALA's {@link CallGraph} interface, which the engine bundle exports, so that a call on it is an
	 * interface call; its implementation's package isn't exported, and a call on that class fails to link.
	 *
	 * @param builder The builder.
	 * @return Its call graph.
	 */
	private static CallGraph callGraph(PropagationCallGraphBuilder builder) {
		return builder.getCallGraph();
	}

	/**
	 * The element index a catalog key names, whatever its encoding.
	 *
	 * @param value The catalog key's value.
	 * @return The index, or {@code null} when the key names no index.
	 */
	private static Integer index(Object value) {
		if (value instanceof Integer i)
			return i >= 0 ? i : null;
		if (value instanceof Long l)
			return l >= 0 && l <= Integer.MAX_VALUE ? Integer.valueOf(l.intValue()) : null;
		if (value instanceof String s) {
			Integer i = parseIntOrNull(s);
			return i != null && i >= 0 ? i : null;
		}
		return null;
	}

	/**
	 * The key of a container's field.
	 *
	 * @param builder The builder.
	 * @param container The container.
	 * @param name The field's name.
	 * @return The key, or {@code null} when the class hierarchy resolves no such field.
	 */
	private static PointerKey fieldKey(PropagationCallGraphBuilder builder, InstanceKey container, String name) {
		IField field = builder.getClassHierarchy().resolveField(FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(name), Root));
		return field == null ? null : builder.getPointerKeyForInstanceField(container, field);
	}

	/**
	 * Whether a container is a list or tuple display that program code allocates: an allocation site in a node whose method is program code
	 * ({@link AstMethod}, not a summary), at a {@code new} of that node's IR. An operator model's allocation keyed at another instruction,
	 * such as a call or a binary operation, is none.
	 *
	 * @param container The container.
	 * @return {@code true} iff the container is such a display.
	 */
	private static boolean isProgramDisplay(InstanceKey container) {
		// A dataset element's structure copies the positions of the structure it was built from
		// (wala/ML#1190), so its catalog holds run-time elements exactly when its source's does.
		if (container instanceof ElementMirrorKey mirror && !mirror.isLeaf())
			return isProgramDisplay(mirror.getSource());
		if (!(container instanceof AllocationSiteInNode asin))
			return false;
		CGNode node = asin.getNode();
		if (!(node.getMethod() instanceof AstMethod) || node.getMethod().isWalaSynthetic())
			return false;
		IR ir = node.getIR();
		if (ir == null)
			return false;
		for (SSAInstruction instruction : ir.getInstructions())
			if (instruction instanceof SSANewInstruction allocation && allocation.getNewSite().equals(asin.getSite()))
				return true;
		return false;
	}

	/**
	 * What the program may do to lists that the analysis does not record among their positioned elements, so that their recorded positions
	 * need not be their run-time ones. Over every node of the call graph, a list is so mutated when it is:
	 * <ul>
	 * <li>the receiver of a read of a repositioning method ({@link #REPOSITIONING_LIST_METHODS}), or an argument of a call of such a read,
	 * as in {@code list.append(l, x)};
	 * <li>the object of a store at a member that is not a constant index or name, such as {@code
	 *       l[i] = x}. When the same body also stores the value in the object's append contents, as a comprehension's body does, the store
	 * adds no element the analysis does not see, so it opens only a dict;
	 * <li>the list of a slice, read or assigned, which the front end lowers to a call of the slice builtin on the list;
	 * <li>the left operand of a {@code +}, {@code *} or {@code |}, which {@code l += m}, {@code l
	 *       *= n} and {@code s |= t} lower to and which mutate a list or set in place. The non-augmented forms build a new container, so
	 * this also takes their left operand, which only loses positions;
	 * <li>an argument of a call in program code that resolves to no target, such as one of {@code
	 *       heapq.heappush}, {@code operator.iadd} or {@code getattr}, which the front end imports rather than summarizes, and so of the
	 * method {@code getattr(l, "append")} returns, unless the callee is a method read off a list (the repositioning methods govern those)
	 * or a function read off a TensorFlow or NumPy module or object ({@link #NON_RESIZING_LIBRARIES}), neither of which resizes a list.
	 * </ul>
	 * A program function the call graph does not reach may still run, as a callback a library summary does not call, such as a {@code key}
	 * of {@code sorted} or a function {@code map} applies. In such a body, the receiver of a repositioning read is a list so mutated when
	 * it is a variable of an enclosing frame or a global, read off what that variable is written. A receiver the body builds itself is none
	 * the analysis has, and a library function that resizes nothing ({@link #NON_RESIZING_LIBRARY_FUNCTIONS}) mutates nothing
	 * (wala/ML#1182). Any other receiver there may be any list, and so is every list.
	 * <p>
	 * The front end drops the target of a {@code del} statement, so a subscript {@code del} anywhere, in a reached node or such a body, may
	 * mutate any list.
	 * <p>
	 * Computed once per builder.
	 *
	 * @param builder The builder.
	 * @return The mutations.
	 */
	private static Mutations mutations(PropagationCallGraphBuilder builder) {
		synchronized (MUTATIONS) {
			Mutations cached = MUTATIONS.get(builder);
			if (cached != null)
				return cached;
		}
		PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
		Marks marks = new Marks();
		Set<IMethod> reached = new HashSet<>();
		for (CGNode node : callGraph(builder)) {
			reached.add(node.getMethod());
			IR ir = node.getIR();
			if (ir == null)
				continue;
			SymbolTable symtab = ir.getSymbolTable();
			DefUse du = node.getDU();
			boolean program = node.getMethod() instanceof AstMethod && !node.getMethod().isWalaSynthetic();
			for (SSAInstruction instruction : ir.getInstructions()) {
				if (instruction instanceof AstPropertyRead read
						&& (isRepositioning(symtab, read.getMemberRef()) || isAdding(symtab, read.getMemberRef())))
					marks.mark(pa, node, read.getObjectRef(), isAdding(symtab, read.getMemberRef()));
				else if (instruction instanceof AstPropertyWrite write && !symtab.isConstant(write.getMemberRef())) {
					// `d[k] = t` or `l[i] = t` with a key the analysis may not name, unless the body also
					// stores the value in the append contents, as a comprehension's body does (wala/ML#1178).
					if (storesInAppendContents(ir, symtab, write))
						marks.markMirrored(pa, node, write.getObjectRef());
					else
						marks.mark(pa, node, write.getObjectRef(), true);
				} else if (instruction instanceof SSABinaryOpInstruction binary
						&& IN_PLACE_OPERATOR_NAMES.contains(String.valueOf(binary.getOperator()).toLowerCase()))
					// `l += m`, `l *= n` and `s |= t` mutate the left operand in place.
					marks.mark(pa, node, binary.getUse(0), true);
				else if (instruction instanceof SSAAbstractInvokeInstruction invoke) {
					Set<CGNode> targets = callGraph(builder).getPossibleTargets(node, invoke.getCallSite());
					SSAInstruction callee = invoke.getNumberOfUses() > 0 ? du.getDef(invoke.getUse(0)) : null;
					if (calls(targets, DELETE_BUILTIN)) {
						if (deletesSubscript(node.getMethod(), invoke))
							marks.markAll(false);
					} else if (calls(targets, SLICE_BUILTIN)) {
						// A slice, `l[i:j]`, read or assigned, is a call of the slice builtin on the list.
						if (invoke.getNumberOfUses() > 1)
							marks.mark(pa, node, invoke.getUse(1), true);
					} else if (callee instanceof AstPropertyRead method && isRepositioning(symtab, method.getMemberRef())) {
						// `list.append(l, x)`: the list is an argument, not the receiver.
						for (int i = 1; i < invoke.getNumberOfUses(); i++)
							marks.mark(pa, node, invoke.getUse(i), true);
					} else if (program && targets.isEmpty() && !isNonResizing(pa, node, callee))
						for (int i = 1; i < invoke.getNumberOfUses(); i++)
							marks.mark(pa, node, invoke.getUse(i), true);
				}
			}
		}
		scanUnreachedBodies(builder, reached, marks);
		Mutations ret = new Mutations(marks.lists, marks.allLists, marks.open, marks.allOpen);
		synchronized (MUTATIONS) {
			MUTATIONS.put(builder, ret);
		}
		return ret;
	}

	/**
	 * Scans the program functions the call graph does not reach; see {@link #mutations}.
	 *
	 * @param builder The builder.
	 * @param reached The methods of the call graph's nodes.
	 * @param marks The mutations, which this adds to.
	 */
	private static void scanUnreachedBodies(PropagationCallGraphBuilder builder, Set<IMethod> reached, Marks marks) {
		// The values a variable is written, by its name and definer, which the receiver checks below
		// read for many reads of the same builtin or module: each is a scan of the whole call graph.
		Map<List<String>, Set<InstanceKey>> writes = new HashMap<>();
		for (IClass cls : builder.getClassHierarchy()) {
			Collection<? extends IMethod> methods = cls.getDeclaredMethods();
			if (methods == null)
				continue;
			for (IMethod method : methods) {
				if (!(method instanceof AstMethod) || method.isWalaSynthetic() || reached.contains(method))
					continue;
				IR ir;
				try {
					ir = builder.getAnalysisCache().getIR(method);
				} catch (RuntimeException e) {
					LOGGER.log(Level.FINE, e, () -> "Cannot read the IR of unreached " + method + ".");
					marks.markAll(true);
					return;
				}
				if (ir == null)
					continue;
				SymbolTable symtab = ir.getSymbolTable();
				DefUse du = builder.getAnalysisCache().getDefUse(ir);
				for (SSAInstruction instruction : ir.getInstructions()) {
					if (instruction instanceof SSAAbstractInvokeInstruction invoke && invoke.getNumberOfUses() > 0
							&& readsName(du.getDef(invoke.getUse(0)), DELETE_NAME) && deletesSubscript(method, invoke))
						marks.markAll(false);
					if (!(instruction instanceof AstPropertyRead read)
							|| !(isRepositioning(symtab, read.getMemberRef()) || isAdding(symtab, read.getMemberRef())))
						continue;
					boolean adds = isAdding(symtab, read.getMemberRef());
					SSAInstruction receiver = du.getDef(read.getObjectRef());
					// A container the body builds itself is none the analysis allocates, and a library
					// function so named may resize none (wala/ML#1182).
					if (isBuiltHere(builder, writes, du, receiver) || isNonResizingLibraryFunction(builder, writes, symtab, du, receiver,
							symtab.getStringValue(read.getMemberRef())))
						continue;
					if (receiver instanceof AstLexicalRead lexical) {
						for (Access access : lexical.getAccesses())
							if (access.valueNumber() == read.getObjectRef())
								markLexicalWrites(builder, access.variableName(), access.variableDefiner(), marks, adds);
					} else if (receiver instanceof AstGlobalRead global)
						markGlobalWrites(builder, global.getGlobalName(), marks, adds);
					// A parameter, or a value the body computes: it may be any container.
					else
						marks.markAll(adds);
				}
			}
		}
	}

	/**
	 * Whether a receiver in a body the call graph does not reach is a container the body builds itself: a display, such as {@code []} or
	 * {@code {}}, or a call of the {@code list}, {@code
	 * dict} or {@code set} builtin, read by a name the call graph's nodes bind to no function of the program's own. The analysis allocates
	 * no such container, so a mutation of it mutates none the analysis has (wala/ML#1182). A rebinding of the name in a body the call graph
	 * does not reach is not seen, which would take a {@code global} statement rebinding a builtin's name there.
	 *
	 * @param builder The builder.
	 * @param writes The values variables are written, by name and definer, filled as read.
	 * @param du The body's def-use.
	 * @param receiver The instruction defining the receiver, or {@code null}.
	 * @return {@code true} iff the body builds the receiver.
	 */
	private static boolean isBuiltHere(PropagationCallGraphBuilder builder, Map<List<String>, Set<InstanceKey>> writes, DefUse du,
			SSAInstruction receiver) {
		if (receiver instanceof SSANewInstruction)
			return true;
		if (!(receiver instanceof SSAAbstractInvokeInstruction call) || call.getNumberOfUses() == 0)
			return false;
		SSAInstruction callee = du.getDef(call.getUse(0));
		String name;
		if (callee instanceof AstGlobalRead global)
			name = bareName(global.getGlobalName());
		else if (callee instanceof AstLexicalRead lexical && lexical.getAccessCount() == 1
				&& isModule(lexical.getAccess(0).variableDefiner()))
			name = lexical.getAccess(0).variableName();
		else
			return false;
		if (!CONTAINER_BUILTINS.contains(name))
			return false;
		// A module may bind the name to a function of its own, which may return any container.
		for (InstanceKey value : variableWrites(builder, writes, name, null))
			if (value.concreteType().getName().toString().startsWith("Lscript "))
				return false;
		return true;
	}

	/**
	 * Whether a method read in a body the call graph does not reach is a library function that resizes no container
	 * ({@link #NON_RESIZING_LIBRARY_FUNCTIONS}): its receiver is a chain of attribute reads, as {@code tf.keras.backend} is, off a variable
	 * that every write in the call graph binds to a TensorFlow or NumPy module ({@link #NON_RESIZING_LIBRARIES}), as {@code import
	 * tensorflow as tf} does (wala/ML#1182).
	 *
	 * @param builder The builder.
	 * @param writes The values variables are written, by name and definer, filled as read.
	 * @param symtab The body's symbol table.
	 * @param du The body's def-use.
	 * @param receiver The instruction defining the receiver, or {@code null}.
	 * @param method The method's name.
	 * @return {@code true} iff the method is such a function.
	 */
	private static boolean isNonResizingLibraryFunction(PropagationCallGraphBuilder builder, Map<List<String>, Set<InstanceKey>> writes,
			SymbolTable symtab, DefUse du, SSAInstruction receiver, String method) {
		// The attributes read off the root variable, outermost first.
		List<String> attributes = new ArrayList<>();
		SSAInstruction current = receiver;
		Set<SSAInstruction> seen = new HashSet<>();
		while (current instanceof AstPropertyRead read && seen.add(current)) {
			if (!symtab.isStringConstant(read.getMemberRef()))
				return false;
			attributes.add(0, symtab.getStringValue(read.getMemberRef()));
			current = du.getDef(read.getObjectRef());
		}
		String variable;
		String definer;
		if (current instanceof AstGlobalRead global) {
			variable = bareName(global.getGlobalName());
			definer = null;
		} else if (current instanceof AstLexicalRead lexical && lexical.getAccessCount() == 1) {
			variable = lexical.getAccess(0).variableName();
			definer = lexical.getAccess(0).variableDefiner();
		} else
			return false;
		String path = (attributes.isEmpty() ? "" : "/" + String.join("/", attributes)) + "/" + method;
		// Only a read some listed function ends in, such as `update`, is worth the root's writes.
		if (NON_RESIZING_LIBRARY_FUNCTIONS.stream().noneMatch(function -> function.endsWith(path)))
			return false;
		boolean any = false;
		for (InstanceKey module : variableWrites(builder, writes, variable, definer)) {
			String type = module.concreteType().getName().toString();
			if (NON_RESIZING_LIBRARIES.stream().noneMatch(type::startsWith) || !NON_RESIZING_LIBRARY_FUNCTIONS.contains(type + path))
				return false;
			any = true;
		}
		return any;
	}

	/**
	 * The values the call graph's nodes write a variable: a global, or a module's variable, which is how a module-level assignment is
	 * lowered, matched by bare name as {@link #markGlobalWrites} matches them; or, given its definer, a variable of that frame.
	 *
	 * @param builder The builder.
	 * @param writes The values variables are written, by name and definer, filled as read.
	 * @param name The variable's bare name.
	 * @param definer The name of the method whose frame defines the variable, or {@code null} for a global.
	 * @return The written values' instance keys.
	 */
	private static Set<InstanceKey> variableWrites(PropagationCallGraphBuilder builder, Map<List<String>, Set<InstanceKey>> writes,
			String name, String definer) {
		List<String> key = Arrays.asList(name, definer);
		Set<InstanceKey> cached = writes.get(key);
		if (cached != null)
			return cached;
		PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
		Set<InstanceKey> ret = new HashSet<>();
		for (CGNode node : callGraph(builder)) {
			IR ir = node.getIR();
			if (ir == null)
				continue;
			for (SSAInstruction instruction : ir.getInstructions())
				if (definer == null && instruction instanceof AstGlobalWrite write && name.equals(bareName(write.getGlobalName())))
					pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, write.getVal())).forEach(ret::add);
				else if (instruction instanceof AstLexicalWrite write)
					for (Access access : write.getAccesses())
						if (name.equals(access.variableName())
								&& (definer == null ? isModule(access.variableDefiner()) : definer.equals(access.variableDefiner())))
							pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, access.valueNumber())).forEach(ret::add);
		}
		writes.put(key, ret);
		return ret;
	}

	/**
	 * Adds the lists a variable of a frame is written, in every node of the frame's method.
	 *
	 * @param builder The builder.
	 * @param name The variable's name.
	 * @param definer The name of the method whose frame defines the variable.
	 * @param marks The mutations, which this adds to.
	 * @param adds Whether the mutation may also add an element the analysis does not record.
	 */
	private static void markLexicalWrites(PropagationCallGraphBuilder builder, String name, String definer, Marks marks, boolean adds) {
		for (CGNode node : callGraph(builder)) {
			IR ir = node.getIR();
			if (ir == null)
				continue;
			for (SSAInstruction instruction : ir.getInstructions())
				if (instruction instanceof AstLexicalWrite write)
					for (Access access : write.getAccesses())
						if (name.equals(access.variableName()) && definer.equals(access.variableDefiner()))
							marks.mark(builder.getPointerAnalysis(), node, access.valueNumber(), adds);
		}
	}

	/**
	 * Adds the lists a global is written, in every node: by a write of the global, or by a module's write of its variable of that name,
	 * which is how a module-level assignment is lowered. The global is matched by its bare name, so a same-named variable of another module
	 * counts as well, which only loses positions.
	 *
	 * @param builder The builder.
	 * @param name The global's name, as its read names it.
	 * @param marks The mutations, which this adds to.
	 * @param adds Whether the mutation may also add an element the analysis does not record.
	 */
	private static void markGlobalWrites(PropagationCallGraphBuilder builder, String name, Marks marks, boolean adds) {
		String bare = bareName(name);
		for (CGNode node : callGraph(builder)) {
			IR ir = node.getIR();
			if (ir == null)
				continue;
			for (SSAInstruction instruction : ir.getInstructions())
				if (instruction instanceof AstGlobalWrite write) {
					if (bare.equals(bareName(write.getGlobalName())))
						marks.mark(builder.getPointerAnalysis(), node, write.getVal(), adds);
				} else if (instruction instanceof AstLexicalWrite write)
					for (Access access : write.getAccesses())
						if (bare.equals(access.variableName()) && isModule(access.variableDefiner()))
							marks.mark(builder.getPointerAnalysis(), node, access.valueNumber(), adds);
		}
	}

	/**
	 * The bare name of a global: what follows its last space or slash, as in {@code global H} or {@code global script m.py/H}.
	 *
	 * @param name The global's name.
	 * @return Its bare name.
	 */
	private static String bareName(String name) {
		if (name == null)
			return "";
		return name.substring(Math.max(name.lastIndexOf(' '), name.lastIndexOf('/')) + 1);
	}

	/**
	 * Whether a frame's name is a module's: a script's, whose name ends in the file's, rather than a function's within it.
	 *
	 * @param definer The frame's name.
	 * @return {@code true} iff it names a module.
	 */
	private static boolean isModule(String definer) {
		return definer != null && definer.startsWith("Lscript ") && definer.endsWith(".py");
	}

	/**
	 * Whether an instruction reads a variable of a name: a global or a variable of an enclosing frame.
	 *
	 * @param instruction The instruction, or {@code null}.
	 * @param name The name.
	 * @return {@code true} iff it reads that name.
	 */
	private static boolean readsName(SSAInstruction instruction, String name) {
		if (instruction instanceof AstGlobalRead global)
			return global.getGlobalName() != null && global.getGlobalName().endsWith(name);
		if (instruction instanceof AstLexicalRead lexical)
			for (Access access : lexical.getAccesses())
				if (name.equals(access.variableName()))
					return true;
		if (instruction instanceof SSANewInstruction allocation)
			return allocation.getConcreteType().getName().toString().endsWith(name);
		return false;
	}

	/**
	 * Whether a body stores the value of a store under a member the analysis cannot name also in the same object's append contents
	 * ({@link PythonSSAPropagationCallGraphBuilder#LIST_APPEND_CONTENTS_FIELD}), where the element readers look. A comprehension's body
	 * stores each element so (wala/ML#1178).
	 *
	 * @param ir The body.
	 * @param symtab Its symbol table.
	 * @param write The store.
	 * @return {@code true} iff the body has such a second store.
	 */
	private static boolean storesInAppendContents(IR ir, SymbolTable symtab, AstPropertyWrite write) {
		for (SSAInstruction other : ir.getInstructions())
			if (other instanceof AstPropertyWrite store && store.getObjectRef() == write.getObjectRef()
					&& store.getValue() == write.getValue() && symtab.isStringConstant(store.getMemberRef())
					&& PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD.equals(symtab.getStringValue(store.getMemberRef())))
				return true;
		return false;
	}

	/**
	 * Whether a member is the name of a repositioning method ({@link #REPOSITIONING_LIST_METHODS}).
	 *
	 * @param symtab The symbol table.
	 * @param member The member's value number.
	 * @return {@code true} iff it is such a name.
	 */
	private static boolean isRepositioning(SymbolTable symtab, int member) {
		return symtab.isStringConstant(member) && REPOSITIONING_LIST_METHODS.contains(symtab.getStringValue(member));
	}

	/**
	 * Whether a member is the name of a method that may add an element unrecorded ({@link #ADDING_METHODS}).
	 *
	 * @param symtab The symbol table.
	 * @param member The member's value number.
	 * @return {@code true} iff it is such a name.
	 */
	private static boolean isAdding(SymbolTable symtab, int member) {
		return symtab.isStringConstant(member) && ADDING_METHODS.contains(symtab.getStringValue(member));
	}

	/**
	 * Whether an unresolved call's callee cannot resize a list passed to it: a method read off a list, which the repositioning methods
	 * govern, or a function read off a TensorFlow or NumPy module or object ({@link #NON_RESIZING_LIBRARIES}), directly or through
	 * attributes, as {@code
	 * tf.image.random_crop} is read off {@code tf}.
	 *
	 * @param pa The pointer analysis.
	 * @param node The calling node.
	 * @param callee The instruction defining the callee, or {@code null}.
	 * @return {@code true} iff the callee cannot resize a list.
	 */
	private static boolean isNonResizing(PointerAnalysis<InstanceKey> pa, CGNode node, SSAInstruction callee) {
		// The owner may be an attribute of a module, `tf.image`, which the analysis allocates as a
		// plain
		// object or not at all: read up the chain of attribute reads for a library's allocation.
		DefUse du = node.getDU();
		Set<SSAInstruction> seen = new HashSet<>();
		for (SSAInstruction current = callee; current instanceof AstPropertyRead read
				&& seen.add(current); current = du.getDef(read.getObjectRef())) {
			OrdinalSet<InstanceKey> owners = pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, read.getObjectRef()));
			for (InstanceKey owner : owners) {
				if (current == callee && owner.concreteType().getReference().equals(list))
					return true;
				String type = owner.concreteType().getName().toString();
				for (String library : NON_RESIZING_LIBRARIES)
					if (type.startsWith(library))
						return true;
			}
		}
		return false;
	}

	/**
	 * Whether a call of the deletion builtin is a subscript {@code del}, as in {@code del l[0]}, read from the statement's source text,
	 * since the front end drops its target. A statement whose text cannot be read is taken to be one.
	 *
	 * @param method The method holding the call.
	 * @param invoke The call.
	 * @return {@code true} unless the statement's text is read and has no subscript.
	 */
	private static boolean deletesSubscript(IMethod method, SSAAbstractInvokeInstruction invoke) {
		if (!(method instanceof AstMethod astMethod) || invoke.iIndex() < 0)
			return true;
		Position at = astMethod.getSourcePosition(invoke.iIndex());
		if (at == null || at.getFirstLine() <= 0)
			return true;
		try (BufferedReader reader = new BufferedReader(at.getReader())) {
			String text = reader.lines().skip(at.getFirstLine() - 1).limit(Math.max(1, at.getLastLine() - at.getFirstLine() + 1))
					.collect(Collectors.joining("\n"));
			return text.indexOf('[') >= 0;
		} catch (IOException | RuntimeException e) {
			LOGGER.log(Level.FINE, e, () -> "Cannot read the source of a del in " + method + ".");
			return true;
		}
	}

	/**
	 * Whether a call's targets include a builtin.
	 *
	 * @param targets The call's targets.
	 * @param builtin The builtin's type name, such as {@link #DELETE_BUILTIN}.
	 * @return {@code true} iff a target is that builtin.
	 */
	private static boolean calls(Set<CGNode> targets, String builtin) {
		for (CGNode target : targets)
			if (target.getMethod().getDeclaringClass().getName().toString().equals(builtin))
				return true;
		return false;
	}

	/**
	 * A container's index in the pointer analysis's instance key mapping.
	 *
	 * @param builder The builder.
	 * @param container The container.
	 * @return The index.
	 */
	private static int mappedIndex(PropagationCallGraphBuilder builder, InstanceKey container) {
		return builder.getPointerAnalysis().getInstanceKeyMapping().getMappedIndex(container);
	}
}
