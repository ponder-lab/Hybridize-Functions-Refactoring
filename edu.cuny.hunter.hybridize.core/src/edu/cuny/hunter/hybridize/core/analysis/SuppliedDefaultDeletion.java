package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.runtime.ILog;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.eclipse.text.edits.DeleteEdit;
import org.eclipse.text.edits.MultiTextEdit;
import org.eclipse.text.edits.TextEdit;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.Call;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.Num;
import org.python.pydev.parser.jython.ast.Str;
import org.python.pydev.parser.jython.ast.UnaryOp;
import org.python.pydev.parser.jython.ast.VisitorBase;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.jython.ast.keywordType;
import org.python.pydev.parser.jython.ast.num_typeType;
import org.python.pydev.parser.jython.ast.str_typeType;
import org.python.pydev.parser.jython.ast.unaryopType;
import org.python.pydev.parser.visitors.NodeUtils;

import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.Iterator2Iterable;

/**
 * Deleting, at each call of a function, an argument that equals its parameter's default, so that the parameter, defaulted and not
 * tensor-typed, can be left out of an inferred input signature even though callers pass it (#1323).
 * <p>
 * A defaulted parameter no call site supplies is omitted from the signature (#787). One that some call site supplies cannot be, since the
 * hybridized function would then accept fewer arguments than that call passes, and it raises {@code TypeError}. Where every such call
 * passes exactly the default, written as a literal, removing the argument leaves the call meaning what it did, and the parameter becomes
 * one no call site supplies. This class decides whether that holds for a parameter and computes the deletions.
 * <p>
 * The conditions, each a reason to decline:
 * <ol>
 * <li>The function carries no decorator: a decorated function's callers reach the callable the decorator returns, not the definition.</li>
 * <li>The parameter's default is a literal: a number, optionally negated, a string that is neither bytes nor an f-string, {@code True},
 * {@code False}, or {@code None}.</li>
 * <li>Every call-graph caller is code the program wrote, reached directly or through the engine's method trampolines, in a module the
 * refactoring reads. A framework summary, such as Keras's {@code Layer.__call__}, or a caller with no source is not.</li>
 * <li>Every call at such a site resolves, through trampolines, only to the function, so no other override is called with the argument
 * deleted.</li>
 * <li>Every call is found in its module's source as the one call on its line naming the function, with the arguments the engine saw, and
 * none spreads {@code *} or {@code **} arguments.</li>
 * <li>A call supplying the parameter passes it as a literal of the default's kind and value, by keyword or as its last positional argument.
 * The value is compared, not the text: {@code 'a'} and {@code "a"} agree, as do {@code 1_000} and {@code 1000}, while {@code 1} and
 * {@code 1.0}, or {@code True} and {@code 1}, do not, and neither do {@code 0.0} and {@code -0.0}.</li>
 * </ol>
 * <p>
 * The call set is the call graph's, so the change rests on the same closed world as the omission of #787: a call the analysis does not see,
 * such as one through {@code getattr} with a string, a framework callback, or a script outside the analyzed projects, keeps its argument,
 * and once the signature is on it raises {@code TypeError}. Typing plays no part: the literals are compared on the parser's tree, and the
 * calls are read from the published call graph.
 */
public final class SuppliedDefaultDeletion {

	private static final ILog LOG = getLog(SuppliedDefaultDeletion.class);

	/**
	 * A module whose calls may be edited: its workspace file, its document, and its parsed tree.
	 *
	 * @param file The workspace file.
	 * @param document The document the offsets index.
	 * @param ast The module's tree.
	 */
	public record SourceModule(IFile file, IDocument document, SimpleNode ast) {
	}

	/**
	 * The removal of one argument from one call.
	 *
	 * @param file The file holding the call.
	 * @param offset Where the removed text begins: the comma before the argument.
	 * @param length The removed text's length, through the argument's last character.
	 */
	public record Deletion(IFile file, int offset, int length) {

		/**
		 * The edit performing the removal, wrapped in a container as every edit the processor collects is.
		 *
		 * @return The edit.
		 */
		public TextEdit toEdit() {
			MultiTextEdit ret = new MultiTextEdit();
			ret.addChild(new DeleteEdit(this.offset, this.length));
			return ret;
		}
	}

	/**
	 * The outcome for one parameter: either the deletions that make it omittable, or why it stays supplied.
	 *
	 * @param deletions The deletions, one per supplying call, or {@code null} when declined.
	 * @param decline Why the relaxation declines, or {@code null} when it applies.
	 */
	public record Result(List<Deletion> deletions, String decline) {

		static Result declined(String why) {
			return new Result(null, why);
		}

		/**
		 * True iff the relaxation applies.
		 *
		 * @return True iff the deletions were computed.
		 */
		public boolean applies() {
			return this.deletions != null;
		}
	}

	/** A literal's value with its kind, so values of different kinds never compare equal. */
	private record Literal(String kind, Object value) {
	}

	/** A call of the function as the engine saw it, in the frame that wrote it. */
	private record SourceSite(CGNode frame, CallSiteReference reference, boolean bound) {
	}

	private SuppliedDefaultDeletion() {
	}

	/**
	 * Decides whether every call supplying {@code parameter} passes its default, and computes the deletions if so.
	 *
	 * @param function The function declaring the parameter.
	 * @param parameter A defaulted, non-{@code self} parameter that some call site supplies.
	 * @param nodes The function's call-graph nodes.
	 * @param callGraph The call graph.
	 * @param modules The modules whose calls may be edited, by the absolute path of their file.
	 * @return The outcome.
	 */
	static Result compute(Function function, Parameter parameter, Set<CGNode> nodes, CallGraph callGraph,
			Map<String, SourceModule> modules) {
		FunctionDef def = function.getFunctionDefinition().getFunctionDef();

		if (def.decs != null && def.decs.length > 0)
			return Result.declined("the function is decorated, so its callers reach what the decorator returns");

		if (!(def.name instanceof NameTok defName))
			return Result.declined("the function's name is not known");

		boolean isMethod = function.getParameters().stream().anyMatch(Parameter::isSelf);

		Optional<Literal> defaultValue = literal(parameter.getDefaultValue());

		if (defaultValue.isEmpty())
			return Result.declined("its default is not a literal");

		List<SourceSite> sites = new ArrayList<>();

		for (CGNode node : nodes)
			for (CGNode predecessor : Iterator2Iterable.make(callGraph.getPredNodes(node)))
				if (isTrampoline(predecessor)) {
					for (CGNode caller : Iterator2Iterable.make(callGraph.getPredNodes(predecessor))) {
						if (isTrampoline(caller) || !(caller.getMethod() instanceof AstMethod))
							return Result.declined("a call reaches it from code with no source of its own");

						for (CallSiteReference reference : Iterator2Iterable.make(callGraph.getPossibleSites(caller, predecessor)))
							sites.add(new SourceSite(caller, reference, true));
					}
				} else if (isMethod)
					// A method reached without a trampoline is called unbound or through a route, such as `super()`, whose binding of
					// `self` the written arguments do not show, so which written argument fills the parameter is not known.
					return Result.declined("a call reaches the method without binding its receiver as the engine's trampolines do");
				else if (predecessor.getMethod() instanceof AstMethod) {
					for (CallSiteReference reference : Iterator2Iterable.make(callGraph.getPossibleSites(predecessor, node)))
						sites.add(new SourceSite(predecessor, reference, false));
				} else
					return Result.declined("a call reaches it from code with no source of its own, such as a framework's");

		if (sites.isEmpty())
			return Result.declined("no call site was found");

		// A call two contexts share is one call in the source, so it is edited once.
		Map<Call, Deletion> deletions = new IdentityHashMap<>();
		Set<Call> omitting = Collections.newSetFromMap(new IdentityHashMap<>());

		for (SourceSite site : sites) {
			if (!resolvesOnlyTo(callGraph, site, nodes))
				return Result.declined("a call supplying it may also reach another function");

			IR ir = site.frame().getIR();
			AstMethod method = (AstMethod) site.frame().getMethod();

			if (ir == null || method.debugInfo() == null)
				return Result.declined("a calling frame has no IR or no source positions");

			SourceModule module = modules.get(method.getDeclaringClass().getSourceFileName());

			if (module == null)
				return Result.declined("a call is in a module the refactoring does not read");

			for (SSAAbstractInvokeInstruction instruction : ir.getCalls(site.reference())) {
				if (!(instruction instanceof PythonInvokeInstruction invoke))
					return Result.declined("a call is not a Python call");

				if (invoke.getStarredPositions().length > 0)
					return Result.declined("a call spreads its arguments");

				Position position = method.debugInfo().getInstructionPosition(instruction.iIndex());

				if (position == null)
					return Result.declined("a call has no source position");

				// The callee takes positional slot 0 of the invoke.
				int positionals = invoke.getNumberOfPositionalParameters() - 1;
				Set<String> keywords = new HashSet<>(invoke.getKeywords());
				Optional<Call> found = findCall(module.ast(), position.getFirstLine(), defName.id, positionals, keywords);

				if (found.isEmpty())
					return Result.declined("a call was not found as the only one of its shape on line " + position.getFirstLine());

				Call call = found.get();

				if (deletions.containsKey(call) || omitting.contains(call))
					continue;

				Optional<exprType> supplied;
				boolean keyword = keywords.contains(parameter.getName());

				if (keyword)
					supplied = keywordValue(call, parameter.getName());
				else {
					// A bound call's receiver fills `self` before the first written argument; an unbound one passes it written.
					int slot = site.bound() ? parameter.getIndex() - 1 : parameter.getIndex();
					// Only a method is reached through a trampoline, and only a plain function directly (above).

					if (slot < 0)
						return Result.declined("a call's binding of the parameter is unclear");

					if (positionals <= slot) {
						omitting.add(call);
						continue;
					}

					if (positionals != slot + 1)
						return Result.declined("a call passes positional arguments after it");

					supplied = Optional.of(call.args[slot]);
				}

				if (supplied.isEmpty() || !defaultValue.equals(literal(supplied.get())))
					return Result.declined("a call passes it a value other than its default literal");

				Optional<Deletion> deletion = deletion(module, supplied.get(), keyword ? parameter.getName() : null);

				if (deletion.isEmpty())
					return Result.declined("a call's argument could not be located in its source");

				deletions.put(call, deletion.get());
			}
		}

		if (deletions.isEmpty())
			return Result.declined("no call supplying it was found in the source");

		return new Result(List.copyOf(deletions.values()), null);
	}

	/**
	 * True iff every target of the site is one of {@code nodes}, through trampolines, so deleting the argument cannot reach another
	 * function.
	 */
	private static boolean resolvesOnlyTo(CallGraph callGraph, SourceSite site, Set<CGNode> nodes) {
		Set<CGNode> targets = callGraph.getPossibleTargets(site.frame(), site.reference());

		if (targets.isEmpty())
			return false;

		for (CGNode target : targets)
			if (isTrampoline(target)) {
				if (!site.bound())
					return false;

				for (CGNode successor : Iterator2Iterable.make(callGraph.getSuccNodes(target)))
					if (!nodes.contains(successor))
						return false;
			} else if (!nodes.contains(target))
				return false;

		return true;
	}

	/**
	 * True iff {@code node} is a synthesized method-dispatch trampoline, which the engine names with a {@code trampoline} prefix.
	 */
	private static boolean isTrampoline(CGNode node) {
		return node.getMethod().getName().toString().startsWith(PythonTypes.TRAMPOLINE_METHOD_NAME);
	}

	/**
	 * The one call on {@code line} naming {@code name} with exactly the given positional count and keyword names and no spread, if there is
	 * exactly one.
	 */
	private static Optional<Call> findCall(SimpleNode module, int line, String name, int positionals, Set<String> keywords) {
		List<Call> found = new ArrayList<>();

		try {
			module.accept(new VisitorBase() {

				@Override
				public Object visitCall(Call node) throws Exception {
					if (node.func != null && node.func.beginLine == line && name.equals(calleeName(node.func)) && node.starargs == null
							&& node.kwargs == null && length(node.args) == positionals && keywordNames(node).equals(keywords))
						found.add(node);

					node.traverse(this);
					return null;
				}

				@Override
				protected Object unhandled_node(SimpleNode node) throws Exception {
					return null;
				}

				@Override
				public void traverse(SimpleNode node) throws Exception {
					node.traverse(this);
				}
			});
		} catch (Exception e) {
			LOG.warn("Can't search for the call of " + name + " on line " + line + ".", e);
			return Optional.empty();
		}

		return found.size() == 1 ? Optional.of(found.get(0)) : Optional.empty();
	}

	private static int length(Object[] array) {
		return array == null ? 0 : array.length;
	}

	private static Set<String> keywordNames(Call call) {
		Set<String> ret = new HashSet<>();

		if (call.keywords != null)
			for (keywordType keyword : call.keywords)
				if (keyword != null && keyword.arg instanceof NameTok tok)
					ret.add(tok.id);

		return ret;
	}

	private static String calleeName(exprType func) {
		if (func instanceof Name name)
			return name.id;

		if (func instanceof Attribute attribute && attribute.attr instanceof NameTok tok)
			return tok.id;

		return null;
	}

	private static Optional<exprType> keywordValue(Call call, String name) {
		if (call.keywords != null)
			for (keywordType keyword : call.keywords)
				if (keyword != null && keyword.arg instanceof NameTok tok && tok.id.equals(name))
					return Optional.ofNullable(keyword.value);

		return Optional.empty();
	}

	/**
	 * The value of a literal with its kind, or empty when the expression is no literal this relaxation compares.
	 *
	 * @param expression The expression, possibly {@code null}.
	 * @return The literal, or empty.
	 */
	static Optional<Literal> literal(exprType expression) {
		if (expression instanceof Num num)
			return number(num, false);

		if (expression instanceof UnaryOp unary && unary.op == unaryopType.USub && unary.operand instanceof Num num)
			return number(num, true);

		if (expression instanceof Str str && !str.binary && !str.fstring && str.s != null)
			// The parser keeps the text between the quotes as written, escapes included, so equal text in the same rawness is an
			// equal value whatever the quotes. Unequal text may still be an equal value (`'\x61'` and `'a'`), which only declines.
			return Optional.of(new Literal(str.raw ? "raw str" : "str", str.s));

		if (expression instanceof Name name && name.reserved)
			return switch (name.id) {
			case "True", "False" -> Optional.of(new Literal("bool", name.id));
			case "None" -> Optional.of(new Literal("None", "None"));
			default -> Optional.empty();
			};

		return Optional.empty();
	}

	private static Optional<Literal> number(Num num, boolean negated) {
		if (num.num == null)
			return Optional.empty();

		switch (num.type) {
		case num_typeType.Int, num_typeType.Long, num_typeType.Hex, num_typeType.Oct:
			if (!(num.n instanceof Number n))
				return Optional.empty();

			BigInteger integer = n instanceof BigInteger big ? big : BigInteger.valueOf(n.longValue());
			return Optional.of(new Literal("int", negated ? integer.negate() : integer));
		case num_typeType.Float:
			// The parser's own value is single precision, so the written text is parsed again in double precision, as Python reads it.
			try {
				double value = Double.parseDouble(num.num.replace("_", ""));
				// Boxed so equality tells -0.0 from 0.0, which Python's `==` equates but which do not behave alike.
				return Optional.of(new Literal("float", Double.valueOf(negated ? -value : value)));
			} catch (NumberFormatException e) {
				LOG.warn("Can't read the number " + num.num + ".", e);
				return Optional.empty();
			}
		default:
			return Optional.empty();
		}
	}

	/**
	 * The removal of {@code argument} from {@code call}: from the comma before it through its last character. Located by reading the text
	 * around the literal, which the parser places, and refused unless only whitespace, the keyword's {@code name =}, and the comma lie
	 * between the previous argument and the literal.
	 */
	private static Optional<Deletion> deletion(SourceModule module, exprType argument, String keyword) {
		IDocument document = module.document();
		exprType literal = argument instanceof UnaryOp unary ? unary.operand : argument;

		try {
			int start = NodeUtils.getOffset(document, literal);
			Optional<Integer> end = literalEnd(document, start, literal);

			if (end.isEmpty())
				return Optional.empty();

			int i = start - 1;
			i = skipWhitespaceBackward(document, i);

			if (argument instanceof UnaryOp) {
				if (i < 0 || document.getChar(i) != '-')
					return Optional.empty();

				i = skipWhitespaceBackward(document, i - 1);
			}

			if (keyword != null) {
				if (i < 0 || document.getChar(i) != '=')
					return Optional.empty();

				i = skipWhitespaceBackward(document, i - 1);
				int nameStart = i - keyword.length() + 1;

				if (nameStart < 0 || !document.get(nameStart, keyword.length()).equals(keyword))
					return Optional.empty();

				i = skipWhitespaceBackward(document, nameStart - 1);
			}

			if (i < 0 || document.getChar(i) != ',')
				return Optional.empty();

			return Optional.of(new Deletion(module.file(), i, end.get() - i));
		} catch (BadLocationException | RuntimeException e) {
			LOG.warn("Can't locate the argument " + argument + " in " + module.file() + ".", e);
			return Optional.empty();
		}
	}

	/** The index of the last non-whitespace character at or before {@code i}, or -1. */
	private static int skipWhitespaceBackward(IDocument document, int from) throws BadLocationException {
		int i = from;

		while (i >= 0 && Character.isWhitespace(document.getChar(i)))
			--i;

		return i;
	}

	/** The offset just past the literal beginning at {@code start}, checked against what the parser recorded for it. */
	private static Optional<Integer> literalEnd(IDocument document, int start, exprType literal) throws BadLocationException {
		if (literal instanceof Name name) {
			int end = start + name.id.length();
			return document.get(start, name.id.length()).equals(name.id) ? Optional.of(end) : Optional.empty();
		}

		if (literal instanceof Num num) {
			int end = start;

			while (end < document.getLength()) {
				char c = document.getChar(end);
				char previous = end > start ? document.getChar(end - 1) : ' ';

				if (Character.isLetterOrDigit(c) || c == '_' || c == '.'
						|| ((c == '+' || c == '-') && (previous == 'e' || previous == 'E') && num.type == num_typeType.Float))
					++end;
				else
					break;
			}

			String text = document.get(start, end - start);
			String recorded = num.type == num_typeType.Float ? text.replace("_", "") : text;
			return recorded.equals(num.num) ? Optional.of(end) : Optional.empty();
		}

		if (literal instanceof Str str) {
			int quotes = str.type == str_typeType.TripleSingle || str.type == str_typeType.TripleDouble ? 3 : 1;
			char quote = str.type == str_typeType.SingleSingle || str.type == str_typeType.TripleSingle ? '\'' : '"';
			int i = start;

			while (i < document.getLength() && Character.isLetter(document.getChar(i)))
				++i;

			for (int q = 0; q < quotes; q++)
				if (document.getChar(i + q) != quote)
					return Optional.empty();

			int contentStart = i + quotes;

			if (!document.get(contentStart, str.s.length()).equals(str.s))
				return Optional.empty();

			int close = contentStart + str.s.length();

			for (int q = 0; q < quotes; q++)
				if (document.getChar(close + q) != quote)
					return Optional.empty();

			return Optional.of(close + quotes);
		}

		return Optional.empty();
	}
}
