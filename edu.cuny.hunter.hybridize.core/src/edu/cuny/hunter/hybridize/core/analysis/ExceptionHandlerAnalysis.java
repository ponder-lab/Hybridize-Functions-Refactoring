package edu.cuny.hunter.hybridize.core.analysis;

import static org.eclipse.core.runtime.Platform.getLog;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.eclipse.core.runtime.ILog;
import org.python.pydev.parser.jython.SimpleNode;
import org.python.pydev.parser.jython.ast.Assign;
import org.python.pydev.parser.jython.ast.Attribute;
import org.python.pydev.parser.jython.ast.ClassDef;
import org.python.pydev.parser.jython.ast.For;
import org.python.pydev.parser.jython.ast.FunctionDef;
import org.python.pydev.parser.jython.ast.If;
import org.python.pydev.parser.jython.ast.Module;
import org.python.pydev.parser.jython.ast.Name;
import org.python.pydev.parser.jython.ast.NameTok;
import org.python.pydev.parser.jython.ast.TryExcept;
import org.python.pydev.parser.jython.ast.TryFinally;
import org.python.pydev.parser.jython.ast.Tuple;
import org.python.pydev.parser.jython.ast.VisitorBase;
import org.python.pydev.parser.jython.ast.While;
import org.python.pydev.parser.jython.ast.With;
import org.python.pydev.parser.jython.ast.excepthandlerType;
import org.python.pydev.parser.jython.ast.exprType;
import org.python.pydev.parser.jython.ast.stmtType;
import org.python.pydev.parser.jython.ast.suiteType;

/**
 * The {@code except} clauses around a line of a module, read from its source (#1014). The front end lowers a {@code try} statement without
 * the classes its clauses name, so which exception a call's {@code except} clause dispatches on is visible only in the source.
 * <p>
 * The statement holding the line is found by descending from the module: in each block, the statement holding it is the last one beginning
 * at or before it, and a compound statement's part holding it is told the same way from where each part begins, since the parser records
 * where a node begins but not where it ends. Entering a function or class body leaves the {@code try} statements around it behind, since a
 * call in that body runs when the function is called, not where it is defined. A line in a {@code try} statement's {@code else} clause or
 * an {@code except} clause is not guarded by that statement.
 * <p>
 * A clause names its classes directly, or through a name the module assigns, such as {@code ERRORS = (ValueError, TypeError)}, whose
 * assigned values are read in its place. A name the module never assigns is taken to be a class, as a builtin or an imported one is.
 */
public final class ExceptionHandlerAnalysis {

	private static final ILog LOG = getLog(ExceptionHandlerAnalysis.class);

	/** The name an {@code except} clause without a class is read as, since it catches every exception. */
	private static final String BARE_EXCEPT_NAME = "BaseException";

	/** The module's AST root, or {@code null} when it is not a module. */
	private final Module module;

	/** The values the module assigns to each plain name by a simple assignment, read once. */
	private final Map<String, List<exprType>> assignments;

	/**
	 * Reads the module whose {@code except} clauses are to be found.
	 *
	 * @param module The module's AST root.
	 */
	public ExceptionHandlerAnalysis(SimpleNode module) {
		this.module = module instanceof Module root ? root : null;
		this.assignments = this.module == null ? Map.of() : assignments(this.module);
	}

	/**
	 * Returns the {@code except} clauses whose {@code try} body holds {@code line} in the same scope, innermost statement first and each
	 * statement's clauses in order, which is the order Python tries them in.
	 *
	 * @param line A line of the module, from one.
	 * @return One guard per {@code except} clause, marked as a handler; empty when the line is guarded by none.
	 */
	List<ExpectedFailureContextAnalysis.Guard> handlersAround(int line) {
		if (this.module == null)
			return List.of();

		Deque<List<ExpectedFailureContextAnalysis.Guard>> enclosing = new ArrayDeque<>();
		this.descend(this.module.body, line, enclosing);

		List<ExpectedFailureContextAnalysis.Guard> ret = new ArrayList<>();
		enclosing.forEach(ret::addAll);
		return ret;
	}

	/**
	 * Descends into the statement of {@code block} holding {@code line}, pushing the clauses of each {@code try} statement whose body holds
	 * it onto {@code enclosing}, so the innermost is first.
	 */
	private void descend(stmtType[] block, int line, Deque<List<ExpectedFailureContextAnalysis.Guard>> enclosing) {
		stmtType statement = holding(block, line);

		if (statement == null || statement.beginLine == line && (statement instanceof FunctionDef || statement instanceof ClassDef))
			return;

		switch (statement) {
		case FunctionDef function -> {
			enclosing.clear();
			this.descend(function.body, line, enclosing);
		}
		case ClassDef type -> {
			enclosing.clear();
			this.descend(type.body, line, enclosing);
		}
		case TryExcept tryExcept -> {
			excepthandlerType[] handlers = tryExcept.handlers == null ? new excepthandlerType[0] : tryExcept.handlers;
			int handlerLine = handlers.length == 0 ? Integer.MAX_VALUE : handlers[0].beginLine;
			int elseLine = firstLine(tryExcept.orelse);

			if (line < handlerLine && line < elseLine) {
				enclosing.push(this.clauses(tryExcept.handlers));
				this.descend(tryExcept.body, line, enclosing);
			} else if (line >= elseLine)
				this.descend(tryExcept.orelse.body, line, enclosing);
			else
				for (int i = handlers.length - 1; i >= 0; i--)
					if (handlers[i].beginLine <= line) {
						this.descend(handlers[i].body, line, enclosing);
						break;
					}
		}
		case TryFinally tryFinally -> this.descend(line < firstLine(tryFinally.finalbody) ? tryFinally.body : tryFinally.finalbody.body,
				line, enclosing);
		case If conditional -> this.descend(line < firstLine(conditional.orelse) ? conditional.body : conditional.orelse.body, line,
				enclosing);
		case For loop -> this.descend(line < firstLine(loop.orelse) ? loop.body : loop.orelse.body, line, enclosing);
		case While loop -> this.descend(line < firstLine(loop.orelse) ? loop.body : loop.orelse.body, line, enclosing);
		case With with when with.body != null -> this.descend(with.body.body, line, enclosing);
		default -> {
			// Any other statement holds no nested block that a try could enclose the line in.
		}
		}
	}

	/** The last statement of {@code block} beginning at or before {@code line}, or {@code null} when there is none. */
	private static stmtType holding(stmtType[] block, int line) {
		stmtType ret = null;

		if (block != null)
			for (stmtType statement : block)
				if (statement != null && statement.beginLine <= line)
					ret = statement;

		return ret;
	}

	/** The line {@code suite}'s first statement begins at, or {@link Integer#MAX_VALUE} when it is absent or empty. */
	private static int firstLine(suiteType suite) {
		if (suite == null || suite.body == null || suite.body.length == 0 || suite.body[0] == null)
			return Integer.MAX_VALUE;

		return suite.body[0].beginLine;
	}

	/** One guard per {@code except} clause of {@code handlers}, in order. */
	private List<ExpectedFailureContextAnalysis.Guard> clauses(excepthandlerType[] handlers) {
		List<ExpectedFailureContextAnalysis.Guard> ret = new ArrayList<>();

		if (handlers != null)
			for (excepthandlerType handler : handlers) {
				if (handler.type == null) {
					ret.add(new ExpectedFailureContextAnalysis.Guard(Set.of(BARE_EXCEPT_NAME), true, true));
					continue;
				}

				Set<String> names = new HashSet<>();
				ret.add(this.addExceptionNames(handler.type, names, new HashSet<>())
						? new ExpectedFailureContextAnalysis.Guard(Set.copyOf(names), true, true)
						: ExpectedFailureContextAnalysis.Guard.UNRESOLVED_HANDLER);
			}

		return ret;
	}

	/**
	 * Adds the simple names of the exception classes {@code expression} denotes to {@code names}: a name ({@code ValueError}), an attribute
	 * ({@code tf.errors.InvalidArgumentError}), or a tuple of either. A name the module assigns denotes what it is assigned.
	 *
	 * @return False iff some class could not be named.
	 */
	private boolean addExceptionNames(exprType expression, Set<String> names, Set<String> seen) {
		if (expression instanceof Name name) {
			List<exprType> values = this.assignments.getOrDefault(name.id, List.of());

			if (values.isEmpty()) {
				names.add(name.id);
				return true;
			}

			if (!seen.add(name.id))
				return false;

			for (exprType value : values)
				if (!this.addExceptionNames(value, names, seen))
					return false;

			return true;
		}

		if (expression instanceof Attribute attribute && attribute.attr instanceof NameTok member) {
			names.add(member.id);
			return true;
		}

		if (expression instanceof Tuple tuple && tuple.elts != null) {
			for (exprType element : tuple.elts)
				if (!this.addExceptionNames(element, names, seen))
					return false;

			return true;
		}

		return false;
	}

	/**
	 * The values {@code module} assigns to each plain name anywhere in it, by a simple assignment {@code name = value}. A name bound
	 * otherwise, as by an import, a loop, or a tuple target, contributes no value here.
	 *
	 * @param module The module's AST root.
	 * @return The assigned values by name; a name with none is absent.
	 */
	private static Map<String, List<exprType>> assignments(Module module) {
		Map<String, List<exprType>> ret = new HashMap<>();

		try {
			module.accept(new VisitorBase() {

				@Override
				public Object visitAssign(Assign node) throws Exception {
					if (node.targets != null)
						for (exprType target : node.targets)
							if (target instanceof Name assigned)
								ret.computeIfAbsent(assigned.id, k -> new ArrayList<>()).add(node.value);

					return super.visitAssign(node);
				}

				@Override
				public void traverse(SimpleNode node) throws Exception {
					node.traverse(this);
				}

				@Override
				protected Object unhandled_node(SimpleNode node) throws Exception {
					return null;
				}
			});
		} catch (Exception e) {
			// An AST walk that fails leaves the names' bindings unknown, which is read as no assignment.
			LOG.warn("Can't read the module's assignments.", e);
			return Map.of();
		}

		return ret;
	}
}
