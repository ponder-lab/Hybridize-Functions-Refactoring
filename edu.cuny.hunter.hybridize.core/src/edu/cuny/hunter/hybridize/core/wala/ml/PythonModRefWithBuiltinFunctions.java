package edu.cuny.hunter.hybridize.core.wala.ml;

import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.string;

import java.util.Collection;

import com.ibm.wala.cast.ipa.callgraph.AstGlobalPointerKey;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.modref.PythonModRef;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.intset.OrdinalSet;

public class PythonModRefWithBuiltinFunctions extends PythonModRef {

	public static class PythonModVisitorWithBuiltinFunctions<T extends InstanceKey> extends PythonModVisitor<T> {

		public static final AstGlobalPointerKey GLOBAL_OUTPUT_STREAM_POINTER_KEY = new AstGlobalPointerKey(
				PythonModVisitorWithBuiltinFunctions.class.getPackageName().replace('.', '/') + "/OUT");

		/**
		 * A fake global variable representing the file system, which a call to the built-in {@code open} reads or writes. Under
		 * {@code tf.function}, the call runs once, while the function is traced, so a file read on every eager call is read on the first
		 * call only, and a write happens only then (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1077).
		 */
		public static final AstGlobalPointerKey GLOBAL_FILE_SYSTEM_POINTER_KEY = new AstGlobalPointerKey(
				PythonModVisitorWithBuiltinFunctions.class.getPackageName().replace('.', '/') + "/FS");

		private static final String PRINT_FUNCTION_VARIABLE_NAME = "print";

		private static final TypeReference PRINT_FUNCTION_TYPE_REFERENCE = TypeReference.findOrCreate(PythonTypes.pythonLoader,
				"Lwala/builtin/" + PRINT_FUNCTION_VARIABLE_NAME);

		/** The type of Ariadne's object for the built-in {@code open} function. */
		private static final TypeReference OPEN_FUNCTION_TYPE_REFERENCE = TypeReference.findOrCreate(PythonTypes.pythonLoader,
				"Lwala/builtin/open");

		public PythonModVisitorWithBuiltinFunctions(CGNode n, Collection<PointerKey> result, ExtendedHeapModel h, PointerAnalysis<T> pa,
				boolean ignoreAllocHeapDefs) {
			super(n, result, h, pa, ignoreAllocHeapDefs);
		}

		@Override
		public void visitPythonInvoke(PythonInvokeInstruction inst) {
			int use = inst.getUse(0); // a reference to the invoked function.
			SSAInstruction def = this.n.getDU().getDef(use); // the definition of the invoked function.

			// A call to the built-in `open`, however the function reaches the call, reads or writes the file system. The callee is
			// recognized by the object it points to, so a script that rebinds `open` to its own function is not mistaken for it.
			for (T ik : this.pa.getPointsToSet(this.h.getPointerKeyForLocal(this.n, use)))
				if (getTypeReference(ik).equals(OPEN_FUNCTION_TYPE_REFERENCE)) {
					this.result.add(GLOBAL_FILE_SYSTEM_POINTER_KEY);
					break;
				}

			if (def instanceof AstLexicalRead) {
				AstLexicalRead read = (AstLexicalRead) def;
				Access[] accesses = read.getAccesses();

				// Use the getName() accessor rather than the variableName field: WALA 1.8.0 encapsulated the field (made it private).
				if (accesses.length > 0 && accesses[0].getName().fst.equals(PRINT_FUNCTION_VARIABLE_NAME)) {
					PointerKey pk = this.h.getPointerKeyForLocal(this.n, use);
					OrdinalSet<T> pointsToSet = this.pa.getPointsToSet(pk);

					pointsToSet.forEach(ik -> {
						TypeReference typeReference = getTypeReference(ik);

						if (typeReference.equals(PRINT_FUNCTION_TYPE_REFERENCE)) {
							// found a call to the built-in print function, which has side effects.
							// add a pointer to a fake global variable representing a modification to the output stream.
							this.result.add(GLOBAL_OUTPUT_STREAM_POINTER_KEY);
						}
					});
				}
			} else if (def instanceof PythonPropertyRead) {
				PythonPropertyRead read = (PythonPropertyRead) def;
				int memberRef = read.getMemberRef();
				PointerKey memberRefPK = this.h.getPointerKeyForLocal(this.n, memberRef);
				OrdinalSet<T> memberRefPointsToSet = this.pa.getPointsToSet(memberRefPK);

				memberRefPointsToSet.forEach(memberRefIK -> {
					TypeReference typeReference = getTypeReference(memberRefIK);

					if (typeReference.equals(string) && memberRefIK instanceof ConstantKey) {
						ConstantKey<?> ck = (ConstantKey<?>) memberRefIK;
						Object value = ck.getValue();

						if (value.equals("append")) {
							// check that the receiver is of type list.
							int objectRef = read.getObjectRef();
							PointerKey objectRefPK = this.h.getPointerKeyForLocal(this.n, objectRef);
							OrdinalSet<T> objectRefPointsToSet = this.pa.getPointsToSet(objectRefPK);

							objectRefPointsToSet.forEach(objectRefIK -> {
								if (objectRefIK.concreteType().getReference().equals(list))
									// it's a list. Add the pointer to the results.
									this.result.add(objectRefPK);
							});
						}
					}
				});
			}

			super.visitPythonInvoke(inst);
		}

		private TypeReference getTypeReference(T ik) {
			return ik.concreteType().getReference();
		}
	}

	@Override
	protected ModVisitor<InstanceKey, ? extends ExtendedHeapModel> makeModVisitor(CGNode n, Collection<PointerKey> result,
			PointerAnalysis<InstanceKey> pa, ExtendedHeapModel h, boolean ignoreAllocHeapDefs) {
		return new PythonModVisitorWithBuiltinFunctions<>(n, result, h, pa, ignoreAllocHeapDefs);
	}
}
