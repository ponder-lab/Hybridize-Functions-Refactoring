package edu.cuny.hunter.hybridize.core.wala.ml;

import java.util.Collection;

import com.ibm.wala.cast.ipa.callgraph.AstGlobalPointerKey;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.modref.PythonModRef;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.intset.OrdinalSet;

public class PythonModRefWithBuiltinFunctions extends PythonModRef {

	public static class PythonModVisitorWithBuiltinFunctions<T extends InstanceKey> extends PythonModVisitor<T> {

		private static final AstGlobalPointerKey GLOBAL_OUTPUT_STREAM_POINTER_KEY = new AstGlobalPointerKey(
				PythonModVisitorWithBuiltinFunctions.class.getPackageName().replace('.', '/') + "/OUT");

		private static final String PRINT_FUNCTION_VARIABLE_NAME = "print";

		private static final TypeReference PRINT_FUNCTION_TYPE_REFERENCE = TypeReference.findOrCreate(PythonTypes.pythonLoader,
				"Lwala/builtin/" + PRINT_FUNCTION_VARIABLE_NAME);

		public PythonModVisitorWithBuiltinFunctions(CGNode n, Collection<PointerKey> result, ExtendedHeapModel h, PointerAnalysis<T> pa,
				boolean ignoreAllocHeapDefs) {
			super(n, result, h, pa, ignoreAllocHeapDefs);
		}

		@Override
		public void visitPythonInvoke(PythonInvokeInstruction inst) {
			int use = inst.getUse(0); // a reference to the invoked function.
			SSAInstruction def = this.n.getDU().getDef(use); // the definition of the invoked function.

			if (def instanceof AstLexicalRead) {
				AstLexicalRead read = (AstLexicalRead) def;
				Access[] accesses = read.getAccesses();

				if (accesses.length > 0 && accesses[0].variableName.equals(PRINT_FUNCTION_VARIABLE_NAME)) {
					PointerKey pk = this.h.getPointerKeyForLocal(this.n, use);
					OrdinalSet<T> pointsToSet = this.pa.getPointsToSet(pk);

					pointsToSet.forEach(ik -> {
						IClass concreteType = ik.getConcreteType();
						TypeReference typeReference = concreteType.getReference();

						if (typeReference.equals(PRINT_FUNCTION_TYPE_REFERENCE)) {
							// found a call to the built-in print function, which has side effects.
							// add a pointer to a fake global variable representing a modification to the output stream.
							this.result.add(GLOBAL_OUTPUT_STREAM_POINTER_KEY);
						}
					});
				}
			}

			super.visitPythonInvoke(inst);
		}
	}

	@Override
	protected ModVisitor<InstanceKey, ? extends ExtendedHeapModel> makeModVisitor(CGNode n, Collection<PointerKey> result,
			PointerAnalysis<InstanceKey> pa, ExtendedHeapModel h, boolean ignoreAllocHeapDefs) {
		return new PythonModVisitorWithBuiltinFunctions<>(n, result, h, pa, ignoreAllocHeapDefs);
	}
}
