package edu.cuny.hunter.hybridize.core.wala.ml;

import java.util.Collection;

import com.ibm.wala.cast.ir.ssa.AstGlobalWrite;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.python.modref.PythonModRef;
import com.ibm.wala.cast.python.modref.PythonModRef.PythonModVisitor;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.shrike.shrikeBT.IInvokeInstruction.IDispatch;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAInvokeInstruction;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.intset.OrdinalSet;

public class PythonModRefWithBuiltinFunctions extends PythonModRef {

	public static class PythonModVisitorWithBuiltinFunctions<T extends InstanceKey> extends PythonModVisitor<T> {

		public PythonModVisitorWithBuiltinFunctions(CGNode n, Collection<PointerKey> result, ExtendedHeapModel h, PointerAnalysis<T> pa,
				boolean ignoreAllocHeapDefs) {
			super(n, result, h, pa, ignoreAllocHeapDefs);
		}

		@Override
		public void visitPythonInvoke(PythonInvokeInstruction inst) {
			int use = inst.getUse(0);
			SSAInstruction def = this.n.getDU().getDef(use);

			if (def instanceof AstLexicalRead) {
				AstLexicalRead read = (AstLexicalRead) def;
				Access[] accesses = read.getAccesses();

				if (accesses.length > 0 && accesses[0].variableName.equals("str")) {
					PointerKey pk = this.h.getPointerKeyForLocal(this.n, use);
					OrdinalSet<T> pointsToSet = this.pa.getPointsToSet(pk);

					pointsToSet.forEach(o -> {
						if (o instanceof InstanceKey) {
							InstanceKey ik = (InstanceKey) o;
							IClass concreteType = ik.getConcreteType();
							TypeReference reference = concreteType.getReference();

							TypeReference strFunction = TypeReference.findOrCreate(PythonTypes.pythonLoader, "Lwala/builtin/str");
							System.out.println(reference.equals(strFunction));
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
