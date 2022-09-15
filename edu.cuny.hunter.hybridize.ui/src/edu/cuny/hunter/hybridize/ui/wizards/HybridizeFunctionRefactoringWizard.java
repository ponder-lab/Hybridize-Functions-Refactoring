package edu.cuny.hunter.hybridize.ui.wizards;

import static edu.cuny.hunter.hybridize.core.messages.Messages.Name;

import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jdt.internal.ui.refactoring.RefactoringMessages;
import org.eclipse.jdt.internal.ui.refactoring.actions.RefactoringStarter;
import org.eclipse.jdt.ui.refactoring.RefactoringSaveHelper;
import org.eclipse.ltk.core.refactoring.Refactoring;
import org.eclipse.ltk.core.refactoring.participants.RefactoringProcessor;
import org.eclipse.ltk.ui.refactoring.RefactoringWizard;
import org.eclipse.swt.widgets.Shell;
import org.python.pydev.parser.jython.ast.FunctionDef;

import edu.cuny.citytech.refactoring.common.ui.InputPage;
import edu.cuny.hunter.hybridize.core.messages.Messages;
import edu.cuny.hunter.hybridize.core.refactorings.HybridizeFunctionRefactoringProcessor;

public class HybridizeFunctionRefactoringWizard extends RefactoringWizard {

	private static class HybridizeFunctionsInputPage extends InputPage {

		private static final String DESCRIPTION = Messages.Name;

		private static final String DIALOG_SETTING_SECTION = "HybridizeFunctions"; // $NON-NLS-1$

		private static final String DO_IT_LABEL_TITLE = "Hybridize imperative TensorFlow functions for increased efficiency.";

		public static final String PAGE_NAME = "HybridizeFunctionsInputPage"; // $NON-NLS-1$

		@SuppressWarnings("unused")
		private HybridizeFunctionRefactoringProcessor processor;

		public HybridizeFunctionsInputPage() {
			super(PAGE_NAME);
			this.setDescription(DESCRIPTION);
		}

		@Override
		protected String getDialoGSettingSectionTitle() {
			return DIALOG_SETTING_SECTION;
		}

		@Override
		protected String getDoItLabelTitle() {
			return DO_IT_LABEL_TITLE;
		}

		@Override
		protected String getHelpContextID() {
			return "hybridize_functions_wizard_page_context";
		}

		@Override
		protected void setProcessor(RefactoringProcessor processor) {
			if (!(processor instanceof HybridizeFunctionRefactoringProcessor))
				throw new IllegalArgumentException("Expecing HybridizeFunctionRefactoringProcessor.");
			this.processor = (HybridizeFunctionRefactoringProcessor) processor;
		}
	}

	public static void startRefactoring(FunctionDef[] functions, Shell shell, IProgressMonitor monitor) {
		Refactoring refactoring = edu.cuny.hunter.hybridize.core.utils.Util.createRefactoring(functions, monitor);
		RefactoringWizard wizard = new HybridizeFunctionRefactoringWizard(refactoring);

		new RefactoringStarter().activate(wizard, shell, RefactoringMessages.OpenRefactoringWizardAction_refactoring,
				RefactoringSaveHelper.SAVE_REFACTORING);
	}

	public HybridizeFunctionRefactoringWizard(Refactoring refactoring) {
		super(refactoring, DIALOG_BASED_USER_INTERFACE & CHECK_INITIAL_CONDITIONS_ON_OPEN);
		this.setWindowTitle(Name);
	}

	@Override
	protected void addUserInputPages() {
		this.addPage(new HybridizeFunctionsInputPage());
	}
}
