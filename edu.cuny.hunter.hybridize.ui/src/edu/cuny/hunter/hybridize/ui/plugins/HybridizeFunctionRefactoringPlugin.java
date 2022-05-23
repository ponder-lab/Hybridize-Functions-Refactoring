package edu.cuny.hunter.hybridize.ui.plugins;

import org.osgi.framework.BundleContext;

import edu.cuny.citytech.refactoring.common.ui.RefactoringPlugin;
import edu.cuny.hunter.hybridize.core.descriptors.HybridizeFunctionRefactoringDescriptor;

public class HybridizeFunctionRefactoringPlugin extends RefactoringPlugin {

	private static HybridizeFunctionRefactoringPlugin plugin;

	public static RefactoringPlugin getDefault() {
		return plugin;
	}

	@Override
	protected String getRefactoringId() {
		return HybridizeFunctionRefactoringDescriptor.REFACTORING_ID;
	}

	@Override
	public void start(BundleContext context) throws Exception {
		plugin = this;
		super.start(context);
	}

	@Override
	public void stop(BundleContext context) throws Exception {
		plugin = null;
		super.stop(context);
	}
}
