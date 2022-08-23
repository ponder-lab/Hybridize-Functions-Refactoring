package edu.cuny.hunter.hybridize.ui.plugins;

import org.eclipse.core.runtime.Plugin;
import org.osgi.framework.BundleContext;

public class EvaluateHybridizePlugin extends Plugin {

	private static EvaluateHybridizePlugin plugin;

	public static Plugin getDefault() {
		return plugin;
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