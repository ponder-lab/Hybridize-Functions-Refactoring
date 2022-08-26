/**
 *
 */
package edu.cuny.hunter.hybridize.core.messages;

import org.eclipse.osgi.util.NLS;

/**
 * @author <a href="mailto:raffi.khatchadourian@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Messages extends NLS {
	private static final String BUNDLE_NAME = "edu.cuny.hunter.hybridize.core.messages.messages"; //$NON-NLS-1$
	public static String CategoryDescription;
	public static String CategoryName;
	public static String CheckingFunctions;
	public static String CheckingPreconditions;
	public static String CreatingChange;
	public static String FunctionsNotSpecified;
	public static String Name;
	public static String NoFunctionsHavePassedThePreconditions;
	public static String NoFunctionsToConvert;
	public static String NoFunctionsToOptimize;
	public static String PreconditionFailed;
	public static String RefactoringNotPossible;

	static {
		// initialize resource bundle
		NLS.initializeMessages(BUNDLE_NAME, Messages.class);
	}

	private Messages() {
	}
}
