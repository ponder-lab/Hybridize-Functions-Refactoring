package edu.cuny.hunter.hybridize.core.wala.ml;

import org.eclipse.core.resources.IProject;

import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;

/**
 * @author <a href="mailto:rk1424@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class EclipsePythonProjectTensorAnalysisEngine extends PythonTensorAnalysisEngine {

	/**
	 * 
	 */
	public EclipsePythonProjectTensorAnalysisEngine() {
		// TODO Auto-generated constructor stub
	}

	public EclipsePythonProjectTensorAnalysisEngine(IProject project) {
		// TODO Auto-generated constructor stub
	}

}
