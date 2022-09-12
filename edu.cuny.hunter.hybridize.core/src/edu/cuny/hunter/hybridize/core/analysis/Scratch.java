				if (decorator.func instanceof Attribute) { // e.g., tf.function
					Attribute decoratorFunction = (Attribute) decorator.func;

					if (decoratorFunction.value instanceof Name) {
						Name decoratorName = (Name) decoratorFunction.value;
						// We have a viable prefix. Get the attribute.
						if (decoratorName.id.equals("tf") && decoratorFunction.attr instanceof NameTok) {
							NameTok decoratorAttribute = (NameTok) decoratorFunction.attr;
							if (decoratorAttribute.id.equals("function")) {
								// Found "tf.function."
								
							}
						}
					}
				} else if (decorator.func instanceof Call) { // e.g., tf.function(...)
					Call decoratorFunction = (Call) decorator.func;
					if (decoratorFunction.func instanceof Attribute) {
						Attribute callFunction = (Attribute) decoratorFunction.func;
						if (callFunction.value instanceof Name) {
							Name decoratorName = (Name) callFunction.value;
							// We have a viable prefix. Get the attribute.
							if (decoratorName.id.equals("tf") && callFunction.attr instanceof NameTok) {
								NameTok decoratorAttribute = (NameTok) callFunction.attr;
								if (decoratorAttribute.id.equals("function")) {
									// Found tf.function(...)
									this.isHybrid = true;
									LOG.info(this + " is hybrid.");
								}
							}
						}
					}
				}
