# Regression fixture for #498 (reverse direction). The function `f` takes two non-tensor
# parameters (ints from a non-tensor call site). After `Parameter.classifyAsTensor` runs,
# both parameters have `isTensor() == FALSE` and `Function.getHasTensorParameter() == FALSE`.
# The function name `f` does not match the speculative-context regex, so the speculative
# override doesn't fire. Pins the reverse direction of the classifier→reflection contract:
# Function.getHasTensorParameter() == FALSE ⇒ no non-self parameter has isTensor() == TRUE.


def f(a, b):
    return a + b


f(1, 2)
