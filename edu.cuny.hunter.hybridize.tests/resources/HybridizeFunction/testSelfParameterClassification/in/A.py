# Regression fixture for #498. A method `f` with only `self` as parameter. After
# `Parameter.classifyAsTensor` runs, `self.isSelf() == TRUE` and `self.isTensor() == FALSE`
# (the early-return in classifyAsTensor for self params). The owning function has only one
# parameter (self), so `Function.getHasTensorParameter() == FALSE` (the speculative-context
# branch is gated on `!onlySelfParam`).


class C:
    def f(self):
        pass


C().f()
