# Regression fixture pinning that `inferInputSignature` emits one INFO per blocking parameter when
# multiple non-tensor parameters block inference, rather than returning after the first blocker.
# Both `m` and `n` are Python `int` literals at the call site; neither is classified as tensor-typed.
# Per #508 category (a), each parameter must surface its own source-side recovery suggestion.
def f(m, n):
    return m + n


f(3, 5)
