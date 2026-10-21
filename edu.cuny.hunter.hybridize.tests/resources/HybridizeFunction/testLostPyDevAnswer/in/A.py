import lost_answer_aborts


@lost_answer_aborts.decorate
def f(x):
    return x


print(f(1))
