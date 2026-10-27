# From https://docs.pytest.org/en/8.0.x/getting-started.html#create-your-first-test.


# content of test_sample.py
def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5


def func2(x):
    return x + 1


def test_answer2():
    assert func2(3) == 4
    assert func2(4) == 5
