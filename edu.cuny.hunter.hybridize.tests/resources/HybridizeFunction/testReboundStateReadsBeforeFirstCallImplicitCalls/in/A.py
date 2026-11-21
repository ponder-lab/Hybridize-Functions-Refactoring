# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1165

# Code that Python runs with no call in the program. Each function fills the object it has just created, but something that reads the
# field may run between the creation and the write, or the object may share a trace with another.

import tensorflow as tf


# An operator whose method reads the field runs between the object's creation and the write.
class Adder:
    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend

    def __add__(self, x):
        return self.add(x)


def make_adder(x, addend):
    adder = Adder()
    print(adder + x)
    adder.addend = addend
    return adder


# A class like it, whose operator the program never applies.
class Unapplied:
    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend

    def __add__(self, x):
        return self.add(x)


def make_unapplied(addend):
    unapplied = Unapplied()
    unapplied.addend = addend
    return unapplied


# The same with the operator's method inherited.
class Operand:
    def __add__(self, x):
        return self.add(x)


class Derived(Operand):
    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend


def make_derived(x, addend):
    derived = Derived()
    print(derived + x)
    derived.addend = addend
    return derived


# The same with the operator's method assigned in the class's body.
def add_to(aliased, x):
    return aliased.add(x)


class Aliased:
    __add__ = add_to

    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend


def make_aliased(x, addend):
    aliased = Aliased()
    print(aliased + x)
    aliased.addend = addend
    return aliased


# A `with` statement enters the object, and the method it runs reads the field.
class Entered:
    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend

    def __enter__(self):
        print(self.add(tf.ones([2])))
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False


def make_entered(addend):
    entered = Entered()
    with entered:
        pass
    entered.addend = addend
    return entered


# The write runs a property's setter, which reads the field first.
class Settered:
    def __init__(self):
        self._addend = tf.constant(1.0)

    @property
    def addend(self):
        return self._addend

    @addend.setter
    def addend(self, value):
        print(self.add(tf.ones([2])))
        self._addend = value

    def add(self, x):
        return x + self.addend


def make_settered(addend):
    settered = Settered()
    settered.addend = addend
    return settered


# Objects that compare equal share a trace of a function they are passed to.
class Equal:
    def __init__(self):
        self.addend = tf.constant(1.0)

    def __eq__(self, other):
        return isinstance(other, Equal)

    def __hash__(self):
        return 7


def add_equal(equal, x):
    return x + equal.addend


def make_equal(addend):
    equal = Equal()
    equal.addend = addend
    return equal


# The same with no equality defined: each object has its own trace.
class Distinct:
    def __init__(self):
        self.addend = tf.constant(1.0)


def add_distinct(distinct, x):
    return x + distinct.addend


def make_distinct(addend):
    distinct = Distinct()
    distinct.addend = addend
    return distinct


# An explicit call that reads the field between the creation and the write.
class Called:
    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend


def make_called(x, addend):
    called = Called()
    print(called.add(x))
    called.addend = addend
    return called


# The same with a call of the object itself, which runs `__call__`.
class Invoked:
    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend

    def __call__(self, x):
        return self.add(x)


def make_invoked(x, addend):
    invoked = Invoked()
    print(invoked(x))
    invoked.addend = addend
    return invoked


# The same with a read of a property, whose getter reads the field.
class Summed:
    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend

    @property
    def preview(self):
        return self.add(tf.ones([2]))


def make_summed(addend):
    summed = Summed()
    print(summed.preview)
    summed.addend = addend
    return summed


# No special method, and nothing between the creation and the write.
class Plain:
    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend


def make_plain(addend):
    plain = Plain()
    plain.addend = addend
    return plain


# A class defining only `__call__`, with nothing between the creation and the write.
class Applied:
    def __init__(self):
        self.addend = tf.constant(1.0)

    def add(self, x):
        return x + self.addend

    def __call__(self, x):
        return self.add(x)


def make_applied(addend):
    applied = Applied()
    applied.addend = addend
    return applied


x = tf.ones([2])
print(make_adder(x, tf.constant(2.0)).add(x))
unapplied = make_unapplied(tf.constant(2.0))
print(unapplied.add(x))
print(make_derived(x, tf.constant(2.0)).add(x))
aliased = make_aliased(x, tf.constant(2.0))
print(aliased.add(x))
print(make_entered(tf.constant(2.0)).add(x))
settered = make_settered(tf.constant(2.0))
print(settered.add(x))
first_equal = make_equal(tf.constant(2.0))
second_equal = make_equal(tf.constant(3.0))
print(add_equal(first_equal, x))
print(add_equal(second_equal, x))
first_distinct = make_distinct(tf.constant(2.0))
second_distinct = make_distinct(tf.constant(3.0))
print(add_distinct(first_distinct, x))
print(add_distinct(second_distinct, x))
print(make_called(x, tf.constant(2.0)).add(x))
invoked = make_invoked(x, tf.constant(2.0))
print(invoked.add(x))
print(invoked(x))
print(make_summed(tf.constant(2.0)).add(x))
plain = make_plain(tf.constant(2.0))
print(plain.add(x))
applied = make_applied(tf.constant(2.0))
print(applied.add(x))
print(applied(x))
