# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1165

# Fields of an object that a class attribute, or a field of an instance, holds. Each object is filled as part of its construction, but
# what holds it is rebound to another object.

import tensorflow as tf


# A class attribute holds an object that the function creating it fills. The class attribute is then rebound.
class Setting:
    pass


def make_setting(scale):
    setting = Setting()
    setting.scale = scale
    return setting


class Scaler:
    setting = make_setting(tf.constant(1.0))

    def scale(self, x):
        return x * self.setting.scale


# The same with a Python scalar.
class Factor:
    pass


def make_factor(scale):
    factor = Factor()
    factor.scale = scale
    return factor


class Multiplier:
    factor = make_factor(1.0)

    def scale(self, x):
        return x * self.factor.scale


# The same with an object that its initializer fills.
class Initialized:
    def __init__(self, scale):
        self.scale = scale


class Initializing:
    setting = Initialized(tf.constant(1.0))

    def scale(self, x):
        return x * self.setting.scale


# The same through a field of the instance, which the program rebinds on the instance.
class Lodged:
    pass


def make_lodged(scale):
    lodged = Lodged()
    lodged.scale = scale
    return lodged


class Lodging:
    def __init__(self):
        self.setting = make_lodged(tf.constant(1.0))

    def scale(self, x):
        return x * self.setting.scale


# A class attribute that is never rebound.
class Fixed:
    pass


def make_fixed(scale):
    fixed = Fixed()
    fixed.scale = scale
    return fixed


class Fixing:
    setting = make_fixed(tf.constant(1.0))

    def scale(self, x):
        return x * self.setting.scale


x = tf.ones([2])

scaler = Scaler()
print(scaler.scale(x))
Scaler.setting = make_setting(tf.constant(2.0))
print(scaler.scale(x))

multiplier = Multiplier()
print(multiplier.scale(x))
Multiplier.factor = make_factor(2.0)
print(multiplier.scale(x))

initializing = Initializing()
print(initializing.scale(x))
Initializing.setting = Initialized(tf.constant(2.0))
print(initializing.scale(x))

lodging = Lodging()
print(lodging.scale(x))
lodging.setting = make_lodged(tf.constant(2.0))
print(lodging.scale(x))

fixing = Fixing()
print(fixing.scale(x))
print(fixing.scale(x))
