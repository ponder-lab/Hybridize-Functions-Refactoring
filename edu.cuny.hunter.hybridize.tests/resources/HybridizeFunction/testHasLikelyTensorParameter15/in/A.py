import tensorflow as tf

# @tf.function
def double(a):
  print("Tracing with", a)
  return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()

# This doesn't print 'Tracing with ...'
print(double(tf.constant("b")))

# print(double.pretty_printed_concrete_signatures())