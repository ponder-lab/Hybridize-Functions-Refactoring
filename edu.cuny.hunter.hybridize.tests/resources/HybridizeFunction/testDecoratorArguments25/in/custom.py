def decorator(input_signature=None):

  def decorated(inner_function):

      def wrapper(*args, **kwargs):
          result = function(*args, **kwargs)
          return result

      return decorated

  return decorator