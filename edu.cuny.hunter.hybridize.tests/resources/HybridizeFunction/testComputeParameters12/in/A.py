import custom


@custom.decorator(input_signature=None)
def func(x):
  print('Tracing with', x)
  return x

 
if __name__ == '__main__':
    func(1)
    
