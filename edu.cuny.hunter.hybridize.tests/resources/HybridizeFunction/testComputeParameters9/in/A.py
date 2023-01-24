import tensorflow as tf

     
@tf.function(experimental_relax_shapes=True)
def func():
    print("Testing")

 
if __name__ == '__main__':
    func()
    
