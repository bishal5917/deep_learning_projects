import tensorflow as tf

#absolute value
oned_arr = tf.constant([1,2,-3,4,5])
print(tf.abs(oned_arr))

#abs of complex value,calculated as underroot of a2+b2
tensor_complex = tf.constant([4+0j])
print(tf.abs(tensor_complex))

#normal addition and multiplication
x_1 = tf.constant([1,2,3,4,5,6],dtype = tf.int32)
x_2 = tf.constant([1,2,3,4,5,6],dtype = tf.int32)
print(tf.add(x_1,x_2))
print(tf.subtract(x_1,x_2))
print(tf.multiply(x_1,x_2))
print(tf.divide(x_1,x_2))

#maximum and minimum value index in 1d arr
oned_arr = tf.constant([1,2,3,4,5])
print(tf.argmax(oned_arr))
print(tf.argmin(oned_arr))

#for two dimensional
tensor_two_d = tf.constant([[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]])
print(tf.argmax(tensor_two_d,0))
print(tf.argmin(tensor_two_d,0))

#power function
print(tf.pow(tf.constant(2),tf.constant(3)))

#power functions
#lets do for three dimensional tensors
tensor_a = tf.constant([[1,2],[4,5]])
tensor_b = tf.constant([[1,2],[0,0]]) 
print(tf.pow(tensor_a,tensor_b))

#reduce
tensor_twod = tf.constant([[1,2,8],[4,5,5]])
print(tf.reduce_sum(tensor_twod,axis=0,keepdims=True,name=None))
print(tf.reduce_max(tensor_twod,axis=0,keepdims=True,name=None))
print(tf.reduce_min(tensor_twod,axis=0,keepdims=True,name=None))

