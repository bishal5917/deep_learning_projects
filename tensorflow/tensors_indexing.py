import tensorflow as tf

#For one dimensional
tensor_one_d = tf.constant([1,2,3,4,5,6])
print(tensor_one_d)
#indexing to get first four
print(tensor_one_d[0:4])
print(tensor_one_d[3:])

#for two dimensional
tensor_two_d = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
#first three rows and first two columns
print(tensor_two_d[0:3,0:2])
#first three rows and all the columns
print(tensor_two_d[0:3,:])

#lets do for three dimensional tensors
tensor_three_d = tf.constant(
    [[[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]]])
print(tensor_three_d[0:2,0:2,0:2])