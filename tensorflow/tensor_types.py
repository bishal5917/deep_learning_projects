import tensorflow as tf

#ragged tensors
t = [[1,2,3],[4,5],[6,7,8]]
t_ragged = tf.ragged.constant(t)
print(t_ragged)
print(t_ragged.shape)

#sparse tensor
#working with zero values
tensor_sparse = tf.sparse.SparseTensor(
    indices=[[1,1],[3,4]],values=[11,56],dense_shape=[5,6]
)
print(tensor_sparse)
print(tf.sparse.to_dense(tensor_sparse))

#string sensors
tensor_string = tf.constant(["Hello","I","am","coder"])
print(tensor_string)
print(tf.strings.join(tensor_string,separator=""))

# VARIABLES
x= tf.constant([1,2])
x_var = tf.Variable(x,name="var1")
print(x_var)
#subtracting
print(x_var.assign_sub([1,2]))
print(x_var.assign_add([2,2]))

#Defining the scopes, CPUs or GPUs
with tf.device('CPU:0'):
    x_1=tf.constant([1,1])
    x_2=tf.constant([3])

with tf.device('GPU:0'):
    x_sum = x_1 + x_2

print(x_1,x_1.device)
print(x_2,x_2.device)
print(x_sum,x_sum.device)
