import tensorflow as tf

#expand dims
tensor_three_d = tf.constant(
    [[[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]]])
print(tensor_three_d)
#by expanding dimensions, 3D Tensor converted to 4D Tensor
print(tf.expand_dims(tensor_three_d,axis=0).shape)

#squeeze
tensor_one_d = tf.constant([[1,2,3]])
print(tensor_one_d)
tensor_oned_expanded = tf.expand_dims(tensor_one_d,axis=0)
print(tensor_oned_expanded.shape)
#Now using squeeze, we can reduce the dimensions
print(tf.squeeze(tensor_oned_expanded,axis=0))

#reshaping
x_original = tf.constant(
    [[[1,2,3],
     [4,5,6]]])
#Note: Doing this doesnot transposes.
print(tf.reshape(x_original,[3,2]))

#concat function
mat_x = tf.constant(
    [[[1,2,3],
     [4,5,6]]])
mat_y = tf.constant(
    [[[1,2,3],
     [4,5,6]]])
tf.concat([mat_x,mat_y],axis=0)

#paddings
t = tf.constant([[1,2,3],[4,5,6]])
paddings = tf.constant([[1,1],[2,2]])
print(tf.pad(t,paddings,"CONSTANT"))
print(tf.pad(t,paddings,"REFLECT"))
print(tf.pad(t,paddings,"SYMMETRIC"))

#Using gather method to obtain same as indexing
t = tf.constant(['t0','t1','t2','t3','t4','t5'])
print(tf.gather(t,[2,3,4]))