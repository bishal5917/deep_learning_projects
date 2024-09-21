import tensorflow as tf
import numpy as np

# creating tensors
tensor_zero_d = tf.constant(1)
print(tensor_zero_d)

tensor_one_d = tf.constant([1,2,3])
print(tensor_one_d)

tensor_two_d = tf.constant([[1,2,3],[4,5,6]])
print(tensor_two_d)

tensor_three_d = tf.constant(
    [[[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]]])
print(tensor_three_d)

# converting numpy array to tensors
np_array = np.array([1,2,3,4])
print(np_array)

converted_tensor = tf.convert_to_tensor(np_array)
print(converted_tensor)

# eye tensor i.e identity matrix
eye_tensor = tf.eye(
    num_rows=3,
    num_columns=None,
    batch_shape=None,
    dtype=tf.int32,
    name=None 
)
print(eye_tensor)

# fill tensor
fill_tensor = tf.fill([2,3],9)
print(fill_tensor)

# uses normal distribution to generate random values
random_tensor = tf.random.normal(
    [3,3],
    mean = 100.0,
    stddev = 3.0,
    dtype = tf.float32,
    seed = None,
    name = None
)
print(random_tensor)

#uses uniform distribution to generate random values
random_tensor_uniform = tf.random.uniform(
    [3,],
    minval = 0,
    maxval = 100,
    dtype = tf.float32,
    seed = None,
    name = None
)
print(random_tensor_uniform)
