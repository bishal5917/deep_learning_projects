import tensorflow as tf

#absolute value
mat_x = tf.constant([[1,2],[4,5]])
mat_y = tf.constant([[1,1],[1,1]])
print(mat_x.shape,mat_y.shape)
print(tf.linalg.matmul(
    mat_x,
    mat_y,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,
    name=None
))

#band part
tensor_twod = tf.constant([[1,2,3],
                           [4,5,6],
                           [3,2,1]])
print(tf.linalg.band_part(tensor_twod,0,-1))
print(tf.linalg.band_part(tensor_twod,-1,0))
print(tf.linalg.band_part(tensor_twod,0,0))

#matrix inverse
mat_x = tf.constant([[1,2],[4,5]],dtype=tf.float32)
mat_x_inv = tf.linalg.inv(mat_x)
print(mat_x@mat_x_inv) # We will get identity matrix
