import tensorflow as tf
import numpy as np


# Build a dataflow graph.
mat1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
mat2 = tf.constant([[1.0, 1.0], [0.0, 1.0]])

op_mat_mul = tf.matmul(mat1, mat2)
op_mat_add = tf.add(mat1, mat2)

# Construct a `Session` to execute the graph.
sess = tf.Session()

# Execute the graph and store the value that `e` represents in `result`.
result_mul = sess.run(op_mat_mul)
result_add = sess.run(op_mat_add)

#print the matrix shape
print(mat1.shape)

#print tensor multiplication result
print(result_mul)

#print tensor addition result
print(result_add)
