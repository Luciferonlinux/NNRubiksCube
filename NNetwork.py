import tensorflow as tf

cubelist = [1, 3, 5, 1, 6, 4, 3, 2, 2, 4, 1, 5]


def convtotensor(arg):
    """
    Takes any list or tuple or nparray and converts it to a tf.Tensor
    """
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


print(tf.config.list_physical_devices())
# cubetensor = convtotensor(cubelist)
# print(create_model(cube))
