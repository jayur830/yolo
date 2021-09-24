import tensorflow as tf

if __name__ == '__main__':
    a = tf.convert_to_tensor([.5, .456, 0., 1.], dtype=tf.float32)
    print(a)
    a += tf.cast(tf.logical_not(tf.cast(a, dtype=tf.bool)), dtype=tf.float32)
    print(a)
