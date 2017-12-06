import unittest
import numpy as np
import tensorflow as tf


from beam_search_optimization import get_entry


def get_entry_np(t, indices_1d, indices_2d, batch_size):
    result = np.zeros(batch_size)
    for i in range(batch_size):
        result[i] = t[i, indices_1d[i], indices_2d[i]]
    return result


class TestBSO(unittest.TestCase):

    def test_get_entry_simple(self):
        num_tests = 10
        sucess = True
        for _ in range(num_tests):
            batch_size, beam_size, vocab_size = map(int, np.random.randint(low=2, high=100, size=3))
            test_input = np.random.random([batch_size, beam_size, vocab_size])
            test_indices_1d = np.random.randint(low=0, high=beam_size-1, size=[batch_size])
            test_indices_2d = np.random.randint(low=0, high=vocab_size-1, size=[batch_size])
            test_result = get_entry_np(test_input, test_indices_1d, test_indices_2d, batch_size)
            with tf.Session() as sess:
                tf_input = tf.constant(test_input, dtype=tf.float32)
                tf_indices_1d = tf.constant(test_indices_1d, dtype=tf.int32)
                tf_indices_2d = tf.constant(test_indices_2d, dtype=tf.int32)
                tf_result = get_entry(tf_input, tf_indices_1d, tf_indices_2d)
                tf_result = sess.run(tf_result)
                sucess = sucess and np.allclose(test_result, tf_result)

        self.assertEqual(sucess, True)


if __name__ == '__main__':
    unittest.main()



