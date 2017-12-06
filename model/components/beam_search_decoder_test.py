import unittest
import numpy as np
import tensorflow as tf


from beam_search_decoder import gather_helper


def gather_helper_np(t, indices, batch_size, beam_size, vocab_size):
    result = np.zeros([batch_size, beam_size, vocab_size])
    for i in range(batch_size):
        for j in range(beam_size):
            result[i, j] = t[i, indices[i, j]]
    return result


class TestGather(unittest.TestCase):

    def test_gather_simple(self):
        num_tests = 10
        sucess = True
        for _ in range(num_tests):
            batch_size, beam_size, vocab_size = map(int, np.random.randint(low=2, high=100, size=3))
            test_input = np.random.random([batch_size, beam_size, vocab_size])
            test_indices = np.random.randint(low=0, high=beam_size-1, size=[batch_size, beam_size])
            test_result = gather_helper_np(test_input, test_indices, batch_size, beam_size, vocab_size)
            with tf.Session() as sess:
                tf_input = tf.constant(test_input, dtype=tf.float32)
                tf_indices = tf.constant(test_indices, dtype=tf.int32)
                tf_result = gather_helper(tf_input, tf_indices, batch_size, beam_size)
                tf_result = sess.run(tf_result)
                sucess = sucess and np.allclose(test_result, tf_result)

        self.assertEqual(sucess, True)


if __name__ == '__main__':
    unittest.main()
