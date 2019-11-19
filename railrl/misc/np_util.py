"""
Utility functions that heavily use or modify numpy.
"""
import numpy as np
import contextlib
import scipy.signal


@contextlib.contextmanager
def np_print_options(*args, **kwargs):
    """
    Locally modify print behavior.

    Usage:
    ```
    x = np.random.random(10)
    with printoptions(precision=3, suppress=True):
        print(x)
        # [ 0.073  0.461  0.689  0.754  0.624  0.901  0.049  0.582  0.557  0.348]
    ```

    http://stackoverflow.com/questions/2891790/how-to-pretty-printing-a-numpy-array-without-scientific-notation-and-with-given
    :param args:
    :param kwargs:
    :return:
    """
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


# TODO(vpong): Test this
def to_onehot(x, num_values):
    """
    Return a one hot vector representing x.
    :param x: Number to represent.
    :param num_values: Size of onehot vector.
    :return: nd.array of shape (num_values,)
    """
    onehot = np.zeros(num_values)
    onehot[x] = 1
    return onehot


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    denom = np.expand_dims(e_x.sum(axis=axis), axis=axis)
    return e_x / denom


def subsequences(tensor, start_indices, length, start_offset=0):
    """
    Return subsequences of a tensor, starting at the indices give by
    `start_indices` plus `start_offset`.
    :param tensor: np.array
        Shape: n x m1 x m2 x ... x md
        where *m could be a number, or
    :param start_indices: list with k elements
    :param length: int
    :param start_offset: int
    :return: np.array
        shape: k x length x m1 x m2 x ... md
    """
    num_indices = len(start_indices)
    indices = np.repeat(
        np.arange(length).reshape((1, length)),
        num_indices,
        axis=0
    ) + np.array(start_indices).reshape((num_indices, 1)) + start_offset
    return tensor[indices]


def assign_subsequences(tensor, new_values, start_indices, length,
                        start_offset=0,
                        keep_old_fraction=0.):
    """
    The same as subseqences, but instead of returning those subsequences,
    this assigns `new_values` to those entries.

    If there's an overlap in places to assign values in `tensors`, the
    subsequences later in `new_values` will overwrite preceding values.
    :param tensor: np.array
        Shape: n x m1 x m2 x ... x md
        where *m could be a number, or
    :param new_values: np.array
        shape: k x `length` x m1 x m2 x ... md
    :param start_indices: list with k elements
    :param length: int, must match second dimension of new_values
    :param start_offset: int
    """
    assert len(new_values) == len(start_indices)
    assert new_values.shape[1] == length
    assert new_values.shape[2:] == tensor.shape[1:]
    assert min(start_indices) >= 0
    assert max(start_indices) + length <= len(tensor)
    num_indices = len(start_indices)
    indices = np.repeat(
        np.arange(length).reshape((1, length)),
        num_indices,
        axis=0
    ) + np.array(start_indices).reshape((num_indices, 1)) + start_offset
    if keep_old_fraction > 0.:
        for new_value, sub_indices in zip(new_values, indices):
            tensor[sub_indices] = (
                (1-keep_old_fraction) * new_value
                + keep_old_fraction * tensor[sub_indices]
            )
    else:
        for new_value, sub_indices in zip(new_values, indices):
            tensor[sub_indices] = new_value


def batch_discounted_cumsum(values, discount):
    """
    Return a matrix of discounted returns.

    output[i, j] = discounted sum of returns of rewards[i, j:]

    So

    output[i, j] = rewards[i, j] + rewards[i, j+1] * discount
                    + rewards[i, j+2] * discount**2 + ...

    Based on rllab.misc.special.discounted_cumsum
    :param rewards: FloatTensor, size [batch_size, sequence_length, 1]
    :param discount: float, discount factor
    :return FloatTensor, size [batch_size, sequence_length, 1]
    """
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or reverse(y)[t] - discount*reverse(y)[t-1] = reverse(x)[t]
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], values.T[::-1], axis=0,
    )[::-1].T


def truncated_geometric(p, truncate_threshold, size, new_value=None):
    """
    Sample from geometric, but truncated values to `truncated_threshold`.

    All values greater than `truncated_threshold` will be set to `new_value`.
    If `new_value` is None, then they will be assigned random integers from 0 to
    `truncate_threshold`.

    :param p: probability parameter for geometric distribution
    :param truncate_threshold: Cut-off
    :param size: size of sample
    :param new_value:
    :return:
    """
    samples = np.random.geometric(p, size)
    samples_too_large = samples > truncate_threshold
    num_bad = sum(samples_too_large)
    if new_value is None:
        samples[samples > truncate_threshold] = (
            np.random.randint(0, truncate_threshold, num_bad)
        )
    else:
        samples[samples > truncate_threshold] = new_value
    return samples