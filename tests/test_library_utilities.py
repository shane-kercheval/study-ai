"""Test the helper functions in the library module."""

from source.library.utilities import softmax_dict


def test__softmax_dict():  # noqa
    d = {'a': 0.1, 'b': 0.3, 'c': 0.6}
    result = softmax_dict(d)
    assert result['a'] == 0.1
    assert result['b'] == 0.3
    assert result['c'] == 0.6
    assert sum(result.values()) == 1

    d = {'a': 1, 'b': 2, 'c': 3}
    result = softmax_dict(d)
    assert result['a'] == 1 / 6
    assert result['b'] == 2 / 6
    assert result['c'] == 3 / 6
    assert sum(result.values()) == 1

    d = {'a': 0, 'b': 2, 'c': 3}
    result = softmax_dict(d)
    assert result['a'] == 0
    assert result['b'] == 2 / 5
    assert result['c'] == 3 / 5
    assert sum(result.values()) == 1
