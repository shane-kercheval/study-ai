"""Contains helper functions that are used in the main program."""


def softmax_dict(d: dict) -> dict:
    """Return a dictionary with softmax applied to values."""
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}
