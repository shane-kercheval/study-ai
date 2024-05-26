"""Contains helper functions that are used in the main program."""
import re


def colorize_markdown(text: str) -> str:
    """Colorizes text (used as the output in the terminal) based on markdown syntax."""
    # Apply red color for text surrounded by **
    text = re.sub(r'\*\*(.*?)\*\*', r'\033[31m\1\033[0m', text)
    # Apply blue color for text surrounded by `
    return re.sub(r'`(.*?)`', r'\033[34m\1\033[0m', text)

def colorize_gray(text: str) -> str:
    """Colorizes text (used as the output in the terminal) to gray."""
    # Apply gray color to all text
    return f'\033[90m{text}\033[0m'


def softmax_dict(d: dict) -> dict:
    """Return a dictionary with softmax values."""
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}
