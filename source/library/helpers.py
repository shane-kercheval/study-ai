import re


def colorize_markdown(text):
    # Apply red color for text surrounded by **
    text = re.sub(r'\*\*(.*?)\*\*', r'\033[31m\1\033[0m', text)
    # Apply blue color for text surrounded by `
    text = re.sub(r'`(.*?)`', r'\033[34m\1\033[0m', text)
    return text

def colorize_gray(text):
    # Apply gray color to all text
    return f'\033[90m{text}\033[0m'

