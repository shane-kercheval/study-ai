"""Utilities for working with PDFs."""
import os
import re
import requests
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str, delete_afterwards: bool = False) -> str:
    """
    Extract text from a PDF.

    Args:
        pdf_path: The path to the PDF.
        delete_afterwards: Whether to delete the PDF after extracting the text.
    """
    local_file_name = pdf_path
    if pdf_path.startswith('http') or 'www.' in pdf_path:
        local_file_name = os.path.join(os.getcwd(), '___temp_pdf___.pdf')
        response = requests.get(pdf_path)
        assert response.status_code == 200
        with open(local_file_name, 'wb') as f:
            f.write(response.content)

    text = ''
    try:
        reader = PdfReader(local_file_name)
        for page in reader.pages:
            text += '\n\n' + page.extract_text()
        if delete_afterwards:
            os.remove(local_file_name)
    finally:
        if '___temp_pdf___' in local_file_name:
            os.remove(local_file_name)
    return text

def clean_text_from_pdf(
        text: str,
        include_at: str | None = None,
        exclude_at: str | None = None,
    ) -> str:
    """
    Clean text from a PDF.

    Args:
        text: The text to clean.
        include_at: Include text starting from this string.
        exclude_at: Exclude text starting from this string.
    """
    text = text.strip()
    if include_at:
        index = text.find(include_at)
        assert index >= 0
        text = text[index:]
    if exclude_at:
        index = text.find(exclude_at)
        assert index >= 0
        text = text[:index]
    # Fix arbitrary newlines
    # Replace newlines not preceded by a sentence-ending punctuation with a space
    # This pattern matches if a newline is not preceded by ., !, ?, or ] (assuming references might
    # end a sentence)
    text = re.sub(r'\n(?=[a-z\[\()])', ' ', text)
    text = text.replace('<EOS>', '\n')
    text = text.replace('<eos>', '\n')
    text = text.replace('\n\n', '\n')
    text = text.replace('<pad>', '')
    footnote_symbols = ['†', '‡', '¶', '§', '‖', '※']
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()  # noqa: PLW2901
        # Remove lines that start with a footnote symbol
        if any(line.startswith(symbol) for symbol in footnote_symbols) \
                or line.isdigit():
                # or re.match(r'^Figure \d+:', line):
            continue
        if not line:
            continue
        processed_lines.append(line)
    return '\n\n'.join(processed_lines).strip()
