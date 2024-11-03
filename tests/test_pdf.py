"""Tests for the pdf module."""
import os
from source.library.pdf import clean_text_from_pdf, extract_text_from_pdf

def test__extract_text_from_pdf():  # noqa
    attention_url = 'https://arxiv.org/pdf/1706.03762.pdf'
    text = extract_text_from_pdf(attention_url)
    assert 'The dominant sequence transduction model' in text
    temp_file_name = './___temp_pdf___.pdf'
    assert not os.path.exists(os.path.join(os.getcwd(), temp_file_name))
    # save the text to a text file so we can track changes to how the text is extracted
    with open('tests/test_files/pdf/attention_is_all_you_need__extracted_url.txt', 'w') as f:
        f.write(text)

    text_from_url = text
    attention_url = 'tests/test_files/pdf/attention_is_all_you_need_short.pdf'
    text = extract_text_from_pdf(attention_url)
    assert text[0:100] == text_from_url[0:100]  # there will be small differences
    # save the text to a text file so we can track changes to how the text is extracted
    with open('tests/test_files/pdf/attention_is_all_you_need__extracted_local.txt', 'w') as f:
        f.write(text)

def test__clean_text_from_pdf():  # noqa
    attention_url = 'https://arxiv.org/pdf/1706.03762.pdf'
    text = extract_text_from_pdf(attention_url)
    assert 'The dominant sequence transduction model' in text
    temp_file_name = './___temp_pdf___.pdf'
    assert not os.path.exists(os.path.join(os.getcwd(), temp_file_name))

    cleaned_text = clean_text_from_pdf(text)
    print(cleaned_text)
    assert cleaned_text.startswith('Provided proper attribution is provided')
    assert cleaned_text.endswith('The heads clearly learned to perform different tasks.')

    cleaned_text = clean_text_from_pdf(text, include_at='Abstract')
    assert cleaned_text.startswith('Abstract')

    cleaned_text = clean_text_from_pdf(text, exclude_at='Acknowledgements')
    assert cleaned_text.endswith("The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.")  # noqa

    cleaned_text = clean_text_from_pdf(
        text,
        include_at='Abstract',
        exclude_at='Acknowledgements',
    )
    assert cleaned_text.startswith('Abstract')
    assert cleaned_text.endswith("The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.")  # noqa
    # save the text to a text file so we can track changes to how the text is cleaned
    with open('tests/test_files/pdf/attention_is_all_you_need_cleaned.txt', 'w') as f:
        f.write(cleaned_text)
