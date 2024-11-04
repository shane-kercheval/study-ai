import os
import re
from textwrap import dedent
from llm_workflow.openai import OpenAIChat
from dotenv import load_dotenv

load_dotenv()

total_cost = 0


def generate_response(notes: str, model_name: str) -> str:
    """Generates notes based on the input notes using the specified model."""
    if notes == "#":
        return ''
    global total_cost
    # Load the prompt template
    model = OpenAIChat(model_name=model_name, temperature=0.5)
    path = "source/library/prompts/format_notes_markdown.txt"
    with open(path) as f:
        prompt_template = f.read()
    prompt = dedent(prompt_template).strip().replace("{{notes}}", notes)
    response = model(prompt)
    total_cost += model.cost
    print(f"Total Cost: {total_cost}; Response Cost: {model.cost}\n")
    return response


def clean_notes(notes: str) -> str:
    """Cleans notes by removing timestamps and line numbers."""
    cleaned_lines = []
    for line in notes.splitlines():
        if not re.match(r"^\d+$", line) and not re.match(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$", line):
            cleaned_lines.append(line.strip())
    cleaned_text = " ".join(cleaned_lines)
    cleaned_text = re.sub(r'\s*\n\s*', ' ', cleaned_text)  # Remove newlines in between text
    # remove double spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text  # noqa: RET504


def extract_number(filename: str) -> int:
    """Extracts the leading number from a filename."""
    match = re.match(r"(\d+)", filename)
    assert match
    return int(match.group(0))

def extract_lecture_title(filename: str) -> str:
    """Extracts the lecture title from a filename."""
    # Remove the leading number and hyphen (if present), then strip whitespace
    return re.sub(r"^\d+\s*-\s*", "", filename).rsplit(".", 1)[0].strip()


if __name__ == "__main__":
    dir_path = "/Users/shanekercheval/Downloads/CS6200_Lectures"
    output_file_path = "output_file.txt"
    model_name = "gpt-4o-mini"
    # for each directory in the path that ends with "_subtitles", get all ".srt" files

    files_with_numbers = []
    # Traverse directories and collect files with their leading numbers
    for subdir, dirs, files in os.walk(dir_path):
        # e.g., P4L1__Remote_Procedure_Calls
        lesson_name = os.path.basename(subdir)\
            .removesuffix("_subtitles")\
            .replace("__", " - ")\
            .replace("_", " ")
        for file in files:
            if file.endswith(".srt"):
                input_file_path = os.path.join(subdir, file)
                number = extract_number(file)
                lecture_title = extract_lecture_title(file)
                files_with_numbers.append((input_file_path, number, lesson_name, lecture_title))

    files_with_numbers.sort(key=lambda x: x[1])

    # Print sorted file paths
    for file_path, lecture_number, lesson_name, lecture_title in files_with_numbers:
        if lecture_number >= 434:
            print(f"{lecture_number} - {lesson_name} - {lecture_title}")

            with open(file_path) as f:
                notes = f.read()

            print(clean_notes(notes))
            break

    # process_file(input_file_path, output_file_path, model_name)