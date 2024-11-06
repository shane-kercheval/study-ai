import os
import re
from textwrap import dedent
from llm_workflow.openai import OpenAIChat
from dotenv import load_dotenv
import asyncio

load_dotenv()

def generate_response_sync(notes: str, model_name: str) -> tuple[str, float]:
    """Generates notes based on the input notes using the specified model."""
    if notes == "#":
        return '', 0.0

    # Load the prompt template
    model = OpenAIChat(model_name=model_name, temperature=0.3)
    path = "source/library/prompts/transcript_to_notes_markdown.txt"
    with open(path) as f:
        prompt_template = f.read()
    prompt = dedent(prompt_template).strip().replace("{{notes}}", notes)
    response = model(prompt)
    return response, model.cost


async def generate_response_async(notes: str, model_name: str) -> tuple[str, float]:
    """Asynchronous wrapper around generate_response_sync."""
    loop = asyncio.get_running_loop()
    response, cost = await loop.run_in_executor(None, generate_response_sync, notes, model_name)
    return response, cost


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
    return cleaned_text


def extract_number(filename: str) -> int:
    """Extracts the leading number from a filename."""
    match = re.match(r"(\d+)", filename)
    assert match
    return int(match.group(0))


def extract_lecture_title(filename: str) -> str:
    """Extracts the lecture title from a filename."""
    return re.sub(r"^\d+\s*-\s*", "", filename).rsplit(".", 1)[0].strip()


async def main():
    dir_path = "/Users/shanekercheval/Downloads/CS6200_Lectures"
    output_file_path = "output_file.txt"
    model_name = "gpt-4o-mini"
    # Set the bounds for lecture numbers to process
    lower_bound = 443
    upper_bound = 467

    files_with_numbers = []
    for subdir, dirs, files in os.walk(dir_path):
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
    filtered_files = [
        (file_path, lecture_number, lesson_name, lecture_title)
        for file_path, lecture_number, lesson_name, lecture_title in files_with_numbers
        if lower_bound <= lecture_number <= upper_bound
    ]
    # Asynchronous processing for the filtered files
    tasks = []
    for file_path, lecture_number, lesson_name, lecture_title in filtered_files:
        print(f"{lecture_number} - {lesson_name} - {lecture_title}")
        with open(file_path) as f:
            notes = f.read()
        notes = clean_notes(notes)
        tasks.append(generate_response_async(notes, model_name))

    responses_with_costs = await asyncio.gather(*tasks)

    # Write results to file and calculate total cost
    total_cost = 0
    with open(output_file_path, "w") as output_handle:
        for (response, cost), (_, _, _, lecture_title) in zip(responses_with_costs, filtered_files):
            response = re.sub(r'^```.*?\n', '', response)
            response = response.removesuffix("```")
            output_handle.write(f"# {lecture_title}\n\n")
            output_handle.write(response)
            output_handle.write("\n\n---\n\n")
            total_cost += cost


    print(f"Total Cost: {total_cost:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
