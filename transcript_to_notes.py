import os
from textwrap import dedent
from llm_workflow.openai import OpenAIChat
from dotenv import load_dotenv

load_dotenv()

total_cost = 0

def process_file(input_file_path, output_file_path, model_name):
    # Open the input file and read the content
    with open(input_file_path, 'r') as file:
        content = file.read()

    # Split the content into sections based on "# " (header-1) and "## " (header-2)
    sections = content.split('# ')
    
    # Open the output file
    with open(output_file_path, 'w') as output_file:
        # Process each section
        for section in sections:
            if section.strip() == "":
                continue
            
            lines = section.strip().splitlines()
            header1 = f"# {lines[0].strip()}"  # First line is the header-1
            output_file.write(header1 + "\n\n")
            
            current_section_text = []
            for line in lines[1:]:
                if line.startswith('## '):  # Detect header-2
                    if current_section_text:
                        # Process the previous section
                        notes = "\n".join(current_section_text).strip()
                        # print(f"`{notes}`")
                        response = generate_response(notes, model_name)
                        # response = notes[0:100]
                        output_file.write(response + "\n\n")
                    
                    # Start a new section
                    header2 = line.strip()
                    output_file.write('#' + header2 + "\n\n")
                    current_section_text = []
                else:
                    current_section_text.append(line)
            
            # Process the last section if it exists
            if current_section_text:
                notes = "\n".join(current_section_text).strip()
                # print(f"`{notes}`")
                response = generate_response(notes, model_name)
                # response = notes[0:100]
                output_file.write(response + "\n\n")
            output_file.write("\n\n---\n\n")
            output_file.flush()  # Flush the output buffer


def generate_response(notes, model_name):
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


if __name__ == "__main__":
    input_file_path = "temp_transcript.md"
    output_file_path = "output_file.txt"
    model_name = "gpt-4o-mini"
    
    process_file(input_file_path, output_file_path, model_name)
