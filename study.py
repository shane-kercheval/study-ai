import glob
import click
import yaml
import os
from textwrap import dedent
from llm_workflow.openai import OpenAIChat, OpenAIServerChat
from llm_workflow.hugging_face import HuggingFaceEndpointChat
from source.library.helpers import colorize_gray, colorize_markdown
from source.library.notes import ClassNotes, Flashcard, History, TestBank
from dotenv import load_dotenv

load_dotenv()


@click.group()
def cli():
    pass


@cli.command()
def create_notes():
    """Create yaml for notes from a text file."""
    pass


# @cli.command()
# @click.option('--category', '-c', help='Only display notes from a specific class category.', default=None)
# @click.option('--ident', '-i', help='Only display notes from a specific class identity.', default=None)
# @click.option('--name', '-n', help='Only display notes from a specific class name.', default=None)
# @click.option('--abbr', '-a', help='Only display notes from a specific class abbreviation.', default=None)
# def search():
#     pass


# @cli.command()
# @click.option('--model', '-m', help='The model to use for chatting.', default='gpt-3.5')
# def chat():
#     pass


# @cli.command()
# def scrape_pdf():
#     pass



@cli.command()
@click.option('--model_type', '-mt', help="The model service to use, e.g. 'openai', 'openai_server', 'hugging_face_endpoint'", default='openai')  # noqa
@click.option('--model_name', '-mn', help="The model name (or endpoint) to use, e.g. 'gpt-3.5-turbo-0125' or 'http://host.docker.internal:1234/v1'", default='gpt-3.5-turbo-0125')  # noqa
@click.option('--temperature', '-t', help='The temperature to set on the model.', default=0.1)
@click.option('--file', '-f', help='The file to use for text-to-notes.', default=None)
def text_to_notes(model_type: str, model_name: str, temperature: float, file: str | None):
    if model_type == 'openai':
        model = OpenAIChat(model_name=model_name, temperature=temperature)
    elif model_type == 'openai_server':
        model = OpenAIServerChat(endpoint_url=model_name, temperature=temperature)
    elif model_type == 'hugging_face_endpoint':
        model = HuggingFaceEndpointChat(endpoint_url=model_name, temperature=temperature)
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented.")

    model.streaming_callback = lambda x: click.echo(x.response, nl=False)
    with open("/code/source/library/prompts/text_to_notes.txt") as f:
        prompt_template = f.read()
    if file:
        with open(file) as f:
            text = f.read()
    else:
        text = click.edit("<replace with text>")
    prompt = dedent(prompt_template).strip().replace("{{text}}", text)
    _ = model(prompt)
    click.echo("\n\n")
    if model.cost:
        click.echo(f"\n\nCost: {model.cost}")


def load_notes(path: str) -> list[ClassNotes]:
    class_notes = []
    # load all yaml files in /code/data/notes via glob
    for file in glob.glob(path):
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
        class_notes.append(ClassNotes.from_dict(data))
    return class_notes


@cli.command()
@click.option('--flash_only', '-f', help='Only display flashcards.', is_flag=True, default=False)
@click.option('--category', '-c', help='Only display notes from a specific class category.', default=None)
@click.option('--ident', '-i', help='Only display notes from a specific class identity.', default=None)
@click.option('--name', '-n', help='Only display notes from a specific class name.', default=None)
@click.option('--abbr', '-a', help='Only display notes from a specific class abbreviation.', default=None)
def cycle(
        flash_only: bool,
        category: str,
        ident: str,
        name: str,
        abbr: str,
    ):
    """Cycle through notes from one or more YAML files."""
    if os.path.exists('/code/data/history.yaml'):
        with open('/code/data/history.yaml', 'r') as f:
            history = yaml.safe_load(f)
            if history is None:
                history = {}
            history = {k: History(**v) for k, v in history.items()}
    else:
        history = {}

    class_notes = load_notes('/code/data/notes/*.yaml')
    test_bank = TestBank(
        class_notes=class_notes,
        history=history,
        flash_only=flash_only,
        class_category=category,
        class_ident=ident,
        class_name=name,
        class_abbr=abbr,
    )
    click.echo(f"Available notes: {len(test_bank.notes)}")
    click.echo("\n\n\n")
    while True:
        note = test_bank.draw()
        if isinstance(note['note'], Flashcard):
            click.echo(f"\n\n{colorize_markdown(note['note'].preview())}\n\n")
            selected = click.prompt(
                colorize_gray("Press any key to reveal answer (q to quit)"),
                type=str,
                default='',
                show_default=False,
            )
            if selected == 'q':
                break

        click.echo(f"\n\n{colorize_markdown(note['note'].text())}\n\n")
        selected = ''
        while selected not in ['y', 'n', 'q']:
            selected = click.prompt(colorize_gray("Invalid input. Correct? (y/n) or q to quit"), type=str)
        if selected == 'q':
            break
        test_bank.correct_answer(uuid=note['uuid'], correct=selected == 'y')
        # save history to a file
        history.update({k: v.to_dict() for k, v in test_bank.history.items()})
        with open('/code/data/history.yaml', 'w') as f:
            yaml.safe_dump(history, f)


if __name__ == '__main__':
    cli()
