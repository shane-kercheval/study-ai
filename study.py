"""CLI for studying notes."""
import click
import yaml
import os
from textwrap import dedent
from llm_workflow.openai import OpenAIChat, OpenAIServerChat
from llm_workflow.hugging_face import HuggingFaceEndpointChat
from source.cli.utilities import colorize_gray, colorize_markdown, filter_notes, load_notes
from source.library.notes import Flashcard, History, NoteBank
from dotenv import load_dotenv

load_dotenv()


@click.group()
def cli():  # noqa
    pass


@cli.command()
def create_notes() -> None:
    """Create yaml for notes from a text file."""
    pass


@cli.command()
@click.option('--flash_only', '-f', help='Only display flashcards.', is_flag=True, default=False)
@click.option('--category', '-c', help='Only display notes from a specific class category.', default=None)  # noqa
@click.option('--ident', '-i', help='Only display notes from a specific class identity.', default=None)  # noqa
@click.option('--name', '-n', help='Only display notes from a specific class name.', default=None)
@click.option('--abbr', '-a', help='Only display notes from a specific class abbreviation.', default=None)  # noqa
@click.option('--notes_path', '-p', help='The path to the notes yaml file(s).', default='data/notes/*.yaml')  # noqa
@click.option('--history_path', '-h', help='The path to the history yaml file.', default='data/history.yaml')  # noqa
def cycle(
        flash_only: bool,
        category: str,
        ident: str,
        name: str,
        abbr: str,
        notes_path: str,
        history_path: str,
    ) -> None:
    """
    Cycle through notes/flashcards from one or more YAML files. The history of the notes will be
    saved to a YAML file. The frequency of notes will be based on the historical accuracy of
    correctly answers.

    Press any key to reveal the answer. Press 'q' to quit.
    """
    history = None
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = yaml.safe_load(f)
    if history is None:
        history = {}

    notes = load_notes(notes_path)
    notes = filter_notes(
        notes=notes,
        flash_only=flash_only,
        category=category,
        ident=ident,
        name=name,
        abbr=abbr,
    )
    test_bank = NoteBank(
        notes=notes,
        history={k: History(**v) for k, v in history.items()},
    )
    click.echo(f"Available notes: {len(test_bank.notes)}")
    click.echo("\n\n\n")
    while True:
        note = test_bank.draw()
        if isinstance(note, Flashcard):
            click.echo(f"\n\n{colorize_markdown(note.preview())}\n\n")
            user_response = click.prompt(
                colorize_gray("Press any key to reveal answer (q to quit)"),
                type=str,
                default='',
                show_default=False,
            )
            if user_response == 'q':
                break
        text = note.answer() if isinstance(note, Flashcard) else note.text()
        click.echo(f"\n\n{colorize_markdown(text)}\n\n")
        user_response = ''
        while user_response not in ['y', 'n', 'q']:
            user_response = click.prompt(
                colorize_gray("Invalid input. Correct? (y/n) or q to quit"),
                type=str,
            )
        if user_response == 'q':
            break
        test_bank.answer(uuid=note.uuid(), correct=user_response == 'y')
        # we need to modify/save our original history dictionary because test_bank may be a subset
        # of the original notes and we want to keep the history of all notes
        history.update(test_bank.history(to_dict=True))
        with open(history_path, 'w') as f:
            yaml.safe_dump(history, f)


@cli.command()
@click.option('--model_type', '-mt', help="The model service to use, e.g. 'openai', 'openai_server', 'hugging_face_endpoint'", default='openai')  # noqa
@click.option('--model_name', '-mn', help="The model name (or endpoint) to use, e.g. 'gpt-3.5-turbo-0125' or 'http://host.docker.internal:1234/v1'", default='gpt-3.5-turbo-0125')  # noqa
@click.option('--temperature', '-t', help='The temperature to set on the model.', default=0.1)
@click.option('--file', '-f', help='The file to use for text-to-notes.', default=None)
def text_to_notes(model_type: str, model_name: str, temperature: float, file: str | None) -> None:
    """Convert text to notes using a language model."""
    if model_type == 'openai':
        model = OpenAIChat(model_name=model_name, temperature=temperature)
    elif model_type == 'openai_server':
        model = OpenAIServerChat(endpoint_url=model_name, temperature=temperature)
    elif model_type == 'hugging_face_endpoint':
        model = HuggingFaceEndpointChat(endpoint_url=model_name, temperature=temperature)
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented.")

    model.streaming_callback = lambda x: click.echo(x.response, nl=False)
    with open("source/library/prompts/text_to_notes.txt") as f:
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


if __name__ == '__main__':
    cli()
