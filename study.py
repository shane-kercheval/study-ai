"""CLI for studying notes."""
import click
import yaml
import os
from textwrap import dedent
from llm_workflow.openai import OpenAIChat, OpenAIServerChat
from source.cli.utilities import (
    colorize_gray,
    colorize_green,
    colorize_red,
    colorize_markdown,
    filter_notes,
    load_notes,
)
from source.library.notes import Flashcard, History, NoteBank, Priority
from dotenv import load_dotenv

from source.library.search import VectorDatabase

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
@click.option('--notes_paths', '-p', multiple=True, help='The path to the notes yaml file(s).', default=['data/notes/*.yaml'])  # noqa
@click.option('--history_path', '-h', help='The path to the history (of correct/incorrect answers) yaml file.', default='data/history.yaml')  # noqa
def cycle(
        flash_only: bool,
        category: str,
        ident: str,
        name: str,
        abbr: str,
        notes_paths: tuple[str],
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

    notes = []
    for path in notes_paths:
        notes.extend(load_notes(path, generate_save_uuids=True))
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
    while True:
        # only consider the last 20 answers; give more weight (linear) to the most recent answers
        weights = list(range(20))
        priority_weights = {Priority.high: 3, Priority.medium: 2, Priority.low: 1}
        note = test_bank.draw(last_n=weights, priority_weights=priority_weights)
        click.echo("--------------------------\n")
        click.echo(colorize_gray(f"{note.uuid}"))
        click.echo(colorize_gray(f"{note.subject_metadata.category} - {note.subject_metadata.ident} - {note.subject_metadata.abbreviation} - {note.subject_metadata.name}"))  # noqa  
        click.echo(colorize_gray(f"{note.note_metadata.source_name}"))
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
        test_bank.answer(uuid=note.uuid, correct=user_response == 'y')
        # we need to modify/save our original history dictionary because test_bank may be a subset
        # of the original notes and we want to keep the history of all notes
        history.update(test_bank.history(to_dict=True))
        with open(history_path, 'w') as f:
            yaml.safe_dump(history, f)


@cli.command()
@click.option('--notes_paths', '-p', multiple=True, help='The path to the notes yaml file(s).', default=['data/notes/*.yaml'])  # noqa
@click.option('--db_path', '-d', help='The path to the database.', default='data/vector_database.parquet')  # noqa
@click.option('--similarity_threshold', '-s', help='The similarity threshold for search results.', default=0.3)  # noqa
@click.option('--top_k', '-k', help='The number of top results to return.', default=5)
def search(notes_paths: tuple[str], db_path: str, similarity_threshold: float, top_k: int) -> None:
    """
    Search the notes database.

    Any yaml files where the notes do not have uuids will be updated with uuids and the original
    files will be overwritten. Static uuids are used to ensure that if the notes are updated, the
    corresponding embeddings are updated in the database.

    Any notes not already in the database will be added (embeddings will be created). Any notes
    that have been modified will have their embeddings (and corresponding text) updated. The
    database will be saved after any changes are made.
    """
    click.echo("Loading notes...")
    notes = []
    for path in notes_paths:
        notes.extend(load_notes(path, generate_save_uuids=True))
    click.echo("Loading database...")
    db = VectorDatabase(db_path=db_path)
    click.echo("Adding/updating notes in vector database...")
    changes = db.add(notes=notes, save=True)
    if changes:
        click.echo("The following changes were made to the database:")
        for uuid, change in changes.items():
            click.echo(f"   `{uuid}`: {change}")
    else:
        # load cached model so first search is faster
        db.model
    while True:
        click.echo("\n")
        query = click.prompt("Enter a search query or 'q' to quit")
        if query == 'q':
            break
        results = db.search(query=query, top_k=top_k)
        if len(results) == 0:
            click.echo("No results found.")
        else:
            results = results[results['cosine_similarity'] > similarity_threshold]
            # get a list of matched notes in the same order as the results (based on uuid)
            matched_uuids = set(results['uuid'].tolist())
            matched_notes = {note.uuid: note for note in notes if note.uuid in matched_uuids}
            # ensure same order as results
            matched_notes = [matched_notes[uuid] for uuid in results['uuid'].tolist()]
            click.echo("\n\n")
            for note, cosine_simiarlity in zip(matched_notes, results['cosine_similarity']):
                click.echo("--------------------------")
                cosine_sim_text = colorize_green(f"Cosine Similarity: {cosine_simiarlity:.2f}")
                uuid_text = colorize_gray(f"; uuid: {note.uuid}")
                click.echo(f"{cosine_sim_text}{uuid_text}")
                click.echo(colorize_gray(f"{note.subject_metadata.category} - {note.subject_metadata.ident} - {note.subject_metadata.abbreviation} - {note.subject_metadata.name}"))  # noqa  
                click.echo(colorize_gray(f"{note.note_metadata.source_name}"))
                click.echo(f"\n{colorize_markdown(str(note))}")
                click.echo("--------------------------\n")


@cli.command()
@click.option('--model_type', '-mt', help="The model service to use, e.g. 'openai' or 'openai_server'", default='openai')  # noqa
@click.option('--model_name', '-mn', help="The model name (or endpoint) to use, e.g. 'gpt-4o-mini' or 'http://host.docker.internal:1234/v1'", default='gpt-4o-mini')  # noqa
@click.option('--temperature', '-t', help='The temperature to set on the model.', default=0.1)
@click.option('--file', '-f', help='The file to use for text-to-notes.', default=None)
def text_to_flashcards(
        model_type: str,
        model_name: str,
        temperature: float,
        file: str | None) -> None:
    """Convert text to notes using a language model."""
    if model_type == 'openai':
        model = OpenAIChat(model_name=model_name, temperature=temperature)
    elif model_type == 'openai_server':
        model = OpenAIServerChat(endpoint_url=model_name, temperature=temperature)
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented.")

    model.streaming_callback = lambda x: click.echo(x.response, nl=False)
    path = "source/library/prompts/text_to_flashcards.txt"
    with open(path) as f:
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


@cli.command()
@click.option('--model_type', '-mt', help="The model service to use, e.g. 'openai' or 'openai_server'", default='openai')  # noqa
@click.option('--model_name', '-mn', help="The model name (or endpoint) to use, e.g. 'gpt-4o-mini', 'http://localhost:1234/v1', or 'http://host.docker.internal:1234/v1'", default='gpt-4o-mini')  # noqa
@click.option('--temperature', '-t', help='The temperature to set on the model.', default=0.1)
@click.option('--file', '-f', help='The file to use for text-to-notes.', default=None)
@click.option('--stream', '-s', help='Stream response.', is_flag=True, default=False)
def quiz(
        model_type: str,
        model_name: str,
        temperature: float,
        file: str | None,
        stream: bool) -> None:
    """Convert text to notes using a language model."""
    if model_type == 'openai':
        model = OpenAIChat(model_name=model_name, temperature=temperature)
    elif model_type == 'openai_server':
        model = OpenAIServerChat(endpoint_url=model_name, temperature=temperature)
        model.model_name = None
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented.")
    if stream:
        model.streaming_callback = lambda x: click.echo(x.response, nl=False)
    path = "source/library/prompts/quiz.txt"
    with open(path) as f:
        prompt_template = f.read()
    with open(file) as f:
        notes = f.read()
    prompt = dedent(prompt_template).strip().replace("{{notes}}", notes)
    click.echo("\n------\n")
    response = model(prompt)
    if not stream:
        click.echo(colorize_markdown(f"{response}"))
    while True:
        user_response = click.prompt(
            colorize_red("\n\nAnswer: "),
            type=str,
        )
        click.echo("\n")
        response = model(user_response)
        if stream:
            click.echo("\n\n")
        else:
            click.echo(colorize_markdown(f"\n{response}\n\n"))
        if model.cost:
            click.echo(colorize_green(f"Cost: {model.cost}"))
        click.echo(colorize_green(f"Total tokens: {model.total_tokens}"))
        click.echo(colorize_green(f"Input tokens: {model.input_tokens}"))
        click.echo(colorize_green(f"Response tokens: {model.response_tokens}"))


@cli.command()
@click.option('--model_type', '-mt', help="The model service to use, e.g. 'openai' or 'openai_server'", default='openai')  # noqa
@click.option('--model_name', '-mn', help="The model name (or endpoint) to use, e.g. 'gpt-4o-mini', 'http://localhost:1234/v1', or 'http://host.docker.internal:1234/v1'", default='gpt-4o-mini')  # noqa
@click.option('--temperature', '-t', help='The temperature to set on the model.', default=0.1)
@click.option('--stream', '-s', help='Stream response.', is_flag=True, default=False)
def format_notes(
        model_type: str,
        model_name: str,
        temperature: float,
        stream: bool) -> None:
    """
    Convert text to notes in markdown format using a language model.

    For example, this is used to convert lector trancripts into notes.
    """
    if model_type == 'openai':

        model = OpenAIChat(model_name=model_name, temperature=temperature)
    elif model_type == 'openai_server':
        model = OpenAIServerChat(endpoint_url=model_name, temperature=temperature)
        model.model_name = None
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented.")
    if stream:
        model.streaming_callback = lambda x: click.echo(x.response, nl=False)
    path = "source/library/prompts/format_notes_markdown.txt"
    with open(path) as f:
        prompt_template = f.read()
    notes = click.edit("")
    click.echo("\n------\n")
    prompt = dedent(prompt_template).strip().replace("{{notes}}", notes)
    response = model(prompt)
    if not stream:
        click.echo(colorize_markdown(f"{response}"))
    click.echo("\n\n")
    click.echo(colorize_green(f"Cost: {model.cost}"))
    click.echo(colorize_green(f"Total tokens: {model.total_tokens}"))
    click.echo(colorize_green(f"Input tokens: {model.input_tokens}"))
    click.echo(colorize_green(f"Response tokens: {model.response_tokens}"))


if __name__ == '__main__':
    cli()
