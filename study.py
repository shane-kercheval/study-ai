import glob
import click
import yaml
import os
from source.library.helpers import colorize_gray, colorize_markdown
from source.library.notes import ClassNotes, Flashcard, History, TestBank

@click.group()
def cli():
    pass

# @cli.command()
# @click.option('--file', '-f', help='The YAML file to load notes from.')
# def load(file):
#     """Load notes from a YAML file."""
#     if not os.path.exists(file):
#         click.echo(f"File {file} does not exist.")
#         return

#     with open(file, 'r') as f:
#         data = yaml.safe_load(f)

#     class_notes = ClassNotes.from_dict(data)
#     click.echo(f"Loaded {len(class_notes.notes)} notes from {file}.")


@cli.command()
def create_notes():
    """Create yaml for notes from a text file."""
    pass


@click.command()
@click.option('--category', '-c', help='Only display notes from a specific class category.', default=None)
@click.option('--ident', '-i', help='Only display notes from a specific class identity.', default=None)
@click.option('--name', '-n', help='Only display notes from a specific class name.', default=None)
@click.option('--abbr', '-a', help='Only display notes from a specific class abbreviation.', default=None)
def search():
    pass


@click.command()
@click.option('--model', '-m', help='The model to use for chatting.', default='gpt-3.5')
def chat():
    pass


@click.command()
def scrape_pdf():
    pass






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
            history = {k: History(**v) for k, v in history.items()}
    else:
        history = {}

    class_notes = []
    # load all yaml files in /code/data/notes via glob
    for file in glob.glob('/code/data/notes/*.yaml'):
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
        class_notes.append(ClassNotes.from_dict(data))
    test_bank = TestBank(
        class_notes=class_notes,
        history=history,
        flash_only=flash_only,
        class_category=category,
        class_ident=ident,
        class_name=name,
        class_abbr=abbr,
    )
    click.echo(f"Available notes: {len(test_bank.test_bank)}")
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

        click.echo(f"\n\n{colorize_markdown(note['note'].note())}\n\n")
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
