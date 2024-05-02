import glob
import click
import yaml
import os
from source.library.notes import ClassNotes, TestBank

@click.group()
def cli():
    pass

@cli.command()
@click.option('--file', '-f', help='The YAML file to load notes from.')
def load(file):
    """Load notes from a YAML file."""
    if not os.path.exists(file):
        click.echo(f"File {file} does not exist.")
        return

    with open(file, 'r') as f:
        data = yaml.safe_load(f)

    class_notes = ClassNotes.from_dict(data)
    click.echo(f"Loaded {len(class_notes.notes)} notes from {file}.")


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

@cli.command()
def cycle():
    """Cycle through specified notes from a YAML file."""
    # load all yaml files in /code/data/notes via glob
    class_notes = []
    for file in glob.glob('/code/data/notes/*.yaml'):
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
        class_notes.append(ClassNotes.from_dict(data))
    test_bank = TestBank(class_notes=class_notes)
    click.echo(f"Available notes: {len(test_bank.test_bank)}")
    click.echo("\n\n\n")
    while True:
        note = test_bank.draw()
        click.echo(f"\n\n{colorize_markdown(note['note'].preview())}\n\n")
        selected = click.prompt(colorize_gray("Use any key to reveal answer, Use q to quit"), type=str)
        if selected == 'q':
            break
        click.echo(f"\n\n{colorize_markdown(note['note'].note())}\n\n")
        selected = click.prompt(colorize_gray("Correct? (y/n)"), type=str)
        



if __name__ == '__main__':
    cli()