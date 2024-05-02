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
    while True:
        # test_bank.correct_answer(selected == 'y')
        note = test_bank.draw()
        print(note)
        click.echo(note['note'].preview())
        # Press 1 to reveal answer press 2 to quit
        selected = click.prompt("Press 1 to reveal answer, press q to quit", type=str)
        if selected == 'q':
            break
        print(note)
        click.echo(note.note().note())
        # ask user to press enter to reveal answer
        selected = click.prompt("Correct? (y/n)", type=str)
        # selected = click.prompt("", type=int)



if __name__ == '__main__':
    cli()