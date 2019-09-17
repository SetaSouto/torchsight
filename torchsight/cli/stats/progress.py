"""Module with commands to show stats using the metrics of the progress logger."""
import click


@click.command()
@click.option('-t', '--train', help='The file with the metrics of the epochs of the training.',
              default='./epochs.csv', show_default=True, type=click.Path(exists=True))
@click.option('-v', '--valid', help='The file with the metrics of the epochs of the validation.',
              default='./validation_epochs.csv', show_default=True, type=click.Path(exists=True))
def progress(train, valid):
    """Show in a table the metrics of the given files."""
    from csv import DictReader

    with open(train, 'r') as file:
        headers = file.readline().replace('\n', '').split(',')
        headers = [h for h in headers if h != 'Time']

    # Show the headers
    show = []
    for header in headers:
        if header in ['Epoch', 'LR']:
            show.append(header.center(6))
        elif header == 'Loss':
            show.append(header.center(19))
        else:
            show.append(header.center(15))
    print('|'.join(show))

    # Start reading the files and show the rows
    with open(train, 'r') as file:
        train = DictReader(file)
        with open(valid, 'r') as file:
            valid = DictReader(file)

            for train, valid in zip(train, valid):
                show = []
                for key in headers:
                    if key in ['Epoch', 'LR']:
                        show.append(f'{train[key].rjust(6)}')
                    elif key == 'Loss':
                        show.append(f'{train[key].rjust(9)} {valid[key].rjust(9)}')
                    else:
                        show.append(f'{train[key].rjust(7)} {valid[key].rjust(7)}')
                print('|'.join(show))
