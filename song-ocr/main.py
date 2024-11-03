
import click
from fix_split import DOIT

@click.command()
@click.argument('filename')
def main(filename):
    DOIT(filename, outdir=f"{filename}_output")

if __name__ == '__main__':
    main()