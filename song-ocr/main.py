#!/usr/bin/env python3
import click
from fix_split import DOIT


@click.command()
@click.argument("filename")
def main(filename):
    output = filename.split(".")[0] + "_output"
    DOIT(
        filename,
        outdir=output,
        vThreshVal=1,
        maxVThreshVal=20,
        hThreshVal=0.5,
        maxHThreshVal=0.5,
        hMergeThresh=100, # means all merge, only vertical split
        hSkipThesh=4,
        debug=False,
    )


if __name__ == "__main__":
    main()
