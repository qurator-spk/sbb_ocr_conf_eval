import json
import sys
import csv

import click

from .ppn_handler import PpnHandler, PpnHandlerConfig
from .ocr_wc_analysis import plot_everything

@click.group('sbb_ocr')
def cli():
    """
    Evaluating OCR @StabiBerlin
    """
    pass


@cli.command('plot')
@click.argument('CSV_FILE')
def plot_cli(csv_file):
    """
    Plot confidence metrics
    """
    plot_everything(csv_file)

@cli.command('ppn2kitodo')
@click.argument('PPN', nargs=-1)
def ppn2kitodo_cli(ppn):
    """
    Translate PPN into Kitodo ID
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    print(json.dumps(ppn_handler.ppn2kitodo(ppn)))

@cli.command('ppn2pagexml')
@click.argument('PPN', nargs=-1)
def ppn2pagexml_cli(ppn):
    """
    Get a list of PAGE-XML files for PPN
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    print(ppn_handler.ppn2pagexml(ppn))

@cli.command('ppn2confs')
@click.option('--format', default='json', type=click.Choice(['csv', 'json']), help="Whether to output csv or json")
@click.argument('PPN', nargs=-1)
def ppn2confs_cli(format, ppn):
    """
    Get the table of line and word confidences for PPN
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    table = ppn_handler.ppn2confs(ppn)
    if format == 'json':
        print(json.dumps(table))
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(['ppn', 'file_id', 'version', 'textline_confs', 'word_confs'])
        for row in table:
            for i in [-1, -2]:
                row[i] = ' '.join(map(str, row[i]))
            writer.writerow(row)
