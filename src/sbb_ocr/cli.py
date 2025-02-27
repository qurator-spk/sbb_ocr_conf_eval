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
@click.option('-M', '--mods-info', 'mods_info_csv', default="mods_info_df_2024-11-27.csv", help='mods_info CSV for PPN metadata')
@click.argument('CSV_FILES', nargs=-1)
@click.argument('PLOT_FILE')
def plot_cli(mods_info_csv, csv_files, plot_file):
    """
    Plot confidence metrics from all CSV_FILES, output to PLOT_FILE.
    """
    plot_everything(csv_files=csv_files, mods_info_csv=mods_info_csv, plot_file=plot_file)

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
@click.option('--format', default='csv', type=click.Choice(['csv', 'json']), help="Whether to output csv or json")
@click.option('--output', type=click.File('w'), default=sys.stdout, help='Print to this file')
@click.argument('PPN', nargs=-1)
def ppn2confs_cli(format, output, ppn):
    """
    Get the table of line and word confidences for PPN
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    table = ppn_handler.ppn2confs(ppn)
    if format == 'json':
        json.dump(table, output)
    else:
        writer = csv.writer(output)
        writer.writerow(['ppn', 'file_id', 'version', 'textline_confs', 'word_confs'])
        for row in table:
            for i in [-1, -2]:
                row[i] = ' '.join(map(str, row[i]))
            writer.writerow(row)
