import json
import sys
import csv
import click
from .ppn_handler import PpnHandler, PpnHandlerConfig
from .ocr_wc_analysis import plot_everything
import pandas as pd
import sqlite3

@click.group('sbb_ocr')
def cli():
    """
    Evaluating OCR @StabiBerlin
    """
    pass


@cli.command('plot')
@click.option('-g', '--genre', 'search_genre', help='genre to be evaluated')
@click.option('-m', '--mods-info', 'mods_info_csv', default="mods_info_df_2024-11-27.csv", help='mods_info CSV for PPN metadata')
@click.option('-d', '--date-range', 'date_range', nargs=2, type=(int, int), help='Year range for filtering data, specify Year_Start Year_End')
@click.argument('CSV_FILES', nargs=-1)
@click.argument('PLOT_FILE')
def plot_cli(search_genre, mods_info_csv, csv_files, plot_file, date_range):
    """
    Plot confidence metrics from all CSV_FILES, output to PLOT_FILE.
    """
    
    if date_range is None:
        year_start, year_end = (None, None)
    else:
        year_start, year_end = date_range
    
    plot_everything(csv_files=csv_files, mods_info_csv=mods_info_csv, search_genre=search_genre, plot_file=plot_file, year_start=year_start, year_end=year_end)

@cli.command('convert-mods-info')
@click.argument('MODS_INFO_SQLITE')
@click.argument('MODS_INFO_CSV')
def convert_mods_info(mods_info_sqlite, mods_info_csv):
    """
    Convert mods_info.parquet.sqlite3 to CSV and remove all non-zero indexed names.
    """
    con = sqlite3.connect(mods_info_sqlite)
    mods_info_df = pd.read_sql("SELECT * FROM mods_info", con, index_col="recordInfo_recordIdentifier")
    columns_to_drop = []
    for c in mods_info_df.columns:
        if c.startswith("name") and not c.startswith("name0"):
            columns_to_drop.append(c)
    mods_info_df.drop(columns=columns_to_drop, inplace=True)
    mods_info_df.to_csv(mods_info_csv, index = False)

@cli.command('ppn2kitodo')
@click.argument('PPN', nargs=-1)
def ppn2kitodo_cli(ppn):
    """
    Translate PPN into Kitodo ID
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    print(json.dumps(ppn_handler.ppn2kitodo(ppn)))

@cli.command('ppn2pagexml')
@click.option('--format', default='csv', type=click.Choice(['csv', 'json']), help="Whether to output csv or json")
@click.option('--output', type=click.File('w'), default=sys.stdout, help='Print to this file')
@click.argument('PPN', nargs=-1)
def ppn2pagexml_cli(format, output, ppn):
    """
    Get a list of PAGE-XML files for PPN
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    table = ppn_handler.ppn2pagexml(ppn)
    if format == 'csv':
        writer = csv.writer(output)
        writer.writerow(['ppn', 'pagexml'])
        for ppn, pagexml_list in table.items():
            for pagexml in pagexml_list:
                writer.writerow([ppn, pagexml])
    else:
        json.dump({k: [str(x) for x in v] for k, v in table.items()}, output)

@cli.command('ppn2mets')
@click.argument('PPN', nargs=-1)
def ppn2mets_cli(ppn):
    """
    Get METS file for PPN
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    print(ppn_handler.ppn2mets(ppn))

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
