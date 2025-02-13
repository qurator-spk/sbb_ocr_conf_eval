import json
import sys
import csv

import click

from .lib import PpnHandler, PpnHandlerConfig

@click.group('sbb_ocr')
def cli():
    pass

@cli.command('extract-confs')
@click.argument('PPN', nargs=-1)
def extract_confs(ppn):
    print(json.dumps(ppn2kitodo(ppn)))

@cli.command('ppn2kitodo')
@click.argument('PPN', nargs=-1)
def ppn2kitodo_cli(ppn):
    ppn_handler = PpnHandler(PpnHandlerConfig())
    print(json.dumps(ppn_handler.ppn2kitodo(ppn)))

@cli.command('ppn2pagexml')
@click.argument('PPN', nargs=-1)
def ppn2kitodo_cli(ppn):
    ppn_handler = PpnHandler(PpnHandlerConfig())
    print(ppn_handler.ppn2pagexml(ppn))

@cli.command('ppn2confs')
@click.option('--format', default='json', type=click.Choice(['csv', 'json']), help="Whether to output csv or json")
@click.argument('PPN', nargs=-1)
def ppn2kitodo_cli(format, ppn):
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
