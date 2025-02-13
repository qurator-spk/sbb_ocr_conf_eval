import click
import requests
import json
from rich.console import Console

console = Console()

@click.group('sbb_ocr')
def cli():
    pass

@cli.command('ppn2kitodo')
@click.argument('PPN', nargs=-1)
def ppn2kitodo(ppn):
    """
    Convert PPN to Kitodo ID by calling ppn2id.pl script
    """
    PPN2ID_URL = 'http://b-lx0129/cgi-bin/kitodo_stat/ppn2id.pl'
    resp =  requests.post(PPN2ID_URL, files={
        'ppn': ' '.join(ppn),
        'PPNs wandeln': 'PPNs wandeln',
    })
    console.log(resp)
    console.log(resp.text)
