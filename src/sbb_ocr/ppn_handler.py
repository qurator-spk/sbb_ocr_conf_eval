import json
import sys
import os
from pathlib import Path
from typing import Optional

from attr import define
import requests
from rich.console import Console
from bs4 import BeautifulSoup

from ocrd_models.ocrd_page import parse

@define
class PpnHandlerConfig():
    ppn2id_url = os.environ['PPN2ID_URL'] if 'PPN2ID_URL' in os.environ else "Must set PPN2ID_URL"
    nfs_goobi: str = f"{os.environ['HOME']}/nfs_goobi"
    nfs_source: str = f"{os.environ['HOME']}/nfs_source"

    def mets_archive(self, ppn: str) -> Path:
        return Path(self.nfs_source, 'new_presentation/dc-indexing/indexed_mets/', f'{ppn}.xml')

    def ocr_archive(self, kitodo_id: str) -> Path:
        return Path(self.nfs_goobi, 'archiv', kitodo_id, 'ocr')

class PpnHandler():
    config : PpnHandlerConfig

    def __init__(self, config):
        self.config = config
        self.console = Console(file=sys.stderr)

    def extract_confs(self, page_fname):
        """
        Load PAGE-XML file, return lists of textline and word confidences
        """
        pcgts = parse(page_fname)

        confs_textline, confs_word = [], []

        for textline in pcgts.get_Page().get_AllTextLines():  # type: ignore
            textline_conf = textline.get_TextEquiv()[0].conf
            if textline_conf is not None:
                confs_textline.append(textline_conf)
            for word in textline.get_Word():
                word_conf = word.get_TextEquiv()[0].conf
                if word_conf is not None:
                    confs_word.append(word_conf)
        return [confs_textline, confs_word]

    def normalize_ppn(self, ppn_list):
        return ['PPN' + x.replace('PPN', '') for x in ppn_list]

    def ppn2confs(self, ppn_list: list[str]):
        """
        Return a list of 5-tuples
        - PPN
        - file_id
        - version timestamp
        - word confidences
        - line confidences
        """
        ppn_list = self.normalize_ppn(ppn_list)
        mapped = self.ppn2pagexml(ppn_list)
        ret = []
        i = 1
        for ppn, pagexml_list in mapped.items():
            self.console.log(f'[{i:3d}/{len(mapped.keys()):3d}] Extracting confidences for {len(pagexml_list)} PAGE-XML files of PPN {ppn}')
            for pagexml in pagexml_list:
                file_id = pagexml.name
                version = pagexml.parent.name
                ret.append([ppn, file_id, version, *self.extract_confs(pagexml)])
            i += 1
        return ret

    def ppn2pagexml(self, ppn_list: list[str]) -> dict[str, list[Path]]:
        """
        Return a mapping of PPN to all PAGE-XML files for that PPN.
        """
        ppn_list = self.normalize_ppn(ppn_list)
        mapped = self.ppn2kitodo(ppn_list)
        ret = {}
        i = 1
        for ppn, kitodo_id in mapped.items():
            self.console.log(f'[{i:3d}/{len(mapped.keys()):3d}] Listing PAGE-XML for PPN {ppn} / Kitodo ID {kitodo_id}')
            ret[ppn] = sorted(self.config.ocr_archive(kitodo_id).glob('*/page/*/*.xml'))
            i += 1
        return ret

    def ppn2kitodo(self, ppn_list: list[str]) -> dict[str, str]:
        """
        Convert PPN to Kitodo ID by calling ppn2id.pl script
        """
        ppn_list = self.normalize_ppn(ppn_list)
      
        self.console.log(f"Retrieving {len(ppn_list)} PPNs from {self.config.ppn2id_url}")
        resp =  requests.post(self.config.ppn2id_url, data={
            'ppn': ' '.join(ppn_list),
            'PPNs wandeln': 'PPNs wandeln',
        })
        if resp.status_code >= 400:
            self.console.log(f"Request to {self.config.ppn2id_url} failed")
            self.console.log(resp.headers)
            self.console.log(resp.text)
            raise ValueError(resp.text)
        soup = BeautifulSoup(resp.text, 'html.parser')
        kitodo_ids = soup.find('input', dict(name='ids')).get('value')  # type: ignore
        kitodo_ids = kitodo_ids.replace('id:', '').replace('"', '').split(' ')  # type: ignore
        retrieved = dict(zip(ppn_list, kitodo_ids))
        self.console.log(f"Retrieved f{len(retrieved)} Kitodo IDs")
        return retrieved

    def ppn2mets(self, ppn_list: list[str]) -> dict[str, str]:
        """
        Retrieve METS-XML for PPN from Kitodo index NFS location.
        """
        ppn_list = self.normalize_ppn(ppn_list)
        ret = {}
        for i, ppn in enumerate(ppn_list):
            self.console.log(f'[{i:3d}/{len(ppn_list):3d}] Retrieving METS for PPN {ppn}')
            ret[ppn] = self.config.mets_archive(ppn)
        return ret
