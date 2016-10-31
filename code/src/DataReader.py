# -*- coding: utf-8 -*-

import pandas as pd
import os
import xml.etree.ElementTree as ET
import itertools
import re
import sys

from dateutil.parser import parse as dt_parse
from datetime import datetime

class DataReader():
    properties = {
        "Id": int,
        "PostTypeId": int,
        "ParentId": int,
        "AcceptedAnswerId": int,
        "CreationDate": dt_parse,
        "Score": int,
        "ViewCount": int,
        "OwnerUserId": int,
#         "LastEditorUserId": int,
#         "LastEditDate": dt_parse,
#         "LastActivityDate": dt_parse,
#         "CommunityOwnedDate": dt_parse,
#         "ClosedDate": dt_parse,
        "Tags": lambda x: re.findall('<(.*?)>', x),
        "AnswerCount": int,
#         "CommentCount": int,
#         "FavoriteCount": int,
        "PostId": int,
        "VoteTypeId": int,
    }

    def __init__(self, fname, verbose=False):
        self.fname = fname
        self.verbose = verbose

    def print_message(self, msg):
        print msg

    def read_file(self):
        if self.verbose:
            self.print_message('Reading file...')

        with open(self.fname) as fl_handle:
            return fl_handle.readlines()

    def gen_dict_from_row(self, row):
        tree = ET.fromstring(row)
        tree_dict = tree.attrib
        ret_dict = {}

        for key in tree.attrib:
            if key in DataReader.properties:
                func = DataReader.properties[key]
                ret_dict[key] = func(tree_dict[key])
            else:
                ret_dict[key] = tree_dict[key]

        return ret_dict

    def get_data_ldict(self, xml):
        if self.verbose:
            self.print_message('Converting...')

        data_rows = []
        count = 0
        ttl = len(xml) / 10
        start = datetime.now()

        for line in xml:
            if 'row' in line:
                dct = self.gen_dict_from_row(line)
                data_rows.append(dct)
                count += 1.

                if count % ttl == 0:
                    it = int(count / ttl)
                    remaining = 10 - it

                    etpi = datetime.now() - start
                    espi = etpi.seconds * 1. / it

                    sys.stdout.write('\rProgress | %s | %s%s || Estimated time remaining: %s seconds' %
                                     ('█' * it + '-' * remaining, it*10, '%', (remaining*espi) ))
                    sys.stdout.flush()

        if self.verbose:
            self.print_message('\nConversion complete...')

        return data_rows

    def convert_to_frame(self, data_rows):
        if self.verbose:
            self.print_message('Forming data frame...')

        self._df = pd.DataFrame(data_rows)

    def read_data(self):
        xml = self.read_file()
        data_rows = self.get_data_ldict(xml)

        self.convert_to_frame(data_rows)

