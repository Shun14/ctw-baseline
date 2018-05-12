# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import json
import settings
from collections import defaultdict
from pythonapi import anno_tools


def main():
    counts = defaultdict(int)
    with open('../data/text_file_pro.txt') as f:
        for line in f.read().splitlines():
            for char in line:
                text = char
                assert 1 == len(text)
                counts[text] += 1
    cates = [{'text': k,'trainval': v,} for k, v in counts.items()]
    # cates.sort(key=lambda o: (-o['trainval'], o['text']))
    for i, o in enumerate(cates):
        o['cate_id'] = i
    with codecs.open(settings.CATES, 'w', 'utf-8') as f:
        json.dump(cates, f, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
