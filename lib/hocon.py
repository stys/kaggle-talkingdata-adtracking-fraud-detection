# -*- coding: utf-8 -*-

from pyhocon import HOCONConverter


def write_config(conf, filename, output_format):
    lines = HOCONConverter.convert(conf, output_format=output_format, indent=4)
    with open(filename, 'w') as fh:
        fh.writelines(lines)


def config2json(conf):
    lines = HOCONConverter.convert(conf, indent=0)
    return ''.join(lines).replace('\n', ' ')