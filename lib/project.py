import sys
import logging
import re
from collections import namedtuple
from argparse import ArgumentParser
from pyhocon import ConfigFactory, ConfigTree

logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S')

Project = namedtuple('Project', ['conf'])
instance = None


def project(argv=sys.argv):
    global instance

    if instance is not None:
        return instance
    else:
        pattern = re.compile('-D(.*)=(.*)')
        conf_override = dict()
        argv_filtered = []
        for a in argv:
            m = pattern.match(a)
            if m is not None:
                conf_override[m.group(1)] = m.group(2)
            else:
                argv_filtered.append(a)

        parser = ArgumentParser()
        parser.add_argument('--conf', default='application.conf')
        args, other = parser.parse_known_args(argv_filtered)

        conf = ConfigFactory.parse_file(args.conf)
        conf_override = ConfigFactory.from_dict(conf_override)
        conf_merged = ConfigTree.merge_configs(conf, conf_override)

        instance = Project(conf=conf_merged)

    return instance
