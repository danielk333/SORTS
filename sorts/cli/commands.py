import argparse
import sys
import logging

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1

    comm = COMM_WORLD()

COMMANDS = dict()


def build_parser():
    parser = argparse.ArgumentParser(description='Space Object Radar Tracking Simulator command line tool')

    parser.add_argument('-v', '--verbose', help='Increase output verbosity', action='count', default=0)

    subparsers = parser.add_subparsers(help='Avalible command line interfaces', dest='command')
    subparsers.required = True

    for name, dat in COMMANDS.items():
        parser_builder, command_help = dat['parser']
        cmd_parser = subparsers.add_parser(name, help=command_help)
        parser_builder(cmd_parser)

    return parser


def add_command(name, function, parser_build, command_help=''):
    global COMMANDS
    COMMANDS[name] = dict()
    COMMANDS[name]['function'] = function
    COMMANDS[name]['parser'] = (parser_build, command_help)


def main():
    parser = build_parser()
    args = parser.parse_args()

    function = COMMANDS[args.command]['function']

    function(args)
