#!/usr/bin/env python

"""Convenience functions for interacting with the Linux terminal

"""
import time
import sys

from tabulate import tabulate

ERASE_LINE = "\x1b[2K"
CURSOR_UP_ONE = "\x1b[1A"


def flush_table(table_data, header, chunksize=10, step_size=10, step_time=0.2):
    """ "Animates" a table by flushing multiple chunks of that table to the terminal with a sleep time between flushes."""

    up_flush = CURSOR_UP_ONE + ERASE_LINE

    lines = chunksize + 3
    for i in range(0, len(table_data), step_size):
        if step_time is not None:
            time.sleep(step_time)
        sub_table = table_data[i : (i + chunksize)]

        str_data = "\n" + tabulate(sub_table, headers=header) + "\n"
        if i > 0:
            for j in range(lines):
                sys.stdout.write(up_flush)
                sys.stdout.flush()

        sys.stdout.write(str_data)

        # last possibly un-even table
        if len(sub_table) < chunksize:
            sys.stdout.write("\n " * (chunksize - len(sub_table)))

        sys.stdout.flush()


def step_flush_table(table_data, header, table_size, step_time=None, first_step=False):
    """ "Animates" a table by flushing multiple chunks of that table to the terminal with a sleep time between flushes."""

    up_flush = CURSOR_UP_ONE + ERASE_LINE

    lines = table_size + 3
    if step_time is not None:
        time.sleep(step_time)

    # last possibly un-even table
    if len(table_data) < table_size:
        fill = "\n" * (table_size - len(table_data))
    else:
        fill = ""

    str_data = "\n" + tabulate(table_data, headers=header) + fill + "\n"

    if not first_step:
        for j in range(lines):
            sys.stdout.write(up_flush)
            sys.stdout.flush()

    sys.stdout.write(str_data)

    sys.stdout.flush()
