#!/usr/bin/env python
import sys
import re
import getpass
import argparse
from datetime import datetime, timedelta
import subprocess

try:
    import spacetrack
except ImportError:
    spacetrack = None

from .commands import add_command

_iso_fmt = "%Y-%m-%d"
_td_regx = re.compile(
    r"^((?P<days>[\.\d]+?)d)? *((?P<hours>[\.\d]+?)h)? "
    + r"*((?P<minutes>[\.\d]+?)m)? *((?P<seconds>[\.\d]+?)s)?$"
)


def parse_timedelta(time_str):
    """
    Parse a time string e.g. (2h13m) into a timedelta object.

    Modified from virhilo's answer at https://stackoverflow.com/a/4628148/851699

    :param time_str: A string identifying a duration.  (eg. 2h13m)
    :return datetime.timedelta: A datetime.timedelta object
    """
    parts = _td_regx.match(time_str)
    assert parts is not None, (
        f"Could not parse any time information from '{time_str}'.  "
        + "Examples of valid strings: '8h', '2d8h5m20s', '2m4s'"
    )
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return timedelta(**time_params)


def main(args):
    assert spacetrack is not None, "spacetrack package not installed, run `pip install spacetrack`"
    # end date argument can be ISO format datetime or 'now'
    if args.end_date == "now":
        dt1 = datetime.now()
    else:
        dt1 = datetime.strptime(args.end_date, _iso_fmt)

    # start date argument can be absolute or timedelta
    try:
        dt0 = dt1 - parse_timedelta(args.start_date)
    except AssertionError:
        dt0 = datetime.strptime(args.start_date, _iso_fmt)

    if args.output is not sys.stdout:
        print(f"Getting TLEs for the range [{dt0} -> {dt1}]")
        print(f"Output to {args.output.name}")

    drange = spacetrack.operators.inclusive_range(dt0, dt1)
    kwargs = {}

    if args.name is not None:
        name_op = spacetrack.operators.like(args.name)
        kwargs["object_name"] = name_op
        kwargs["epoch"] = drange
    else:
        kwargs["publish_epoch"] = drange

    if args.secret_tool_key is not None:
        res = subprocess.run(
            ["secret-tool", "lookup", "username"] + args.secret_tool_key,
            capture_output=True,
            text=True,
        )
        user = res.stdout
        res = subprocess.run(
            ["secret-tool", "lookup", "password"] + args.secret_tool_key,
            capture_output=True,
            text=True,
        )
        passwd = res.stdout
    elif args.credentials is not None:
        raise NotImplementedError("Add input of username/password from file")
    else:
        user = input("Username for space-track.org:")
        passwd = getpass.getpass("Password for " + user + ":")

    st = spacetrack.SpaceTrackClient(user, passwd)

    if args.name is not None:
        print('Using CLASS "tle"...')
        lines = st.tle(iter_lines=True, orderby="TLE_LINE1", format="tle", **kwargs)
    else:
        print('Using CLASS "tle_publish"...')
        lines = st.tle_publish(iter_lines=True, orderby="TLE_LINE1", format="tle", **kwargs)
    lineno = 0
    for line in lines:
        args.output.write(line + "\n")
        lineno += 1

    if args.output is not sys.stdout:
        print(f"Wrote {lineno} lines to {args.output.name}")


def parser_build(parser):
    parser.add_argument(
        "start_date",
        type=str,
        nargs="?",
        default="7d",
        help='Start date of snapshot [ISO] or timedelta ("24h", "12d", etc)',
    )
    parser.add_argument(
        "end_date", type=str, nargs="?", default="now", help="End date of snapshot [ISO]"
    )
    parser.add_argument("output", nargs="?", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("--secret-tool-key", "-k", nargs=1)
    parser.add_argument(
        "--credentials",
        "-c",
        nargs=1,
        help="File containing username and password for space-track.org",
    )
    parser.add_argument(
        "--name", "-n", default=None, help='Name of the object to match with the "like" operator'
    )

    return parser


add_command(
    name="spacetrack",
    function=main,
    parser_build=parser_build,
    command_help="Download tle snapshot from space-track",
)
