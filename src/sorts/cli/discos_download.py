#!/usr/bin/env python
import pathlib
import sys
import time
import getpass
import json
import subprocess
import requests

from .commands import add_command


URL = "https://discosweb.esoc.esa.int"

_ID_MAP = {
    "NORAD": "satno",
    "COSPAR": "cosparId",
    "DISCOS": "id",
}


def main(args):
    if isinstance(args.object_id, str):
        args.object_id = [args.object_id]

    if len(args.output) == 0:
        output_pth = sys.stdout
    else:
        output_pth = pathlib.Path(args.output)
        if output_pth.exists() and not args.override:
            raise FileExistsError(f"{output_pth} exists")
        elif output_pth.exists() and args.override:
            print("Overriding existing file..")
            output_pth.unlink()
        output_pth = open(output_pth, "w")

    if args.type == "NORAD":
        tmp_oids = []
        for oid in args.object_id:
            oid = oid[:-1] if oid[-1].isalpha() else oid
            tmp_oids.append(oid)
        args.object_id = tmp_oids

    elif args.type == "COSPAR":
        args.object_id = [f"'{oid}'" for oid in args.object_id]

    if args.secret_tool_key is not None:
        res = subprocess.run(
            ["secret-tool", "lookup", "token"] + args.secret_tool_key,
            capture_output=True,
            text=True,
        )
        token = res.stdout
    elif args.credentials is not None:
        raise NotImplementedError("Add input of username/password from file")
    else:
        token = getpass.getpass("API token for:")

    current_page = 1
    params = {
        "sort": "id",
        "page[size]": 100,
    }
    if len(args.object_id) > 0:
        oids = ",".join(args.object_id)
        if len(args.object_id) == 1:
            filt_str = f"{_ID_MAP[args.type]}={oids}"
        else:
            filt_str = f"in({_ID_MAP[args.type]},({oids}))"
        params["filter"] = filt_str
        print(f'Fetching data for "{filt_str}"')
    else:
        print("Fetching all data")

    objs = []
    while True:
        print("getting page ", current_page)
        params["page[number]"] = current_page
        response = requests.get(
            f"{URL}/api/objects",
            headers={
                "Authorization": f"Bearer {token}",
                "DiscosWeb-Api-Version": "2",
            },
            params=params,
        )

        if response.status_code == 429:
            retry_interval = int(response.headers["Retry-After"])
            print(f"API requests exceeded, sleeping for {retry_interval} s")
            time.sleep(retry_interval + 1)
            continue
        else:
            current_page += 1

        doc = response.json()
        if response.ok:
            objs += doc["data"]
            print(f"{len(doc['data'])} Entries found...")
            if doc["links"]["next"] is None:
                print("No more pages, exiting...")
                break
        else:
            print("Error...")
            print(doc["errors"])

    print(f"{len(objs)} Entries downloaded...")
    json.dump(objs, output_pth, indent=2)
    if len(args.output) > 0:
        output_pth.close()


def parser_build(parser):
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="NORAD",
        choices=["NORAD", "COSPAR", "DISCOS"],
        help='Input ID, ID variant can be set with "--type"',
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        default="",
        help="Path to file where output should be written.",
    )
    parser.add_argument(
        "--secret-tool-key",
        "-k",
        nargs=1,
        type=str,
        metavar="KEY",
        help='Attribute [named "token"] value [key] to fetch secret from',
    )
    parser.add_argument(
        "--credentials",
        "-c",
        nargs=1,
        metavar="FILE",
        help="File containing DISCOSweb token",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override output file",
    )
    parser.add_argument(
        "object_id",
        metavar="ID",
        type=str,
        nargs="+",
        help='Input ID(s), ID variant can be set with "--type"',
    )

    return parser


add_command(
    name="discos",
    function=main,
    parser_build=parser_build,
    command_help="Download object information from ESA discosweb",
)
