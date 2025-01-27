"""Utilities for bbtautau package."""

from __future__ import annotations


def parse_common_run_args(parser):
    parser.add_argument(
        "--processor",
        required=True,
        help="processor",
        type=str,
        choices=["skimmer"],
    )

    parser.add_argument(
        "--region",
        help="region",
        default="signal",
        choices=["signal"],
        type=str,
    )

    parser.add_argument(
        "--nano-version",
        type=str,
        default="v12_private",
        choices=["v12_private"],
        help="NanoAOD version",
    )

    parser.add_argument(
        "--fatjet-pt-cut",
        type=float,
        default=None,
        help="pt cut for fatjets in skimmer",
    )
