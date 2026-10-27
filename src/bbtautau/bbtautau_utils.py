"""Utilities for bbtautau package."""

from __future__ import annotations

from boostedhh.utils import add_bool_arg


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
        choices=["v12_private", "v15"],
        help="NanoAOD version",
    )

    parser.add_argument(
        "--fatjet-pt-cut",
        type=float,
        default=None,
        help="pt cut for fatjets in skimmer",
    )

    parser.add_argument(
        "--fatjet-bb-preselection-cut",
        type=float,
        default=None,
        help="if set, require >=1 fatjet with ParTXbbvsQCDTop >= this cut",
    )

    add_bool_arg(
        parser,
        "save-kinematic-systematics",
        default=False,
        help="save and propagate kinematic systematic variations (lepton/tau scale and resolution, JES/JER) through object selection, derived variables, and shifted SF matching",
    )

    parser.add_argument(
        "--fatjet-tt-preselection-cut",
        type=float,
        default=None,
        help=(
            "if set, require >=1 fatjet with "
            "max(ParTXtauhtauevsQCDTop, ParTXtauhtaumvsQCDTop, ParTXtauhtauhvsQCDTop) >= this cut"
        ),
    )

    parser.add_argument(
        "--prescale-factor",
        type=int,
        default=None,
        help="prescale factor for skimmer",
    )
