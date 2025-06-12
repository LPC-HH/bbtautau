"""Utilities for bbtautau package."""

from __future__ import annotations

from dataclasses import dataclass

from boostedhh.utils import add_bool_arg

from bbtautau.HLTs import HLTs


@dataclass
class Channel:
    """Channel."""

    key: str  # key in dictionaries etc.
    label: str  # label for plotting
    data_samples: list[str]  # datasets for this channel
    hlt_types: list[str]  # list of HLT types
    isLepton: bool  # lepton channel or fully hadronic
    tagger_label: str  # label for tagger score used
    txbb_cut: float  # cut on bb tagger score
    txtt_cut: float  # cut on tt tagger score
    txtt_BDT_cut: float  # cut on tt BDT score
    tt_mass_cut: tuple[str, list[float]]  # cut on tt mass
    lepton_dataset: str = None  # lepton dataset (if applicable)

    def triggers(
        self,
        year: str,
        **hlt_kwargs,
    ):
        """Get triggers for a given year for this channel."""
        return HLTs.hlts_by_type(year, self.hlt_types, **hlt_kwargs)

    def lepton_triggers(
        self,
        year: str,
        **hlt_kwargs,
    ):
        """Get lepton triggers for a given year for this channel."""
        if self.lepton_dataset is None:
            return None

        return HLTs.hlts_by_dataset(year, self.lepton_dataset, **hlt_kwargs)


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

    add_bool_arg(
        parser, "fatjet-bb-preselection", default=False, help="apply bb preselection to fatjets"
    )
