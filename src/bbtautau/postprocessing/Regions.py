"""
Defines all the analysis regions.
****Important****: Region names used in the analysis cannot have underscores because of a rhalphalib convention.
Author(s): Raghav Kansal
"""

from __future__ import annotations

from boostedhh.utils import CUT_MAX_VAL

from bbtautau.bbtautau_utils import Channel
from bbtautau.postprocessing.utils import Region


def get_selection_regions(channel: Channel):
    regions = {
        # {label: {cutvar: [min, max], ...}, ...}
        "pass": Region(
            cuts={
                "bbFatJetPt": [250, CUT_MAX_VAL],
                "ttFatJetPt": [200, CUT_MAX_VAL],
                f"ttFatJet{channel.tt_mass_cut[0]}": channel.tt_mass_cut[1],
                "bbFatJetParTXbbvsQCD": [channel.txbb_cut, CUT_MAX_VAL],
                f"ttFatJetParTX{channel.tagger_label}vsQCDTop": [channel.txtt_cut, CUT_MAX_VAL],
            },
            signal=True,
            label="Pass",
        ),
        "fail": Region(
            cuts={
                "bbFatJetPt": [250, CUT_MAX_VAL],
                "ttFatJetPt": [200, CUT_MAX_VAL],
                f"ttFatJet{channel.tt_mass_cut[0]}": channel.tt_mass_cut[1],
                # invert at least one of the cuts
                f"bbFatJetParTXbbvsQCD+ttFatJetParTX{channel.tagger_label}vsQCDTop": [
                    [-CUT_MAX_VAL, channel.txbb_cut],
                    [-CUT_MAX_VAL, channel.txtt_cut],
                ],
                f"ttFatJetParTX{channel.tagger_label}vsQCDTop": [0.3, CUT_MAX_VAL],
            },
            signal=False,
            label="Fail",
        ),
    }

    return regions
