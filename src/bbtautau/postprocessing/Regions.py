"""
Defines all the analysis regions.
****Important****: Region names used in the analysis cannot have underscores because of a rhalphalib convention.
Author(s): Raghav Kansal
"""

from __future__ import annotations

from boostedhh.utils import CUT_MAX_VAL

from bbtautau.bbtautau_utils import Channel
from bbtautau.postprocessing.utils import Region


def get_selection_regions(channel: Channel, use_bdt: bool = False):
    pass_cuts = {
        "bbFatJetPt": [250, CUT_MAX_VAL],
        "ttFatJetPt": [200, CUT_MAX_VAL],
        "bbFatJetParTXbbvsQCD": [channel.txbb_cut, CUT_MAX_VAL],
    }

    fail_cuts = {
        "bbFatJetPt": [250, CUT_MAX_VAL],
        "ttFatJetPt": [200, CUT_MAX_VAL],
    }

    if use_bdt:
        pass_cuts[f"BDTScore{channel.tagger_label}vsAll"] = [channel.txtt_BDT_cut, CUT_MAX_VAL]
        fail_cuts[f"bbFatJetParTXbbvsQCD+BDTScore{channel.tagger_label}vsAll"] = [
            [-CUT_MAX_VAL, channel.txbb_cut],
            [-CUT_MAX_VAL, channel.txtt_BDT_cut],
        ]
    else:
        pass_cuts[f"ttFatJetParTX{channel.tagger_label}vsQCDTop"] = [channel.txtt_cut, CUT_MAX_VAL]
        pass_cuts[channel.tt_mass_cut[0]] = channel.tt_mass_cut[1]
        fail_cuts[channel.tt_mass_cut[0]] = channel.tt_mass_cut[1]
        fail_cuts[f"bbFatJetParTXbbvsQCD+ttFatJetParTX{channel.tagger_label}vsQCDTop"] = [
            [-CUT_MAX_VAL, channel.txbb_cut],
            [-CUT_MAX_VAL, channel.txtt_cut],
        ]

    regions = {
        # {label: {cutvar: [min, max], ...}, ...}
        "pass": Region(
            cuts=pass_cuts,
            signal=True,
            label="Pass",
        ),
        "fail": Region(
            cuts=fail_cuts,
            signal=False,
            label="Fail",
        ),
    }

    return regions
