"""
Creates datacards for Higgs Combine using hist.Hist templates output from postprocessing.py
(1) Adds systematics for all samples,
(2) Sets up data-driven QCD background estimate ('rhalphabet' method)

Based on https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py

Authors: Raghav Kansal
"""

from __future__ import annotations

# from utils import add_bool_arg
import argparse
import logging
import pickle
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rhalphalib as rl
from boostedhh.hh_vars import (
    LUMI,
    data_key,
    jecs,
    jmsr,
    jmsr_keys,
)
from boostedhh.hh_vars import years as hh_years
from hist import Hist

from bbtautau.bbtautau_utils import Channel
from bbtautau.postprocessing.datacardHelpers import (
    ShapeVar,
    Syst,
    add_bool_arg,
    combine_templates,
    rem_neg,
    sum_templates,
)
from bbtautau.postprocessing.Samples import (
    CHANNELS,
    SIGNALS,
    SIGNALS_CHANNELS,
    sig_keys_ggf,
    sig_keys_vbf,
    single_h_keys,
    ttbar_keys,
)

try:
    rl.util.install_roofit_helpers()
    rl.ParametericSample.PreferRooParametricHist = False
except:
    print("rootfit install failed - not an issue for VBF")

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
adjust_posdef_yields = False


warnings.filterwarnings("ignore", message="CMS_bbtautau_boosted_.*mcstat.*")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--templates-dir",
    default="",
    type=str,
    help="input pickle file of dict of hist.Hist templates",
)

add_bool_arg(parser, "sig-separate", "separate templates for signals and bgs", default=False)

parser.add_argument("--cards-dir", default="cards", type=str, help="output card directory")

parser.add_argument("--mcstats-threshold", default=100, type=float, help="mcstats threshold n_eff")
parser.add_argument(
    "--epsilon",
    default=1e-2,
    type=float,
    help="epsilon to avoid numerical errs - also used to decide whether to add mc stats error",
)
parser.add_argument(
    "--scale-templates", default=None, type=float, help="scale all templates for bias tests"
)
parser.add_argument(
    "--min-qcd-val", default=1e-3, type=float, help="clip the pass QCD to above a minimum value"
)

add_bool_arg(parser, "only-sm", "Only add SM HH samples", default=False)

parser.add_argument(
    "--sigs",
    default=SIGNALS,
    nargs="*",
    type=str,
    help="specify signals",
)

parser.add_argument(
    "--channels",
    default="all",
    nargs="*",
    choices=list(CHANNELS.keys()) + ["all"],
    type=str,
    help="specify channels",
)

parser.add_argument(
    "--nTF",
    default=[2],
    type=int,
    nargs="*",
    help="order of polynomial for TF.",
)
parser.add_argument(
    "--regions",
    default=["pass"],
    nargs="*",
    type=str,
    help="regions for which to make cards",
    choices=["pass"],
)
parser.add_argument("--model-name", default=None, type=str, help="output model name")
parser.add_argument(
    "--year",
    type=str,
    default="2022-2023",
    choices=hh_years + ["2022-2023"],
    help="years to make datacards for",
)
add_bool_arg(parser, "dd-dyjets", "use data-driven DY jets estimate", default=True)
add_bool_arg(parser, "mcstats", "add mc stats nuisances", default=True)
add_bool_arg(parser, "bblite", "use barlow-beeston-lite method", default=True)
add_bool_arg(parser, "unblinded", "unblinded so skip blinded parts", default=False)
add_bool_arg(parser, "jmsr", "Do JMS/JMR uncertainties", default=False)
add_bool_arg(parser, "jesr", "Do JES/JER uncertainties", default=False)
add_bool_arg(
    parser, "thu-hh", "Add THU_HH uncertainty; remove for HH inference framework", default=True
)
args = parser.parse_args()


CMS_PARAMS_LABEL = "CMS_bbtautau_boosted"
MCB_LABEL = "MCBlinded"
qcd_data_key = "qcd_datadriven"
blind_window = [110, 140]

# if args.nTF is None:
#     if args.regions == "all":
#         args.nTF = [0, 0, 0, 0]
#     else:
#         args.nTF = [0]

print("Transfer factors:", args.nTF)

signal_regions = ["pass"] if args.regions == "pass" else args.regions

channels = (
    list(CHANNELS.values()) if args.channels == "all" else [CHANNELS[ch] for ch in args.channels]
)

# (name in templates, name in cards)
mc_samples = OrderedDict(
    [
        ("ttbarsl", "ttbarsl"),
        ("ttbarll", "ttbarll"),
        ("ttbarhad", "ttbarhad"),
        # ("dyjets", "dyjets"),
        ("wjets", "wjets"),
        ("zjets", "zjets"),
        ("hbb", "hbb"),
    ]
)

mc_samples_sig = OrderedDict(
    [
        ("bbtt", "ggHH_kl_1_kt_1_13p6TeV_hbbhtauau"),
        ("bbtt-kl0", "ggHH_kl_0_kt_1_13p6TeV_hbbhtauau"),
        ("bbtt-kl2p45", "ggHH_kl_2p45_kt_1_13p6TeV_hbbhtauau"),
        ("bbtt-kl5", "ggHH_kl_5_kt_1_13p6TeV_hbbhtauau"),
        ("vbfbbtt", "qqHH_CV_1_C2V_1_kl_1_13p6TeV_hbbhtauau"),
        ("vbfbbtt-k2v0", "qqHH_CV_1_C2V_0_kl_1_13p6TeV_hbbhtauau"),
        ("vbfbbtt-kv1p74-k2v1p37-kl14p4", "qqHH_CV_1p74_C2V_1p37_kl_14p4_13p6TeV_hbbhtauau"),
        ("vbfbbtt-kvm0p012-k2v0p03-kl10p2", "qqHH_CV_m0p012_C2V_0p03_kl_10p2_13p6TeV_hbbhtauau"),
        ("vbfbbtt-kvm0p758-k2v1p44-klm19p3", "qqHH_CV_m0p758_C2V_1p44_kl_m19p3_13p6TeV_hbbhtauau"),
        (
            "vbfbbtt-kvm0p962-k2v0p959-klm1p43",
            "qqHH_CV_m0p962_C2V_0p959_kl_m1p43_13p6TeV_hbbhtauau",
        ),
        ("vbfbbtt-kvm1p21-k2v1p94-klm0p94", "qqHH_CV_m1p21_C2V_1p94_kl_m0p94_13p6TeV_hbbhtauau"),
        ("vbfbbtt-kvm1p6-k2v2p72-klm1p36", "qqHH_CV_m1p6_C2V_2p72_kl_m1p36_13p6TeV_hbbhtauau"),
        ("vbfbbtt-kvm1p83-k2v3p57-klm3p39", "qqHH_CV_m1p83_C2V_3p57_kl_m3p39_13p6TeV_hbbhtauau"),
        ("vbfbbtt-kvm2p12-k2v3p87-klm5p96", "qqHH_CV_m2p12_C2V_3p87_kl_m5p96_13p6TeV_hbbhtauau"),
    ]
)

bg_keys = list(mc_samples.keys())

if args.only_sm:
    sig_keys_ggf = [f"bbtt{channel.key}" for channel in channels]
    sig_keys_vbf = [f"vbfbbtt{channel.key}" for channel in channels]

all_sig_keys = SIGNALS_CHANNELS
sig_keys = []
hist_names = {}  # names of hist files for the samples

for key in all_sig_keys:
    # check in case specific sig samples are specified
    if args.sigs is None or key in args.sigs:
        for channel in channels:
            # change names to match HH combination convention
            mc_samples[f"{key}{channel.key}"] = mc_samples_sig[key]
            sig_keys.append(f"{key}{channel.key}")


all_mc = list(mc_samples.keys())


years = hh_years if args.year == "2022-2023" else [args.year]
full_lumi = LUMI[args.year]

# jmsr_keys = sig_keys + ["vhtobb", "zz", "nozzdiboson"]


br_hbb_values = {key: 1.0124**2 for key in sig_keys}
br_hbb_values.update({key: 1.0124 for key in single_h_keys})
br_hbb_values_down = {key: 0.9874**2 for key in sig_keys}
br_hbb_values_down.update({key: 0.9874 for key in single_h_keys})
# dictionary of nuisance params -> (modifier, samples affected by it, value)
nuisance_params = {
    # https://gitlab.cern.ch/hh/naming-conventions#experimental-uncertainties
    # https://gitlab.cern.ch/hh/naming-conventions#theory-uncertainties
    "BR_hbb": Syst(
        prior="lnN",
        samples=sig_keys + single_h_keys,
        value=br_hbb_values,
        value_down=br_hbb_values_down,
        diff_samples=True,
    ),
    "pdf_gg": Syst(prior="lnN", samples=ttbar_keys, value=1.042),
    # "pdf_qqbar": Syst(prior="lnN", samples=["ST"], value=1.027),
    "pdf_Higgs_ggHH": Syst(prior="lnN", samples=sig_keys_ggf, value=1.030),
    "pdf_Higgs_qqHH": Syst(prior="lnN", samples=sig_keys_vbf, value=1.021),
    # TODO: add these for single Higgs backgrounds
    # "pdf_Higgs_gg": Syst(prior="lnN", samples=ggfh_keys, value=1.019),
    "QCDscale_ttbar": Syst(
        prior="lnN",
        samples=ttbar_keys,
        # value={"ST": 1.03, "ttbar": 1.024},
        value={"ttbarhad": 1.03, "ttbarsl": 1.03, "ttbarll": 1.03},
        # value_down={"ST": 0.978, "ttbar": 0.965},
        value_down={"ttbarhad": 0.978, "ttbarsl": 0.978, "ttbarll": 0.978},
        diff_samples=True,
    ),
    "QCDscale_qqHH": Syst(prior="lnN", samples=sig_keys_vbf, value=1.0003, value_down=0.9996),
    "THU_HH": Syst(
        prior="lnN",
        samples=sig_keys_ggf,
        value={"bbtt": 1.06, "bbtt-kl0": 1.08, "bbtt-kl2p45": 1.06, "bbtt-kl5": 1.18},
        value_down={"bbtt": 0.77, "bbtt-kl0": 0.82, "bbtt-kl2p45": 0.75, "bbtt-kl5": 0.87},
        diff_samples=True,
    ),
    # apply 2022 uncertainty to all MC (until 2023 rec.)
    "lumi_2022": Syst(prior="lnN", samples=all_mc, value=1.014),
}
if not args.thu_hh:
    del nuisance_params["THU_HH"]

nuisance_params_dict = {
    param: rl.NuisanceParameter(param, syst.prior) for param, syst in nuisance_params.items()
}

# dictionary of correlated shape systematics: name in templates -> name in cards, etc.
corr_year_shape_systs = {
    # "JES": Syst(name="CMS_scale_j", prior="shape", samples=all_mc),
    "trigger": Syst(name=f"{CMS_PARAMS_LABEL}_trigger", prior="shape", samples=all_mc),
    "TXbbSF_correlated": Syst(
        name=f"{CMS_PARAMS_LABEL}_txbb_sf_correlated",
        prior="shape",
        samples=sig_keys,
        pass_only=True,
        convert_shape_to_lnN=True,
    ),
    # "FSRPartonShower": Syst(name="ps_fsr", prior="shape", samples=sig_keys, samples_corr=True),
    # "ISRPartonShower": Syst(name="ps_isr", prior="shape", samples=sig_keys, samples_corr=True),
    "scale": Syst(
        name=f"{CMS_PARAMS_LABEL}_QCDScaleacc",
        prior="shape",
        samples=sig_keys,  # + ["ttbar"],  # FIXME: add back ttbar later
        samples_corr=True,
        separate_prod_modes=True,
    ),
    "pdf": Syst(
        name=f"{CMS_PARAMS_LABEL}_PDFacc",
        prior="shape",
        samples=sig_keys,
        samples_corr=True,
        separate_prod_modes=True,
    ),
}

uncorr_year_shape_systs = {
    # "pileup": Syst(name="CMS_pileup", prior="shape", samples=all_mc),
    "JER": Syst(
        name="CMS_res_j",
        prior="shape",
        samples=all_mc,
        convert_shape_to_lnN=True,
        uncorr_years={
            "2022": ["2022"],
            "2022EE": ["2022EE"],
            "2023": ["2023"],
            "2023BPix": ["2023BPix"],
        },
    ),
    "JMS": Syst(
        name=f"{CMS_PARAMS_LABEL}_jms",
        prior="shape",
        samples=jmsr_keys,
        uncorr_years={
            "2022": ["2022"],
            "2022EE": ["2022EE"],
            "2023": ["2023"],
            "2023BPix": ["2023BPix"],
        },
    ),
    "JMR": Syst(
        name=f"{CMS_PARAMS_LABEL}_jmr",
        prior="shape",
        samples=jmsr_keys,
        uncorr_years={
            "2022": ["2022"],
            "2022EE": ["2022EE"],
            "2023": ["2023"],
            "2023BPix": ["2023BPix"],
        },
    ),
}

if not args.jmsr:
    del uncorr_year_shape_systs["JMR"]
    del uncorr_year_shape_systs["JMS"]

if not args.jesr:
    # del corr_year_shape_systs["JES"]
    del uncorr_year_shape_systs["JER"]

shape_systs_dict = {}
for skey, syst in corr_year_shape_systs.items():
    if not syst.samples_corr:
        # separate nuisance param for each affected sample
        for sample in syst.samples:
            if sample not in mc_samples:
                continue
            shape_systs_dict[f"{skey}_{sample}"] = rl.NuisanceParameter(
                f"{syst.name}_{mc_samples[sample]}", "lnN" if syst.convert_shape_to_lnN else "shape"
            )
    elif syst.decorrelate_regions:
        # separate nuisance param for each region
        for region in signal_regions + ["fail"]:
            shape_systs_dict[f"{skey}_{region}"] = rl.NuisanceParameter(
                f"{syst.name}_{region}", "lnN" if syst.convert_shape_to_lnN else "shape"
            )
    elif syst.separate_prod_modes:
        # separate nuisance param for each production mode
        for prod_mode in ["ggHH", "qqHH"]:
            shape_systs_dict[f"{skey}_{prod_mode}"] = rl.NuisanceParameter(
                f"{syst.name}_{prod_mode}", "lnN" if syst.convert_shape_to_lnN else "shape"
            )
    else:
        shape_systs_dict[skey] = rl.NuisanceParameter(
            syst.name, "lnN" if syst.convert_shape_to_lnN else "shape"
        )
for skey, syst in uncorr_year_shape_systs.items():
    for uncorr_label in syst.uncorr_years:
        shape_systs_dict[f"{skey}_{uncorr_label}"] = rl.NuisanceParameter(
            f"{syst.name}_{uncorr_label}", "lnN" if syst.convert_shape_to_lnN else "shape"
        )


def get_templates(
    templates_dir: str,
    years: list[str],
    sig_separate: bool,
    scale: float | None = None,
):
    """Loads templates, combines bg and sig templates if separate, sums across all years"""
    templates_dict: dict[str, dict[str, Hist]] = {}

    if not sig_separate:
        # signal and background templates in same hist, just need to load and sum across years
        for year in years:
            with Path(f"{templates_dir}/{year}_templates.pkl").open("rb") as f:
                templates_dict[year] = rem_neg(pickle.load(f))
    else:
        # signal and background in different hists - need to combine them into one hist
        for year in years:
            with Path(f"{templates_dir}/backgrounds/{year}_templates.pkl").open("rb") as f:
                bg_templates = rem_neg(pickle.load(f))

            sig_templates = []

            for sig_key in sig_keys:
                with Path(f"{templates_dir}/{hist_names[sig_key]}/{year}_templates.pkl").open(
                    "rb"
                ) as f:
                    sig_templates.append(rem_neg(pickle.load(f)))

            templates_dict[year] = combine_templates(bg_templates, sig_templates)

    if scale is not None and scale != 1:
        for year in templates_dict:
            for key in templates_dict[year]:
                templates_dict[year][key] = templates_dict[year][key] * scale

    templates_summed: dict[str, Hist] = sum_templates(templates_dict, years)  # sum across years
    return templates_dict, templates_summed


def get_year_updown(
    templates_dict, sample, region, region_noblinded, blind_str, years_to_shift, skey
):
    """
    Return templates with only the given year's shapes shifted up and down by the ``skey`` systematic.
    Returns as [up templates, down templates]
    """
    updown = []

    for shift in ["up", "down"]:
        sshift = f"{skey}_{shift}"
        # get nominal templates for each year
        templates = {y: templates_dict[y][region][sample, ...] for y in years}

        # replace template for this year with the shifted template
        for year in years_to_shift:
            if skey in jecs or skey in jmsr:
                # JEC/JMCs saved as different "region" in dict
                reg_name = f"{region_noblinded}_{sshift}{blind_str}"
                templates[year] = templates_dict[year][reg_name][sample, ...]
            else:
                # weight uncertainties saved as different "sample" in dict
                templates[year] = templates_dict[year][region][f"{sample}_{sshift}", ...]

        # sum templates with year's template replaced with shifted
        updown.append(sum(list(templates.values())).values())

    return updown


def fill_regions(
    model: rl.Model,
    channel: rl.Channel,
    regions: list[str],
    templates_dict: dict,  # noqa: ARG001
    templates_summed: dict,
    mc_samples: dict[str, str],
    nuisance_params: dict[str, Syst],
    nuisance_params_dict: dict[str, rl.NuisanceParameter],
    corr_year_shape_systs: dict[str, Syst],  # noqa: ARG001
    uncorr_year_shape_systs: dict[str, Syst],  # noqa: ARG001
    shape_systs_dict: dict[str, rl.NuisanceParameter],  # noqa: ARG001
    bblite: bool = True,
):
    """Fill samples per region including given rate, shape and mcstats systematics.
    Ties "MCBlinded" and "nonblinded" mc stats parameters together.

    Args:
        model (rl.Model): rhalphalib model
        regions (List[str]): list of regions to fill
        templates_dict (Dict): dictionary of all templates
        templates_summed (Dict): dictionary of templates summed across years
        mc_samples (Dict[str, str]): dict of mc samples and their names in the given templates -> card names
        nuisance_params (Dict[str, Tuple]): dict of nuisance parameter names and tuple of their
          (modifier, samples affected by it, value)
        nuisance_params_dict (Dict[str, rl.NuisanceParameter]): dict of nuisance parameter names
          and NuisanceParameter object
        corr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (correlated across years)
          and tuple of their (name in cards, samples affected by it)
        uncorr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (unccorrelated across years)
          and tuple of their (name in cards, samples affected by it)
        shape_systs_dict (Dict[str, rl.NuisanceParameter]): dict of shape syst names and
          NuisanceParameter object
        pass_only (List[str]): list of systematics which are only applied in the pass region(s)
        bblite (bool): use Barlow-Beeston-lite method or not (single mcstats param across MC samples)
    """

    logging.info(channel.key)
    for region in regions:
        region_templates = templates_summed[region]

        pass_region = "pass" in region
        region_noblinded = region.split(MCB_LABEL)[0]
        blind_str = MCB_LABEL if region.endswith(MCB_LABEL) else ""  # noqa: F841

        logging.info(f"\tStarting region: {region}")
        ch = rl.Channel(channel.key + region.replace("_", ""))  # can't have '_'s in name
        model.addChannel(ch)

        for sample_name, card_name in mc_samples.items():
            # don't add signals in fail regions
            if sample_name in sig_keys:
                if not pass_region:
                    logging.info(f"\t\tSkipping {sample_name} in {region} region")
                    continue

                if sample_name[-2:] != channel.key:
                    continue

            logging.info(f"\t\tTemplates for: {sample_name}")

            sample_template = region_templates[sample_name, :]

            stype = rl.Sample.SIGNAL if sample_name in sig_keys else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + card_name, stype, sample_template)

            # ttbar rate_param
            # if sample_name in rate_params and region_noblinded in rate_params[sample_name]:
            #     rate_param = rate_params[sample_name][region_noblinded]
            #     sample.setParamEffect(rate_param, 1 * rate_param)

            # # rate params per signal to freeze them for individual limits
            # if stype == rl.Sample.SIGNAL and len(sig_keys) > 1:
            #     srate = rate_params[sample_name]
            #     sample.setParamEffect(srate, 1 * srate)

            # nominal values, errors
            values_nominal = np.maximum(sample_template.values(), 0.0)

            mask = values_nominal > 0
            errors_nominal = np.ones_like(values_nominal)
            errors_nominal[mask] = (
                1.0 + np.sqrt(sample_template.variances()[mask]) / values_nominal[mask]
            )

            logging.debug(f"nominal   : {values_nominal}")
            logging.debug(f"error     : {errors_nominal}")

            if not bblite and args.mcstats:
                # set mc stat uncs
                logging.info(f"setting autoMCStats for {sample_name} in {region}")

                # tie MC stats parameters together in blinded and "unblinded" region
                stats_sample_name = f"{CMS_PARAMS_LABEL}_{region}_{card_name}"
                sample.autoMCStats(
                    sample_name=stats_sample_name,
                    # this function uses a different threshold convention from combine
                    threshold=np.sqrt(1 / args.mcstats_threshold),
                    epsilon=args.epsilon,
                )

            # rate systematics
            for skey, syst in nuisance_params.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(f"\t\t\tGetting {skey} rate")

                param = nuisance_params_dict[skey]

                val, val_down = syst.value, syst.value_down
                if syst.diff_regions:
                    val = val[region]
                    val_down = val_down[region] if val_down is not None else val_down
                if syst.diff_samples:
                    val = val[sample_name]
                    val_down = val_down[sample_name] if val_down is not None else val_down

                sample.setParamEffect(param, val, effect_down=val_down)

            # correlated shape systematics
            # for skey, syst in corr_year_shape_systs.items():
            #     if sample_name not in syst.samples or (not pass_region and syst.pass_only):
            #         continue

            #     logging.info(f"Getting {skey} shapes")

            #     if skey in jecs or skey in jmsr:
            #         # JEC/JMCs saved as different "region" in dict
            #         up_hist = templates_summed[f"{region_noblinded}_{skey}_up{blind_str}"][
            #             sample_name, :
            #         ]
            #         down_hist = templates_summed[f"{region_noblinded}_{skey}_down{blind_str}"][
            #             sample_name, :
            #         ]

            #         values_up = up_hist.values()
            #         values_down = down_hist.values()
            #     else:
            #         # weight uncertainties saved as different "sample" in dict
            #         values_up = region_templates[f"{sample_name}_{skey}_up", :].values()
            #         values_down = region_templates[f"{sample_name}_{skey}_down", :].values()

            #     logger = logging.getLogger(f"validate_shapes_{region}_{sample_name}_{skey}")

            #     effect_up, effect_down = get_effect_updown(
            #         values_nominal,
            #         values_up,
            #         values_down,
            #         mask,
            #         logger,
            #         args.epsilon,
            #         syst.convert_shape_to_lnN,
            #     )
            #     if not syst.samples_corr:
            #         # separate syst if not correlated across samples
            #         sdkey = f"{skey}_{sample_name}"
            #     elif syst.decorrelate_regions:
            #         # separate syst if not correlated across regions
            #         sdkey = f"{skey}_{region_noblinded}"
            #     elif syst.separate_prod_modes:
            #         # separate syst if not correlated across production modes
            #         if sample_name in sig_keys_ggf:
            #             prod_mode = "ggHH"
            #         elif sample_name in sig_keys_vbf:
            #             prod_mode = "qqHH"
            #         else:
            #             raise NotImplementedError(
            #                 f"Splitting Syst by production mode for Sample {sample_name} not yet implemented"
            #             )
            #         sdkey = f"{skey}_{prod_mode}"
            #     else:
            #         sdkey = skey
            #     sample.setParamEffect(shape_systs_dict[sdkey], effect_up, effect_down)

            # # uncorrelated shape systematics
            # for skey, syst in uncorr_year_shape_systs.items():
            #     if sample_name not in syst.samples or (not pass_region and syst.pass_only):
            #         continue

            #     logging.info(f"Getting {skey} shapes")

            #     for uncorr_label, years_to_shift in syst.uncorr_years.items():

            #         values_up, values_down = get_year_updown(
            #             templates_dict,
            #             sample_name,
            #             region,
            #             region_noblinded,
            #             blind_str,
            #             years_to_shift,
            #             skey,
            #         )
            #         logger = logging.getLogger(f"validate_shapes_{region}_{sample_name}_{skey}")

            #         effect_up, effect_down = get_effect_updown(
            #             values_nominal,
            #             values_up,
            #             values_down,
            #             mask,
            #             logger,
            #             args.epsilon,
            #             syst.convert_shape_to_lnN,
            #         )
            #         sample.setParamEffect(
            #             shape_systs_dict[f"{skey}_{uncorr_label}"], effect_up, effect_down
            #         )

            ch.addSample(sample)

        if bblite and args.mcstats:
            # tie MC stats parameters together in blinded and "unblinded" region
            channel_name = region_noblinded
            ch.autoMCStats(
                channel_name=f"{CMS_PARAMS_LABEL}_{channel.key}{channel_name}",
                threshold=args.mcstats_threshold,
                epsilon=args.epsilon,
            )

        # data observed
        ch.setObservation(region_templates[data_key, :])


def alphabet_fit(
    model: rl.Model,
    channel: Channel,
    shape_vars: list[ShapeVar],
    templates_summed: dict,
    scale: float | None = None,
    min_qcd_val: float | None = None,
    unblinded: bool = False,
):
    shape_var = shape_vars[0]
    m_obs = rl.Observable(shape_var.name, shape_var.bins)

    ##########################
    # Setup fail region first
    ##########################

    # Independent nuisances to float QCD in each fail bin
    qcd_params = np.array(
        [
            rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_tf_dataResidual_{channel.key}Bin{i}", 0)
            for i in range(m_obs.nbins)
        ]
    )

    fail_qcd_samples = {}

    blind_strs = [""] if unblinded else ["", MCB_LABEL]
    for blind_str in blind_strs:
        failChName = f"{channel.key}fail{blind_str}".replace("_", "")
        logging.info(f"Setting up fail region {failChName}")
        failCh = model[failChName]

        # sideband fail
        # was integer, and numpy complained about subtracting float from it
        initial_qcd = failCh.getObservation().astype(float)
        for sample in failCh:
            # don't subtract signals (#TODO: do we want to subtract SM signal?)
            if sample.sampletype == rl.Sample.SIGNAL:
                continue
            logging.debug(
                f"subtracting {sample._name}={sample.getExpectation(nominal=True)} from qcd"
            )
            initial_qcd -= sample.getExpectation(nominal=True)

        if np.any(initial_qcd < 0.0):
            logging.warning(f"initial_qcd negative for some bins.. {initial_qcd}")
            initial_qcd = np.maximum(initial_qcd, 1e-6)

        # idea here is that the error should be 1/sqrt(N), so parametrizing it as (1 + 1/sqrt(N))^qcdparams
        # will result in qcdparams errors ~±1
        # but because qcd is poorly modelled we're scaling sigma scale

        sigmascale = 10  # to scale the deviation from initial
        if scale is not None:
            sigmascale *= scale

        scaled_params = (
            initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcd_params
        )

        # add samples
        fail_qcd = rl.ParametericSample(
            f"{failChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
            rl.Sample.BACKGROUND,
            m_obs,
            scaled_params,
        )
        failCh.addSample(fail_qcd)

        fail_qcd_samples[blind_str] = fail_qcd

    ##########################
    # Now do signal regions
    ##########################

    for sr in signal_regions:
        # QCD overall pass / fail efficiency
        qcd_eff = (
            templates_summed[sr][data_key, :].sum().value
            - np.sum([templates_summed[sr][bg_key, :].sum().value for bg_key in bg_keys])
        ) / (
            templates_summed["fail"][data_key, :].sum().value
            - np.sum([templates_summed["fail"][bg_key, :].sum().value for bg_key in bg_keys])
        )
        # qcd_eff = (
        #     templates_summed[sr][qcd_key, :].sum().value
        #     / templates_summed["fail"][qcd_key, :].sum().value
        # )
        logging.info(f"qcd eff {qcd_eff:.5f}")

        # transfer factor
        tf_dataResidual = rl.BasisPoly(
            f"{CMS_PARAMS_LABEL}_tf_dataResidual_{channel.key}{sr}",
            (shape_var.orders[sr],),
            [shape_var.name],
            basis="Bernstein",
            limits=(-20, 20),
            square_params=True,
        )
        tf_dataResidual_params = tf_dataResidual(shape_var.scaled)
        tf_params_pass = qcd_eff * tf_dataResidual_params  # scale params initially by qcd eff

        for blind_str in blind_strs:
            passChName = f"{channel.key}{sr}{blind_str}".replace("_", "")
            passCh = model[passChName]

            pass_qcd = rl.TransferFactorSample(
                f"{passChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
                rl.Sample.BACKGROUND,
                tf_params_pass,
                fail_qcd_samples[blind_str],
                min_val=min_qcd_val,
            )
            passCh.addSample(pass_qcd)


def createDatacardAlphabet(
    args,
    model: rl.Model,
    channel: rl.Channel,
    templates_dict: dict,
    templates_summed: dict,
    shape_vars: list[ShapeVar],
):
    # (pass, fail) x (unblinded, blinded)
    blind_strs = [""] if args.unblinded else ["", MCB_LABEL]

    regions: list[str] = [
        f"{pf}{blind_str}" for pf in [*signal_regions, "fail"] for blind_str in blind_strs
    ]

    # Fill templates per sample, incl. systematics
    fill_args = [
        model,
        channel,
        regions,
        templates_dict,
        templates_summed,
        mc_samples,
        nuisance_params,
        nuisance_params_dict,
        corr_year_shape_systs,
        uncorr_year_shape_systs,
        shape_systs_dict,
        args.bblite,
    ]

    fit_args = [
        model,
        channel,
        shape_vars,
        templates_summed,
        args.scale_templates,
        args.min_qcd_val,
        args.unblinded,
    ]

    fill_regions(*fill_args)
    alphabet_fit(*fit_args)


def main(args):
    model = rl.Model("HHModel")

    for channel in channels:
        # templates per region per year, templates per region summed across years
        templates_dict, templates_summed = get_templates(
            f"{args.templates_dir}/{channel.key}", years, args.sig_separate, args.scale_templates
        )

        # random template from which to extract shape vars
        sample_templates: Hist = templates_summed[next(iter(templates_summed.keys()))]

        # [mH(bb)]
        shape_vars = [
            ShapeVar(
                name=axis.name,
                bins=axis.edges,
                orders={sr: args.nTF[i] for i, sr in enumerate(signal_regions)},
            )
            for _, axis in enumerate(sample_templates.axes[1:])
        ]

        createDatacardAlphabet(args, model, channel, templates_dict, templates_summed, shape_vars)

    ##############################################
    # Save model
    ##############################################

    logging.info("rendering combine model")
    Path(args.cards_dir).mkdir(parents=True, exist_ok=True)
    out_dir = (
        Path(args.cards_dir) / args.model_name if args.model_name is not None else args.cards_dir
    )
    model.renderCombine(out_dir)

    with Path(f"{out_dir}/model.pkl").open("wb") as fout:
        pickle.dump(model, fout, 2)  # use python 2 compatible protocol


main(args)
