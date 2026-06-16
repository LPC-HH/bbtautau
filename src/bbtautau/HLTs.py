"""HLTs for bbtautau analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from boostedhh import utils

years_2022 = ["2022", "2022EE"]
years_2023 = ["2023", "2023BPix"]
years = years_2022 + years_2023 + ["2024"]


@dataclass
class HLT(utils.HLT):
    """Same as boostedhh.utils.HLT but with channel."""

    # which channel? defaults to all.
    channel: list[str] = field(default_factory=lambda: ["hh", "hm", "he"])


# Short labels for overlap / trigger-study plots (full path -> axis label).
HLT_SHORT_LABELS: dict[str, str] = {
    "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35": "AK8Jet250_PNetBB0p35",
    "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30": "AK8Jet230_PNetTT0p30",
    "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06": "AK8Jet230_PNetBB0p06",
    "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03": "AK8Jet230_PNetTT0p03",
    "HLT_AK8PFJet420_MassSD30": "AK8Jet420_MassSD30",
    "HLT_AK8PFJet425_SoftDropMass40": "AK8Jet425_SD40",
    "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65": "Quad70_PNet2B0p65",
    "HLT_QuadPFJet70_50_40_35_PNet2BTagMean0p65": "Quad70_PNet2B0p65",
    "HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2": "Quad103_DJ1p3_VBF2",
    "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1": "Quad103_2xDJ1p3_VBF1",
    "HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1": "LooseDeepTau180",
    "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1": "DiMedDeepTau35",
    "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60": "DiMedDeepTau30_Jet60",
    "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75": "DiMedDeepTau30_Jet75",
    "HLT_IsoMu24": "IsoMu24",
    "HLT_Mu50": "Mu50",
    "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1": "IsoMu20_LooseTau27",
    "HLT_Ele30_WPTight_Gsf": "Ele30_WPTight",
    "HLT_Ele115_CaloIdVT_GsfTrkIdT": "Ele115",
    "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165": "Ele50_Jet165",
    "HLT_Photon200": "Photon200",
    "HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1": "Ele24_LooseTau30",
    "HLT_PFMET120_PFMHT120_IDTight": "MET120",
    "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55": "PFHT280_Quad30_PNet2B0p55",
    "HLT_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70": "PFHT340_Quad70_PNet2B0p70",
}


class HLTs:
    HLTs: ClassVar[dict[str, list[HLT]]] = {
        "pnet": [
            # 2022 + 6fb-1 of 2023
            HLT(
                name="HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                mc_years=years_2022,
                data_years=years_2022 + ["2023"],
                dataset="JetMET",
            ),
            HLT(
                name="HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                mc_years=years_2022,
                data_years=years_2022 + ["2023"],
                dataset="JetMET",
            ),
            # 2023 after 6fb-1, that is from Run2023C_0v2 to Run2023C_0v3
            HLT(
                name="HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                years=years_2023 + ["2024"],
                dataset="JetMET",
            ),
            HLT(
                name="HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                years=years_2023 + ["2024"],
                dataset="JetMET",
            ),
        ],
        "pfjet": [
            HLT(
                name="HLT_AK8PFJet420_MassSD30",
                years=years_2022 + years_2023,  # years_2023  makes it work in 25Mar7 data samples
                dataset="JetMET",
            ),
            HLT(
                name="HLT_AK8PFJet425_SoftDropMass40",
                years=years_2022 + years_2023,
                dataset="JetMET",
            ),
            HLT(
                name="HLT_AK8PFJet425_SoftDropMass30",
                years=["2024"],
                dataset="JetMET",
            ),
        ],
        "quadjet": [
            # 2022 + 6fb-1 of 2023 (moves to Parking after this)
            HLT(
                name="HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                mc_years=years_2022,
                data_years=years_2022,
                dataset="JetMET",
                channel=["hh"],
            ),
            HLT(
                name="HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2",
                years=years_2022 + years_2023,
                dataset="JetMET",
                channel=["hh"],
            ),
            HLT(
                name="HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                years=years_2022 + years_2023,
                dataset="JetMET",
                channel=["hh"],
            ),
            HLT(
                name="HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1",
                years=["2024"],
                dataset="JetMET",
                channel=["hh"],
            ),
            HLT(
                name="HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2",
                years=["2024"],
                dataset="JetMET",
                channel=["hh"],
            ),
        ],
        "singletau": [
            HLT(
                name="HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1",
                years=years,
                dataset="Tau",
            ),
        ],
        "ditau": [
            HLT(
                name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
                years=years,
                dataset="Tau",
            ),
        ],
        "ditaujet": [
            HLT(
                name="HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
                years=years,
                dataset="Tau",
                channel=["hh"],
            ),
            # HLT(
            #     name="HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75",
            #     years=years,
            #     dataset="Tau",
            #     channel=["hh"],
            # ),
        ],
        "muon": [
            HLT(
                name="HLT_IsoMu24",
                years=years,
                dataset="Muon",
                channel=["hm"],
            ),
            # TODO: check sensitivity without below triggers
            HLT(
                name="HLT_Mu50",
                years=years,
                dataset="Muon",
                channel=["hm"],
            ),
        ],
        "muontau": [
            HLT(
                name="HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
                years=years,
                dataset="Muon",
                channel=["hm"],
            ),
        ],
        "egamma": [
            HLT(
                name="HLT_Ele30_WPTight_Gsf",
                years=years,
                dataset="EGamma",
                channel=["he"],
            ),
            HLT(
                name="HLT_Ele115_CaloIdVT_GsfTrkIdT",
                years=years,
                dataset="EGamma",
                channel=["he"],
            ),
            HLT(
                name="HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                years=years,
                dataset="EGamma",
                channel=["he"],
            ),
            HLT(
                name="HLT_Photon200",
                years=years,
                dataset="EGamma",
                channel=["he"],
            ),
        ],
        "etau": [
            HLT(
                name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
                years=years,
                dataset="EGamma",
                channel=["he"],
            ),
        ],
        "met": [
            HLT(
                name="HLT_PFMET120_PFMHT120_IDTight",
                years=years,
                dataset="JetMET",
            ),
        ],
        "parking": [
            # Moved to Parking in 2023 after 6fb-1
            HLT(
                name="HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55",
                years=["2023BPix"] + ["2024"],
                dataset="ParkingHH",
                channel=["hh"],
            ),
            HLT(
                name="HLT_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
                years=years_2023 + ["2024"],
                dataset="ParkingHH",
                channel=["hh"],
            ),
        ],
    }

    # Analysis trigger menu: maps a single logical, year-agnostic key to the
    # year-tagged variants of ONE physical trigger. Each channel's ``hlt_menu``
    # references these keys; ``resolve_menu`` turns them into the year-correct
    # HLT path(s) to OR when applying triggers.

    MENU: ClassVar[dict[str, list[HLT]]] = {
        # PNet BB single-jet trigger. The path was retuned partway through 2023
        # (after 6 fb-1), so the "2023" data sample sees BOTH variants -> both
        # resolve and are OR'd.
        "pnetbb": [
            HLT(
                name="HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                mc_years=years_2022,
                data_years=years_2022 + ["2023"],
                dataset="JetMET",
            ),
            HLT(
                name="HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                years=years_2023 + ["2024"],
                dataset="JetMET",
            ),
        ],
        "ditau": [
            HLT(
                name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
                years=years,
                dataset="Tau",
            ),
        ],
        "muiso24": [
            HLT(
                name="HLT_IsoMu24",
                years=years,
                dataset="Muon",
            ),
        ],
        "ele30": [
            HLT(
                name="HLT_Ele30_WPTight_Gsf",
                years=years,
                dataset="EGamma",
            ),
        ],
    }

    @classmethod
    def resolve_menu(
        cls,
        year: str,
        keys: str | list[str],
        as_str: bool = True,
        hlt_prefix: bool = True,
        data_only: bool = False,
        mc_only: bool = False,
    ) -> list[HLT | str]:
        """Resolve logical analysis-menu key(s) to year-correct HLT path(s).

        Args:
            year (str): year to resolve for.
            keys (str | list[str]): one or more keys from ``HLTs.MENU``.
            as_str (bool): return HLT names (True) or HLT objects (False).
            hlt_prefix (bool): keep the ``HLT_`` prefix on names.
            data_only / mc_only: filter variants by data/MC validity for the year.

        Returns:
            list[HLT | str]: the concrete trigger(s) for this menu in this year.
            A single key may yield more than one path during a transition year
            (e.g. ``pnetbb`` in 2023).
        """
        if isinstance(keys, str):
            keys = [keys]

        out: list[HLT | str] = []
        for key in keys:
            if key not in cls.MENU:
                raise ValueError(f"Menu key {key!r} not found in HLTs.MENU")
            for hlt in cls.MENU[key]:
                if hlt.check_year(year, data_only, mc_only):
                    out.append(hlt.get_name(hlt_prefix) if as_str else hlt)
        return out

    @classmethod
    def hlt_dict(
        cls,
        year: str,
        as_str: bool = True,
        hlt_prefix: bool = True,
        data_only: bool = False,
        mc_only: bool = False,
    ) -> dict[str, list[HLT | str]]:
        """
        Convert into a dictionary of HLTs per year, optionally filtered by data or MC.

        Args:
            year (str): year to filter by.
            as_str (bool): if True, return HLT names only. If False, return HLT objects. Defaults to True.
            data_only (bool): filter by HLTs in data for that year. Defaults to False.
            mc_only (bool): filter by HLTs in MC for that year. Defaults to False.

        Returns:
            dict[str, list[HLT | str]]: format is ``{hlt_type: [hlt, ...]}``
        """
        if data_only and mc_only:
            raise ValueError("Cannot filter by both data and MC")

        return {
            hlt_type: [
                (hlt.get_name(hlt_prefix) if as_str else hlt)
                for hlt in hlt_list
                if hlt.check_year(year, data_only, mc_only)
            ]
            for hlt_type, hlt_list in cls.HLTs.items()
        }

    @classmethod
    def hlt_list(
        cls, as_str: bool = True, hlt_prefix: bool = True, **hlt_kwargs
    ) -> dict[str, list[HLT | str]]:
        """
        Combine into a dict of lists of HLTs per year.

        Args:
            as_str (bool): if True, return HLT names only. If False, return HLT objects. Defaults to True.
            hlt_prefix (bool): if True, return HLT names with "HLT_" prefix. If False, return HLT names without "HLT_" prefix. Defaults to True.
            **hlt_kwargs: additional kwargs to pass to the hlt_dict function.

        Returns:
            dict[str, list[HLT | str]]: format is ``{year: [hlt, ...]}``
        """
        return {
            year: [
                (hlt.get_name(hlt_prefix) if as_str else hlt)
                for sublist in cls.hlt_dict(year, as_str=False, **hlt_kwargs).values()
                for hlt in sublist
            ]
            for year in years
        }

    @classmethod
    def hlts_by_type(
        cls,
        year: str,
        hlt_type: str | list[str],
        **hlt_kwargs,
    ) -> list[HLT | str]:
        """
        HLTs per year and type(s), with optional filters.

        Args:
            year (str): year to filter by.
            hlt_type (str | list[str]): filter by HLT type(s) out of ["PNet", "PFJet", "QuadJet", "DiTau", "SingleTau", "Muon", "EGamma", "MET", "Parking"].
            **hlt_kwargs: additional kwargs to pass to the hlt_dict function.

        Returns:
            list[HLT | str]: list of HLTs. Returns strings if as_str=True is passed in hlt_kwargs, otherwise returns HLT objects.
        """
        hlts = cls.hlt_dict(year, **hlt_kwargs)

        if isinstance(hlt_type, str):
            return hlts[hlt_type.lower()]
        else:
            return [hlt for ht in hlt_type for hlt in hlts[ht.lower()]]

    @classmethod
    def hlts_by_dataset(
        cls,
        year: str,
        dataset: str,
        as_str: bool = True,
        hlt_prefix: bool = True,
        **hlt_kwargs,
    ) -> list[HLT | str]:
        """
        HLTs per year and dataset, with optional filters.

        Args:
            year (str): year to filter by.
            dataset (str): filter by dataset out of ["JetMET", "Tau", "Muon", "EGamma", "ParkingHH"].
            as_str (bool): if True, return HLT names only. If False, return HLT objects. Defaults to True.
            hlt_prefix (bool): if True, return HLT names with "HLT_" prefix. If False, return HLT names without "HLT_" prefix. Defaults to True.
            **hlt_kwargs: additional kwargs to pass to the hlt_list function.

        Returns:
            list[HLT | str]: list of HLTs
        """
        hlts = cls.hlt_list(False, **hlt_kwargs)[year]
        ret_hlts = [
            (hlt.get_name(hlt_prefix) if as_str else hlt)
            for hlt in hlts
            if hlt.dataset.lower() == dataset.lower()
        ]

        if len(ret_hlts) == 0:
            raise ValueError(f"Dataset {dataset} not found in HLTs")

        return ret_hlts

    @classmethod
    def hlts_list_by_dtype(
        cls,
        year: str,
        as_str: bool = True,
        hlt_prefix: bool = True,
        **hlt_kwargs,
    ) -> list[HLT | str]:
        """
        HLTs per year, with optional filters.

        Args:
            year (str): year to filter by.
            as_str (bool): if True, return HLT names only. If False, return HLT objects. Defaults to True.
            hlt_prefix (bool): if True, return HLT names with "HLT_" prefix. If False, return HLT names without "HLT_" prefix. Defaults to True.
            **hlt_kwargs: additional kwargs to pass to the hlt_list function.

        Returns:
            dict[str, list[HLT | str]]: format is ``{data: [hlt, ...], signal: [...]}``
        """
        return {
            "signal": [
                (hlt.get_name(hlt_prefix) if as_str else hlt)
                for sublist in cls.hlt_dict(year, as_str=False, mc_only=True, **hlt_kwargs).values()
                for hlt in sublist
            ],
            "bg": [
                (hlt.get_name(hlt_prefix) if as_str else hlt)
                for sublist in cls.hlt_dict(year, as_str=False, mc_only=True, **hlt_kwargs).values()
                for hlt in sublist
            ],
            "data": [
                (hlt.get_name(hlt_prefix) if as_str else hlt)
                for sublist in cls.hlt_dict(
                    year, as_str=False, data_only=True, **hlt_kwargs
                ).values()
                for hlt in sublist
            ],
        }

    @classmethod
    def get_hlt(cls, name: str) -> HLT:
        for cat in cls.HLTs.values():
            for hlt in cat:
                if hlt.get_name() == name:
                    return hlt
        raise ValueError(f"HLT {name} not found in HLTs")

    @classmethod
    def short_label(cls, name: str) -> str:
        """Short axis label for *name* (with or without ``HLT_`` prefix)."""
        key = name if name.startswith("HLT_") else f"HLT_{name}"
        if key in HLT_SHORT_LABELS:
            return HLT_SHORT_LABELS[key]
        return key.removeprefix("HLT_")
