"""Common variables for bbtautau analysis."""

from __future__ import annotations

HLT_dict = {
    "PNet": [
        "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
    ],
    "PFJet": [
        "HLT_AK8PFJet425_SoftDropMass40",
    ],
    "QuadJet": [
        "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
        "HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2",
        "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
    ],
    "SingleTau": [
        "HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1",
    ],
    "DiTau": [
        "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75",
    ],
    "Muon": [
        "HLT_IsoMu24",
        "HLT_Mu50",
    ],
    # These are in Muon
    "MuonTau": [
        "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS180_eta2p1",
        "HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS45_L2NN_eta2p1_CrossL1",
        "HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_CrossL1",
        "HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_TightID_CrossL1",
    ],
    "EGamma": [
        "HLT_Ele30_WPTight_Gsf",
        "HLT_Ele115_CaloIdVT_GsfTrkIdT",
        "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
        "HLT_Photon200",
    ],
    # This is in EGamma
    "ETau": [
        "HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        "HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1",
    ],
}

# combine into a single list
HLT_list = [hlt for sublist in HLT_dict.values() for hlt in sublist]

# HLTs per dataset
HLT_jets = [hlt for key in ["PNet", "PFJet", "QuadJet"] for hlt in HLT_dict[key]]
HLT_taus = [hlt for key in ["DiTau", "SingleTau"] for hlt in HLT_dict[key]]
HLT_egammas = [hlt for key in ["EGamma", "ETau"] for hlt in HLT_dict[key]]
HLT_muons = [hlt for key in ["Muon", "MuonTau"] for hlt in HLT_dict[key]]

# HLTs per channel
HLT_hh = [
    hlt for key in ["PNet", "PFJet", "QuadJet", "DiTau", "SingleTau"] for hlt in HLT_dict[key]
]
HLT_hmu = [
    hlt
    for key in ["PNet", "PFJet", "Muon", "MuonTau", "DiTau", "SingleTau"]
    for hlt in HLT_dict[key]
]
HLT_he = [
    hlt
    for key in ["PNet", "PFJet", "EGamma", "ETau", "DiTau", "SingleTau"]
    for hlt in HLT_dict[key]
]

HLT_channels = {
    "hh": HLT_hh,
    "he": HLT_he,
    "hm": HLT_hmu,
}
