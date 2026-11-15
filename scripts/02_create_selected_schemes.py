import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

schemes_data = [
    # --- HEALTH & CHRONIC CARE ---
    {"scheme_id": "ab_pmjay", "scheme_name": "Ayushman Bharat PM-JAY", "category": "health", "priority": 1, "official_url": "https://pmjay.gov.in/"},
    {"scheme_id": "ayushman_vay_vandana", "scheme_name": "Ayushman Vay Vandana", "category": "health", "priority": 1, "official_url": "https://pmjay.gov.in/"},
    {"scheme_id": "pmndp", "scheme_name": "Pradhan Mantri National Dialysis Programme", "category": "chronic_care", "priority": 1, "official_url": "https://nhm.gov.in/"},
    {"scheme_id": "npcdcs", "scheme_name": "National Programme for Prevention & Control of Cancer, Diabetes, CVD and Stroke", "category": "chronic_care", "priority": 2, "official_url": "https://nhm.gov.in/"},
    {"scheme_id": "abdm", "scheme_name": "Ayushman Bharat Digital Mission (ABHA)", "category": "digital_health", "priority": 1, "official_url": "https://abdm.gov.in/"},
    {"scheme_id": "ntep", "scheme_name": "National Tuberculosis Elimination Programme (Nikshay)", "category": "health", "priority": 1, "official_url": "https://tbcindia.gov.in/"},
    {"scheme_id": "e_sanjeevani", "scheme_name": "eSanjeevani Teleconsultation", "category": "digital_health", "priority": 2, "official_url": "https://esanjeevani.mohfw.gov.in/"},
    {"scheme_id": "nmhp", "scheme_name": "National Mental Health Programme (Tele-MANAS)", "category": "health", "priority": 2, "official_url": "https://telemanas.mohfw.gov.in/"},

    # --- MATERNAL & CHILD HEALTH ---
    {"scheme_id": "jsy", "scheme_name": "Janani Suraksha Yojana", "category": "maternal_health", "priority": 1, "official_url": "https://nhm.gov.in/"},
    {"scheme_id": "jssk", "scheme_name": "Janani Shishu Suraksha Karyakram", "category": "maternal_child_health", "priority": 1, "official_url": "https://nhm.gov.in/"},
    {"scheme_id": "pmsma", "scheme_name": "Pradhan Mantri Surakshit Matritva Abhiyan", "category": "maternal_health", "priority": 2, "official_url": "https://pmsma.nhp.gov.in/"},
    {"scheme_id": "rbsk", "scheme_name": "Rashtriya Bal Swasthya Karyakram", "category": "child_health", "priority": 2, "official_url": "https://rbsk.gov.in/"},
    {"scheme_id": "uip", "scheme_name": "Universal Immunisation Programme (Mission Indradhanush)", "category": "child_health", "priority": 1, "official_url": "https://nhm.gov.in/"},
    {"scheme_id": "poshan", "scheme_name": "POSHAN Abhiyaan (National Nutrition Mission)", "category": "nutrition", "priority": 1, "official_url": "https://poshanabhiyaan.gov.in/"},

    # --- FINANCIAL INCLUSION & SOCIAL SECURITY ---
    {"scheme_id": "pmjdy", "scheme_name": "Pradhan Mantri Jan Dhan Yojana", "category": "financial_inclusion", "priority": 1, "official_url": "https://pmjdy.gov.in/"},
    {"scheme_id": "pmjjby", "scheme_name": "Pradhan Mantri Jeevan Jyoti Bima Yojana", "category": "social_security", "priority": 2, "official_url": "https://financialservices.gov.in/"},
    {"scheme_id": "pmsby", "scheme_name": "Pradhan Mantri Suraksha Bima Yojana", "category": "social_security", "priority": 2, "official_url": "https://financialservices.gov.in/"},
    {"scheme_id": "apy", "scheme_name": "Atal Pension Yojana", "category": "social_security", "priority": 2, "official_url": "https://npscra.nsdl.co.in/"},
    {"scheme_id": "pm_mudra", "scheme_name": "Pradhan Mantri Mudra Yojana", "category": "financial_inclusion", "priority": 2, "official_url": "https://www.mudra.org.in/"},
    {"scheme_id": "stand_up_india", "scheme_name": "Stand-Up India Scheme", "category": "financial_inclusion", "priority": 2, "official_url": "https://www.standupmitra.in/"},

    # --- HOUSING & UTILITIES ---
    {"scheme_id": "pm_awas_g", "scheme_name": "Pradhan Mantri Awas Yojana - Gramin", "category": "housing", "priority": 1, "official_url": "https://pmayg.nic.in/"},
    {"scheme_id": "pm_awas_u", "scheme_name": "Pradhan Mantri Awas Yojana - Urban", "category": "housing", "priority": 1, "official_url": "https://pmayuclap.gov.in/"},
    {"scheme_id": "pm_ujjwala", "scheme_name": "Pradhan Mantri Ujjwala Yojana (LPG)", "category": "utilities", "priority": 1, "official_url": "https://www.pmuy.gov.in/"},
    {"scheme_id": "saubhagya", "scheme_name": "Pradhan Mantri Sahaj Bijli Har Ghar Yojana", "category": "utilities", "priority": 2, "official_url": "https://saubhagya.gov.in/"},
    {"scheme_id": "jjm", "scheme_name": "Jal Jeevan Mission (Har Ghar Jal)", "category": "utilities", "priority": 2, "official_url": "https://jaljeevanmission.gov.in/"},

    # --- AGRICULTURE & RURAL WELFARE ---
    {"scheme_id": "pm_kisan", "scheme_name": "Pradhan Mantri Kisan Samman Nidhi", "category": "agriculture", "priority": 1, "official_url": "https://pmkisan.gov.in/"},
    {"scheme_id": "pm_fasal_bima", "scheme_name": "Pradhan Mantri Fasal Bima Yojana", "category": "agriculture", "priority": 1, "official_url": "https://pmfby.gov.in/"},
    {"scheme_id": "kcc", "scheme_name": "Kisan Credit Card (KCC)", "category": "agriculture", "priority": 2, "official_url": "https://agricoop.nic.in/"},
    {"scheme_id": "mgnrega", "scheme_name": "Mahatma Gandhi National Rural Employment Guarantee Act", "category": "livelihood", "priority": 1, "official_url": "https://nrega.nic.in/"},
    {"scheme_id": "soil_health", "scheme_name": "Soil Health Card Scheme", "category": "agriculture", "priority": 3, "official_url": "https://soilhealth.dac.gov.in/"},

    # --- SKILLS, EDUCATION & EMPLOYMENT ---
    {"scheme_id": "pmkvy", "scheme_name": "Pradhan Mantri Kaushal Vikas Yojana", "category": "skill_development", "priority": 2, "official_url": "https://www.pmkvyofficial.org/"},
    {"scheme_id": "pm_egp", "scheme_name": "Prime Minister's Employment Generation Programme", "category": "livelihood", "priority": 2, "official_url": "https://www.kviconline.gov.in/"},
    {"scheme_id": "nsp_minority", "scheme_name": "National Scholarship Portal - Central Sector Schemes", "category": "education", "priority": 2, "official_url": "https://scholarships.gov.in/"},
    {"scheme_id": "pm_shri", "scheme_name": "PM SHRI Schools Initiative", "category": "education", "priority": 3, "official_url": "https://pmshri.education.gov.in/"},
    {"scheme_id": "national_apprentice", "scheme_name": "National Apprenticeship Promotion Scheme (NAPS)", "category": "skill_development", "priority": 2, "official_url": "https://www.apprenticeshipindia.gov.in/"},

    # --- VULNERABLE GROUPS & PENSIONS ---
    {"scheme_id": "nsap_old_age", "scheme_name": "Indira Gandhi National Old Age Pension Scheme (IGNOAPS)", "category": "social_security", "priority": 1, "official_url": "https://nsap.nic.in/"},
    {"scheme_id": "nsap_widow", "scheme_name": "Indira Gandhi National Widow Pension Scheme (IGNWPS)", "category": "social_security", "priority": 1, "official_url": "https://nsap.nic.in/"},
    {"scheme_id": "nsap_disability", "scheme_name": "Indira Gandhi National Disability Pension Scheme (IGNDPS)", "category": "social_security", "priority": 1, "official_url": "https://nsap.nic.in/"},
    {"scheme_id": "disability_card", "scheme_name": "Unique Disability ID (UDID) Scheme", "category": "social_security", "priority": 2, "official_url": "https://www.swavlambancard.gov.in/"},
    {"scheme_id": "pm_daksh", "scheme_name": "PM-DAKSH (Skill Training for Marginalized Groups)", "category": "skill_development", "priority": 3, "official_url": "https://pmdaksh.dosje.gov.in/"}
]

def main():
    df = pd.DataFrame(schemes_data)
    out_file = DATA / "selected_schemes.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} schemes to {out_file}")
    assert len(df) == 40
    print("40 core schemes verified across 7 domains.")

if __name__ == "__main__":
    main()
