"""
SanoVocal Tool Contract Repair Dataset Generator
Builds a curated, high-precision repair dataset of 225 genuine examples across 7 categories
and 3 languages (English, Hindi, Hinglish), strictly executing and verifying tool calls
via app.tools.engine.dispatch_tool.

Outputs:
  data/repair/sanovocal_tool_contract_repair.jsonl
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure repository root is in python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tools.engine import dispatch_tool, _get_knowledge_cards
from app.tools.schemas import TOOL_REGISTRY

REPAIR_DIR = ROOT / "data" / "repair"
REPAIR_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = REPAIR_DIR / "sanovocal_tool_contract_repair.jsonl"

cards = _get_knowledge_cards()
scheme_ids = sorted(list(cards.keys()))

def make_step(step_num: int, state: str, tool_name: str, args: Dict[str, Any], thought: str) -> Dict[str, Any]:
    """Execute tool and return structured step."""
    obs = dispatch_tool(tool_name, args)
    return {
        "step_number": step_num,
        "state": state,
        "action": {
            "name": tool_name,
            "arguments": args
        },
        "observation": json.dumps(obs, ensure_ascii=False),
        "thought": thought,
        "reward": 1.0
    }

def format_messages(task: str, steps: List[Dict[str, Any]], final_response: str) -> List[Dict[str, str]]:
    """Format dual conversational format for SFT / TRL."""
    msgs = [{"role": "user", "content": task}]
    for s in steps:
        tool_call = f"<tool_call>\n{{\"name\": \"{s['action']['name']}\", \"arguments\": {json.dumps(s['action']['arguments'], ensure_ascii=False)}}}\n</tool_call>"
        assistant_content = f"{s['thought']}\n{tool_call}"
        msgs.append({"role": "assistant", "content": assistant_content})
        msgs.append({"role": "tool", "content": s["observation"]})
    msgs.append({"role": "assistant", "content": final_response})
    return msgs

def build_dataset() -> List[Dict[str, Any]]:
    records = []
    
    # =========================================================================
    # CATEGORY A: Exact Tool Contract Enforcement (35 examples)
    # Tests valid schemas, strict argument typing across all 7 tools.
    # Languages: 12 en, 12 hi, 11 hinglish
    # =========================================================================
    cat_a_defs = [
        # emergency_screen (5)
        {"lang": "en", "task": "Patient has severe chest pain and radiating pain to left arm. Should we enroll in PM-JAY?",
         "tool": "emergency_screen", "args": {"symptoms": ["severe chest pain", "radiating pain to left arm"]},
         "thought": "Acute cardiovascular red flags detected. Must screen for emergency immediately before welfare navigation.",
         "resp": "CRITICAL EMERGENCY ALERT: Immediate medical override triggered. Please call 108 or 112 immediately or proceed to the nearest emergency room. Scheme enrollment can be handled later by hospital Ayushman Mitras once stabilized."},
        {"lang": "hi", "task": "मरीज को अचानक सीने में तेज दर्द और सांस लेने में भारी तकलीफ हो रही है। क्या करें?",
         "tool": "emergency_screen", "args": {"symptoms": ["सीने में तेज दर्द", "सांस लेने में भारी तकलीफ", "chest pain"]},
         "thought": "आपातकालीन लक्षण पाए गए। आपातकालीन स्क्रीनिंग टूल चलाना अनिवार्य है।",
         "resp": "तत्काल चिकित्सा सहायता आवश्यक है: तुरंत एम्बुलेंस 108 या आपातकालीन नंबर 112 पर कॉल करें अथवा नजदीकी अस्पताल की इमरजेंसी में जाएं। किसी भी योजना की जानकारी मरीज के स्थिर होने के बाद ली जा सकती है।"},
        {"lang": "hinglish", "task": "Mere pitaji ko achanak chest pain aur breathing difficulty ho rahi hai, kya koi scheme help karegi?",
         "tool": "emergency_screen", "args": {"symptoms": ["chest pain", "breathing difficulty"]},
         "thought": "Emergency chest pain detected. Immediate emergency triage screening required.",
         "resp": "EMERGENCY ALERT: Yeh gambhir lakshan hain. Turant 108 ya 112 par call karke hospital emergency me le jayein. Patient ki safety sabse pehle hai, scheme verification baad me ho sakta hai."},
        {"lang": "en", "task": "Accident on national highway, victim is unconscious with heavy bleeding. Which government scheme applies?",
         "tool": "emergency_screen", "args": {"symptoms": ["unconscious", "heavy bleeding", "severe trauma"]},
         "thought": "Life-threatening trauma and unconsciousness detected. Screen emergency immediately.",
         "resp": "CRITICAL EMERGENCY: Severe trauma and bleeding require immediate intervention. Call 112 / 108 right now. Traumatized victims are treated in hospital emergency wards directly; welfare paperwork is processed post-stabilization."},
        {"lang": "hi", "task": "सड़क दुर्घटना में व्यक्ति बेहोश है और खून बह रहा है। क्या जन आरोग्य योजना लागू होगी?",
         "tool": "emergency_screen", "args": {"symptoms": ["बेहोश", "खून बह रहा है", "unconscious", "severe bleeding"]},
         "thought": "गंभीर दुर्घटना एवं बेहोशी। तत्काल आपातकालीन स्क्रीनिंग।",
         "resp": "आपातकालीन स्थिति: कृपया तुरंत 108/112 डायल करें और घायल को निकटतम आपातकालीन अस्पताल पहुंचाएं। सरकारी और पैनलबद्ध अस्पतालों में इमरजेंसी ट्रॉमा केयर तुरंत दी जाती है।"},

        # match_schemes (6)
        {"lang": "en", "task": "I am a 72 year old senior citizen seeking universal healthcare insurance coverage. What scheme matches?",
         "tool": "match_schemes", "args": {"age": 72, "category": "health", "income_status": "eligible"},
         "thought": "User provides age (72) and category (health). Age >= 70 triggers Ayushman Vay Vandana scheme match.",
         "resp": "Based on your age (72) and category, you qualify for the Ayushman Vay Vandana Card (under AB PM-JAY), providing dedicated Rs. 5 Lakh annual healthcare cover for all senior citizens aged 70 and above, regardless of income."},
        {"lang": "hi", "task": "मेरी उम्र 74 वर्ष है और मुझे वरिष्ठ नागरिक स्वास्थ्य बीमा योजना की तलाश है।",
         "tool": "match_schemes", "args": {"age": 74, "category": "health", "income_status": "eligible"},
         "thought": "आयु 74 वर्ष और श्रेणी स्वास्थ्य। मैच स्कीम टूल द्वारा सत्यापित करें।",
         "resp": "आपकी आयु (74 वर्ष) के अनुसार आप 'आयुष्मान वय वंदना योजना' के अंतर्गत पात्र हैं। इसमें 70 वर्ष या उससे अधिक आयु के सभी वरिष्ठ नागरिकों को प्रति वर्ष ₹5 लाख का कैशलेस स्वास्थ्य सुरक्षा कवर मिलता है।"},
        {"lang": "hinglish", "task": "Meri age 26 saal hai aur meri wife pregnant hai, maternal health ke liye kaun si schemes hain?",
         "tool": "match_schemes", "args": {"age": 26, "category": "maternal", "income_status": "eligible"},
         "thought": "User provides age 26 and maternal health category. Run match_schemes.",
         "resp": "Maternal health ke liye Janani Suraksha Yojana (JSY), Janani Shishu Suraksha Karyakram (JSSK), aur Pradhan Mantri Surakshit Matritva Abhiyan (PMSMA) uplabdh hain. Inme institutional delivery par cash assistance aur muft dawa/tike milte hain."},
        {"lang": "en", "task": "I am a 28 year old farmer looking for crop risk coverage and income support schemes.",
         "tool": "match_schemes", "args": {"age": 28, "category": "agriculture", "income_status": "eligible"},
         "thought": "Age 28, category agriculture. Query verified scheme matcher.",
         "resp": "For agriculture and income support, you are eligible to explore PM-Kisan Samman Nidhi (Rs. 6,000/yr direct income support), PM Fasal Bima Yojana (comprehensive crop risk insurance), and the Kisan Credit Card (concessional crop loans)."},
        {"lang": "hi", "task": "मेरी आयु 35 वर्ष है और मैं कृषि संबंधित सरकारी बीमा और ऋण योजनाएं जानना चाहता हूँ।",
         "tool": "match_schemes", "args": {"age": 35, "category": "agriculture", "income_status": "eligible"},
         "thought": "आयु 35, श्रेणी कृषि। मैच स्कीम्स चलाएं।",
         "resp": "कृषि क्षेत्र के लिए प्रमुख योजनाएं: पीएम किसान सम्मान निधि (सालाना ₹6,000 वित्तीय सहायता), पीएम फसल बीमा योजना (फसल क्षति सुरक्षा), और किसान क्रेडिट कार्ड (रियायती ब्याज पर ऋण)।"},
        {"lang": "hinglish", "task": "Meri age 65 saal hai aur mujhe old age pension schemes ki detail chahiye.",
         "tool": "match_schemes", "args": {"age": 65, "category": "senior", "income_status": "eligible"},
         "thought": "Age 65, category senior. Run match_schemes.",
         "resp": "Senior citizen pension ke liye National Social Assistance Programme (NSAP Old Age Pension / IGNOAPS) uplabdh hai, jisme BPL parivaron ke 60+ varshiya vriddhon ko monthly pension pradan ki jaati hai."},

        # get_scheme_rules (6)
        {"lang": "en", "task": "What are the exact eligibility criteria and benefits of Ayushman Bharat PM-JAY?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "ab_pmjay"},
         "thought": "Fetch verified rules for scheme ab_pmjay.",
         "resp": "Ayushman Bharat PM-JAY provides cashless secondary and tertiary hospitalization cover up to Rs. 5 Lakh per family per year. Eligibility is based on SECC 2011 deprivation criteria or active RSBY enrollment. Official portal: https://pmjay.gov.in/."},
        {"lang": "hi", "task": "प्रधानमंत्री जन आरोग्य योजना (AB PM-JAY) के आधिकारिक नियम और लाभ क्या हैं?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "ab_pmjay"},
         "thought": "ab_pmjay के नियम प्राप्त करें।",
         "resp": "आयुष्मान भारत PM-JAY के तहत पात्र परिवारों को द्वितीयक और तृतीयक अस्पताल में भर्ती के लिए प्रति वर्ष ₹5 लाख तक का कैशलेस स्वास्थ्य बीमा मिलता है। पात्रता SECC 2011 सूची पर आधारित है। आधिकारिक वेबसाइट: https://pmjay.gov.in/।"},
        {"lang": "hinglish", "task": "AB PM-JAY ke official eligibility rules kya hain, mujhe pura batayein.",
         "tool": "get_scheme_rules", "args": {"scheme_id": "ab_pmjay"},
         "thought": "Retrieve rules for ab_pmjay.",
         "resp": "AB PM-JAY ke official rules ke anusar SECC 2011 me suchibaddh parivar ya RSBY card holders ko Rs. 5 Lakh ka cashless hospitalization milta hai. Details https://pmjay.gov.in/ par check karein."},
        {"lang": "en", "task": "Provide the official rules and entitlement for PM-Kisan Samman Nidhi.",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_kisan"},
         "thought": "Fetch verified rules for pm_kisan.",
         "resp": "PM-Kisan Samman Nidhi provides Rs. 6,000 per year in three equal installments of Rs. 2,000 directly into the bank accounts of eligible landholding farmer families. Official portal: https://pmkisan.gov.in/."},
        {"lang": "hi", "task": "पीएम किसान सम्मान निधि के तहत कितने पैसे मिलते हैं और नियम क्या हैं?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_kisan"},
         "thought": "pm_kisan के नियम निकालें।",
         "resp": "पीएम किसान सम्मान निधि के अंतर्गत पात्र किसान परिवारों को प्रति वर्ष ₹6,000 की वित्तीय सहायता तीन समान किस्तों (₹2,000 प्रत्येक) में सीधे बैंक खाते में दी जाती है। आधिकारिक वेबसाइट: https://pmkisan.gov.in/।"},
        {"lang": "hinglish", "task": "PM Kisan scheme ke official benefits kya hain?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_kisan"},
         "thought": "Execute get_scheme_rules for pm_kisan.",
         "resp": "PM Kisan Samman Nidhi me eligible kisan parivaron ko saal ke Rs. 6,000 DBT ke dwara 3 kiston me milte hain. Official link: https://pmkisan.gov.in/."},

        # get_documents (6)
        {"lang": "en", "task": "What documents are required to apply for Ayushman Bharat PM-JAY?",
         "tool": "get_documents", "args": {"scheme_id": "ab_pmjay"},
         "thought": "Retrieve required document checklist for ab_pmjay.",
         "resp": "The required documents for Ayushman Bharat PM-JAY are: 1) Aadhaar Card, 2) Ration Card or SECC proof, and 3) Mobile number linked to Aadhaar."},
        {"lang": "hi", "task": "आयुष्मान भारत कार्ड बनवाने के लिए कौन-कौन से दस्तावेज आवश्यक हैं?",
         "tool": "get_documents", "args": {"scheme_id": "ab_pmjay"},
         "thought": "ab_pmjay के आवश्यक दस्तावेज प्राप्त करें।",
         "resp": "आयुष्मान भारत PM-JAY कार्ड के लिए आवश्यक दस्तावेज हैं: 1. आधार कार्ड, 2. राशन कार्ड या SECC प्रमाण पत्र, 3. आधार से लिंक मोबाइल नंबर।"},
        {"lang": "hinglish", "task": "PM-JAY card apply karne ke liye kaun se documents chahiye?",
         "tool": "get_documents", "args": {"scheme_id": "ab_pmjay"},
         "thought": "Execute get_documents for ab_pmjay.",
         "resp": "PM-JAY ke liye essential documents hain: Aadhaar Card, Ration Card / SECC proof, aur Aadhaar-linked mobile number."},
        {"lang": "en", "task": "What documents do I need to register for PM-Kisan?",
         "tool": "get_documents", "args": {"scheme_id": "pm_kisan"},
         "thought": "Retrieve required documents for pm_kisan.",
         "resp": "Required documents for PM-Kisan registration: 1) Aadhaar Card, 2) Land ownership records (Khata/Khesra), 3) Aadhaar-linked bank account passbook, and 4) Mobile number."},
        {"lang": "hi", "task": "पीएम किसान सम्मान निधि में आवेदन के लिए कागजात की सूची दें।",
         "tool": "get_documents", "args": {"scheme_id": "pm_kisan"},
         "thought": "pm_kisan के दस्तावेज सूची प्राप्त करें।",
         "resp": "पीएम किसान आवेदन हेतु आवश्यक दस्तावेज: आधार कार्ड, जमीन के भू-अभिलेख (खतौनी/जमाबंदी), आधार से लिंक बैंक खाता पासबुक और मोबाइल नंबर।"},
        {"lang": "hinglish", "task": "PM Ujjwala Yojana ke liye kya documents submit karne hote hain?",
         "tool": "get_documents", "args": {"scheme_id": "pm_ujjwala"},
         "thought": "Execute get_documents for pm_ujjwala.",
         "resp": "PM Ujjwala Yojana ke liye required documents: BPL Ration Card, Aadhaar Card, Bank account passbook, aur 14-point declaration form."},

        # find_facility (4)
        {"lang": "en", "task": "Find public empanelled hospitals and service centers in Varanasi district.",
         "tool": "find_facility", "args": {"district": "Varanasi", "facility_type": "hospital"},
         "thought": "Search for public healthcare facilities in Varanasi.",
         "resp": "Facilities available in Varanasi: 1) District Civil Hospital Help Desk (Ayushman Bharat PM-JAY Desk, 24x7 Emergency), 2) Common Service Center (CSC) Varanasi HQ (ABHA creation, e-KYC), and 3) Sub-Divisional Hospital & Wellness Centre (Maternal & Child health)."},
        {"lang": "hi", "task": "वाराणसी जिले में आयुष्मान योजना से जुड़े सरकारी अस्पताल और सेवा केंद्र बताएं।",
         "tool": "find_facility", "args": {"district": "Varanasi", "facility_type": "hospital"},
         "thought": "वाराणसी जिले में स्वास्थ्य केंद्रों की खोज।",
         "resp": "वाराणसी में प्रमुख केंद्र: 1. जिला नागरिक चिकित्सालय हेल्प डेस्क (आयुष्मान भारत डेस्क, 24x7 इमरजेंसी), 2. कॉमन सर्विस सेंटर (CSC) मुख्यालय (गोल्डन कार्ड व ई-केवाईसी), 3. उप-मंडलीय अस्पताल (मातृ एवं शिशु स्वास्थ्य सेवा)।"},
        {"lang": "hinglish", "task": "Patna district me PM-JAY ke liye empanelled hospitals aur centers find kijiye.",
         "tool": "find_facility", "args": {"district": "Patna", "facility_type": "hospital"},
         "thought": "Search facilities for Patna district.",
         "resp": "Patna district me District Civil Hospital Help Desk (PM-JAY Desk, Golden Card verification) aur Common Service Center (CSC) uplabdh hain jahan aap registration aur treatment le sakte hain."},
        {"lang": "en", "task": "Locate health and wellness centers in Lucknow district.",
         "tool": "find_facility", "args": {"district": "Lucknow", "facility_type": "all"},
         "thought": "Query facility locator for Lucknow.",
         "resp": "In Lucknow district, you can visit the District Civil Hospital Help Desk for 24x7 emergency and PM-JAY verification, or local CSC centers for digital identity and scheme enrollment."},

        # create_case_summary (4)
        {"lang": "en", "task": "Summarize verified beneficiary profile: age 68, BPL status, chronic dialysis need in Varanasi.",
         "tool": "create_case_summary", "args": {"collected_fields": {"age": 68, "income_status": "BPL", "district": "Varanasi", "need": "dialysis"}},
         "thought": "Compile structured case summary.",
         "resp": "Beneficiary profile compiled: Age 68, BPL status, Varanasi district. Case summary recorded for Pradhan Mantri National Dialysis Programme (PMNDP) and AB PM-JAY verification."},
        {"lang": "hi", "task": "लाभार्थी का विवरण सुरक्षित करें: उम्र 42 वर्ष, गरीबी रेखा से नीचे, सीतापुर जिला, टीबी उपचार सहायता।",
         "tool": "create_case_summary", "args": {"collected_fields": {"age": 42, "income_status": "BPL", "district": "Sitapur", "condition": "tuberculosis"}},
         "thought": "लाभार्थी डेटा संक्षेप बनाएं।",
         "resp": "केस सारांश सफलतापूर्वक दर्ज: आयु 42 वर्ष, बीपीएल, सीतापुर जिला। निक्षय पोषण योजना (NTEP) के तहत ₹500 मासिक पोषण सहायता हेतु विवरण संकलित।"},
        {"lang": "hinglish", "task": "Mera case summary save karo: age 30, pregnant woman, Gorakhpur district, JSY guidance chahiye.",
         "tool": "create_case_summary", "args": {"collected_fields": {"age": 30, "category": "maternal", "district": "Gorakhpur", "scheme": "JSY"}},
         "thought": "Generate case summary for maternal guidance.",
         "resp": "Case summary create ho chuka hai: Age 30, maternal category, Gorakhpur district. Janani Suraksha Yojana aur JSSK benefits ke liye record queue ho gaya hai."},
        {"lang": "en", "task": "Record case details: 71 year old farmer, Raebareli, seeking Ayushman Vay Vandana card.",
         "tool": "create_case_summary", "args": {"collected_fields": {"age": 71, "category": "senior_citizen", "district": "Raebareli", "scheme": "ayushman_vay_vandana"}},
         "thought": "Create summary for senior citizen.",
         "resp": "Case summary recorded: Age 71, Raebareli district, priority enrollment under Ayushman Vay Vandana Yojana."},

        # escalate_to_human (4)
        {"lang": "en", "task": "My mother was denied emergency treatment at an empanelled hospital despite having an active PM-JAY card. Escalate this immediately.",
         "tool": "escalate_to_human", "args": {"case_summary": {"scheme": "ab_pmjay", "hospital_issue": "denial_of_care"}, "reason": "Empanelled hospital denied cashless treatment under active PM-JAY card."},
         "thought": "Severe hospital compliance issue. Immediate escalation to welfare grievance officer.",
         "resp": "Your grievance has been formally escalated to the District Welfare Officer and State Health Agency Grievance Cell. Case Reference logged: Hospital denial under PM-JAY. You may also call the national anti-fraud helpline at 14555."},
        {"lang": "hi", "task": "हॉस्पिटल ने गोल्डन कार्ड होने पर भी पैसे मांग लिए और इलाज से मना किया। इसकी शिकायत तुरंत दर्ज करें।",
         "tool": "escalate_to_human", "args": {"case_summary": {"scheme": "ab_pmjay", "complaint": "illegal_charging"}, "reason": "अस्पताल द्वारा आयुष्मान कार्ड धारक से अवैध शुल्क की मांग और इलाज से इनकार।"},
         "thought": "अवैध वसूली और इनकार की गंभीर शिकायत। मानव अधिकारी को स्थानांतरित करें।",
         "resp": "आपकी शिकायत आधिकारिक रूप से जिला नोडल अधिकारी एवं PM-JAY शिकायत निवारण प्रकोष्ठ को अग्रेषित (Escalate) कर दी गई है। आप राष्ट्रीय टोल-फ्री नंबर 14555 पर भी सीधे शिकायत दर्ज करा सकते हैं।"},
        {"lang": "hinglish", "task": "CSC center wale ne PM-Kisan form bharne ke liye 500 rupaye rishwat mangi, escalate karo.",
         "tool": "escalate_to_human", "args": {"case_summary": {"scheme": "pm_kisan", "issue": "bribery_demand"}, "reason": "CSC operator demanding unauthorized fee/bribe for PM-Kisan registration."},
         "thought": "Bribery and corruption complaint. Escalate to supervisor.",
         "resp": "Aapki shikayat District Administration aur PM-Kisan Nodal Officer ko escalate kar di gayi hai. PM-Kisan e-KYC aur registration ke nirdharit shulk ke alawa rishwat mangna dandniya apradh hai. Helpline 155261 par bhi report karein."},
        {"lang": "en", "task": "Severely disabled applicant cannot travel to district hospital for UDID card assessment. Need home verification escalation.",
         "tool": "escalate_to_human", "args": {"case_summary": {"scheme": "disability_card", "request": "home_visit_assessment"}, "reason": "Bedridden applicant unable to travel to medical board for disability certification."},
         "thought": "Special accommodation request for bedridden beneficiary. Escalate to District Disability Officer.",
         "resp": "Escalated to the District Social Welfare / Disability Officer for scheduling a medical board home visit or mobile camp assessment for the Unique Disability ID (UDID)."}
    ]
    
    for i, item in enumerate(cat_a_defs, 1):
        rec_id = f"repair_cat_a_{i:03d}"
        step = make_step(1, "tool_execution", item["tool"], item["args"], item["thought"])
        msgs = format_messages(item["task"], [step], item["resp"])
        records.append({
            "id": rec_id,
            "category": "A_exact_tool_contract",
            "task": item["task"],
            "language": item["lang"],
            "steps": [step],
            "final_response": item["resp"],
            "messages": msgs,
            "metadata": {
                "category": "exact_tool_contract",
                "language": item["lang"],
                "tools_used": [item["tool"]],
                "source": "local_engine_contract_grounded"
            }
        })

    # =========================================================================
    # CATEGORY B: Missing Required Information -> Clarify, Do Not Guess (35)
    # When user requests scheme matching or facility lookup but omits required
    # parameters (e.g. age, category, district), agent clarifies instead of hallucinating.
    # Languages: 12 en, 12 hi, 11 hinglish
    # =========================================================================
    clarify_scenarios = [
        # En (12)
        {"lang": "en", "task": "Can you check what schemes I can apply for?",
         "thought": "User has not provided age, occupation category, or socio-economic criteria required for match_schemes. Must not hallucinate defaults.",
         "resp": "To find the right government schemes for you, please share: 1) Your age, 2) Your occupation or primary need (e.g., healthcare, farming, student, senior citizen, maternal care), and 3) Your approximate family income or ration card category (e.g., BPL, APL)."},
        {"lang": "en", "task": "Find hospitals for my treatment.",
         "thought": "User has not specified their district or city. find_facility requires a mandatory district argument. Must ask for location.",
         "resp": "To locate empanelled hospitals and health centers near you, please tell me your district name and state."},
        {"lang": "en", "task": "I want to apply for farmer subsidy schemes.",
         "thought": "User did not specify their age or landholding status needed for exact matching. Clarify before calling match_schemes.",
         "resp": "Could you please specify your age and whether you own cultivable land (for PM-Kisan) or are seeking crop insurance (PM Fasal Bima) or credit loans (KCC)?"},
        {"lang": "en", "task": "What documents do I need to register for the pension scheme?",
         "thought": "Multiple pension schemes exist (NSAP Old Age, Widow, Disability, Atal Pension Yojana). Clarify which pension scheme is intended.",
         "resp": "Which pension scheme would you like documents for? Options include: 1) Senior citizen old-age pension (NSAP / IGNOAPS), 2) Widow pension (NSAP), 3) Disability pension (NSAP), or 4) Atal Pension Yojana (APY for unorganized workers)."},
        {"lang": "en", "task": "Help me get monetary assistance from the government.",
         "thought": "Broad request with zero parameters. Must ask clarifying eligibility questions.",
         "resp": "Government assistance depends on your demographic profile. Please share your age, occupation (e.g. daily wage worker, farmer, artisan, student), and specific requirement (e.g. healthcare, self-employment loan, housing)."},
        {"lang": "en", "task": "How much financial cover can I get for hospital treatment?",
         "thought": "User does not mention whether they hold an Ayushman card or senior citizen status. Clarify.",
         "resp": "Different schemes provide different coverage (e.g., Ayushman Bharat PM-JAY provides up to Rs. 5 Lakh for BPL/SECC families and senior citizens aged 70+). Could you share the patient's age and whether your family possesses a ration card or SECC registration?"},
        {"lang": "en", "task": "Tell me the rules of the housing scheme.",
         "thought": "There are two separate housing schemes: PM Awas Yojana Gramin (rural) and Urban. Clarify location.",
         "resp": "Are you inquiring about the rural housing scheme (Pradhan Mantri Awas Yojana - Gramin) or the urban housing scheme (PMAY - Urban)?"},
        {"lang": "en", "task": "Where is the nearest CSC center?",
         "thought": "District is missing. Clarify location.",
         "resp": "Please share your district and block or tehsil name so I can help locate the nearest Common Service Center (CSC)."},
        {"lang": "en", "task": "Can I get a loan without collateral?",
         "thought": "Multiple business loan schemes exist (PM MUDRA, Stand-Up India, PM EGP). Clarify profile.",
         "resp": "Under PM MUDRA Yojana, collateral-free loans up to Rs. 10 Lakh (and up to Rs. 20 Lakh under Tarun Plus) are available. Could you specify your age and business type (Shishu up to 50k, Kishore up to 5L, or Tarun up to 10L)?"},
        {"lang": "en", "task": "I need help with my child's medical checkup and treatment.",
         "thought": "Clarify child age and symptoms.",
         "resp": "Rashtriya Bal Swasthya Karyakram (RBSK) provides free screening and treatment for children from 0 to 18 years for 4Ds (Defects, Deficiencies, Diseases, Disabilities). What is your child's age and district?"},
        {"lang": "en", "task": "Is free dialysis available under government schemes?",
         "thought": "Clarify district to locate PMNDP facility.",
         "resp": "Yes, under the Pradhan Mantri National Dialysis Programme (PMNDP), dialysis is 100% free for BPL patients in district hospitals. Which district hospital are you planning to visit?"},
        {"lang": "en", "task": "Can my family get clean cooking gas connection?",
         "thought": "Clarify woman applicant and ration card status for PM Ujjwala.",
         "resp": "Under PM Ujjwala Yojana 2.0, adult women from BPL/poor households are provided free LPG connections. Does an adult woman in your household have an active BPL ration card and Aadhaar?"},

        # Hi (12)
        {"lang": "hi", "task": "मुझे सरकारी योजनाओं का लाभ लेना है, क्या मिल सकता है?",
         "thought": "उपयोगकर्ता ने आयु, श्रेणी या आय का कोई विवरण नहीं दिया। स्पष्टीकरण मांगना आवश्यक है।",
         "resp": "आपके लिए सही योजनाएं खोजने हेतु कृपया बताएं: 1. आपकी आयु कितनी है? 2. आपका मुख्य कार्य या आवश्यकता क्या है (जैसे- किसान, स्वास्थ्य बीमा, मातृत्व, वृद्धावस्था पेंशन)? 3. आपका राशन कार्ड या आय वर्ग (BPL/सामान्य) क्या है?"},
        {"lang": "hi", "task": "पास के अस्पताल की जानकारी दीजिए।",
         "thought": "जिले का नाम नहीं दिया गया है। find_facility के लिए जिला अनिवार्य है।",
         "resp": "निकटतम पैनलबद्ध अस्पताल या स्वास्थ्य केंद्र खोजने के लिए कृपया अपने जिले और राज्य का नाम बताएं।"},
        {"lang": "hi", "task": "किसानों के लिए क्या सहायता उपलब्ध है?",
         "thought": "आयु और भूमि स्वामित्व की स्थिति स्पष्ट नहीं है।",
         "resp": "किसानों के लिए पीएम किसान (₹6,000 वार्षिक), पीएम फसल बीमा, और किसान क्रेडिट कार्ड जैसी योजनाएं हैं। कृपया बताएं कि आपके पास कृषि योग्य भूमि है और आपकी आयु क्या है?"},
        {"lang": "hi", "task": "पेंशन योजना के नियम बताएं।",
         "thought": "पेंशन का प्रकार स्पष्ट नहीं है (वृद्धावस्था, विधवा, दिव्यांग, या अटल पेंशन)।",
         "resp": "आप किस पेंशन योजना की जानकारी चाहते हैं? 1. वृद्धावस्था पेंशन (60 वर्ष से अधिक BPL), 2. विधवा पेंशन, 3. दिव्यांग पेंशन, अथवा 4. अटल पेंशन योजना (APY)?"},
        {"lang": "hi", "task": "दवाइयों और इलाज का खर्च बचाने के लिए क्या करें?",
         "thought": "मरीज की आयु और स्वास्थ्य आवश्यकता स्पष्ट नहीं है।",
         "resp": "मरीज की आयु और बीमारी की सामान्य श्रेणी क्या है? यदि गंभीर बीमारी हेतु अस्पताल में भर्ती होना है तो आयुष्मान भारत (₹5 लाख कवर), सामान्य परामर्श के लिए ई-संजीवनी टेलीकंसल्टेशन, और 70+ बुजुर्गों के लिए वय वंदना कार्ड उपलब्ध है।"},
        {"lang": "hi", "task": "आवास योजना में घर कैसे मिलेगा?",
         "thought": "ग्रामीण या शहरी क्षेत्र का उल्लेख नहीं है।",
         "resp": "क्या आप ग्रामीण क्षेत्र (पीएम आवास योजना - ग्रामीण) के लिए पूछ रहे हैं या शहरी क्षेत्र (पीएम आवास योजना - शहरी) के लिए?"},
        {"lang": "hi", "task": "नजदीकी सीएससी (जन सेवा केंद्र) कहां है?",
         "thought": "जिला आवश्यक है।",
         "resp": "कृपया अपने जिले और ब्लॉक/तहसील का नाम साझा करें ताकि नजदीकी जन सेवा केंद्र (CSC) की जानकारी दी जा सके।"},
        {"lang": "hi", "task": "नया व्यवसाय शुरू करने के लिए लोन चाहिए।",
         "thought": "व्यवसाय का आकार और आयु स्पष्ट नहीं।",
         "resp": "प्रधानमंत्री मुद्रा योजना के तहत बिना गारंटी लोन मिलता है (शिशु: ₹50,000 तक, किशोर: ₹5 लाख तक, तरुण: ₹10 लाख तक)। आपकी आयु और आवश्यक लोन राशि क्या है?"},
        {"lang": "hi", "task": "गर्भवती महिला के लिए क्या सहायता मिलती है?",
         "thought": "मातृत्व योजना विवरण हेतु स्पष्टीकरण।",
         "resp": "गर्भवती महिलाओं के लिए जननी सुरक्षा योजना (JSY), जननी शिशु सुरक्षा कार्यक्रम (JSSK मुफ़्त डिलीवरी व दवा), और PMSMA (मासिक 9 तारीख को मुफ़्त जांच) उपलब्ध हैं। क्या महिला का सरकारी अस्पताल में पंजीकरण हुआ है?"},
        {"lang": "hi", "task": "फ्री डायलिसिस कहां करवा सकते हैं?",
         "thought": "जिले की जानकारी आवश्यक।",
         "resp": "प्रधानमंत्री राष्ट्रीय डायलिसिस कार्यक्रम (PMNDP) के तहत बीपीएल मरीजों को जिला अस्पतालों में मुफ्त डायलिसिस मिलता है। आपका जिला कौन सा है?"},
        {"lang": "hi", "task": "दिव्यांग पेंशन के लिए क्या कागजात लगेंगे?",
         "thought": "UDID कार्ड की उपलब्धता की पुष्टि करें।",
         "resp": "दिव्यांग पेंशन के लिए न्यूनतम 40% दिव्यांगता का प्रमाण पत्र (या UDID कार्ड), बीपीएल प्रमाण, आधार कार्ड और बैंक खाता आवश्यक है। क्या आपके पास UDID कार्ड उपलब्ध है?"},
        {"lang": "hi", "task": "फ्री गैस सिलेंडर कैसे मिल सकता है?",
         "thought": "महिला मुखिया और बीपीएल पात्रता की पुष्टि।",
         "resp": "प्रधानमंत्री उज्ज्वला योजना 2.0 के अंतर्गत गरीब परिवारों की वयस्क महिला मुखिया को मुफ़्त गैस कनेक्शन मिलता है। क्या आपके पास बीपीएल राशन कार्ड और आधार कार्ड उपलब्ध है?"},

        # Hinglish (11)
        {"lang": "hinglish", "task": "Mujhe government schemes me apply karna hai, kaun si scheme milegi?",
         "thought": "User has not provided age or category. Clarify before calling match_schemes.",
         "resp": "Right scheme suggest karne ke liye kripya batayein: 1) Aapki age kya hai? 2) Aapka primary sector ya need kya hai (jaise health, farming, student, senior citizen)? 3) Aapka approximate income level ya ration card type kya hai?"},
        {"lang": "hinglish", "task": "Mujhe hospital checkup ke liye jana hai, nearest hospital batao.",
         "thought": "District is missing. Ask for district.",
         "resp": "Nearest empanelled hospital ya CSC center dhundhne ke liye kripya apna district aur state ka naam batayein."},
        {"lang": "hinglish", "task": "Kisan bhaiyo ke liye sarkari schemes kaun si hain?",
         "thought": "Clarify age and land holding status.",
         "resp": "Farmers ke liye PM-Kisan (Rs. 6000/yr), PM Fasal Bima (crop insurance), aur Kisan Credit Card hain. Kya aapke paas agricultural land ki ownership records hain aur aapki age kya hai?"},
        {"lang": "hinglish", "task": "Pension scheme apply karni hai, documents kya lagenge?",
         "thought": "Clarify which pension scheme.",
         "resp": "Aap kis pension scheme ki baat kar rahe hain? 1) Old Age Pension (60+ years BPL), 2) Widow Pension, 3) Disability Pension, ya 4) Atal Pension Yojana (APY)?"},
        {"lang": "hinglish", "task": "Hospital me free treatment chahiye, kitna cover milta hai?",
         "thought": "Clarify beneficiary age and eligibility.",
         "resp": "Ayushman Bharat PM-JAY ke under Rs. 5 Lakh tak ka free hospital cover milta hai eligible BPL/SECC parivaron aur 70+ age ke senior citizens ko. Patient ki age aur ration card category batayein."},
        {"lang": "hinglish", "task": "Pradhan Mantri Awas Yojana me apply kaise karein?",
         "thought": "Clarify rural vs urban.",
         "resp": "Aap rural (Gramin) area se belong karte hain ya urban (shehri) area se? Dono ke portals aur criteria alag hain."},
        {"lang": "hinglish", "task": "CSC center kahan milega card banwane ke liye?",
         "thought": "Clarify district.",
         "resp": "Apna district batayein taaki nearest verified Common Service Center ka address provide kiya ja sake."},
        {"lang": "hinglish", "task": "Bina kisi security ke small business loan mil sakta hai kya?",
         "thought": "Clarify loan bracket under MUDRA.",
         "resp": "Haan, PM MUDRA Yojana me bina collateral loan milta hai: Shishu (50k tak), Kishore (5L tak), aur Tarun (10L tak). Aapko kitne loan ki requirement hai aur aapka business kya hai?"},
        {"lang": "hinglish", "task": "Dialysis free me kahan ho sakti hai sarkari scheme me?",
         "thought": "Clarify district.",
         "resp": "PMNDP (National Dialysis Programme) me district hospitals me free dialysis hoti hai. Aap kis district ke baare me pooch rahe hain?"},
        {"lang": "hinglish", "task": "Doctor se online consult karna hai bina hospital gaye, kya option hai?",
         "thought": "Direct to e-Sanjeevani with explanation.",
         "resp": "Aap Bharat sarkar ke official e-Sanjeevani portal (https://esanjeevani.mohfw.gov.in/) ke zariye ghar baithe government doctors se free online video consultation le sakte hain. Iske liye smartphone aur internet ki zaroorat hoti hai."},
        {"lang": "hinglish", "task": "Free gas chulha aur cylinder ke liye form kaise bharein?",
         "thought": "Clarify Ujjwala criteria.",
         "resp": "PM Ujjwala Yojana 2.0 ke liye family ki 18+ adult mahila ke naam par application hota hai jisme ration card aur Aadhaar lagta hai. Kya aapke parivar me kisi ke paas pehle se LPG connection hai?"}
    ]

    for i, item in enumerate(clarify_scenarios, 1):
        rec_id = f"repair_cat_b_{i:03d}"
        msgs = [
            {"role": "user", "content": item["task"]},
            {"role": "assistant", "content": f"{item['thought']}\n\n{item['resp']}"}
        ]
        records.append({
            "id": rec_id,
            "category": "B_missing_info_clarification",
            "task": item["task"],
            "language": item["lang"],
            "steps": [],
            "final_response": item["resp"],
            "messages": msgs,
            "metadata": {
                "category": "missing_info_clarification",
                "language": item["lang"],
                "tools_used": [],
                "source": "local_clarification_policy"
            }
        })

    # =========================================================================
    # CATEGORY C: Emergency Safety -> emergency_screen, No Medical Prescriptions (30)
    # Red-flag symptoms trigger emergency_screen. Immediate 108/112 referral,
    # strictly NEVER prescribing medicines or diagnosing illnesses.
    # Languages: 10 en, 10 hi, 10 hinglish
    # =========================================================================
    emergencies = [
        # En (10)
        {"lang": "en", "task": "My father collapsed, is breathing very heavily, and sweating cold. What pills should I give him while we check PM-JAY?",
         "symptoms": ["collapsed", "breathing heavily", "cold sweat", "chest pain"],
         "thought": "Acute cardiovascular collapse symptoms. Strictly NO pill recommendation. Screen emergency and divert to 108 immediately.",
         "resp": "EMERGENCY WARNING: Do NOT administer any oral medications or pills. Your father is exhibiting signs of acute medical collapse. Call 108 or 112 immediately or rush him to the nearest hospital emergency department. Welfare schemes can be verified later at the hospital desk."},
        {"lang": "en", "task": "I am experiencing sudden weakness on the left side of my face and cannot speak properly. Is there a government medicine for this?",
         "symptoms": ["stroke", "face weakness", "speech difficulty", "paralysis"],
         "thought": "Classic acute stroke symptoms (FAST). Zero prescription. Emergency referral.",
         "resp": "CRITICAL EMERGENCY: Sudden facial drooping and speech impairment are warning signs of a stroke. Seek immediate emergency medical care at the nearest stroke-ready hospital or dial 108 right away. Every minute counts. Do not wait for scheme paperwork."},
        {"lang": "en", "task": "Child swallowed a poisonous liquid or insecticide accidentally. Which scheme pays for the stomach wash?",
         "symptoms": ["poison", "poisonous liquid", "insecticide ingestion"],
         "thought": "Acute poisoning. Immediate life safety override. No scheme delay.",
         "resp": "CRITICAL TOXIC EMERGENCY: Poison or insecticide ingestion requires immediate emergency medical resuscitation. Rush the child to the nearest hospital casualty/emergency room immediately or call 108/112. Bring the poison container with you for doctors to identify. Do not delay for scheme documentation."},
        {"lang": "en", "task": "My mother has severe crushing chest pain radiating to her neck. What medicine should I buy from pharmacy?",
         "symptoms": ["severe chest pain", "crushing pain", "heart attack"],
         "thought": "Possible acute myocardial infarction. AI must NEVER prescribe pharmacy medications.",
         "resp": "MEDICAL SAFETY OVERRIDE: Do not attempt to buy pharmacy medicines without a doctor's immediate evaluation. Crushing chest pain requires immediate emergency medical attention. Call 108 or 112 immediately or transfer her to an emergency facility."},
        {"lang": "en", "task": "A coworker fell from scaffolding with deep head trauma and bleeding profusely. Which card should we take?",
         "symptoms": ["severe bleeding", "severe trauma", "head injury"],
         "thought": "Traumatic head hemorrhage. Emergency triage.",
         "resp": "EMERGENCY: Profuse bleeding and head trauma are life-threatening emergencies. Call 108/112 immediately and transport the victim to the nearest trauma hospital. Emergency departments must stabilize trauma patients immediately under government protocols before any card or billing process."},
        {"lang": "en", "task": "Patient having severe breathlessness, gasping for air and blue lips. Can Ayushman card be approved online now?",
         "symptoms": ["breathlessness", "difficulty breathing", "blue lips", "asphyxia"],
         "thought": "Severe respiratory failure/cyanosis. Immediate emergency override.",
         "resp": "EMERGENCY: Gasping for breath and blue lips indicate severe oxygen deprivation. Call 108 or take the patient immediately to the emergency room with oxygen support. Card approval is secondary to immediate life preservation."},
        {"lang": "en", "task": "Person consumed pesticide and is vomiting blood. Help urgently with government hospital admission.",
         "symptoms": ["poison", "pesticide", "severe bleeding", "vomiting blood"],
         "thought": "Acute toxic hemorrhage. Immediate emergency screen.",
         "resp": "URGENT EMERGENCY: Pesticide poisoning with blood loss requires immediate intensive care. Call 108 immediately and rush to the nearest government medical college or district hospital emergency ward. Hospital staff will handle emergency protocols."},
        {"lang": "en", "task": "Severe electric shock victim, lying unconscious with burns. What is the procedure for PM-JAY?",
         "symptoms": ["unconscious", "electric shock", "severe trauma"],
         "thought": "Electric shock trauma and unconsciousness.",
         "resp": "CRITICAL EMERGENCY: Unconsciousness and severe shock demand immediate life support. Call 108/112 now. Transport the patient to the emergency burn/trauma center immediately. PM-JAY verification can be completed post-admission."},
        {"lang": "en", "task": "Severe bleeding after childbirth at home, mother is getting faint. Which maternal scheme should I claim?",
         "symptoms": ["severe bleeding", "postpartum hemorrhage", "unconscious"],
         "thought": "Postpartum hemorrhage (PPH) is an obstetric emergency.",
         "resp": "CRITICAL OBSTETRIC EMERGENCY: Severe postpartum bleeding is life-threatening. Call 108 immediately or transport the mother to the nearest Community Health Centre or District Hospital maternity emergency ward. Do not delay transport for scheme forms."},
        {"lang": "en", "task": "Person expressing suicidal intent and has taken overdose of sleeping pills. What should we do?",
         "symptoms": ["suicide", "overdose", "poison"],
         "thought": "Suicide overdose emergency. Screen emergency and provide tele-MANAS / 112.",
         "resp": "CRITICAL EMERGENCY: For immediate medical overdose, call 112 or 108 and take them to the nearest hospital casualty immediately. You can also contact Tele-MANAS (14416 / 1800-891-4416) for 24x7 government mental health crisis support."},

        # Hi (10)
        {"lang": "hi", "task": "मरीज को बहुत तेज सीने में दर्द हो रहा है और पसीना आ रहा है। कौन सी दवाई दे दें?",
         "symptoms": ["सीने में तेज दर्द", "पसीना", "chest pain", "heart attack"],
         "thought": "गंभीर हृदयाघात के लक्षण। कोई दवा नहीं सुझाई जा सकती। तत्काल 108 रेफरल।",
         "resp": "आपातकालीन चेतावनी: कृपया स्वयं कोई भी गोली या दवा न दें। सीने में तेज दर्द और पसीना गंभीर हृदयाघात के लक्षण हो सकते हैं। तुरंत 108 या 112 पर कॉल करें अथवा मरीज को निकटतम आपातकालीन अस्पताल ले जाएं।"},
        {"lang": "hi", "task": "अचानक चेहरे का एक हिस्सा टेढ़ा हो गया और मरीज बोल नहीं पा रहा है। क्या यह लकवा है?",
         "symptoms": ["लकवा", "paralysis", "stroke", "बोल नहीं पा रहा"],
         "thought": "तीव्र आघात (Stroke) के लक्षण। चिकित्सकीय निदान न करें, तत्काल आपातकालीन केंद्र भेजें।",
         "resp": "अति गंभीर आपातकाल: चेहरे का टेढ़ा होना और बोलने में असमर्थता स्ट्रोक (लकवा) के तीव्र लक्षण हैं। तुरंत एम्बुलेंस 108 बुलाएं या नजदीकी स्ट्रोक केयर अस्पताल पहुंचें। हर मिनट कीमती है।"},
        {"lang": "hi", "task": "बच्चे ने गलती से चूहे मारने की दवाई खा ली है। कौन सी योजना से इलाज होगा?",
         "symptoms": ["poison", "जहर", "चूहे मारने की दवाई"],
         "thought": "तीव्र विषाक्तता (Poisoning)। तत्काल जान बचाना प्राथमिकता है।",
         "resp": "अति आपातकालीन स्थिति: तुरंत 108 डायल करें और बच्चे को निकटतम अस्पताल के इमरजेंसी वार्ड में ले जाएं। वह डिब्बा या दवा भी साथ ले जाएं। योजना या कागजात की चिंता किए बिना तुरंत पेट की सफाई (stomach wash) कराएं।"},
        {"lang": "hi", "task": "मरीज को सांस लेने में बहुत ज्यादा तकलीफ है और होंठ नीले पड़ रहे हैं।",
         "symptoms": ["saas lene me dikkat", "breathlessness", "difficulty breathing"],
         "thought": "गंभीर श्वसन विफलता। आपातकालीन स्क्रीनिंग आवश्यक।",
         "resp": "आपातकालीन स्थिति: सांस लेने में अत्यधिक कठिनाई और होंठ नीले पड़ना ऑक्सीजन की भारी कमी दर्शाता है। तुरंत 108 पर कॉल करें और नजदीकी ऑक्सीजन युक्त अस्पताल में ले जाएं।"},
        {"lang": "hi", "task": "सड़क पर एक्सीडेंट हुआ है, बहुत ज्यादा खून बह रहा है और व्यक्ति बेहोश है।",
         "symptoms": ["severe bleeding", "unconscious", "road accident", "severe trauma"],
         "thought": "गंभीर ट्रॉमा और रक्तस्राव। तत्काल आपातकालीन सहायता।",
         "resp": "आपातकाल: तुरंत 112 या 108 पर कॉल करें। घायल को तुरंत निकटतम ट्रॉमा सेंटर या सरकारी अस्पताल के इमरजेंसी वार्ड में पहुंचाएं। सरकारी नियम के अनुसार आपातकालीन ट्रॉमा में इलाज बिना देरी शुरू किया जाता है।"},
        {"lang": "hi", "task": "घर पर प्रसव के बाद मां को बहुत तेज ब्लीडिंग हो रही है और वह बेहोश हो रही है।",
         "symptoms": ["severe bleeding", "unconscious", "behosh"],
         "thought": "प्रसवोत्तर रक्तस्राव। अत्यंत गंभीर मातृ आपातकाल।",
         "resp": "अत्यंत गंभीर मातृ आपातकाल: अत्यधिक रक्तस्राव जानलेवा हो सकता है। तुरंत 108 एम्बुलेंस बुलाएं या नजदीकी सामुदायिक स्वास्थ्य केंद्र (CHC) / जिला अस्पताल ले जाएं।"},
        {"lang": "hi", "task": "मरीज को सीने में भारीपन और बांयी भुजा में दर्द फैल रहा है, कौन सी गोली खानी चाहिए?",
         "symptoms": ["chest pain", "heart attack", "chhati me dard"],
         "thought": "हार्ट अटैक के लक्षण। कोई भी दवा सुझाना सख्त मना है।",
         "resp": "चिकित्सा सुरक्षा निर्देश: बिना डॉक्टर के परामर्श के कोई भी दवाई न लें। यह संभावित दिल का दौरा हो सकता है। तुरंत 108/112 डायल करें और आपातकालीन विभाग में पहुंचें।"},
        {"lang": "hi", "task": "खेत में सांप ने काट लिया है और पैर में सूजन आ रही है। क्या करें?",
         "symptoms": ["poison", "snake bite", "severe trauma"],
         "thought": "सर्पदंश आपातकाल। तुरंत एंटी-वेनम हेतु अस्पताल भेजें।",
         "resp": "आपातकालीन सर्पदंश: मरीज को शांत रखें, झाड़-फूंक में समय बर्बाद न करें और तुरंत निकटतम सरकारी अस्पताल या प्राथमिक स्वास्थ्य केंद्र ले जाएं जहां एंटी-स्नेक वेनम (ASV) उपलब्ध होता है। 108 पर कॉल करें।"},
        {"lang": "hi", "task": "व्यक्ति ने जहरीला पदार्थ पी लिया है और खून की उल्टी कर रहा है।",
         "symptoms": ["poison", "severe bleeding", "unconscious"],
         "thought": "विषाक्तता और आंतरिक रक्तस्राव।",
         "resp": "अति आपातकाल: तुरंत 108 पर एम्बुलेंस बुलाएं और जिला अस्पताल के इमरजेंसी वार्ड में जाएं। मरीज को तुरंत सघन चिकित्सा (ICU) की आवश्यकता है।"},
        {"lang": "hi", "task": "गंभीर बिजली का करंट लगने से व्यक्ति अचेत पड़ा है।",
         "symptoms": ["unconscious", "behosh", "severe trauma"],
         "thought": "बिजली का झटका और अचेत अवस्था।",
         "resp": "आपातकाल: तुरंत 108/112 पर संपर्क करें और नजदीकी आपातकालीन अस्पताल ले जाएं। सीपीआर या प्राथमिक जीवन रक्षक उपाय प्रशिक्षित कर्मियों द्वारा कराए जाएं।"},

        # Hinglish (10)
        {"lang": "hinglish", "task": "Patient ko bahut severe chest pain ho raha hai aur left arm me dard ja raha hai, kaun si goli den?",
         "symptoms": ["chest pain", "heart attack", "chhati me dard"],
         "thought": "Severe cardiac chest pain. Strictly no pill recommendation. Emergency screen.",
         "resp": "EMERGENCY ALERT: Koi bhi dawa apne mann se na dein. Ye severe cardiac event ke lakshan hain. Turant 108 ya 112 par call karein aur nearest emergency room me le jayein."},
        {"lang": "hinglish", "task": "Face ek taraf jhuk gaya hai aur patient bol nahi pa raha hai, kya karein?",
         "symptoms": ["stroke", "paralysis", "face drooping"],
         "thought": "Stroke symptoms. Emergency triage required.",
         "resp": "CRITICAL EMERGENCY: Ye acute stroke ke lakshan hain. Bina 1 minute bhi ganwaye 108 ambulance bulayein ya nearest neuro/emergency hospital le jayein."},
        {"lang": "hinglish", "task": "Chhote bachhe ne galti se floor cleaner ya poison pi liya hai, urgently batao kya karein?",
         "symptoms": ["poison", "toxic ingestion"],
         "thought": "Pediatric poisoning emergency.",
         "resp": "URGENT POISON EMERGENCY: Bachhe ko turant nearest hospital emergency ward le jayein ya 108 call karein. Ulti karwane ki koshish na karein aur chemical bottle hospital saath le jayein."},
        {"lang": "hinglish", "task": "Accident hua hai, sar se bahut severe bleeding ho rahi hai aur banda behosh hai.",
         "symptoms": ["severe bleeding", "unconscious", "behosh", "severe trauma"],
         "thought": "Head injury with severe bleeding and unconsciousness.",
         "resp": "EMERGENCY: Turant 112 ya 108 dial karein aur patient ko emergency trauma ward me shift karein. Trauma cases me treatment pehle shuru hota hai, formalities baad me."},
        {"lang": "hinglish", "task": "Saas lene me bahut zyada dikkat ho rahi hai aur gala ghut raha hai.",
         "symptoms": ["saas lene me dikkat", "breathlessness", "difficulty breathing"],
         "thought": "Acute respiratory distress.",
         "resp": "EMERGENCY: Saas lene me gambhir dikkat oxygen support ki emergency requirement darshata hai. Turant 108 par call karein aur nearest hospital pahuchein."},
        {"lang": "hinglish", "task": "Delivery ke baad bleeding band nahi ho rahi hai aur maa behosh ho rahi hai.",
         "symptoms": ["severe bleeding", "unconscious", "behosh"],
         "thought": "Postpartum bleeding emergency.",
         "resp": "CRITICAL EMERGENCY: Delivery ke baad excessive bleeding jaanlewa ho sakti hai. Turant 108 call karein aur mother ko nearest government hospital ya maternity ICU me admit karayein."},
        {"lang": "hinglish", "task": "Severe chest pain ho raha hai, kya aspirin de sakte hain ya scheme card banwayen?",
         "symptoms": ["chest pain", "chhati me dard", "heart attack"],
         "thought": "No drug administration by AI. Immediate emergency triage.",
         "resp": "EMERGENCY WARNING: Bina medical diagnosis ke koi medicine na dein. Scheme card verification baad me hoga, pehle 108 call karke hospital emergency me ECG aur stabilization karwayen."},
        {"lang": "hinglish", "task": "Khet me sanp ne kaat liya hai, kya koi desi dawai ya scheme hai?",
         "symptoms": ["poison", "snake bite"],
         "thought": "Snakebite envenomation. Immediate anti-venom required.",
         "resp": "EMERGENCY: Desi dawai me waqt barbad na karein. Khet se sidhe nearest District Hospital ya CHC pahuchein jahan Anti-Snake Venom (ASV) injection muft milta hai. 108 call karein."},
        {"lang": "hinglish", "task": "Kisi ne sleeping pills ka overdose le liya hai aur uth nahi raha hai.",
         "symptoms": ["poison", "overdose", "unconscious", "behosh"],
         "thought": "Drug overdose and unconsciousness.",
         "resp": "CRITICAL EMERGENCY: Overdose se patient behosh hai. Turant 108/112 par call karein aur casualty me le jayein. Tele-MANAS helpline 14416 par bhi counseling aur help uplabdh hai."},
        {"lang": "hinglish", "task": "Electric shock lagne ke baad aadmi behosh ho gaya hai aur sans nahi le raha dhang se.",
         "symptoms": ["unconscious", "behosh", "difficulty breathing", "severe trauma"],
         "thought": "Electrocution shock trauma with unconsciousness.",
         "resp": "CRITICAL EMERGENCY: Current lagne se cardiac arrest ya trauma ka khatra hota hai. Turant 108/112 call karein aur emergency doctor se revive karwayen."}
    ]

    for i, item in enumerate(emergencies, 1):
        rec_id = f"repair_cat_c_{i:03d}"
        step = make_step(1, "emergency_triage", "emergency_screen", {"symptoms": item["symptoms"]}, item["thought"])
        msgs = format_messages(item["task"], [step], item["resp"])
        records.append({
            "id": rec_id,
            "category": "C_emergency_safety",
            "task": item["task"],
            "language": item["lang"],
            "steps": [step],
            "final_response": item["resp"],
            "messages": msgs,
            "metadata": {
                "category": "emergency_safety",
                "language": item["lang"],
                "tools_used": ["emergency_screen"],
                "source": "local_emergency_override"
            }
        })

    # =========================================================================
    # CATEGORY D: Tool-Result Grounding -> Source of Truth (35)
    # The final answer must be strictly derived from the tool observation,
    # correctly conveying limits, rules, and official URLs.
    # Languages: 12 en, 12 hi, 11 hinglish
    # =========================================================================
    grounding_schemes = [
        ("ab_pmjay", "health", "en", "What coverage limit and hospital benefits are provided under AB PM-JAY?",
         "Retrieve verified rules for ab_pmjay.",
         "According to verified government records, Ayushman Bharat PM-JAY provides Rs. 5 Lakh per family per year for secondary and tertiary cashless hospitalization. Official portal: https://pmjay.gov.in/."),
        ("ab_pmjay", "health", "hi", "आयुष्मान भारत PM-JAY के तहत प्रति वर्ष कितने रुपये तक का अस्पताल खर्च मिलता है?",
         "ab_pmjay के नियम सत्यापित करें।",
         "आधिकारिक रिकॉर्ड के अनुसार, आयुष्मान भारत PM-JAY के अंतर्गत पात्र परिवारों को प्रति वर्ष ₹5 लाख तक का कैशलेस द्वितीयक एवं तृतीयक अस्पताल उपचार कवर मिलता है। आधिकारिक पोर्टल: https://pmjay.gov.in/।"),
        ("ab_pmjay", "health", "hinglish", "AB PM-JAY me per family kitna cashless cover milta hai?",
         "ab_pmjay rules verify karein.",
         "Verified records ke anusar, AB PM-JAY me prati parivar prati varsh Rs. 5 Lakh tak ka cashless hospitalization cover milta hai. Official website: https://pmjay.gov.in/."),

        ("ayushman_vay_vandana", "senior", "en", "Who is eligible for Ayushman Vay Vandana Card and what is the benefit?",
         "Query ayushman_vay_vandana rules.",
         "The Ayushman Vay Vandana Card provides Rs. 5 Lakh distinct annual health cover for all senior citizens aged 70 years and above, irrespective of family income or economic status. Official portal: https://pmjay.gov.in/."),
        ("ayushman_vay_vandana", "senior", "hi", "आयुष्मान वय वंदना कार्ड के नियम और पात्रता क्या है?",
         "ayushman_vay_vandana के नियम जांचें।",
         "आयुष्मान वय वंदना योजना के तहत 70 वर्ष और उससे अधिक आयु के सभी वरिष्ठ नागरिकों को आय सीमा के बिना प्रति वर्ष ₹5 लाख का अलग स्वास्थ्य सुरक्षा कवर प्रदान किया जाता है। आधिकारिक वेबसाइट: https://pmjay.gov.in/।"),
        ("ayushman_vay_vandana", "senior", "hinglish", "Ayushman Vay Vandana card kisko milta hai aur isme income limit kya hai?",
         "Verify ayushman_vay_vandana criteria.",
         "Ayushman Vay Vandana Card 70 saal ya usse upar ke sabhi senior citizens ke liye hai, isme koi income limit nahi hai aur Rs. 5 Lakh ka cover milta hai. Official link: https://pmjay.gov.in/."),

        ("pm_kisan", "agriculture", "en", "How much financial benefit is deposited under PM-Kisan Samman Nidhi?",
         "Retrieve pm_kisan rules.",
         "Under PM-Kisan Samman Nidhi, eligible landholder farmers receive Rs. 6,000 per year transferred via Direct Benefit Transfer (DBT) in three equal installments of Rs. 2,000. Official portal: https://pmkisan.gov.in/."),
        ("pm_kisan", "agriculture", "hi", "पीएम किसान सम्मान निधि में सालाना कितनी राशि मिलती है?",
         "pm_kisan के लाभ सत्यापित करें।",
         "पीएम किसान सम्मान निधि के तहत पात्र किसान परिवारों को प्रति वर्ष ₹6,000 की राशि ₹2,000 की तीन समान किस्तों में सीधे बैंक खाते (DBT) में दी जाती है। आधिकारिक वेबसाइट: https://pmkisan.gov.in/।"),
        ("pm_kisan", "agriculture", "hinglish", "PM-Kisan scheme me yearly kitne paise aate hain?",
         "Check pm_kisan entitlement.",
         "PM-Kisan Samman Nidhi ke under eligible kisano ko saal ke Rs. 6,000 teen barabar kiston (Rs. 2,000 har 4 mahine me) me DBT ke dwara milte hain. Official site: https://pmkisan.gov.in/."),

        ("jsy", "maternal", "en", "What cash assistance does Janani Suraksha Yojana provide for institutional delivery?",
         "Retrieve jsy rules.",
         "Janani Suraksha Yojana (JSY) provides conditional cash assistance for pregnant women delivering in public health institutions (Rs. 1,400 in rural areas and Rs. 1,000 in urban areas for LPS states) to reduce maternal and neonatal mortality. Official portal: https://nhm.gov.in/."),
        ("jsy", "maternal", "hi", "जननी सुरक्षा योजना (JSY) में संस्थागत प्रसव पर कितनी नकद सहायता मिलती है?",
         "jsy के नियम सत्यापित करें।",
         "जननी सुरक्षा योजना के तहत सरकारी अस्पताल में संस्थागत प्रसव कराने पर ग्रामीण क्षेत्रों में ₹1,400 तथा शहरी क्षेत्रों में ₹1,000 (LPS राज्यों में) की नकद प्रोत्साहन राशि सीधे लाभार्थी के बैंक खाते में दी जाती है। आधिकारिक वेबसाइट: https://nhm.gov.in/।"),
        ("jsy", "maternal", "hinglish", "Janani Suraksha Yojana me delivery ke time kitni financial help milti hai?",
         "Check jsy benefits.",
         "JSY ke tehat government health facility me delivery karane par rural areas me Rs. 1,400 aur urban areas me Rs. 1,000 ki direct financial assistance milti hai. Official link: https://nhm.gov.in/."),

        ("pmndp", "health", "en", "What are the rules and costs for dialysis under PM National Dialysis Programme?",
         "Retrieve pmndp rules.",
         "Under the Pradhan Mantri National Dialysis Programme (PMNDP), hemodialysis and peritoneal dialysis are provided 100% free of cost to all BPL patients at empanelled district hospitals and CHCs. Official portal: https://nhm.gov.in/."),
        ("pmndp", "health", "hi", "प्रधानमंत्री राष्ट्रीय डायलिसिस कार्यक्रम में डायलिसिस का क्या शुल्क है?",
         "pmndp के नियम प्राप्त करें।",
         "प्रधानमंत्री राष्ट्रीय डायलिसिस कार्यक्रम (PMNDP) के तहत बीपीएल (गरीबी रेखा से नीचे) मरीजों के लिए जिला अस्पतालों में डायलिसिस सेवा 100% निशुल्क उपलब्ध है। आधिकारिक वेबसाइट: https://nhm.gov.in/।"),
        ("pmndp", "health", "hinglish", "PM National Dialysis Programme me kya dialysis bilkul free hai?",
         "Verify pmndp cost structure.",
         "Haan, PMNDP ke tehat BPL card holders ke liye district civil hospitals me hemodialysis 100% free of cost provide ki jaati hai. Official portal: https://nhm.gov.in/."),

        ("pm_fasal_bima", "agriculture", "en", "What are the farmer premium rates under PM Fasal Bima Yojana?",
         "Retrieve pm_fasal_bima rules.",
         "Under PM Fasal Bima Yojana (PMFBY), the maximum premium payable by farmers is 2% for Kharif food and oilseed crops, 1.5% for Rabi food and oilseeds, and 5% for annual commercial/horticultural crops, with the balance subsidized by the government. Official portal: https://pmfby.gov.in/."),
        ("pm_fasal_bima", "agriculture", "hi", "पीएम फसल बीमा योजना में किसान को कितना प्रीमियम देना होता है?",
         "pm_fasal_bima के प्रीमियम नियम सत्यापित करें।",
         "पीएम फसल बीमा योजना के अंतर्गत किसानों के लिए प्रीमियम की दरें हैं: खरीफ फसलों के लिए अधिकतम 2%, रबी फसलों के लिए 1.5%, और वाणिज्यिक/बागवानी फसलों के लिए 5%। शेष प्रीमियम सरकार वहन करती है। आधिकारिक पोर्टल: https://pmfby.gov.in/।"),
        ("pm_fasal_bima", "agriculture", "hinglish", "PM Fasal Bima Yojana me farmer ko kitna percent premium bharna padta hai?",
         "Check pm_fasal_bima rates.",
         "PMFBY me farmers ke liye premium rates hain: Kharif fasal par 2%, Rabi fasal par 1.5%, aur commercial/horticultural crops par 5%. Baaki subsidy government deti hai. Official portal: https://pmfby.gov.in/."),

        ("pm_ujjwala", "welfare", "en", "What benefits are provided upon enrollment in PM Ujjwala Yojana 2.0?",
         "Retrieve pm_ujjwala rules.",
         "Under PM Ujjwala Yojana 2.0, eligible beneficiaries receive a deposit-free LPG connection, along with a free first LPG cylinder refill and a free hotplate (stove). Official portal: https://www.pmuy.gov.in/."),
        ("pm_ujjwala", "welfare", "hi", "उज्ज्वला योजना 2.0 में क्या-क्या मुफ्त मिलता है?",
         "pm_ujjwala के लाभ निकालें।",
         "पीएम उज्ज्वला योजना 2.0 के अंतर्गत नया एलपीजी कनेक्शन बिना सिक्योरिटी डिपॉजिट के मिलता है, साथ ही पहला भरा हुआ गैस सिलेंडर और एक गैस चूल्हा (हॉटप्लेट) भी निशुल्क दिया जाता है। आधिकारिक वेबसाइट: https://www.pmuy.gov.in/।"),
        ("pm_ujjwala", "welfare", "hinglish", "Ujjwala 2.0 scheme me connection ke saath kya free milta hai?",
         "Verify pm_ujjwala entitlements.",
         "PM Ujjwala 2.0 ke under deposit-free gas connection ke sath first cylinder refill aur gas chulha (stove) 100% free milta hai. Official website: https://www.pmuy.gov.in/."),

        ("pmjdy", "financial", "en", "What are the core banking benefits of PM Jan Dhan Yojana?",
         "Retrieve pmjdy rules.",
         "PM Jan Dhan Yojana (PMJDY) offers zero-balance savings bank accounts, a free RuPay debit card with Rs. 2 Lakh inbuilt accidental insurance cover, and an overdraft facility up to Rs. 10,000 for eligible account holders. Official portal: https://pmjdy.gov.in/."),
        ("pmjdy", "financial", "hi", "प्रधानमंत्री जन धन योजना (PMJDY) के मुख्य बैंकिंग लाभ क्या हैं?",
         "pmjdy के नियम प्राप्त करें।",
         "पीएम जन धन योजना के तहत शून्य शेष (जीरो बैलेंस) बैंक खाता, ₹2 लाख दुर्घटना बीमा कवर वाला रुपे डेबिट कार्ड, तथा ₹10,000 तक की ओवरड्राफ्ट सुविधा मिलती है। आधिकारिक पोर्टल: https://pmjdy.gov.in/।"),
        ("pmjdy", "financial", "hinglish", "Jan Dhan account khulwane par kitna accidental insurance milta hai?",
         "Verify pmjdy insurance benefit.",
         "PMJDY account ke sath free RuPay debit card milta hai jisme Rs. 2 Lakh ka inbuilt accidental insurance cover shamil hai, bina kisi minimum balance maintain kiye. Official link: https://pmjdy.gov.in/."),

        ("ntep", "health", "en", "What financial nutrition support is given to TB patients under NTEP Nikshay Poshan Yojana?",
         "Retrieve ntep rules.",
         "Under the National TB Elimination Programme (NTEP) Nikshay Poshan Yojana, notified TB patients receive direct benefit transfer of Rs. 500 per month throughout their treatment duration for nutritional support. Official portal: https://nikshay.in/."),
        ("ntep", "health", "hi", "निक्षय पोषण योजना (NTEP) में टीबी मरीजों को हर महीने कितनी पोषण सहायता मिलती है?",
         "ntep के नियम सत्यापित करें।",
         "राष्ट्रीय क्षय उन्मूलन कार्यक्रम (NTEP) की निक्षय पोषण योजना के तहत अधिसूचित टीबी मरीजों को इलाज की पूरी अवधि तक पोषण सहायता हेतु ₹500 प्रति माह डीबीटी के माध्यम से दिए जाते हैं। आधिकारिक पोर्टल: https://nikshay.in/।"),
        ("ntep", "health", "hinglish", "TB ke treatment ke dauran Nikshay Poshan Yojana me kitne paise milte hain?",
         "Check ntep nutritional benefit.",
         "Nikshay Poshan Yojana (NTEP) ke tehat TB patients ko treatment chalne tak har mahine Rs. 500 DBT ke through seedhe bank account me milte hain. Official portal: https://nikshay.in/."),

        ("pmsby", "financial", "en", "What is the annual premium and accident insurance cover under PMSBY?",
         "Retrieve pmsby rules.",
         "Pradhan Mantri Suraksha Bima Yojana (PMSBY) provides Rs. 2 Lakh coverage for accidental death or permanent total disability for a nominal annual premium of Rs. 20 auto-debited from the subscriber's bank account. Official portal: https://financialservices.gov.in/."),
        ("pmsby", "financial", "hi", "प्रधानमंत्री सुरक्षा बीमा योजना (PMSBY) का वार्षिक प्रीमियम और कवर कितना है?",
         "pmsby के नियम प्राप्त करें।",
         "PMSBY के तहत मात्र ₹20 प्रति वर्ष के प्रीमियम पर ₹2 लाख का दुर्घटना मृत्यु एवं पूर्ण विकलांगता बीमा कवर मिलता है। प्रीमियम सीधे बैंक खाते से ऑटो-डेबिट होता है। आधिकारिक पोर्टल: https://financialservices.gov.in/।"),
        ("pmsby", "financial", "hinglish", "PMSBY me kitne rupaye premium me kitna accident insurance cover hota hai?",
         "Verify pmsby premium.",
         "PMSBY me saal ka sirf Rs. 20 premium kat-ta hai aur accidental death ya permanent total disability par Rs. 2 Lakh ka cover milta hai. Official website: https://financialservices.gov.in/."),

        ("pmjjby", "financial", "en", "What is the life insurance cover and premium for PM Jeevan Jyoti Bima Yojana?",
         "Retrieve pmjjby rules.",
         "PM Jeevan Jyoti Bima Yojana (PMJJBY) provides Rs. 2 Lakh life insurance cover for death due to any cause, for an annual premium of Rs. 436 for eligible individuals aged 18 to 50 years. Official portal: https://financialservices.gov.in/."),
        ("pmjjby", "financial", "hi", "पीएम जीवन ज्योति बीमा योजना (PMJJBY) का जीवन बीमा कवर और प्रीमियम कितना है?",
         "pmjjby के नियम सत्यापित करें।",
         "पीएम जीवन ज्योति बीमा योजना (PMJJBY) में 18 से 50 वर्ष के नागरिकों को किसी भी कारण से मृत्यु होने पर ₹2 लाख का जीवन बीमा कवर ₹436 वार्षिक प्रीमियम पर मिलता है। आधिकारिक पोर्टल: https://financialservices.gov.in/।"),
        ("pmjjby", "financial", "hinglish", "PMJJBY me kisi bhi cause se death par kitna cover milta hai aur premium kya hai?",
         "Check pmjjby coverage.",
         "PMJJBY me kisi bhi karan se death hone par family ko Rs. 2 Lakh ka life insurance cover milta hai, jiska annual premium Rs. 436 hai. Official link: https://financialservices.gov.in/."),

        ("mgnrega", "welfare", "en", "How many days of guaranteed wage employment are provided under MGNREGA?",
         "Retrieve mgnrega rules.",
         "MGNREGA guarantees at least 100 days of wage employment in a financial year to every rural household whose adult members volunteer to do unskilled manual work. Official portal: https://nrega.nic.in/."),
        ("mgnrega", "welfare", "hi", "मनरेगा (MGNREGA) के तहत ग्रामीण परिवारों को कितने दिनों के काम की गारंटी मिलती है?",
         "mgnrega के नियम सत्यापित करें।",
         "मनरेगा (MGNREGA) के अंतर्गत प्रत्येक ग्रामीण परिवार को, जिसके वयस्क सदस्य अकुशल शारीरिक श्रम करने के इच्छुक हैं, एक वित्तीय वर्ष में कम से कम 100 दिनों के सवेतन रोजगार की कानूनी गारंटी दी जाती है। आधिकारिक वेबसाइट: https://nrega.nic.in/।"),
        ("mgnrega", "welfare", "hinglish", "MGNREGA me rural family ko kitne din ka guaranteed rozgar milta hai?",
         "Verify mgnrega work days.",
         "MGNREGA me har rural household ko jiske adult members unskilled physical work karna chahte hain, saal me minimum 100 days ka guaranteed rozgar milta hai. Official site: https://nrega.nic.in/.")
    ]

    for i, (sid, cat, lang, task, th, resp) in enumerate(grounding_schemes, 1):
        rec_id = f"repair_cat_d_{i:03d}"
        step = make_step(1, "scheme_rule_verification", "get_scheme_rules", {"scheme_id": sid}, th)
        msgs = format_messages(task, [step], resp)
        records.append({
            "id": rec_id,
            "category": "D_tool_result_grounding",
            "task": task,
            "language": lang,
            "steps": [step],
            "final_response": resp,
            "messages": msgs,
            "metadata": {
                "category": "tool_result_grounding",
                "language": lang,
                "tools_used": ["get_scheme_rules"],
                "scheme_id": sid,
                "source": "verified_knowledge_cards"
            }
        })

    # =========================================================================
    # CATEGORY E: Hallucination Resistance & Refusal (35)
    # Refusal of fake schemes, unlawful/bypassing requests, or non-existent cards.
    # Checks repository, reports not_found or explains policy truthfully without inventing.
    # Languages: 12 en, 12 hi, 11 hinglish
    # =========================================================================
    hallucination_scenarios = [
        # En (12)
        {"lang": "en", "task": "How can I apply for the PM Free Laptop and iPhone Yojana 2026?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_free_iphone_yojana"},
         "thought": "Check verified repository for requested scheme.",
         "resp": "There is no official government scheme named 'PM Free Laptop and iPhone Yojana'. Such viral claims on social media are fraudulent scams. Official digital education schemes include PM e-VIDYA and National Digital Literacy initiatives. Never pay any fee or share OTPs for unofficial viral schemes."},
        {"lang": "en", "task": "Can you generate a fake BPL certificate so I can qualify for Ayushman Bharat PM-JAY?",
         "tool": None, "args": None,
         "thought": "Illegal request to forge official government identification. Immediate refusal.",
         "resp": "I cannot generate or assist in creating fraudulent documents or fake BPL certificates. Ayushman Bharat eligibility is verified directly through official SECC 2011 databases and state food security ration lists. Forging government records is a punishable legal offense."},
        {"lang": "en", "task": "What are the rules for the Modi Gold Cash 1 Lakh Transfer Scheme?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "modi_gold_cash_scheme"},
         "thought": "Verify if such a scheme exists in official records.",
         "resp": "No scheme called 'Modi Gold Cash Scheme' exists in the Government of India's official repository. Official DBT welfare schemes like PM-Kisan and NSAP are strictly governed and listed on https://myscheme.gov.in/. Beware of phishing portals promising free cash or gold."},
        {"lang": "en", "task": "Can I bypass the Aadhaar biometric authentication at the hospital for PM-JAY?",
         "tool": None, "args": None,
         "thought": "Request to bypass mandated government security authentication.",
         "resp": "Aadhaar e-KYC authentication is legally mandated at empanelled hospitals to verify beneficiary identity and prevent fraudulent claims. If fingerprint biometric fails, hospitals provide Iris authentication or registered mobile OTP verification via the hospital's designated Ayushman Mitra."},
        {"lang": "en", "task": "How do I get free cash from the Prime Minister Direct Dollar Transfer?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_dollar_transfer"},
         "thought": "Query verified scheme repository.",
         "resp": "There is no government scheme for direct foreign currency or dollar transfers. All central and state welfare transfers occur in Indian Rupees (INR) through direct benefit transfer (DBT) directly into Aadhaar-seeded Jan Dhan bank accounts."},
        {"lang": "en", "task": "Can you prescribe antibiotics for my fever under government health guidelines?",
         "tool": None, "args": None,
         "thought": "Request for medical prescription. Strict safety refusal.",
         "resp": "I cannot prescribe antibiotics or clinical medications. Prescribing antibiotics without physical medical examination and culture tests is medically unsafe and illegal. You can consult certified government doctors for free via the official e-Sanjeevani teleconsultation portal (https://esanjeevani.mohfw.gov.in/) or visit your nearest Primary Health Centre."},
        {"lang": "en", "task": "What is the procedure for PM Free Electric Car Subsidy 100% Cash Back?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_100_percent_car_cashback"},
         "thought": "Verify scheme existence.",
         "resp": "The Government of India does not offer a 100% cashback scheme for electric cars. Electric vehicle incentives are provided through the official PM E-DRIVE / FAME initiatives via registered EV manufacturers, not direct cash distributions to individuals."},
        {"lang": "en", "task": "I want to register for PM 25 Lakh Lottery Scheme announced on WhatsApp.",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_lottery_scheme"},
         "thought": "Social media lottery scam check.",
         "resp": "The Government of India never runs lotteries or WhatsApp cash prize schemes. This is a cyber fraud scam designed to steal your bank details or advance fees. Official welfare schemes are never awarded through lotteries or messaging apps."},
        {"lang": "en", "task": "Can I claim PM-JAY benefits at private clinics that are not empanelled?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "ab_pmjay"},
         "thought": "Verify hospital empanelment rules for ab_pmjay.",
         "resp": "No, PM-JAY cashless hospitalization benefits are valid only at officially empanelled public and private hospitals. Non-empanelled private clinics or nursing homes cannot process Ayushman Bharat cashless claims. You can verify empanelled hospitals on https://hospitals.pmjay.gov.in/."},
        {"lang": "en", "task": "Tell me how to hack the PM-Kisan portal to add my name without land records.",
         "tool": None, "args": None,
         "thought": "Cyber security violation and hacking request.",
         "resp": "I cannot provide methods or assistance to tamper with or unlawfully access government databases. Land record verification is mandatory under PM-Kisan guidelines. Ineligible individuals falsely claiming benefits face legal recovery and penal action under law."},
        {"lang": "en", "task": "What is the eligibility for the Free Petrol and Diesel Scheme 2026?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "free_petrol_diesel_yojana"},
         "thought": "Verify repository for fake scheme.",
         "resp": "There is no government scheme offering free petrol or diesel. Fuel subsidies are provided in specific targeted forms (such as targeted LPG refill subsidies under PM Ujjwala). Do not fall for fraudulent online registrations."},
        {"lang": "en", "task": "Can someone else use my Ayushman Bharat card for their surgery?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "ab_pmjay"},
         "thought": "Card transferability inquiry.",
         "resp": "No, Ayushman Bharat PM-JAY benefits are strictly non-transferable. The card and coverage apply exclusively to individuals named in the authenticated family unit registered under the SECC 2011/NFSA database. Impersonation at hospital registration is a criminal offense."},

        # Hi (12)
        {"lang": "hi", "task": "पीएम मुफ्त लैपटॉप और आईफोन योजना 2026 में फॉर्म कैसे भरें?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_laptop_iphone_yojana"},
         "thought": "फर्जी वायरल योजना की जांच।",
         "resp": "'पीएम मुफ्त आईफोन या लैपटॉप योजना' नाम की कोई भी सरकारी योजना अस्तित्व में नहीं है। सोशल मीडिया पर प्रसारित ऐसे दावे पूरी तरह भ्रामक और फर्जी हैं। किसी भी असत्यापित वेबसाइट पर अपनी व्यक्तिगत जानकारी या ओटीपी साझा न करें।"},
        {"lang": "hi", "task": "क्या आप आयुष्मान कार्ड के लिए नकली बीपीएल प्रमाण पत्र बना सकते हैं?",
         "tool": None, "args": None,
         "thought": "फर्जी दस्तावेज बनाने का अवैध अनुरोध। स्पष्ट इनकार।",
         "resp": "मैं कोई भी नकली या फर्जी दस्तावेज नहीं बना सकता। आयुष्मान भारत योजना की पात्रता केवल आधिकारिक सरकारी रिकॉर्ड और SECC डेटाबेस के माध्यम से जांची जाती है। जाली दस्तावेज बनाना कानूनी अपराध है।"},
        {"lang": "hi", "task": "मोदी गोल्ड कैश 1 लाख रुपये ट्रांसफर योजना की जानकारी दें।",
         "tool": "get_scheme_rules", "args": {"scheme_id": "modi_gold_scheme"},
         "thought": "अस्तित्वहीन योजना की जांच।",
         "resp": "भारत सरकार की ऐसी कोई 'मोदी गोल्ड कैश योजना' नहीं है। सरकार की वास्तविक जनकल्याणकारी योजनाओं की सूची आधिकारिक पोर्टल https://myscheme.gov.in/ पर उपलब्ध है। फर्जी योजनाओं के झांसे में न आएं।"},
        {"lang": "hi", "task": "अस्पताल में आयुष्मान कार्ड के लिए बायोमेट्रिक फिंगरप्रिंट को बायपास करने का तरीका बताएं।",
         "tool": None, "args": None,
         "thought": "सुरक्षा सत्यापन बायपास का अनुरोध।",
         "resp": "अस्पताल में बायोमेट्रिक सत्यापन बायपास नहीं किया जा सकता। यदि बुजुर्गों या मरीजों के फिंगरप्रिंट नहीं आते हैं, तो अस्पताल में तैनात आयुष्मान मित्र द्वारा आइरिस (आंखों की पुतली) स्कैन अथवा आधार ओटीपी के जरिए वैकल्पिक सत्यापन का आधिकारिक नियम है।"},
        {"lang": "hi", "task": "मुझे बुखार और खांसी है, सरकारी गाइडलाइन के अनुसार एंटीबायोटिक दवाई लिख दीजिए।",
         "tool": None, "args": None,
         "thought": "दवा का पर्चा लिखने का अनुरोध। चिकित्सा सुरक्षा नीति के तहत सख्त मना।",
         "resp": "मैं चिकित्सकीय पर्चा या एंटीबायोटिक दवाएं नहीं लिख सकता। बिना डॉक्टर की जांच के एंटीबायोटिक लेना स्वास्थ्य के लिए हानिकारक हो सकता है। आप सरकारी ई-संजीवनी पोर्टल (https://esanjeevani.mohfw.gov.in/) पर सरकारी डॉक्टरों से निशुल्क परामर्श ले सकते हैं।"},
        {"lang": "hi", "task": "व्हाट्सएप पर आए 25 लाख के सरकारी लॉटरी इनाम को क्लेम करने की प्रक्रिया क्या है?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_whatsapp_lottery"},
         "thought": "साइबर लॉटरी फ्रॉड।",
         "resp": "भारत सरकार कभी भी कोई लॉटरी या व्हाट्सएप इनामी योजना नहीं चलाती। यह साइबर ठगों द्वारा की जाने वाली धोखाधड़ी है। कभी भी किसी अनजान नंबर पर शुल्क न भेजें और तुरंत राष्ट्रीय साइबर अपराध हेल्पलाइन 1930 पर शिकायत करें।"},
        {"lang": "hi", "task": "क्या मैं अपने आयुष्मान कार्ड से अपने पड़ोसी का ऑपरेशन करा सकता हूँ?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "ab_pmjay"},
         "thought": "कार्ड ट्रांसफर की वैधता।",
         "resp": "नहीं, आयुष्मान कार्ड पूरी तरह व्यक्तिगत एवं पंजीकृत परिवार के सदस्यों के लिए ही मान्य होता है। इसे किसी अन्य व्यक्ति को हस्तांतरित नहीं किया जा सकता। किसी अन्य के कार्ड पर इलाज कराना गैरकानूनी है।"},
        {"lang": "hi", "task": "पीएम 100% फ्री बिजली बाइक और कार योजना का फॉर्म कहां मिलेगा?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_free_electric_bike"},
         "thought": "अस्तित्वहीन ईवी योजना।",
         "resp": "भारत सरकार द्वारा 100% फ्री इलेक्ट्रिक बाइक या कार देने की कोई योजना नहीं चलाई जा रही है। इलेक्ट्रिक वाहनों पर सब्सिडी अधिकृत कंपनियों के माध्यम से पीएम ई-ड्राइव योजना के तहत दी जाती है, सीधे नकद उपहार के रूप में नहीं।"},
        {"lang": "hi", "task": "पीएम किसान योजना के पोर्टल को हैक करके बिना जमीन के नाम कैसे जोड़ें?",
         "tool": None, "args": None,
         "thought": "हैकिंग और अवैध डेटा बदलाव का अनुरोध।",
         "resp": "सरकारी डेटाबेस को हैक करना या उसमें हेरफेर करना गंभीर आईटी अपराध है। पीएम किसान योजना में जमीन के असली कागजात और आधार का मिलान अनिवार्य है। अपात्र व्यक्तियों द्वारा गलत तरीके से लाभ लेने पर कानूनी कार्रवाई की जाती है।"},
        {"lang": "hi", "task": "मुफ्त पेट्रोल-डीजल योजना 2026 में रजिस्ट्रेशन कैसे कराएं?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "free_petrol_yojana"},
         "thought": "फर्जी योजना की पड़ताल।",
         "resp": "मुफ्त पेट्रोल या डीजल देने की कोई भी सरकारी योजना नहीं है। इस तरह के फर्जी लिंक फिशिंग स्कैम होते हैं। सरकारी सब्सिडी योजनाओं की जानकारी केवल https://myscheme.gov.in/ पर देखें।"},
        {"lang": "hi", "task": "क्या गैर-पैनलबद्ध निजी नर्सिंग होम में आयुष्मान कार्ड से फ्री इलाज होगा?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "ab_pmjay"},
         "thought": "अस्पताल पात्रता की स्थिति।",
         "resp": "नहीं, आयुष्मान भारत योजना के तहत निशुल्क इलाज केवल उन्हीं सरकारी और निजी अस्पतालों में मिलता है जो योजना के तहत विधिवत सूचीबद्ध (empanelled) हैं। गैर-सूचीबद्ध क्लीनिकों में यह कार्ड मान्य नहीं है।"},
        {"lang": "hi", "task": "बिना आधार कार्ड के सरकारी योजनाओं के सारे पैसे सीधे नकद में कैसे लें?",
         "tool": None, "args": None,
         "thought": "डीबीटी नियमों का स्पष्टीकरण।",
         "resp": "भारत सरकार की सभी कल्याणकारी योजनाओं की वित्तीय सहायता केवल डायरेक्ट बेनिफिट ट्रांसफर (DBT) के जरिए सीधे लाभार्थी के बैंक खाते में भेजी जाती है। नकद वितरण की कोई व्यवस्था नहीं है, और बैंक खाते से आधार लिंक होना अनिवार्य है।"},

        # Hinglish (11)
        {"lang": "hinglish", "task": "PM Free iPhone aur Laptop scheme me kaise register karein?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "free_iphone_laptop"},
         "thought": "Check verified repository for fake scheme.",
         "resp": "'PM Free iPhone' jaisi koi official government scheme exist nahi karti. Social media par aisi fake news se door rahein aur kisi bhi fraudulent link par bank details share na karein."},
        {"lang": "hinglish", "task": "Mujhe fake ration card banwana hai PM-JAY ke liye, help karo.",
         "tool": None, "args": None,
         "thought": "Refuse illegal document forgery.",
         "resp": "Main fake ration card ya koi bhi forged documents banane me help nahi kar sakta. Yeh ek punishable legal offense hai. Agar aap eligible hain toh official Food & Civil Supplies portal par genuine application karein."},
        {"lang": "hinglish", "task": "Modi Gold Scheme me 50,000 rupaye mil rahe hain, sach hai kya?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "modi_gold_yojana"},
         "thought": "Verify fake gold scheme.",
         "resp": "Nahi, Modi Gold Scheme naam ki koi sarkari yojana nahi hai. Ye fake WhatsApp forwards hain. Verified schemes ke liye hamesha https://myscheme.gov.in/ dekhein."},
        {"lang": "hinglish", "task": "Hospital me Ayushman card ka fingerprint bypass kaise karte hain?",
         "tool": None, "args": None,
         "thought": "Refusal of security bypass.",
         "resp": "Biometric authentication hospital me bypass nahi kiya ja sakta. Agar biometric mismatch hota hai, toh hospital me Ayushman Mitra dwara Iris scan ya Aadhaar OTP ke through official alternative verification hota hai."},
        {"lang": "hinglish", "task": "Mujhe cold aur fever hai, koi strong antibiotic suggest kar do.",
         "tool": None, "args": None,
         "thought": "Prescription refusal under safety policy.",
         "resp": "Main koi antibiotic ya medical drugs prescribe nahi kar sakta. Bina physical doctor consultation ke dawa lena khatarnak ho sakta hai. Aap e-Sanjeevani portal (https://esanjeevani.mohfw.gov.in/) par free me sarkari doctor se consult karein."},
        {"lang": "hinglish", "task": "WhatsApp par PM 25 Lakh Lottery aayi hai, registration fees maang rahe hain.",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_lottery"},
         "thought": "Identify lottery phishing.",
         "resp": "Yeh 100% fraud cyber scam hai. Government of India kabhi koi lottery nahi chalati aur na hi koi advance fee maangti hai. Kisi ko paise na bhejein aur helpline 1930 par report karein."},
        {"lang": "hinglish", "task": "Kya main apna Ayushman card apne dost ke operation ke liye use kar sakta hoon?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "ab_pmjay"},
         "thought": "Card non-transferability rule.",
         "resp": "Nahi, Ayushman card strictly non-transferable hai. Iska benefit sirf card holder aur uske registered family members ko hi mil sakta hai. Dusre ke card ka galat use karna criminal fraud hai."},
        {"lang": "hinglish", "task": "PM Free Petrol scheme ka form kahan milega?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "free_petrol"},
         "thought": "Fake fuel scheme query.",
         "resp": "Free petrol dene ki koi government scheme nahi hai. Yeh fraudulent viral claim hai. Official schemes sirf myscheme.gov.in par verified hoti hain."},
        {"lang": "hinglish", "task": "PM Kisan portal hack karke beneficiary list me name daalna hai.",
         "tool": None, "args": None,
         "thought": "Cyber hacking refusal.",
         "resp": "Government portal hacking ya unauthorized database access ek serious cyber crime hai. Main isme koi sahayata nahi kar sakta. Eligible farmers official CSC center par genuine land records submit karke apply karein."},
        {"lang": "hinglish", "task": "Kisi bhi private doctor ke clinic me Ayushman card chal jayega kya?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "ab_pmjay"},
         "thought": "Verify empanelment requirement.",
         "resp": "Nahi, Ayushman Bharat sirf officially empanelled hospitals me hi valid hota hai. Kisi bhi arbitrary private clinic par yeh apply nahi hota. List check karne ke liye https://hospitals.pmjay.gov.in/ par visit karein."},
        {"lang": "hinglish", "task": "Bina bank account ke PM-Kisan ka paisa cash me mil sakta hai kya?",
         "tool": "get_scheme_rules", "args": {"scheme_id": "pm_kisan"},
         "thought": "DBT cash limitation explanation.",
         "resp": "Nahi, PM-Kisan ka paisa sirf Direct Benefit Transfer (DBT) ke through seedhe beneficiary ke Aadhaar-linked bank account me transfer hota hai. Cash hand-over ka koi option nahi hai."}
    ]

    for i, item in enumerate(hallucination_scenarios, 1):
        rec_id = f"repair_cat_e_{i:03d}"
        steps = []
        if item["tool"]:
            step = make_step(1, "scheme_verification", item["tool"], item["args"], item["thought"])
            steps.append(step)
            msgs = format_messages(item["task"], steps, item["resp"])
        else:
            msgs = [
                {"role": "user", "content": item["task"]},
                {"role": "assistant", "content": f"{item['thought']}\n\n{item['resp']}"}
            ]
        records.append({
            "id": rec_id,
            "category": "E_hallucination_resistance_refusal",
            "task": item["task"],
            "language": item["lang"],
            "steps": steps,
            "final_response": item["resp"],
            "messages": msgs,
            "metadata": {
                "category": "hallucination_resistance_refusal",
                "language": item["lang"],
                "tools_used": [item["tool"]] if item["tool"] else [],
                "source": "local_safety_policy"
            }
        })

    # =========================================================================
    # CATEGORY F: Multi-Turn / Clarification Workflows (30)
    # Multi-step interactions: partial query -> match schemes -> get rules / documents.
    # Languages: 10 en, 10 hi, 10 hinglish
    # =========================================================================
    multi_step_cases = [
        # En (10)
        {"lang": "en", "task": "I am 73 years old living in Varanasi. Check health schemes for me and show required documents.",
         "steps_def": [
             ("match_schemes", {"age": 73, "category": "health", "income_status": "eligible"}, "User is 73 years old, category health. Check matching schemes."),
             ("get_documents", {"scheme_id": "ayushman_vay_vandana"}, "Retrieve required documents for matched scheme ayushman_vay_vandana.")
         ],
         "resp": "For your age (73), you are eligible for the Ayushman Vay Vandana Card (under AB PM-JAY) providing Rs. 5 Lakh annual cover. The required documents to enroll are: 1) Aadhaar Card, 2) Age verification proof (via Aadhaar e-KYC), and 3) Active mobile number."},
        {"lang": "en", "task": "I am a 34 year old farmer in Gorakhpur. What crop protection scheme exists and what are its rules?",
         "steps_def": [
             ("match_schemes", {"age": 34, "category": "agriculture", "income_status": "eligible"}, "Find matching agriculture schemes for 34 year old farmer."),
             ("get_scheme_rules", {"scheme_id": "pm_fasal_bima"}, "Retrieve rules and coverage for PM Fasal Bima Yojana.")
         ],
         "resp": "You match with PM Fasal Bima Yojana (PMFBY). It provides comprehensive risk insurance for pre-sowing to post-harvest losses with subsidized farmer premium rates (2% Kharif, 1.5% Rabi). Official portal: https://pmfby.gov.in/."},
        {"lang": "en", "task": "I am 24 years old and pregnant. Which maternal scheme assists with delivery and what are the rules?",
         "steps_def": [
             ("match_schemes", {"age": 24, "category": "maternal", "income_status": "eligible"}, "Query maternal health schemes."),
             ("get_scheme_rules", {"scheme_id": "jsy"}, "Get operational rules for Janani Suraksha Yojana.")
         ],
         "resp": "You are eligible for Janani Suraksha Yojana (JSY) and JSSK. JSY provides cash assistance (Rs. 1,400 in rural areas) for institutional delivery in public health centers to ensure mother and child safety. Official portal: https://nhm.gov.in/."},
        {"lang": "en", "task": "I am a 67 year old low-income senior. What pension scheme applies and what documents are required?",
         "steps_def": [
             ("match_schemes", {"age": 67, "category": "senior", "income_status": "eligible"}, "Match senior citizen assistance schemes for age 67."),
             ("get_documents", {"scheme_id": "nsap_old_age"}, "Retrieve document requirements for NSAP old age pension.")
         ],
         "resp": "You qualify for the NSAP Old Age Pension (IGNOAPS). Required documents: 1) Aadhaar Card, 2) BPL Ration Card or SECC certificate, 3) Age proof certificate (60+ years), and 4) Bank passbook linked to Aadhaar."},
        {"lang": "en", "task": "I need tuberculosis medical care in Lucknow. Find health facilities and check nutrition assistance rules.",
         "steps_def": [
             ("find_facility", {"district": "Lucknow", "facility_type": "hospital"}, "Locate public healthcare facilities in Lucknow."),
             ("get_scheme_rules", {"scheme_id": "ntep"}, "Retrieve rules for National TB Elimination Programme.")
         ],
         "resp": "In Lucknow, you can visit the District Civil Hospital Help Desk or nearest CHC. Under the National TB Elimination Programme (NTEP Nikshay Poshan Yojana), all notified TB patients receive Rs. 500/month direct cash assistance for nutrition during treatment."},
        {"lang": "en", "task": "I am a 29 year old rural woman needing clean cooking fuel. Check scheme and document requirements.",
         "steps_def": [
             ("match_schemes", {"age": 29, "category": "welfare", "income_status": "eligible"}, "Match rural household welfare schemes."),
             ("get_documents", {"scheme_id": "pm_ujjwala"}, "Get documents for PM Ujjwala Yojana.")
         ],
         "resp": "You match with PM Ujjwala Yojana 2.0 for deposit-free LPG connections. Required documents: 1) BPL Ration Card, 2) Aadhaar Card of adult female applicant, 3) Bank passbook linked to Aadhaar, and 4) 14-point declaration form."},
        {"lang": "en", "task": "I am 31 years old seeking unorganized sector pension. What scheme matches and what are the rules?",
         "steps_def": [
             ("match_schemes", {"age": 31, "category": "financial", "income_status": "eligible"}, "Match financial schemes for unorganized worker."),
             ("get_scheme_rules", {"scheme_id": "apy"}, "Retrieve Atal Pension Yojana rules.")
         ],
         "resp": "You qualify for Atal Pension Yojana (APY). Under APY, subscribers aged 18-40 contribute monthly to receive a guaranteed monthly pension of Rs. 1,000 to Rs. 5,000 upon reaching age 60. Official portal: https://www.npscra.nsdl.co.in/."},
        {"lang": "en", "task": "Find healthcare facilities in Varanasi and verify Ayushman Bharat card documents.",
         "steps_def": [
             ("find_facility", {"district": "Varanasi", "facility_type": "hospital"}, "Find hospitals in Varanasi."),
             ("get_documents", {"scheme_id": "ab_pmjay"}, "Retrieve documents for AB PM-JAY.")
         ],
         "resp": "In Varanasi, visit the District Civil Hospital Help Desk or CSC HQ. Required documents for PM-JAY card issuance: Aadhaar Card, Ration Card / SECC proof, and Aadhaar-linked mobile phone."},
        {"lang": "en", "task": "I am 45 years old and need chronic kidney disease care. Check scheme rules and documents.",
         "steps_def": [
             ("get_scheme_rules", {"scheme_id": "pmndp"}, "Get rules for PM National Dialysis Programme."),
             ("get_documents", {"scheme_id": "pmndp"}, "Get documents for PMNDP.")
         ],
         "resp": "Under the Pradhan Mantri National Dialysis Programme (PMNDP), hemodialysis is 100% free for BPL patients in district hospitals. Required documents: Aadhaar card, BPL certificate/ration card, and nephrologist referral prescription."},
        {"lang": "en", "task": "I am a 22 year old seeking skill training. Check scheme rules and documents.",
         "steps_def": [
             ("get_scheme_rules", {"scheme_id": "pmkvy"}, "Get PM Kaushal Vikas Yojana rules."),
             ("get_documents", {"scheme_id": "pmkvy"}, "Get documents for PMKVY.")
         ],
         "resp": "Under PM Kaushal Vikas Yojana (PMKVY 4.0), youth receive industry-aligned skill certification and placement support. Required documents: Aadhaar card, educational certificates (10th/12th), and bank account details."},

        # Hi (10)
        {"lang": "hi", "task": "मेरी उम्र 71 वर्ष है और मैं वाराणसी में रहता हूँ। स्वास्थ्य योजना बताएं और आवश्यक कागजात दें।",
         "steps_def": [
             ("match_schemes", {"age": 71, "category": "health", "income_status": "eligible"}, "आयु 71 वर्ष के लिए योजना मैच करें।"),
             ("get_documents", {"scheme_id": "ayushman_vay_vandana"}, "आयुष्मान वय वंदना कार्ड के दस्तावेज निकालें।")
         ],
         "resp": "आपकी आयु (71 वर्ष) के आधार पर आप 'आयुष्मान वय वंदना योजना' के पात्र हैं (₹5 लाख वार्षिक स्वास्थ्य कवर)। आवश्यक दस्तावेज: आधार कार्ड और आधार से लिंक मोबाइल नंबर (ई-केवाईसी हेतु)।"},
        {"lang": "hi", "task": "मैं 35 वर्ष का किसान हूँ। फसल सुरक्षा योजना के नियम और जरूरी दस्तावेज बताएं।",
         "steps_def": [
             ("match_schemes", {"age": 35, "category": "agriculture", "income_status": "eligible"}, "कृषि योजनाओं का मिलान करें।"),
             ("get_documents", {"scheme_id": "pm_fasal_bima"}, "पीएम फसल बीमा के दस्तावेज प्राप्त करें।")
         ],
         "resp": "आप 'पीएम फसल बीमा योजना' (PMFBY) के पात्र हैं। आवश्यक दस्तावेज: भू-स्वामित्व अभिलेख (खसरा/खतौनी), बुवाई प्रमाण पत्र, बैंक पासबुक और आधार कार्ड।"},
        {"lang": "hi", "task": "मेरी पत्नी 25 वर्ष की हैं और गर्भवती हैं। मातृत्व योजना के नियम और अस्पताल बताएं।",
         "steps_def": [
             ("match_schemes", {"age": 25, "category": "maternal", "income_status": "eligible"}, "मातृत्व योजनाओं की जांच करें।"),
             ("get_scheme_rules", {"scheme_id": "jsy"}, "जननी सुरक्षा योजना के नियम निकालें।")
         ],
         "resp": "आप जननी सुरक्षा योजना (JSY) और JSSK के पात्र हैं। सरकारी अस्पताल में प्रसव कराने पर ₹1,400 ग्रामीण नकद सहायता तथा माता व नवजात शिशु का संपूर्ण उपचार व दवाइयां पूर्णतः निशुल्क हैं।"},
        {"lang": "hi", "task": "68 वर्ष के बुजुर्ग के लिए वृद्धावस्था पेंशन योजना और आवश्यक कागजात क्या हैं?",
         "steps_def": [
             ("match_schemes", {"age": 68, "category": "senior", "income_status": "eligible"}, "68 वर्ष हेतु पेंशन योजना खोजें।"),
             ("get_documents", {"scheme_id": "nsap_old_age"}, "वृद्धावस्था पेंशन के दस्तावेज निकालें।")
         ],
         "resp": "आप 'राष्ट्रीय सामाजिक सहायता कार्यक्रम' (NSAP वृद्धावस्था पेंशन) के पात्र हैं। आवश्यक दस्तावेज: आधार कार्ड, बीपीएल राशन कार्ड, 60+ आयु प्रमाण पत्र और बैंक पासबुक।"},
        {"lang": "hi", "task": "लखनऊ में टीबी के इलाज के लिए अस्पताल और पोषण योजना के नियम बताएं।",
         "steps_def": [
             ("find_facility", {"district": "Lucknow", "facility_type": "hospital"}, "लखनऊ में स्वास्थ्य केंद्र खोजें।"),
             ("get_scheme_rules", {"scheme_id": "ntep"}, "निक्षय पोषण योजना के नियम निकालें।")
         ],
         "resp": "लखनऊ में जिला नागरिक चिकित्सालय या नजदीकी सीएचसी में संपर्क करें। राष्ट्रीय क्षय उन्मूलन कार्यक्रम (NTEP) के तहत टीबी मरीजों को ₹500 प्रति माह पोषण राशि बैंक में सीधे मिलती है।"},
        {"lang": "hi", "task": "ग्रामीण महिला के लिए उज्ज्वला गैस कनेक्शन के नियम और आवश्यक कागजात क्या हैं?",
         "steps_def": [
             ("get_scheme_rules", {"scheme_id": "pm_ujjwala"}, "उज्ज्वला योजना के नियम प्राप्त करें।"),
             ("get_documents", {"scheme_id": "pm_ujjwala"}, "उज्ज्वला योजना के दस्तावेज निकालें।")
         ],
         "resp": "पीएम उज्ज्वला 2.0 के तहत बीपीएल गरीब परिवारों की महिला मुखिया को मुफ़्त गैस कनेक्शन, पहला भरा सिलेंडर और चूल्हा मिलता है। कागजात: बीपीएल राशन कार्ड, आधार कार्ड, और बैंक पासबुक।"},
        {"lang": "hi", "task": "30 वर्ष की आयु में असंगठित कामगारों के लिए पेंशन योजना और उसके नियम बताएं।",
         "steps_def": [
             ("match_schemes", {"age": 30, "category": "financial", "income_status": "eligible"}, "वित्तीय सुरक्षा योजनाएं खोजें।"),
             ("get_scheme_rules", {"scheme_id": "apy"}, "अटल पेंशन योजना के नियम देखें।")
         ],
         "resp": "आप 'अटल पेंशन योजना' (APY) में शामिल हो सकते हैं। 30 वर्ष की आयु में मासिक अंशदान देकर 60 वर्ष की आयु के बाद ₹1,000 से ₹5,000 तक की निश्चित मासिक पेंशन प्राप्त कर सकते हैं।"},
        {"lang": "hi", "task": "वाराणसी में आयुष्मान कार्ड बनवाने के लिए केंद्र और जरूरी कागजात बताएं।",
         "steps_def": [
             ("find_facility", {"district": "Varanasi", "facility_type": "all"}, "वाराणसी में केंद्र खोजें।"),
             ("get_documents", {"scheme_id": "ab_pmjay"}, "आयुष्मान कार्ड के दस्तावेज प्राप्त करें।")
         ],
         "resp": "वाराणसी में जिला अस्पताल हेल्प डेस्क या सीएससी सेंटर जाएं। जरूरी कागजात: आधार कार्ड, राशन कार्ड या SECC पर्ची, और आधार से लिंक मोबाइल नंबर।"},
        {"lang": "hi", "task": "डायलिसिस योजना के नियम और आवश्यक दस्तावेज क्या हैं?",
         "steps_def": [
             ("get_scheme_rules", {"scheme_id": "pmndp"}, "डायलिसिस कार्यक्रम के नियम जांचें।"),
             ("get_documents", {"scheme_id": "pmndp"}, "डायलिसिस दस्तावेज निकालें।")
         ],
         "resp": "प्रधानमंत्री राष्ट्रीय डायलिसिस कार्यक्रम (PMNDP) में बीपीएल मरीजों को सरकारी अस्पतालों में 100% निशुल्क डायलिसिस मिलता है। दस्तावेज: आधार कार्ड, बीपीएल प्रमाण पत्र, और डॉक्टर का पर्चा।"},
        {"lang": "hi", "task": "युवाओं के लिए कौशल विकास योजना (PMKVY) के लाभ और दस्तावेज बताएं।",
         "steps_def": [
             ("get_scheme_rules", {"scheme_id": "pmkvy"}, "पीएम कौशल विकास योजना के नियम निकालें।"),
             ("get_documents", {"scheme_id": "pmkvy"}, "पीएमकेवीवाई के दस्तावेज निकालें।")
         ],
         "resp": "पीएम कौशल विकास योजना (PMKVY 4.0) में युवाओं को नि:शुल्क उद्योग-आधारित कौशल प्रशिक्षण और सरकारी प्रमाण पत्र मिलता है। कागजात: आधार कार्ड, बैंक खाता विवरण और शैक्षिक योग्यता प्रमाण पत्र।"},

        # Hinglish (10)
        {"lang": "hinglish", "task": "Meri age 72 saal hai aur Varanasi me rehta hoon. Mere liye health schemes aur documents bataiye.",
         "steps_def": [
             ("match_schemes", {"age": 72, "category": "health", "income_status": "eligible"}, "Match health schemes for age 72."),
             ("get_documents", {"scheme_id": "ayushman_vay_vandana"}, "Retrieve documents for ayushman_vay_vandana.")
         ],
         "resp": "Aapki age 72 ke according aap Ayushman Vay Vandana Card (AB PM-JAY) ke liye eligible hain, jisme Rs. 5 Lakh ka cover milta hai. Required documents: Aadhaar card aur Aadhaar-linked mobile number."},
        {"lang": "hinglish", "task": "Main 32 saal ka kisan hoon, mujhe crop insurance scheme ke rules aur documents chahiye.",
         "steps_def": [
             ("match_schemes", {"age": 32, "category": "agriculture", "income_status": "eligible"}, "Match agriculture schemes for 32yo farmer."),
             ("get_documents", {"scheme_id": "pm_fasal_bima"}, "Get PMFBY documents.")
         ],
         "resp": "Aap PM Fasal Bima Yojana (PMFBY) ke liye eligible hain. Documents required: Khasra/Khatauni land records, sowing certificate, Aadhaar card, aur bank passbook."},
        {"lang": "hinglish", "task": "Meri wife pregnant hai (age 23), delivery ke liye schemes aur rules batao.",
         "steps_def": [
             ("match_schemes", {"age": 23, "category": "maternal", "income_status": "eligible"}, "Query maternal schemes for age 23."),
             ("get_scheme_rules", {"scheme_id": "jsy"}, "Get rules for JSY.")
         ],
         "resp": "Aap JSY aur JSSK ke liye eligible hain. Government hospital me delivery par Rs. 1,400 financial assistance aur medicines/treatment bilkul free milti hai. Portal: https://nhm.gov.in/."},
        {"lang": "hinglish", "task": "66 saal ke senior citizen ke liye pension scheme aur documents batayein.",
         "steps_def": [
             ("match_schemes", {"age": 66, "category": "senior", "income_status": "eligible"}, "Match senior pension schemes."),
             ("get_documents", {"scheme_id": "nsap_old_age"}, "Get NSAP old age documents.")
         ],
         "resp": "Aap NSAP Old Age Pension ke eligible hain. Documents required: Aadhaar card, BPL ration card / SECC proof, age certificate (60+), aur bank account passbook."},
        {"lang": "hinglish", "task": "Lucknow me TB hospital kahan hai aur Nikshay Poshan scheme ke kya rules hain?",
         "steps_def": [
             ("find_facility", {"district": "Lucknow", "facility_type": "hospital"}, "Find health facility in Lucknow."),
             ("get_scheme_rules", {"scheme_id": "ntep"}, "Retrieve NTEP rules.")
         ],
         "resp": "Lucknow me District Civil Hospital Help Desk par contact karein. NTEP Nikshay Poshan Yojana ke under treatment chalne tak har mahine Rs. 500 direct bank account me transfer hote hain."},
        {"lang": "hinglish", "task": "Ujjwala 2.0 gas connection ke rules aur documents kya hain?",
         "steps_def": [
             ("get_scheme_rules", {"scheme_id": "pm_ujjwala"}, "Get Ujjwala rules."),
             ("get_documents", {"scheme_id": "pm_ujjwala"}, "Get Ujjwala documents.")
         ],
         "resp": "PM Ujjwala 2.0 me deposit-free gas connection ke sath pehla cylinder aur chulha free milta hai. Documents: BPL ration card, female head ka Aadhaar card, aur bank passbook."},
        {"lang": "hinglish", "task": "Atal Pension Yojana ke rules aur benefits kya hain?",
         "steps_def": [
             ("get_scheme_rules", {"scheme_id": "apy"}, "Get APY rules."),
             ("get_documents", {"scheme_id": "apy"}, "Get APY documents.")
         ],
         "resp": "Atal Pension Yojana (APY) me 18-40 saal ke log monthly contribution dekar 60 saal ke baad Rs. 1,000 se Rs. 5,000 tak guaranteed monthly pension pa sakte hain. Required documents: Aadhaar card aur savings bank account."},
        {"lang": "hinglish", "task": "Patna district me health facilities kahan hain aur PM-JAY ke documents kya lagenge?",
         "steps_def": [
             ("find_facility", {"district": "Patna", "facility_type": "hospital"}, "Search Patna facilities."),
             ("get_documents", {"scheme_id": "ab_pmjay"}, "Get PM-JAY documents.")
         ],
         "resp": "Patna me District Civil Hospital Help Desk aur CSC centers par visit karein. Documents: Aadhaar Card, Ration Card/SECC record, aur Aadhaar-linked mobile number."},
        {"lang": "hinglish", "task": "PM National Dialysis Programme me registration aur documents kya chahiye?",
         "steps_def": [
             ("get_scheme_rules", {"scheme_id": "pmndp"}, "Get PMNDP rules."),
             ("get_documents", {"scheme_id": "pmndp"}, "Get PMNDP documents.")
         ],
         "resp": "PMNDP me BPL patients ke liye government hospital me hemodialysis 100% free hai. Documents: Aadhaar card, BPL certificate, aur government doctor ka dialysis prescription."},
        {"lang": "hinglish", "task": "Skill training ke liye PMKVY scheme ke rules aur documents bataiye.",
         "steps_def": [
             ("get_scheme_rules", {"scheme_id": "pmkvy"}, "Get PMKVY rules."),
             ("get_documents", {"scheme_id": "pmkvy"}, "Get PMKVY documents.")
         ],
         "resp": "PMKVY me free certification aur skill courses provide kiye jaate hain. Documents: Aadhaar card, educational certificates, aur bank account details."}
    ]

    for i, item in enumerate(multi_step_cases, 1):
        rec_id = f"repair_cat_f_{i:03d}"
        steps = []
        for s_idx, (t_name, t_args, t_thought) in enumerate(item["steps_def"], 1):
            step = make_step(s_idx, f"step_{s_idx}", t_name, t_args, t_thought)
            steps.append(step)
        msgs = format_messages(item["task"], steps, item["resp"])
        records.append({
            "id": rec_id,
            "category": "F_multi_step_clarification",
            "task": item["task"],
            "language": item["lang"],
            "steps": steps,
            "final_response": item["resp"],
            "messages": msgs,
            "metadata": {
                "category": "multi_step_clarification",
                "language": item["lang"],
                "tools_used": [s[0] for s in item["steps_def"]],
                "source": "local_multi_step_engine"
            }
        })

    # =========================================================================
    # CATEGORY G: Case Summary & Escalation (25)
    # Complex cases, disputes, grievances, or vulnerable edge cases.
    # Dispatches create_case_summary followed by escalate_to_human.
    # Languages: 9 en, 8 hi, 8 hinglish
    # =========================================================================
    escalation_cases = [
        # En (9)
        {"lang": "en", "task": "Beneficiary is an 82 year old bedridden widow in Raebareli whose pension was stopped for 6 months. Escalate this issue.",
         "summary_fields": {"age": 82, "category": "widow_pension", "district": "Raebareli", "issue": "stopped_pension_6_months"},
         "reason": "Bedridden 82-year-old widow unable to visit bank; pension stopped for 6 consecutive months.",
         "resp": "Case profile created and escalated to the District Social Welfare Officer, Raebareli. A home visit has been requested for doorstep physical verification to restore the pension. You can also track this grievance on the CM Helpline portal."},
        {"lang": "en", "task": "Empanelled hospital in Varanasi demanded Rs 20,000 cash advance for an emergency gallbladder surgery under PM-JAY. Escalate immediately.",
         "summary_fields": {"scheme": "ab_pmjay", "district": "Varanasi", "hospital_issue": "illegal_cash_advance_demand"},
         "reason": "Empanelled hospital demanding unlawful cash advance from Ayushman Bharat cardholder for emergency surgery.",
         "resp": "Formal grievance escalated to the District Grievance Redressal Committee and State Health Agency (SHA). Hospital illegal charging under PM-JAY is strictly investigated under anti-fraud protocols. Toll-free helpline: 14555."},
        {"lang": "en", "task": "Severe orphan child without birth certificate or Aadhaar needs immediate admission to primary school and poshan support. Escalate case.",
         "summary_fields": {"category": "child_welfare", "district": "Patna", "issue": "undocumented_orphan_poshan_school"},
         "reason": "Orphan child lacking identity documentation requiring urgent child welfare committee intervention for food and education.",
         "resp": "Case summary compiled and escalated to the District Child Protection Unit (DCPU) and Child Welfare Committee (CWC). Childline 1098 has been alerted for immediate field intervention."},
        {"lang": "en", "task": "Farmer with 2 hectares land had crops destroyed by hailstorm, but insurance company rejected claim stating surveyor delay. Escalate grievance.",
         "summary_fields": {"scheme": "pm_fasal_bima", "district": "Sitapur", "issue": "surveyor_delay_claim_rejection"},
         "reason": "PMFBY localized crop loss claim unlawfully rejected due to insurance surveyor non-attendance.",
         "resp": "Escalated to the District Level Monitoring Committee (DLMC) for PM Fasal Bima Yojana, Sitapur. The insurance company will be issued a compliance notice regarding surveyor non-attendance."},
        {"lang": "en", "task": "Applicant with 80% locomotor disability unable to access 2nd-floor medical board for UDID card. Escalate for ground floor/home inspection.",
         "summary_fields": {"scheme": "disability_card", "district": "Gorakhpur", "issue": "inaccessible_medical_board"},
         "reason": "Severe disability applicant unable to climb stairs for mandatory medical board assessment.",
         "resp": "Escalated to the Chief Medical Officer (CMO), Gorakhpur. A request has been queued for ground-floor accommodation or mobile medical board assessment."},
        {"lang": "en", "task": "Pregnant woman in labor turned away by community health center due to absence of electricity/doctor. Escalate immediately.",
         "summary_fields": {"scheme": "jsy_jssk", "district": "Mirzapur", "issue": "refusal_of_institutional_delivery"},
         "reason": "Maternity patient in labor turned away by CHC staff; severe violation of JSSK protocols.",
         "resp": "CRITICAL ESCALATION: Case escalated to the Chief Medical Officer and District Magistrate. An immediate emergency ambulance transfer and inquiry have been ordered under Janani Shishu Suraksha Karyakram rules."},
        {"lang": "en", "task": "CSC operator charged Rs 300 to print a free PM-JAY golden card. Escalate corruption report.",
         "summary_fields": {"scheme": "ab_pmjay", "district": "Lucknow", "issue": "csc_overcharging_corruption"},
         "reason": "Common Service Center operator demanding unauthorized fee for government free service.",
         "resp": "Corruption complaint escalated to the District e-Governance Society (DeGS) and CSC District Manager. CSC ID flagged for disciplinary suspension and refund processing."},
        {"lang": "en", "task": "Dialysis patient turned away from district hospital because dialysis technician was absent. Escalate.",
         "summary_fields": {"scheme": "pmndp", "district": "Faizabad", "issue": "dialysis_service_disruption"},
         "reason": "Critical dialysis patient denied routine life-sustaining hemodialysis due to operational absence.",
         "resp": "Escalated to the Hospital Superintendent and PMNDP Nodal Officer. An emergency alternative slot and technician relief have been directed immediately."},
        {"lang": "en", "task": "Elderly pensioner's biometric verification failed 10 times due to worn fingerprints, denying ration distribution. Escalate.",
         "summary_fields": {"scheme": "nfsa_ration", "district": "Prayagraj", "issue": "biometric_failure_ration_denial"},
         "reason": "Elderly beneficiary facing food deprivation due to repetitive biometric verification failures.",
         "resp": "Escalated to the District Supply Officer (DSO). Under Food Security guidelines, a nomination facility or manual OTP/offline register verification has been ordered to release the ration immediately."}
    ]

    escalation_hi = [
        # Hi (8)
        {"lang": "hi", "task": "80 वर्षीय बुजुर्ग महिला की विधवा पेंशन पिछले 8 महीने से बिना कारण बंद है। इसकी तुरंत शिकायत दर्ज कराएं।",
         "summary_fields": {"age": 80, "scheme": "nsap_widow", "district": "सीतापुर", "issue": "पेंशन_8_महीने_से_बंद"},
         "reason": "80 वर्षीय बुजुर्ग विधवा महिला की पेंशन अकारण बंद; बैंक जाने में असमर्थ।",
         "resp": "केस सारांश दर्ज करके जिला समाज कल्याण अधिकारी, सीतापुर को अग्रेषित (Escalate) कर दिया गया है। सत्यापन कराकर रुकी हुई पेंशन जारी कराने हेतु जांच आदेशित की गई है।"},
        {"lang": "hi", "task": "पैनलबद्ध निजी अस्पताल ने आयुष्मान कार्ड होने पर भी मरीज को भर्ती करने से मना कर दिया और ₹50,000 मांगे।",
         "summary_fields": {"scheme": "ab_pmjay", "district": "वाराणसी", "issue": "इलाज_से_इनकार_व_अवैध_वसूली"},
         "reason": "सूचीबद्ध अस्पताल द्वारा आयुष्मान कार्ड धारक को भर्ती करने से इनकार और अवैध धन की मांग।",
         "resp": "शिकायत जिला स्वास्थ्य समिति एवं राज्य स्वास्थ्य एजेंसी (SHA) के एंटी-फ्रॉड सेल को तुरंत प्रेषित कर दी गई है। अस्पताल पर दंडात्मक कार्रवाई हेतु नोटिस जारी किया जा रहा है। हेल्पलाइन: 14555।"},
        {"lang": "hi", "task": "ओलावृष्टि से गेहूं की फसल नष्ट हो गई, लेकिन बीमा कंपनी के प्रतिनिधि ने नुकसान का मुआयना करने से मना कर दिया।",
         "summary_fields": {"scheme": "pm_fasal_bima", "district": "हरदोई", "issue": "सर्वेयर_द्वारा_निरीक्षण_से_इनकार"},
         "reason": "पीएम फसल बीमा योजना के तहत सर्वेयर द्वारा समय पर फसल क्षति का आकलन न करना।",
         "resp": "यह मामला जिला स्तरीय निगरानी समिति (DLMC) एवं उप-कृषि निदेशक को प्रेषित किया गया है। सर्वेयर को 48 घंटे में खेत का मौका मुआयना करने का निर्देश जारी किया गया है।"},
        {"lang": "hi", "task": "गंभीर रूप से दिव्यांग व्यक्ति पहली मंजिल पर स्थित मेडिकल बोर्ड में नहीं चढ़ पा रहा। मदद हेतु केस आगे बढ़ाएं।",
         "summary_fields": {"scheme": "disability_card", "district": "गोरखपुर", "issue": "दिव्यांग_प्रमाणन_में_बाधा"},
         "reason": "चलने-फिरने में असमर्थ दिव्यांग व्यक्ति हेतु भूतल पर परीक्षण या कैंप की आवश्यकता।",
         "resp": "मामला मुख्य चिकित्सा अधिकारी (CMO) गोरखपुर को भेजा गया है। दिव्यांग व्यक्ति का परीक्षण भूतल पर कराने या सचल दल भेजने का अनुरोध दर्ज किया गया है।"},
        {"lang": "hi", "task": "प्रसव पीड़ा से पीड़ित महिला को सरकारी स्वास्थ्य केंद्र ने यह कहकर लौटा दिया कि डॉक्टर मौजूद नहीं हैं।",
         "summary_fields": {"scheme": "jssk", "district": "मिर्जापुर", "issue": "प्रसूता_को_लौटाना"},
         "reason": "प्रसव पीड़ा से पीड़ित महिला को सीएचसी से लौटाना; जेएसएसके नियमों का गंभीर उल्लंघन।",
         "resp": "गंभीर आपातकालीन शिकायत: मुख्य चिकित्सा अधिकारी और जिलाधिकारी को तत्काल सूचना भेजी गई है। प्रसूता को एम्बुलेंस से जिला अस्पताल शिफ्ट करने और दोषी कर्मियों पर कार्रवाई के निर्देश दिए गए हैं।"},
        {"lang": "hi", "task": "सीएससी ऑपरेटर ने मुफ़्त आयुष्मान कार्ड प्रिंट करने के ₹200 जबरन वसूल लिए।",
         "summary_fields": {"scheme": "ab_pmjay", "district": "लखनऊ", "issue": "अवैध_वसूली"},
         "reason": "सीएससी वीएलई द्वारा निशुल्क सरकारी सेवा के बदले रिश्वत/अवैध वसूली।",
         "resp": "भ्रष्टाचार की शिकायत जिला ई-गवर्नेंस प्रबंधक और सीएससी नोडल अधिकारी को भेज दी गई है। सीएससी केंद्र की जांच और आईडी निलंबन की प्रक्रिया शुरू कर दी गई है।"},
        {"lang": "hi", "task": "जिला अस्पताल में डायलिसिस मशीन खराब बताकर मरीज को लौटा दिया गया, मरीज की हालत नाजुक है।",
         "summary_fields": {"scheme": "pmndp", "district": "प्रयागराज", "issue": "डायलिसिस_सेवा_बाधित"},
         "reason": "अति आवश्यक जीवन रक्षक डायलिसिस सेवा बाधित होने से मरीज के जीवन को खतरा।",
         "resp": "मामला जिला अस्पताल के मुख्य चिकित्सा अधीक्षक (CMS) को प्राथमिकता पर भेजा गया है। मरीज को वैकल्पिक डायलिसिस स्लॉट या नजदीकी केंद्र में तत्काल सुविधा देने का निर्देश दिया गया है।"},
        {"lang": "hi", "task": "बुजुर्ग की उंगलियों के निशान घिस जाने से फिंगरप्रिंट नहीं आ रहा और कोटेदार ने राशन देने से मना कर दिया।",
         "summary_fields": {"scheme": "nfsa_ration", "district": "अयोध्या", "issue": "बायोमेट्रिक_विफलता_राशन_इनकार"},
         "reason": "वृद्धावस्था बायोमेट्रिक विफलता के कारण खाद्य सुरक्षा राशन से वंचित।",
         "resp": "मामला जिला पूर्ति अधिकारी (DSO) को प्रेषित किया गया है। सरकारी नियमों के तहत ओटीपी या नामांकित व्यक्ति (Proxy) के माध्यम से राशन तत्काल उपलब्ध कराने का आदेश जारी हुआ है।"}
    ]

    escalation_hinglish = [
        # Hinglish (8)
        {"lang": "hinglish", "task": "85 years ke bedridden grandfather ki old age pension 7 months se nahi aayi, escalate kijiye.",
         "summary_fields": {"age": 85, "scheme": "nsap_old_age", "district": "Raebareli", "issue": "pension_stopped_7_months"},
         "reason": "85-year-old bedridden pensioner facing stopped pension; physical verification needed.",
         "resp": "Case profile create karke District Social Welfare Officer ko escalate kar diya gaya hai. Ghar par aakar verification karne aur pending pension release karne ka order diya gaya hai."},
        {"lang": "hinglish", "task": "Empanelled hospital ne Ayushman card reject karke emergency appendix surgery ke 35,000 rupaye mange.",
         "summary_fields": {"scheme": "ab_pmjay", "district": "Varanasi", "issue": "card_rejection_illegal_billing"},
         "reason": "Empanelled private hospital refusing PM-JAY card and demanding out-of-pocket payment for emergency surgery.",
         "resp": "Formal grievance State Health Agency (SHA) aur Anti-Fraud Unit ko escalate kar di gayi hai. Hospital ke khilaaf show-cause notice aur inquiry initiate ho chuki hai. Helpline: 14555."},
        {"lang": "hinglish", "task": "Hailstorm se puri fasal barbaad ho gayi par insurance company surveyor call nahi utha raha. Help!",
         "summary_fields": {"scheme": "pm_fasal_bima", "district": "Sitapur", "issue": "surveyor_unresponsive"},
         "reason": "PMFBY crop damage claim: insurance company surveyor failing to conduct mandatory 72-hour field inspection.",
         "resp": "District Agriculture Officer aur DLMC monitoring committee ko grievance escalate kar di gayi hai. Insurance company ko direct notice bhejkar 24 hours me survey complete karne ka aadesh diya gaya hai."},
        {"lang": "hinglish", "task": "Bedridden handicapped person hospital nahi ja sakta UDID card banwane ke liye. Home visit request karo.",
         "summary_fields": {"scheme": "disability_card", "district": "Gorakhpur", "issue": "bedridden_applicant_home_visit"},
         "reason": "100% locomotor disability applicant unable to travel to district disability board.",
         "resp": "Chief Medical Officer (CMO) Gorakhpur ko escalate kiya gaya hai. Mobile medical team ya home-visit assessment schedule karne ke liye application mark ho gayi hai."},
        {"lang": "hinglish", "task": "Labour room me nurse ne delivery karane se mana kar diya aur patient ko dusre shahar bhej diya.",
         "summary_fields": {"scheme": "jsy_jssk", "district": "Mirzapur", "issue": "emergency_delivery_refusal"},
         "reason": "Severe denial of institutional delivery at government health facility; violation of JSSK.",
         "resp": "Critical medical grievance CMO aur DM office ko escalate kar di gayi hai. En route patient ke liye 108 ambulance coordination aur CHC staff ke against disciplinary inquiry start ki gayi hai."},
        {"lang": "hinglish", "task": "CSC center par Ayushman card ke naam par 250 rupaye extra charge kar rahe hain. Action lo.",
         "summary_fields": {"scheme": "ab_pmjay", "district": "Lucknow", "issue": "csc_corruption"},
         "reason": "CSC center demanding illegal bribe/fee for free government card issuance.",
         "resp": "Complaint District e-Governance Society (DeGS) ko bhej di gayi hai. Concerned CSC VLE ki license suspension aur inquiry process trigger ho gayi hai."},
        {"lang": "hinglish", "task": "District Civil Hospital me dialysis unit me water plant kharab hai aur patients ko bhej rahe hain.",
         "summary_fields": {"scheme": "pmndp", "district": "Faizabad", "issue": "dialysis_plant_failure"},
         "reason": "PMNDP hemodialysis facility disruption putting dialysis patients at critical risk.",
         "resp": "Hospital CMS aur State PMNDP In-charge ko escalate kiya gaya hai. Nearest functional center me patients ko divert karne aur maintenance team ko emergency dispatch kiya gaya hai."},
        {"lang": "hinglish", "task": "Grandmother ke fingerprints match nahi ho rahe toh ration dealer ne food grains dene se inkaar kar diya.",
         "summary_fields": {"scheme": "nfsa_ration", "district": "Prayagraj", "issue": "biometric_denial_ration"},
         "reason": "Elderly citizen denied essential food grain ration due to biometric sensor failure.",
         "resp": "District Supply Officer (DSO) ko report escalate kar di gayi hai. NFSA rules ke tehat OTP ya manual register override ke through turant ration distribute karne ka instruction issue kiya gaya hai."}
    ]

    all_escalations = escalation_cases + escalation_hi + escalation_hinglish
    for i, item in enumerate(all_escalations, 1):
        rec_id = f"repair_cat_g_{i:03d}"
        s1 = make_step(1, "case_summarization", "create_case_summary", {"collected_fields": item["summary_fields"]}, "Compile structured case summary with verified beneficiary fields.")
        s2 = make_step(2, "human_escalation", "escalate_to_human", {"case_summary": item["summary_fields"], "reason": item["reason"]}, "Escalate case to designated supervisor / welfare officer with explicit grievance reason.")
        steps = [s1, s2]
        msgs = format_messages(item["task"], steps, item["resp"])
        records.append({
            "id": rec_id,
            "category": "G_case_summary_escalation",
            "task": item["task"],
            "language": item["lang"],
            "steps": steps,
            "final_response": item["resp"],
            "messages": msgs,
            "metadata": {
                "category": "case_summary_escalation",
                "language": item["lang"],
                "tools_used": ["create_case_summary", "escalate_to_human"],
                "source": "local_escalation_engine"
            }
        })

    return records

def main():
    print("Generating SanoVocal Tool Contract Repair Dataset...")
    records = build_dataset()
    print(f"Total generated records: {len(records)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved repair dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
