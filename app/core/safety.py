"""
SanoVocal Safety & Policy Enforcement Engine
Implements deterministic, clinical safety pre-checks, fraud refusal,
and clinical non-diagnosis boundaries.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

# Acute emergency medical keywords (Multilingual: English, Hindi, Hinglish, Devanagari)
EMERGENCY_RED_FLAGS = [
    "chest pain", "heart attack", "difficulty breathing", "breathlessness",
    "severe bleeding", "unconscious", "behosh", "poison", "zeher",
    "stroke", "paralysis", "suicide", "severe trauma", "road accident",
    "saas lene me dikkat", "saas lehne me dikkat", "saas lene mein dikkat",
    "chhati me dard", "chhati mein dard", "chaati me dard", "chaati me",
    "khun beh raha", "chhati me bahut dard", "chhati me tez dard",
    "breathing trouble", "heavy bleeding", "gasping",
    "छाती में दर्द", "सांस लेने में तकलीफ", "सांस लेने में दिक्कत", "बेहोश"
]

# Medical diagnosis / prescription keywords
PRESCRIPTION_KEYWORDS = [
    "prescribe", "medicine for", "which antibiotic", "what drug should i take",
    "dosage of", "diagnose my", "dawa batao", "konsi goli lu", "treatment dose"
]

# Fraud / Bribery / Corruption keywords
FRAUD_KEYWORDS = [
    "fake document", "bribe", "bypass rule", "fake certificate", "false income",
    "jaali certificate", "rishwat", "fraud", "forge certificate", "counterfeit"
]

class SafetyVerdict:
    def __init__(self, is_safe: bool, reason: str, category: str, override_response: Optional[str] = None):
        self.is_safe = is_safe
        self.reason = reason
        self.category = category  # 'CLEARED', 'EMERGENCY_OVERRIDE', 'MEDICAL_REFUSAL', 'FRAUD_REFUSAL'
        self.override_response = override_response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "category": self.category,
            "reason": self.reason,
            "has_override": bool(self.override_response)
        }

def evaluate_deterministic_safety(user_text: str) -> SafetyVerdict:
    """
    Evaluates input text against deterministic safety policies before LLM invocation.
    Priority 1: Acute Medical Emergency Red-Flags (Life-threatening situation)
    Priority 2: Medical Prescription / Diagnosis Requests
    Priority 3: Fraud / Illegal Scheme Exploitation
    """
    text_lower = user_text.lower()

    # 1. Clinical Emergency Red-Flags
    detected_emergencies = [kw for kw in EMERGENCY_RED_FLAGS if kw in text_lower]
    if detected_emergencies:
        override_msg = (
            "🚨 **EMERGENCY MEDICAL ALERT / आपातकालीन चिकित्सा चेतावनी**\n\n"
            f"You have reported symptoms indicating a potential life-threatening emergency ({', '.join(detected_emergencies)}).\n\n"
            "- **IMMEDIATE ACTION**: Call **112** (National Emergency) or **108** (Ambulance) immediately.\n"
            "- Please proceed to the nearest emergency department or government civil hospital.\n"
            "- Welfare scheme navigation has been temporarily halted to prioritize your safety."
        )
        return SafetyVerdict(
            is_safe=False,
            reason=f"Emergency red-flags detected: {detected_emergencies}",
            category="EMERGENCY_OVERRIDE",
            override_response=override_msg
        )

    # 2. Medical Prescription / Clinical Diagnosis Refusal
    detected_rx = [kw for kw in PRESCRIPTION_KEYWORDS if kw in text_lower]
    if detected_rx:
        override_msg = (
            "⚕️ **CLINICAL BOUNDARY NOTICE / चिकित्सा परामर्श सूचना**\n\n"
            "SanoVocal is an AI scheme navigator and **cannot prescribe medication or diagnose medical conditions**.\n\n"
            "- Please consult a registered medical practitioner (MBBS/Ayush) or visit your nearest Primary Health Centre (PHC) / Community Health Centre (CHC).\n"
            "- I can help you find nearby empanelled healthcare facilities or check if your treatment is covered under schemes like Ayushman Bharat (PM-JAY)."
        )
        return SafetyVerdict(
            is_safe=False,
            reason=f"Prescription/diagnosis request detected: {detected_rx}",
            category="MEDICAL_REFUSAL",
            override_response=override_msg
        )

    # 3. Fraud / Illegal Exploitation Refusal
    detected_fraud = [kw for kw in FRAUD_KEYWORDS if kw in text_lower]
    if detected_fraud:
        override_msg = (
            "🛡️ **COMPLIANCE & LEGAL NOTICE / अनुपालन सूचना**\n\n"
            "SanoVocal strictly adheres to Indian statutory guidelines and anti-fraud policies.\n\n"
            "- I cannot assist with falsifying documents, bypassing eligibility verifications, or paying unauthorized intermediaries.\n"
            "- All government welfare benefits require authentic verification through Aadhaar e-KYC and statutory documentation."
        )
        return SafetyVerdict(
            is_safe=False,
            reason=f"Fraudulent intent detected: {detected_fraud}",
            category="FRAUD_REFUSAL",
            override_response=override_msg
        )

    return SafetyVerdict(
        is_safe=True,
        reason="Passed all deterministic safety checks.",
        category="CLEARED",
        override_response=None
    )
