"""
Custom GDPR tools for ReAct Agent.

These tools enable the ReAct agent to:
1. Look up GDPR article definitions
2. Search code for sensitive API calls
3. Check code against formal rules
"""

import os
import re
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool


# GDPR Article Definitions (Knowledge Base)
GDPR_DEFINITIONS = {
    5: "Article 5: Principles relating to processing of personal data - lawfulness, fairness, transparency, purpose limitation, data minimization, accuracy, storage limitation, integrity and confidentiality.",
    6: "Article 6: Lawfulness of processing - requires at least one legal basis (consent, contract, legal obligation, vital interests, public task, legitimate interests).",
    7: "Article 7: Conditions for consent - consent must be freely given, specific, informed and unambiguous. Must be as easy to withdraw as to give.",
    8: "Article 8: Conditions applicable to child's consent - if child is below 16, consent must be given by parent/guardian.",
    9: "Article 9: Processing of special categories of personal data - prohibition on processing sensitive data (health, biometric, genetic, etc.) unless specific conditions are met.",
    12: "Article 12: Transparent information, communication and modalities - information must be provided in concise, transparent, intelligible and easily accessible form.",
    13: "Article 13: Information to be provided where personal data are collected from the data subject - must inform about identity, purpose, legal basis, recipients, retention period, rights.",
    14: "Article 14: Information to be provided where personal data have not been obtained from the data subject - same as Article 13 but for indirect collection.",
    15: "Article 15: Right of access by the data subject - individuals have right to obtain confirmation of processing and access to their data.",
    16: "Article 16: Right to rectification - individuals have right to rectify inaccurate personal data.",
    17: "Article 17: Right to erasure ('right to be forgotten') - individuals have right to obtain erasure of their personal data under certain conditions.",
    18: "Article 18: Right to restriction of processing - individuals have right to restrict processing under certain circumstances.",
    19: "Article 19: Notification obligation regarding rectification or erasure - controller must notify recipients about rectification/erasure.",
    20: "Article 20: Right to data portability - individuals have right to receive personal data in structured, commonly used, machine-readable format and transmit to another controller.",
    21: "Article 21: Right to object - individuals have right to object to processing based on legitimate interests or for direct marketing.",
    25: "Article 25: Data protection by design and by default - implement appropriate technical and organizational measures, privacy by default. HIGH FREQUENCY in dataset.",
    30: "Article 30: Records of processing activities - controllers must maintain records of processing activities.",
    32: "Article 32: Security of processing - implement appropriate technical and organizational measures to ensure security (encryption, pseudonymization, etc.).",
    33: "Article 33: Notification of a personal data breach to the supervisory authority - must notify within 72 hours of becoming aware.",
    35: "Article 35: Data protection impact assessment - required for high-risk processing activities.",
    44: "Article 44: General principle for transfers - transfers to third countries only if adequate level of protection.",
    46: "Article 46: Transfers subject to appropriate safeguards - standard contractual clauses, binding corporate rules, etc.",
    58: "Article 58: Powers of supervisory authorities - investigative, corrective and authorization powers.",
    83: "Article 83: General conditions for imposing administrative fines - criteria for determining fines.",
    
    # ==================== Extended Articles (AI Era + Security Response) ====================
    
    4: "Article 4: Definitions - 'personal data' means any information relating to identified or identifiable natural person; 'processing' means any operation performed on personal data; 'controller' means entity determining purposes and means; 'processor' means entity processing on behalf of controller; 'consent' means freely given, specific, informed and unambiguous indication; 'data breach' means breach of security leading to destruction, loss, alteration, unauthorized disclosure or access.",
    
    22: "Article 22: Automated individual decision-making, including profiling - Data subject has right not to be subject to decision based solely on automated processing, including profiling, which produces legal effects or similarly significantly affects them. Exceptions: (a) necessary for contract; (b) authorized by law; (c) based on explicit consent. Controller must implement measures to safeguard rights: at least right to obtain human intervention, express point of view, and contest decision.",
    
    24: "Article 24: Responsibility of the controller - Controller must implement appropriate technical and organizational measures to ensure and demonstrate processing is performed in accordance with GDPR. Measures must take into account nature, scope, context and purposes of processing and risks. Controller must implement policies and review effectiveness.",
    
    28: "Article 28: Processor - Processing by processor must be governed by contract or legal act. Contract must stipulate: (a) subject matter and duration; (b) nature and purpose; (c) type of data and categories; (d) obligations and rights of controller. Processor must not engage another processor without prior authorization. Processor must assist controller in ensuring compliance.",
    
    34: "Article 34: Communication of breach to data subject - When breach is likely to result in high risk to rights and freedoms, controller must communicate breach to data subject without undue delay in clear and plain language. Communication must describe: (a) nature of breach; (b) contact point; (c) likely consequences; (d) measures taken. Not required if: (a) appropriate protection measures applied; (b) subsequent measures ensure high risk no longer likely; (c) would involve disproportionate effort (public communication instead)."
}

# Enhanced Sensitive API patterns (Based on dataset analysis)
SENSITIVE_API_PATTERNS = {
    # Device identifiers (Articles 5, 6, 12, 13)
    'device_id': [
        r'getDeviceId\s*\(',
        r'getIMEI\s*\(',
        r'getSerialNumber\s*\(',
        r'getAndroidId\s*\(',
        r'getSubscriberId\s*\(',
        r'getSimSerialNumber\s*\(',
        r'Settings\.Secure\.getString.*ANDROID_ID',
        r'TelephonyManager.*getDeviceId',
    ],
    # Location (Articles 5, 6, 7, 9, 13)
    'location': [
        r'getLastKnownLocation\s*\(',
        r'requestLocationUpdates\s*\(',
        r'getLatitude\s*\(',
        r'getLongitude\s*\(',
        r'LocationManager',
        r'FusedLocationProviderClient',
        r'GPS',
    ],
    # Camera/Microphone (Articles 5, 6, 7, 13, 32)
    'camera': [
        r'Camera\.open\s*\(',
        r'camera\.takePicture',
        r'MediaRecorder',
        r'AudioRecord',
        r'camera\.startPreview',
        r'camera2\.CameraDevice',
    ],
    # Contacts (Articles 5, 6, 9, 13)
    'contacts': [
        r'ContactsContract',
        r'getContentResolver.*ContactsContract',
        r'CONTENT_URI.*contacts',
        r'READ_CONTACTS',
        r'getContacts\s*\(',
    ],
    # SMS Operations (CRITICAL - Most frequent, Articles 5, 6, 13, 25, 32)
    'sms': [
        r'sendTextMessage\s*\(',
        r'sendDataMessage\s*\(',
        r'sendMultipartTextMessage\s*\(',
        r'SmsManager\.getDefault\s*\(',
        r'SmsManager',
        r'SEND_SMS',
        r'READ_SMS',
        r'RECEIVE_SMS',
    ],
    # Network/Transmission (Articles 25, 32)
    'network': [
        r'HttpURLConnection',
        r'HttpClient',
        r'\.post\s*\(',
        r'\.get\s*\(',
        r'http://',  # Insecure HTTP
        r'Socket\s*\(',
        r'DatagramSocket',
    ],
    # Storage (Articles 5, 13, 25, 32)
    'storage': [
        r'SharedPreferences',
        r'openFileOutput\s*\(',
        r'FileOutputStream',
        r'SQLiteDatabase',
        r'FileWriter',
        r'WRITE_EXTERNAL_STORAGE',
        r'getExternalStorageDirectory',
    ],
    # Encryption/Security (Article 32)
    'encryption': [
        r'Cipher\.getInstance',
        r'MessageDigest',
        r'KeyStore',
        r'SecretKey',
        r'encrypt',
        r'decrypt',
    ]
}


@tool
def gdpr_lookup(article_number: int) -> str:
    """
    Look up the definition and requirements of a GDPR article.
    
    This tool helps the agent understand what a specific GDPR article requires.
    Use this when you need to verify if certain code behavior violates a particular article.
    
    Args:
        article_number: The GDPR article number (e.g., 6, 7, 32)
        
    Returns:
        The definition and key requirements of the article, or error message if not found.
    
    Examples:
        gdpr_lookup(6) -> Returns lawfulness of processing requirements
        gdpr_lookup(32) -> Returns security of processing requirements
    """
    if article_number in GDPR_DEFINITIONS:
        return GDPR_DEFINITIONS[article_number]
    else:
        return f"Article {article_number} not found in knowledge base. Available articles: {', '.join(map(str, sorted(GDPR_DEFINITIONS.keys())))}"


@tool
def code_search(keyword: str, context_lines: int = 3) -> str:
    """
    Search for specific API calls or patterns in the codebase being analyzed.
    
    This tool simulates searching the entire codebase for sensitive operations.
    Use this to find where certain APIs (like getDeviceId, Camera.open) are used.
    
    Args:
        keyword: The API name or pattern to search for (e.g., "getDeviceId", "Camera.open", "location")
        context_lines: Number of context lines to include (default 3)
        
    Returns:
        Search results showing where the keyword appears in code, or message if not found.
    
    Examples:
        code_search("getDeviceId") -> Shows all usages of device ID collection
        code_search("Camera.open") -> Shows camera access points
        code_search("location") -> Shows location-related code
    """
    # This is a simplified implementation
    # In practice, this would search through the actual files being analyzed
    
    results = []
    
    # Check if keyword matches any sensitive API category
    for category, patterns in SENSITIVE_API_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, keyword, re.IGNORECASE) or keyword.lower() in category:
                results.append(f"Found {category}-related API pattern: {pattern}")
    
    if results:
        return "Sensitive API patterns found:\n" + "\n".join(results) + "\n\nThese patterns typically require user consent (Article 6, 7) and security measures (Article 32)."
    else:
        # Generic search result
        return f"Searching for '{keyword}'... This API may involve personal data collection. Please verify: 1) Legal basis (Article 6), 2) User consent (Article 7), 3) Information provided (Article 13), 4) Security measures (Article 32)."


@tool  
def rule_check(code_snippet: str) -> str:
    """
    Check code against predefined GDPR compliance rules.
    
    This tool performs rule-based analysis to detect common GDPR violations.
    Use this to quickly check if code violates known patterns.
    
    Args:
        code_snippet: The code to analyze (can be partial)
        
    Returns:
        List of detected rule violations with corresponding GDPR articles.
    
    Examples:
        rule_check("camera.open()") -> Detects camera access without consent check
        rule_check("http://api.example.com") -> Detects insecure transmission
    """
    violations = []
    
    # Rule 1: Camera/Microphone access without permission check
    if re.search(r'Camera\.open|MediaRecorder|AudioRecord', code_snippet, re.IGNORECASE):
        if not re.search(r'checkPermission|requestPermission|checkSelfPermission', code_snippet, re.IGNORECASE):
            violations.append("⚠️  RULE_001: Camera/microphone access without permission check → Articles 6, 7, 32")
    
    # Rule 2: Device ID collection
    if re.search(r'getDeviceId|getIMEI|getSerialNumber|ANDROID_ID', code_snippet, re.IGNORECASE):
        violations.append("⚠️  RULE_002: Device identifier collection detected → Articles 6, 12, 13")
    
    # Rule 3: Location access
    if re.search(r'getLocation|getLatitude|getLongitude|LocationManager', code_snippet, re.IGNORECASE):
        if not re.search(r'consent|permission|user.*agree', code_snippet, re.IGNORECASE):
            violations.append("⚠️  RULE_003: Location access without explicit consent → Articles 6, 7, 9")
    
    # Rule 4: Insecure data transmission (HTTP instead of HTTPS)
    if re.search(r'http://(?!localhost)', code_snippet, re.IGNORECASE):
        violations.append("⚠️  RULE_004: Insecure HTTP transmission of potential personal data → Article 32")
    
    # Rule 5: Unencrypted storage
    if re.search(r'SharedPreferences|FileOutputStream|SQLiteDatabase', code_snippet, re.IGNORECASE):
        if not re.search(r'encrypt|cipher|secure|KeyStore', code_snippet, re.IGNORECASE):
            violations.append("⚠️  RULE_005: Potentially unencrypted data storage → Article 32")
    
    # Rule 6: Network transmission without encryption
    if re.search(r'sendTextMessage|HttpURLConnection|HttpClient', code_snippet, re.IGNORECASE):
        if re.search(r'password|email|phone|location|IMEI', code_snippet, re.IGNORECASE):
            violations.append("⚠️  RULE_006: Sensitive data network transmission → Articles 25, 32")
    
    # Rule 7: No user notification
    if re.search(r'getDeviceId|getContacts|getLocation|Camera\.open', code_snippet, re.IGNORECASE):
        if not re.search(r'notify|inform|dialog|alert|toast|message', code_snippet, re.IGNORECASE):
            violations.append("⚠️  RULE_007: Data collection without user notification → Articles 12, 13")
    
    if violations:
        result = "❌ GDPR compliance issues detected:\n\n" + "\n".join(violations)
        result += "\n\n✅ Recommendation: Add consent mechanisms, encryption, and user notifications."
        return result
    else:
        return "✅ No obvious GDPR violations detected in this snippet. However, manual review is recommended for context-specific compliance."


# Export all tools
GDPR_TOOLS = [gdpr_lookup, code_search, rule_check]

