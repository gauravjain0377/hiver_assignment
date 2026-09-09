"""
Intent Definitions — Phase 2
Defines the 10 intents for the AI support agent based on EDA findings.
"""

# ─── Intent Taxonomy ──────────────────────────────────────────────────────────
# Defined from EDA on AppleSupport Twitter data.
# Each intent has: name, description, keywords (for labelling), examples

INTENTS = {
    "login_problem": {
        "label": "Login / Account Access Problem",
        "description": "User cannot log in, forgot password, or is locked out of their account.",
        "keywords": ["can't login", "cant login", "cannot login", "sign in", "password", "locked out", "forgot password", "account access", "can't log in", "cant log in", "cannot log in", "not logging"],
        "escalate_if": False,
        "auto_reply_template": "We understand login issues are frustrating. Let's get you back in. Have you tried resetting your password at {reset_url}? If the issue persists, DM us your account details.",
    },
    "billing_issue": {
        "label": "Billing / Payment / Subscription Issue",
        "description": "User has a charge dispute, refund request, or subscription problem.",
        "keywords": ["charge", "refund", "payment", "bill", "subscription", "cancel", "charged", "money", "purchase"],
        "escalate_if": True,  # Money issues → human review
        "auto_reply_template": "We're sorry about the billing issue. Our team will review your account. Please DM us your order/transaction details and we'll resolve this ASAP.",
    },
    "bug_report": {
        "label": "Bug / Technical Error",
        "description": "User reports a bug, crash, error message, or broken feature.",
        "keywords": ["not working", "broken", "error", "crash", "bug", "glitch", "freeze", "stuck", "fail"],
        "escalate_if": False,
        "auto_reply_template": "Thanks for letting us know! Can you share: (1) your device/OS version, (2) app version, (3) steps to reproduce? This helps us investigate faster.",
    },
    "feature_request": {
        "label": "Feature Request / Suggestion",
        "description": "User requests a new feature or improvement.",
        "keywords": ["would be great", "please add", "feature", "suggestion", "wish", "should have", "why can't", "add option"],
        "escalate_if": False,
        "auto_reply_template": "Great suggestion! We've noted this feedback. You can also submit feature requests at {feedback_url} — our team reviews these regularly.",
    },
    "account_suspension": {
        "label": "Account Suspended / Banned",
        "description": "User's account has been suspended, banned, or disabled.",
        "keywords": ["suspended", "banned", "disabled", "blocked", "removed", "locked"],
        "escalate_if": True,  # Suspensions always need human review
        "auto_reply_template": "Account suspension reviews require human attention. Please DM us your account details and we'll have our Trust & Safety team look into this within 24 hours.",
    },
    "content_issue": {
        "label": "Content / Media Issue",
        "description": "User has issues with specific content (songs, apps, media not loading/missing).",
        "keywords": ["song", "playlist", "music", "video", "content", "missing", "disappeared", "app", "download"],
        "escalate_if": False,
        "auto_reply_template": "Sorry about the content issue! Try: (1) Sign out and back in, (2) Clear app cache, (3) Reinstall the app. Still happening? Tell us which content and your device.",
    },
    "app_performance": {
        "label": "App Performance / Speed Issue",
        "description": "User experiences slowness, lag, battery drain, or performance problems.",
        "keywords": ["slow", "lag", "loading", "performance", "battery", "drain", "memory", "heating", "fast"],
        "escalate_if": False,
        "auto_reply_template": "Performance issues are no fun! Try: (1) Restart the app, (2) Restart your device, (3) Check for updates. What device and app version are you using?",
    },
    "connectivity": {
        "label": "Connectivity / Sync Issue",
        "description": "User has offline, internet, sync, or network-related issues.",
        "keywords": ["offline", "internet", "connection", "wifi", "network", "sync", "not syncing", "server"],
        "escalate_if": False,
        "auto_reply_template": "Connectivity troubles are frustrating! Check: (1) Your internet connection, (2) Our status page at {status_url}, (3) Restart your device. Are others in your area affected?",
    },
    "general_inquiry": {
        "label": "General Question / How-To",
        "description": "User asks a general how-to, informational, or help question.",
        "keywords": ["how do", "can i", "is it possible", "how to", "what is", "help me", "where", "when"],
        "escalate_if": False,
        "auto_reply_template": "Happy to help! For detailed how-to guides, check our support center at {support_url}. What specifically are you trying to do? We'll guide you through it.",
    },
    "positive_feedback": {
        "label": "Positive Feedback / Compliment",
        "description": "User expresses satisfaction, thanks, or positive feedback.",
        "keywords": ["love", "great", "amazing", "thank", "awesome", "best", "perfect", "wonderful", "fantastic"],
        "escalate_if": False,
        "auto_reply_template": "Thank you so much! We're thrilled you're having a great experience. Your feedback means the world to us! 😊",
    },
}

INTENT_NAMES = list(INTENTS.keys())
NUM_INTENTS = len(INTENTS)

# Intents that should always trigger escalation
ALWAYS_ESCALATE = [k for k, v in INTENTS.items() if v.get("escalate_if", False)]

# Mapping for display
INTENT_LABELS = {k: v["label"] for k, v in INTENTS.items()}
