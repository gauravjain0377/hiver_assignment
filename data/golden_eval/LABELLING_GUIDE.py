"""
Labelling Guidelines for Golden Eval Set
=========================================

This document explains how to label data/golden_eval/golden_eval_set.csv

OPEN THE FILE IN:
- Excel / Google Sheets (easiest)
- VS Code with CSV editor extension
- Any text editor

COLUMNS TO FILL IN:
====================

1. human_intent (REQUIRED)
   Valid values (copy exactly, no spaces):
   - login_problem        → Can't login, forgot password, locked out
   - billing_issue        → Charge dispute, refund, subscription cancel
   - bug_report           → App crash, error, broken feature, glitch
   - feature_request      → Suggestions, "please add", "why can't you"
   - account_suspension   → Suspended, banned, account disabled
   - content_issue        → Missing song/app/content, download problem
   - app_performance      → Slow, lag, battery drain, crashing on use
   - connectivity         → Offline, won't sync, server error
   - general_inquiry      → How-to questions, general help requests
   - positive_feedback    → Thanks, compliments, love the product

2. should_escalate (REQUIRED)
   Valid values: Y or N
   
   Escalate (Y) when:
   - Billing disputes involving money (always escalate)
   - Account suspension appeals (always escalate)
   - Legal threats or mentions of lawyers
   - User mentions harm to self
   - Third escalation attempt from same user (visible in thread)
   - Very high anger/frustration level
   - Privacy/data breach concerns
   
   Auto-handle (N) when:
   - Standard tech support (bugs, performance)
   - General how-to questions
   - Positive feedback
   - Feature requests
   - First-touch connectivity or login issues

3. escalation_reason (fill ONLY if should_escalate = Y)
   Write a brief reason in plain English. Examples:
   - "billing dispute - charge without consent"
   - "account suspension - user claims wrongful ban"
   - "legal threat - user mentions filing complaint"
   - "high anger - repeat contact, threatening to leave"

4. notes (OPTIONAL)
   Any other observations. Examples:
   - "ambiguous - could be bug_report or connectivity"
   - "sarcastic tone - not actually positive feedback"

TIPS FOR LABELLING:
===================
- Look at BOTH customer_message AND brand_reply for context
- The suggested_intent is a keyword guess — it's often right but not always
- When in doubt between two intents, pick the PRIMARY issue
- Aim for consistency: if two messages say similar things, label them the same
- Don't overthink it — your gut is often right

QUALITY CHECK:
==============
After labelling all 200, run:
  python src/classifier/build_eval_set.py validate

This checks for:
- Missing labels
- Invalid intent names (typos)
- Missing should_escalate values
"""
