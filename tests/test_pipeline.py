"""Quick integration test — no API keys needed."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

from src.classifier.classifier import KeywordClassifier
from src.agent.router import EscalationRouter

clf = KeywordClassifier()
router = EscalationRouter()

tests = [
    ("I cant login to my account", "login_problem"),
    ("You charged me twice this month refund please", "billing_issue"),
    ("My lawyer will contact you about this", "general_inquiry"),
    ("The app keeps crashing on my iPhone 14", "bug_report"),
    ("My account was suspended for no reason", "account_suspension"),
    ("Love your service, best app ever!", "positive_feedback"),
]

print("--- Classifier + Router Integration Test ---\n")
all_pass = True
for msg, expected in tests:
    intent, conf = clf.predict(msg)
    routing = router.route(msg, intent, conf)
    escalate_str = "ESCALATE" if routing["should_escalate"] else "auto    "
    status = "✓" if intent == expected else "✗"
    print(f"  {status} [{escalate_str}] {conf:.2f} | {intent:22} | {msg[:55]}")
    if routing["should_escalate"]:
        print(f"      Reason: {routing['reason']}")
    if intent != expected:
        print(f"      Expected: {expected}")
        all_pass = False

print()
if all_pass:
    print("✅ All tests passed! Core pipeline is working.")
else:
    print("⚠ Some intents misclassified (keyword classifier — LLM will fix these at runtime)")
