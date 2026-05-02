# Topics that should always be escalated to a human agent
HARD_ESCALATION_KEYWORDS = [
    "fraud", "unauthorized", "hacked", "compromised",
    "lawsuit", "legal action", "legal notice", "court", "police",
    "death", "deceased", "discrimination", "harassment", "abuse", "threat",
]

# Soft risk — escalate only if retrieval also fails
SOFT_ESCALATION_KEYWORDS = [
    "stolen", "chargeback", "dispute", "refund",
    "cannot log in", "locked out", "account suspended", "banned",
]

DOMAIN_HINTS = {
    "hackerrank": ["hackerrank", "coding test", "assessment", "challenge", "interview", "hiring", "recruiter", "test link", "proctoring"],
    "claude": ["claude", "anthropic", "ai assistant", "subscription", "pro plan", "claude.ai", "artifact", "conversation"],
    "visa": ["visa", "card payment", "transaction", "merchant", "chargeback", "cvv", "atm withdrawal", "international payment"],
}

# Maps (domain, keyword) -> product_area label.
# Checked in order — first match wins.
PRODUCT_AREA_RULES: list[tuple[str, list[str], str]] = [
    # HackerRank
    ("hackerrank", ["variant", "test version", "default role", "best practice", "new test",
                    "active", "expir", "start date", "end date", "time limit", "duration",
                    "reinvite", "re-invite", "extra time", "accommodation", "proctoring",
                    "webcam", "plagiarism", "submission", "coding challenge", "assessment",
                    "test link", "hiring", "recruiter", "candidate", "invite", "score",
                    "IDE", "environment", "report"],               "screen"),
    ("hackerrank", ["community", "google login", "delete account", "password",
                    "sign up", "signup", "profile", "username"],   "community"),
    ("hackerrank", ["billing", "subscription", "invoice", "charge", "payment",
                    "plan", "upgrade", "downgrade"],               "billing"),

    # Claude
    ("claude",     ["conversation", "chat history", "delete", "private", "temporary",
                    "memory", "forget", "export"],                  "privacy"),
    ("claude",     ["artifact", "render", "image", "upload", "file"],  "artifacts"),
    ("claude",     ["subscription", "pro", "billing", "charge",
                    "invoice", "plan", "team"],                    "billing"),
    ("claude",     ["api", "rate limit", "token", "sdk", "integration"],  "api"),
    ("claude",     ["slow", "down", "outage", "not responding",
                    "error", "bug", "broken"],                     "reliability"),

    # Visa
    ("visa",       ["traveller", "travelers", "cheque", "check",
                    "citicorp", "foreign", "abroad", "international",
                    "lisbon", "lost abroad", "stolen abroad"],     "travel_support"),
    ("visa",       ["lost", "stolen", "report", "block", "emergency",
                    "replace", "replacement"],                     "general_support"),
    ("visa",       ["decline", "declined", "transaction", "merchant",
                    "contactless", "atm", "pin", "cvv",
                    "payment", "purchase", "charge"],              "card_services"),
    ("visa",       ["dispute", "chargeback", "refund",
                    "unauthorized", "fraud"],                      "disputes"),

    # Cross-domain fallbacks
    ("general",    ["down", "outage", "not loading", "inaccessible",
                    "broken", "crash"],                            "reliability"),
    ("general",    ["account", "login", "password", "access",
                    "locked", "suspended"],                        "account_access"),
]

# Request type detection signals
BUG_SIGNALS = [
    "down",
    "not loading",
    "inaccessible",
    "pages are accessible",
    "crash",
    "broken",
    "outage",
    "not working",
    "error",
    "500",
    "404",
    "unexpected behaviour",
    "unexpected behavior",
    "problem",
    "issue",
    "failed",
    "fail",
]

FEATURE_REQUEST_SIGNALS = [
    "would be great if", "request", "wish you could", "feature", "suggestion",
    "suggest adding", "can you add", "please add", "it would help if",
]

INVALID_SIGNALS = [
    "thank you", "thanks for", "great service", "you're welcome",
    "celebrity question", "delete files"
]

TRAINING_DATA = [
    # hackerrank
    ("my coding test link expired", "hackerrank"),
    ("the assessment won't load in my browser", "hackerrank"),
    ("I cannot submit my solution", "hackerrank"),
    ("the proctoring webcam is not working", "hackerrank"),
    ("recruiter sent me a broken test link", "hackerrank"),
    ("my hackerrank score is wrong", "hackerrank"),
    ("interview challenge is showing an error", "hackerrank"),
    ("coding challenge time ran out too early", "hackerrank"),
    ("I failed the hiring assessment unfairly", "hackerrank"),
    ("test environment keeps crashing", "hackerrank"),
    ("IDE inside hackerrank is broken", "hackerrank"),
    ("my candidate report is missing", "hackerrank"),
    ("plagiarism flag on my submission is incorrect", "hackerrank"),
    ("remote proctoring disconnected mid-test", "hackerrank"),
    ("cannot access my hackerrank dashboard", "hackerrank"),

    # claude
    ("claude is not responding to my messages", "claude"),
    ("my claude pro subscription was charged twice", "claude"),
    ("how do I cancel my claude.ai plan", "claude"),
    ("artifacts are not rendering in the conversation", "claude"),
    ("context window limit reached too quickly", "claude"),
    ("I lost my conversation history in claude", "claude"),
    ("anthropic charged me but I cannot access pro", "claude"),
    ("claude is giving wrong answers about recent events", "claude"),
    ("how do I export my claude conversations", "claude"),
    ("claude keeps forgetting things mid chat", "claude"),
    ("my claude account is locked", "claude"),
    ("claude api rate limit exceeded", "claude"),
    ("image upload is not working in claude", "claude"),
    ("claude is much slower than usual today", "claude"),
    ("team plan seat not showing up for my colleague", "claude"),

    # visa
    ("my visa card was declined at the supermarket", "visa"),
    ("international transaction fee on my statement", "visa"),
    ("I did not make this visa purchase", "visa"),
    ("ATM would not give me cash with my visa card", "visa"),
    ("how do I dispute a charge on my visa card", "visa"),
    ("contactless payment not working on my card", "visa"),
    ("visa card expired and new one not arrived", "visa"),
    ("my card PIN is blocked after wrong attempts", "visa"),
    ("merchant charged me twice on visa", "visa"),
    ("online payment declined but card is valid", "visa"),
    ("visa virtual card not accepted by merchant", "visa"),
    ("foreign currency conversion rate seems wrong", "visa"),
    ("card transaction pending for too many days", "visa"),
    ("lost my visa card abroad what do I do", "visa"),
    ("cvv not accepted during checkout", "visa"),

    # general
    ("hello I have a question", "general"),
    ("please help me with my account", "general"),
    ("I need support urgently", "general"),
    ("something is not working properly", "general"),
    ("can someone assist me please", "general"),
]