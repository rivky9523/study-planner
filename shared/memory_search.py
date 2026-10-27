"""Keyword ranking over memory entries (used by the recall_memory tool and by
the API harness when it picks which memories go into the turn context)."""
import re

# Hebrew/English hints so a Hebrew question can still reach english keys.
SYNONYMS = {
    "study_time": ["בוקר", "בבוקר", "בקרים", "ערב", "בערב", "ערבים", "צהריים", "שעה", "מתי", "time", "morning", "evening"],
    "max_hours_per_day": ["שעות", "כמה", "hours", "max"],
    "days_off": ["שבת", "שישי", "חופש", "יום", "ימים", "off", "day", "days"],
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 1}


def rank(entries: dict, query: str) -> list[tuple[str, dict]]:
    q = _words(query)
    scored = []
    for key, e in entries.items():
        vocab = _words(key.replace("_", " ")) | _words(str(e["value"])) | set(SYNONYMS.get(key, []))
        score = len(q & vocab)
        if e["kind"] == "preference":
            score += 0.5  # preferences are small and almost always relevant
        if score > 0:
            scored.append((score, key, e))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [(k, e) for _, k, e in scored]
