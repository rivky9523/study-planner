# study-planner: סוכן אחד, שני harnesses

מתכנן לימודים אישי שבנוי פעמיים:
- `API/`: על ה-Messages API הגולמי. כל ה-harness כתוב ידנית ב-`API/harness/`.
- `SDK/`: על Claude Agent SDK. Claude Code מריץ את ה-harness, והקוד רק מגדיר אותו.

קוד הדומיין (כלים ונתונים) נמצא ב-`shared/`, ושתי הגרסאות משתמשות בו.

## מבנה

```
shared/      domain.py (7 כלים + system prompt), store.py (מצב בדיסק), memory_search.py,
             scenario.py (בדיקות, תוצאות, ספירת שורות), data/syllabi/*.md
API/         main.py  agent.py  providers/gemini.py  harness/{loop,tools,history,memory,context,compaction,caching,tokens}.py
SDK/         main.py  agent.py  tools.py        (כל שורה שמחליפה קובץ harness מסומנת # replaces: <part>)
tests/       בדיקות offline עם client מדומה, בלי מפתח
scenario.json  7 פניות בשני סשנים + 9 בדיקות pass/fail
results/     api.json  sdk.json
scripts/make_report.py   ממלא את הטבלאות ב-REPORT.md
```

## התקנה

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## בדיקות offline (בלי מפתח)

```bash
python -m pytest -q
```

## ריצה

**API**: צריך מפתח מ-Claude Console, כמשתנה סביבה בלבד:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python API/main.py --scenario scenario.json --out results/api.json
unset ANTHROPIC_API_KEY
```

**API עם מפתח Gemini (לפיתוח ובדיקות, יש מסלול חינמי)**:
```bash
export GEMINI_API_KEY=...
python API/main.py --provider gemini --out results/api-gemini.json
# מודל אחר: --model <model-id>
```
אותו harness בדיוק. מתאם קטן ב-`API/providers/gemini.py` מתרגם בין הפורמט של Anthropic לפורמט של Gemini.
שימו לב: ההשוואה שהתרגיל מבקש היא Claude מול Claude, ולכן `results/api.json` צריך להגיע מריצה עם Anthropic.

**SDK**: רץ על ה-login של Claude Code (`claude` ואז `/login`). המפתח לא צריך להיות מוגדר:
```bash
python SDK/main.py --scenario scenario.json --out results/sdk.json
```

**דוח:**
```bash
python scripts/make_report.py
```

כל ריצה מוחקת את `state/api` או `state/sdk` ומתחילה ממצב נקי. בריצת ה-API כל פעולה מודפסת עם שם הקובץ שביצע אותה, למשל `[loop.py] model call #2`.
