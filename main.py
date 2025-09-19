from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import os
from dotenv import load_dotenv  # pip install python-dotenv
import logging
load_dotenv()

user_state = {
    "current_step": 0,
    "last_question": None,
    "awaiting_response": False,
    "module_progress": {},
}

def continue_conversation(user_input=None):
    if user_input:
        user_state["awaiting_response"] = False
        # Save user response somewhere if needed
    if not user_state["awaiting_response"]:
        next_question = get_next_question()
        user_state["last_question"] = next_question
        user_state["awaiting_response"] = True
        return next_question
    else:
        return "Please answer the previous question first."

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# ---------- Reply helper (logs assistant turns to session history) ----------
def _reply(state: Dict[str, Any], text: str) -> Dict[str, str]:
    try:
        state.setdefault("history", []).append({"role": "assistant", "content": text})
    except Exception:
        pass
    return {"message": text}
# ---------- UI/formatting helpers ----------
def _soft_wrap(text: str, max_len: int = 420) -> str:
    """
    Insert paragraph breaks so long blocks are easier to read in the UI.
    We split on sentence boundaries and rebuild into short paragraphs.
    """
    import re as _r
    sents = _r.split(r'(?<=[.!?])\s+', text.strip())
    out, cur = [], []
    acc = 0
    for s in sents:
        if not s:
            continue
        if acc + len(s) > max_len and cur:
            out.append(" ".join(cur))
            cur, acc = [s], len(s)
        else:
            cur.append(s)
            acc += len(s) + 1
    if cur:
        out.append(" ".join(cur))
    return "\n\n".join(out)

def _de_heading(text: str) -> str:
    """
    Remove stiff headings like 'Definition:', 'Parts:', 'Example:', 'Quick check:'.
    Replace with natural phrasing + light bulleting.
    """
    repl = {
        r'^\s*(definition)\s*[:\-]\s*': '',
        r'^\s*(parts?)\s*[:\-]\s*': 'Two main ideas: ',
        r'^\s*(example)\s*[:\-]\s*': 'For example, ',
        r'^\s*(quick\s*check)\s*[:\-]\s*': 'Quick check: ',
    }
    lines = [l.strip() for l in text.splitlines()]
    out = []
    import re as _r
    for line in lines:
        ln = line
        for pat, sub in repl.items():
            ln = _r.sub(pat, sub, ln, flags=_r.I)
        # Turn simple (1)(2)(3) into bullets for readability
        ln = _r.sub(r'\(\s*1\s*\)', '•', ln)
        ln = _r.sub(r'\(\s*2\s*\)', '•', ln)
        ln = _r.sub(r'\(\s*3\s*\)', '•', ln)
        out.append(ln)
    cleaned = "\n".join(out).strip()
    cleaned = _r.sub(r'\n{3,}', '\n\n', cleaned)
    return _soft_wrap(cleaned)

def _naturalize(text: str) -> str:
    """Pipeline to make any teaching text friendlier for the UI."""
    return _de_heading(text)

from pdfminer.high_level import extract_text as pdfminer_extract  # pip install pdfminer.six

# Optional OpenAI client (for reasoning + retrieval)
try:
    from openai import OpenAI  # pip install openai
except Exception:
    OpenAI = None

# ----------------- App & CORS -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- Static assets (for background textures, icons, etc.) ----
STATIC_DIR = (Path(__file__).parent / "static").resolve()
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Back-compat route: serve the metal texture at the old root path if requested
@app.get("/silver-textured-sheet-metal-texture.jpeg")
def _metal_texture():
    file_path = STATIC_DIR / "silver-textured-sheet-metal-texture.jpeg"
    if file_path.exists():
        return FileResponse(str(file_path))
    return FileResponse(str(file_path), status_code=404)

# Optional: avoid noisy 404s from browsers requesting a favicon
@app.get("/favicon.ico")
def _favicon():
    fav = STATIC_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(str(fav))
    # Return 204 No Content if not present
    from fastapi import Response
    return Response(status_code=204)

# ----------------- Design guardrails -----------------
DESIGN_RULES = (
    "Follow Hendrick’s course skeleton and rationale. "
    "Initiate with an opening message and require confirmation to begin. "
    "Teach by small steps (scaffolding), use plain language, and check for understanding. "
    "Personalize when possible (echo the learner’s words), but stay on topic. "
    "Guided Breakdown: define → parts → short example → one quick check. "
    "Text only. Keep paragraphs short. Bold only a few key terms."
)

OPENING_MESSAGE = (
    "Hello there — today I’ll be leading you through a course about AI literacy.\n\n"
    "We’ll cover four modules:\n"
    "• **Foundational AI** — what AI is, how it works, and common limits.\n"
    "• **Prompting & Conversational AI** — how to give clear, structured instructions.\n"
    "• **Applied AI Technology** — real tools, choosing the right one, and the basic pipeline.\n"
    "• **Ethics & Responsible AI** — bias, privacy, and guardrails.\n\n"
    "Before we start, please confirm you’ve taken the two pre-course quizzes your instructor shared. "
    "When you’re ready, type 'begin' and I’ll kick off Module 1."
)

FRIENDLY_MODULE_NAMES = {
    0: "Foundational AI",
    1: "Prompting & Conversational AI",
    2: "Applied AI Technology",
    3: "Ethics & Responsible AI",
}

FRIENDLY_UNIT_LINES = {
    0: "First, we’ll begin with **Foundational AI** — what AI is, how it works, and common limits.",
    1: "Next, we’ll practice **Prompting & Conversational AI** — how to give clear, structured instructions.",
    2: "Then we’ll explore **Applied AI Technology** — real tools, choosing the right one, and the basic pipeline.",
    3: "Finally, we’ll cover **Ethics & Responsible AI** — bias, privacy, and guardrails.",
}

# ----------------- Knowledge loading (PDF -> text) -----------------
KB_TEXT = ""
KB_DIR_ENV = os.getenv("KB_DIR", "")
CANDIDATE_PATHS: List[Path] = []
if KB_DIR_ENV:
    CANDIDATE_PATHS.append(Path(KB_DIR_ENV))
CANDIDATE_PATHS += [
    Path(__file__).parent / "Course Content",
    Path.home() / "Downloads" / "AI Literacy Chatbot" / "Course Content",
    Path.home() / "Documents" / "ai-chatbot-draft" / "AI Literacy Chatbot" / "Course Content",
    Path(__file__).parent / "course-content",
]

def read_pdf(path: Path) -> str:
    try:
        text = pdfminer_extract(str(path))
        print(f"📘 Total extracted from {path.name}: {len(text)} characters")
        return text
    except Exception as e:
        print(f"❌ Failed to extract text from {path.name}: {e}")
        return ""

PDF_TEXTS: List[str] = []
for base in CANDIDATE_PATHS:
    if base.exists() and base.is_dir():
        for p in base.rglob("*.pdf"):
            print(f"📄 Loading PDF: {p}")
            PDF_TEXTS.append(read_pdf(p))

_joined = "\n\n".join([t for t in PDF_TEXTS if t.strip()])
MAX_KB_CHARS = 1_200_000
KB_TEXT = _joined[:MAX_KB_CHARS]

# ----------------- Optional retrieval via embeddings -----------------
CLIENT: Optional[OpenAI] = None
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

if OpenAI and os.getenv("OPENAI_API_KEY"):
    CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if CLIENT:
    logging.info(f"OpenAI client active: chat_model={CHAT_MODEL}")
else:
    logging.warning("OpenAI client NOT active; running offline fallback (no API key or openai package missing).")
def chunk(text: str, size: int = 600, overlap: int = 150) -> List[str]:
    """Naive character-based chunking with overlap."""
    parts: List[str] = []
    i = 0
    n = len(text)
    step = max(1, size - overlap)
    while i < n:
        parts.append(text[i:i+size])
        i += step
    return [s for s in parts if s.strip()]

class KBIndex:
    def __init__(self):
        self.chunks: List[str] = []
        self.vecs: List[List[float]] = []
        self.model = EMBED_MODEL

    def build(self, text: str):
        self.chunks = []
        print("⏳ Sending embedding request batch to OpenAI...")
        self.vecs = []
        if not CLIENT or not text.strip():
            print(f"✅ Embedding complete — {len(self.vecs)} vectors from {len(self.chunks)} chunks")
            return
        all_chunks = chunk(text, size=700, overlap=100)
        MAX_CHUNKS = 1500
        if len(all_chunks) > MAX_CHUNKS:
            all_chunks = all_chunks[:MAX_CHUNKS]
        self.chunks = all_chunks
        BATCH_SIZE = 100
        for i in range(0, len(self.chunks), BATCH_SIZE):
            print(f"🧪 Sending batch {i}–{i + BATCH_SIZE} to OpenAI...")
            batch = self.chunks[i:i + BATCH_SIZE]
            try:
                resp = CLIENT.embeddings.create(model=self.model, input=batch)
                self.vecs.extend(d.embedding for d in resp.data)
            except Exception as e:
                print(f"❌ Embedding batch failed at i={i}")
                import traceback
                traceback.print_exc()
        print(f"✅ Embedding complete — {len(self.vecs)} vectors from {len(self.chunks)} chunks")

    def _cos(self, a: List[float], b: List[float]) -> float:
        s_ab = sum(x*y for x, y in zip(a, b))
        s_a = sum(x*x for x in a) ** 0.5
        s_b = sum(x*x for x in b) ** 0.5
        if s_a == 0 or s_b == 0:
            return 0.0
        return s_ab / (s_a * s_b)

    def search(self, query: str, k: int = 5) -> List[str]:
        if not CLIENT or not self.vecs:
            return []
        q = CLIENT.embeddings.create(model=self.model, input=query).data[0].embedding
        sims = [(self._cos(v, q), i) for i, v in enumerate(self.vecs)]
        sims.sort(reverse=True)
        return [self.chunks[i] for _, i in sims[:k]]
def kb_supporting_sentences(query: str, k: int = 2) -> List[str]:
    """
    Return up to k short supporting sentences (40–180 chars) from the course readings.
    Uses the embedding index when available; otherwise falls back to a light
    substring search over KB_TEXT. No citations are added here.
    """
    import re as _r
    out: List[str] = []

    # Try vector search first
    try:
        hits = KB.search(query, k=4)
    except Exception:
        hits = []

    def _pick_sents(text: str):
        sents = _r.split(r'(?<=[.!?])\s+', (text or "").strip())
        for s in sents:
            s = s.strip()
            if 40 <= len(s) <= 180 and not s.lower().startswith("figure"):
                out.append(s)
                if len(out) >= k:
                    return True
        return False

    for h in hits:
        if _pick_sents(h):
            break
    if out:
        return out[:k]

    # Fallback: scan KB_TEXT without embeddings
    if KB_TEXT:
        words = [w.lower() for w in _r.findall(r"[A-Za-z]{4,}", query)]
        sents = _r.split(r'(?<=[.!?])\s+', KB_TEXT)
        for s in sents:
            s_l = s.lower().strip()
            if 40 <= len(s) <= 180 and any(w in s_l for w in words):
                out.append(s.strip())
                if len(out) >= k:
                    break
    return out[:k]
KB = KBIndex()
KB.build(KB_TEXT)
logging.info(f"KB loaded: {len(KB.chunks)} chunks from PDFs")

# ----------------- Course -----------------
COURSE: List[Dict[str, Any]] = [
    {
        "title": "Module 1: Understanding AI",
        "overview": (
            "Artificial intelligence (AI) is the simulation of human intelligence in machines through data processing and algorithms. "
            "It has specific capabilities and limitations that are often misunderstood. Knowing what AI can and cannot do sets realistic expectations."
        ),
        "subtopics": [
            {"title": "Definition & Types of AI"},
            {"title": "How AI Works: data, algorithms, predictions"},
            {"title": "Capabilities, limits, and misconceptions"},
        ],
        "prior": [
            "Have you learned about how AI works before? (yes/no)",
            "Do you think AI can think or feel like a human? (yes/no)",
            "Name one thing you think AI is NOT able to do. (short answer)",
        ],
    },
    {
        "title": "Module 2: Prompting & Conversational AI",
        "overview": (
            "Prompting is how you phrase instructions to guide an AI chatbot’s response. Clear prompts set roles, define tasks, and provide context."
        ),
        "subtopics": [
            {"title": "Prompt basics and anatomy (role, task, context)"},
            {"title": "Iterative refinement and examples"},
            {"title": "Common pitfalls and how to debug prompts"},
        ],
        "prior": [
            "Have you used an AI chatbot before (e.g., ChatGPT)? (yes/no)",
            "How often do you use it? (rarely / sometimes / often)",
            "Do you think about how you phrase your questions? (yes/no)",
        ],
    },
    {
        "title": "Module 3: Applied AI Technology",
        "overview": (
            "AI applicational technology refers to software that uses AI to perform specific tasks, often autonomously. This module focuses on AI tools for specific domains."
        ),
        "subtopics": [
            {"title": "AI tools in education, healthcare, and business"},
            {"title": "Choosing the right tool: capabilities vs. needs"},
            {"title": "Basic AI pipeline: data → training → deployment"},
        ],
        "prior": [
            "Have you used an app or tool you believe is powered by AI? (yes/no)",
            "If yes, what did it do? (short answer)",
            "Did you know it was using AI at the time? (yes/no)",
        ],
    },
    {
        "title": "Module 4: Ethics & Responsible AI",
        "overview": (
            "Ethical AI involves fairness, privacy, transparency, and accountability. We consider risks like bias, misinformation, and misuse, and how to act responsibly."
        ),
        "subtopics": [
            {"title": "Bias and fairness"},
            {"title": "Privacy and data protection"},
            {"title": "Responsible use and governance"},
        ],
        "prior": [
            "Do you think AI systems can be biased? (yes/no)",
            "Should AI explain its decisions in high-stakes settings? (yes/no)",
            "Name one risk of using AI irresponsibly. (short answer)",
        ],
    },
]

# ----------------- Session state -----------------
sessions: Dict[str, Dict[str, Any]] = {}

# Require answering the current check before advancing to the next subtopic
REQUIRE_ANSWER_BEFORE_NEXT = True

class ChatIn(BaseModel):
    message: str
    session_id: str

ACK_PAT = re.compile(
    r"(?:^|\b)(?:y|yes|yeah|yep|ok|okay|alright|sure|cool|got it|gotcha|understood|makes sense|that makes sense|sounds good|works for me|i see|follow|right)(?:\b|$)",
    re.I
)
def is_ack(msg: str) -> bool:
    return bool(ACK_PAT.search((msg or "").strip()))

import re as _re
def has_cmd(msg: str, word: str) -> bool:
    if not msg or not word:
        return False
    return _re.search(rf"\b{_re.escape(word)}\b", (msg or "").strip().lower()) is not None

CLAIM_ANSWER_PHRASES = {
    "i just did", "i already did", "i did", "i answered", "i just answered",
    "i said it", "i told you", "i gave it", "i responded", "i responded already"
}
def is_claim_answer(msg: str) -> bool:
    return (msg or "").strip().lower() in CLAIM_ANSWER_PHRASES

NON_ANSWERS = {"", "next", "begin", "ready", "learn", "start", "go", "continue", "skip"}
def is_non_answer(msg: str) -> bool:
    return (msg or "").strip().lower() in NON_ANSWERS

def _is_yes_no(text: str) -> Optional[bool]:
    t = (text or "").strip().lower()
    if t in {"y", "yes", "yeah", "yep"}:
        return True
    if t in {"n", "no", "nope"}:
        return False
    return None

def _valid_short_answer(text: str, min_words: int = 2) -> bool:
    import re as _r
    t = (text or "").strip()
    if t.lower() in NON_ANSWERS:
        return False
    words = [w for w in _r.findall(r"[A-Za-z]+", t)]
    return len(words) >= min_words

def normalize(msg: str) -> str:
    return (msg or "").strip().lower()

# ---------- Quick-check question banks ----------
CHECK_QUESTIONS = {
    "Definition & Types of AI": [
        "One short phrase: how is **narrow AI** different from **human intelligence**?",
        "If a spam filter is narrow AI, name one thing a human can do that the filter can’t (one short phrase).",
        "Fill the blanks: narrow AI = **specialist** in one job; humans = **general** learners — say one difference in your own words.",
    ],
    "How AI Works: data, algorithms, predictions": [
        "Choose one to focus on — type **data**, **training**, or **prediction** (one word).",
        "Tell me which part you want explained more: **data**, **training**, or **prediction** (one word).",
        "Which step feels foggiest right now: **data**, **training**, or **prediction**? (one word)",
    ],
    "Capabilities, limits, and misconceptions": [
        "Give one **limit** of today’s AI systems (one short sentence).",
        "Name one situation where an AI making up facts — a '**hallucination**' — would cause problems.",
        "Share one **misconception** people have about AI."
    ],
    "Prompt basics and anatomy (role, task, context)": [
        "Draft a quick prompt using **role + task + context**.",
        "What **role** would you assign the model for your next task?",
        "Give one piece of **context** you’d include in a good prompt."
    ],
    "Iterative refinement and examples": [
        "What would you try first when a prompt underperforms?",
        "Share a prompt you’d like to **tighten**.",
        "Which knob would you tweak: be more **specific**, add **examples**, or split **steps**?"
    ],
    "Common pitfalls and how to debug prompts": [
        "Name one **pitfall** to avoid when prompting.",
        "What **constraint** would you add (length, format, steps)?",
        "What’s one detail of **context** you’d include next time?"
    ],
    "AI tools in education, healthcare, and business": [
        "Which domain is most relevant to you (education / healthcare / business)?",
        "Name a task where an AI tool might help in your field.",
        "What **outcome** would you expect from a domain tool?"
    ],
    "Choosing the right tool: capabilities vs. needs": [
        "What’s one **non-negotiable** requirement for your use case?",
        "Is **privacy**, **accuracy**, or **integration** your top constraint?",
        "Describe a failure you cannot accept (so we pick the right tool)."
    ],
    "Basic AI pipeline: data → training → deployment": [
        "Pick a stage you want an example of: **data**, **training**, or **deployment**.",
        "Which stage do you own today in your work?",
        "Where would you put **monitoring** in this pipeline?"
    ],
    "Bias and fairness": [
        "How would you **test** a model for subgroup gaps?",
        "Name one way **bias** can enter a system.",
        "What **metric** would you track for fairness?"
    ],
    "Privacy and data protection": [
        "What personal data in your projects would need **masking**?",
        "Would you prefer **on-device** or **server-side** processing? Why?",
        "Name one **access control** you’d enforce."
    ],
    "Responsible use and governance": [
        "Where should **human review** sit in your workflow?",
        "Name one **guardrail** you’d add before deployment.",
        "When would you require an **explanation** from the system?"
    ],
}

def next_check_question(state: Dict[str, Any]) -> str:
    m = COURSE[state["module"]]
    sub = m["subtopics"][state["subtopic"]]["title"]
    bank = CHECK_QUESTIONS.get(sub, ["What would you like me to clarify next?"])
    state["retry_count"] = (state.get("retry_count", 0) + 1) % len(bank)
    return bank[state["retry_count"]]

def get_state(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "module_turns": 0,              # counts tutor teaching messages this module
            "module_turn_goal": 15,         # target tutor messages per module (excludes Q&A answers)
            "how_steps_done": set(),        # tracks covered steps: {"data","training","prediction"}
            "stage": "intro",      # intro → overview → prior → learn → teaching → await_next → done
            "module": 0,
            "prior_idx": 0,
            "subtopic": 0,
            "prior_answers": [],
            "prior_tries": [0, 0, 0],
            "retry_count": 0,
            "level": "standard",
            "mastery": {},
            "awaiting_check": False,
            "last_subtopic": None,
            "misconceptions": set(),
            "review_queue": [],
            "history": [],
        }
    return sessions[session_id]

# ---- Rubrics ----
SUBTOPIC_RUBRICS: Dict[str, Dict[str, Any]] = {
    "Definition & Types of AI": {
        "correct_any": [
            ["narrow", "single", "task"],
            ["single", "job"],
            ["one", "job"],
            ["specific", "job"],
            ["single", "task"],
            ["human", "general", "many"],
            ["specialist", "narrow"],
            ["human", "many"],
            ["versatile"],
            ["adaptive"],
            ["not", "adaptive"],
            ["not", "adapt"],         # added
            ["humans", "many", "jobs"],
            ["not", "versatile"],
        ],
        "misconceptions": [
            (re.compile(r"\b(ai|it)\s*(can)?\s*(think|feel|have\s*emotions)\b", re.I), "thinks AI can think/feel like a human"),
            (re.compile(r"\bnarrow\s*ai.*(is|=).*(general|human[-\s]*level)\b", re.I), "thinks narrow AI is general/human-level"),
        ],
    },
    "How AI Works: data, algorithms, predictions": {
        "correct_any": [
            ["data", "train"],
            ["train", "predict"],
            ["data", "predict"],
        ],
        "misconceptions": [],
    },
    "Capabilities, limits, and misconceptions": {
        "correct_any": [
            ["hallucination"],
            ["bias"],
            ["out", "of", "distribution"],
            ["not", "general"],
        ],
        "misconceptions": [],
    },
    "Prompt basics and anatomy (role, task, context)": {
        "correct_any": [["role", "task", "context"], ["constraints"], ["example"]],
        "misconceptions": [],
    },
    "Iterative refinement and examples": {
        "correct_any": [["iterate"], ["refine"], ["test", "adjust"]],
        "misconceptions": [],
    },
    "Common pitfalls and how to debug prompts": {
        "correct_any": [["vague"], ["missing", "context"], ["too", "many", "steps"]],
        "misconceptions": [],
    },
    "AI tools in education, healthcare, and business": {
        "correct_any": [["education"], ["healthcare"], ["business"]],
        "misconceptions": [],
    },
    "Choosing the right tool: capabilities vs. needs": {
        "correct_any": [["privacy"], ["accuracy"], ["integration"]],
        "misconceptions": [],
    },
    "Basic AI pipeline: data → training → deployment": {
        "correct_any": [["data"], ["training"], ["deployment"]],
        "misconceptions": [],
    },
    "Bias and fairness": {
        "correct_any": [["bias"], ["subgroup"], ["fairness"]],
        "misconceptions": [],
    },
    "Privacy and data protection": {
        "correct_any": [["privacy"], ["masking"], ["encryption"], ["access", "control"]],
        "misconceptions": [],
    },
    "Responsible use and governance": {
        "correct_any": [["human", "review"], ["guardrail"], ["explain"]],
        "misconceptions": [],
    },
}

def _keyword_score(text: str, patterns: List[List[str]]) -> float:
    t = (text or "").lower()
    best = 0.0
    for group in patterns:
        present = sum(1 for w in group if w.lower() in t)
        if present:
            best = max(best, present / max(1, len(group)))
    return best
# Detect if the learner explicitly mentions adaptivity / specialist-vs-general ideas
_DEFN_ADAPT_TOKENS = (
    "adapt", "adaptive", "not adaptive", "specialist", "generalist",
    "single job", "one job", "specific job", "versatile", "not versatile",
    "humans can learn many", "many jobs", "general"
)

def mentions_adaptivity(text: str) -> bool:
    t = (text or "").lower()
    return any(tok in t for tok in _DEFN_ADAPT_TOKENS)
# ---- Optional LLM-backed grading for adaptive feedback ----
def grade_with_llm(state: Dict[str, Any], answer: str) -> Optional[float]:
    """
    Optional LLM-backed grading for adaptive feedback.
    Returns a float score in [0,1] or None if grading fails/unavailable.
    Stores a one-sentence feedback string in state['last_llm_feedback'] when available.
    """
    if not CLIENT:
        return None
    try:
        m = COURSE[state["module"]]
        sub = m["subtopics"][state["subtopic"]]["title"]
        rubric = SUBTOPIC_RUBRICS.get(sub, {})
        # Light retrieval to anchor grading
        try:
            sources = KB.search(f"{m['title']} — {sub}", k=4)
            context = "\n\n".join(sources)
        except Exception:
            context = ""
        import json as _json
        prompt = (
            "You are grading a one-sentence learner answer for an AI Literacy course.\n"
            f"Subtopic: {sub}\n"
            f"Rubric keyword groups (any ONE group suffices for full credit): {_json.dumps(rubric.get('correct_any', []))}\n"
            "Return STRICT JSON with keys: score (0..1) and short_feedback (<= 18 words, friendly, specific).\n"
            f"Learner answer: {answer}\n\n"
            "Course excerpts for grounding:\n" + context
        )
        resp = CLIENT.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=140,
        )
        import json as _json2
        js = _json2.loads(resp.choices[0].message.content.strip())
        score = float(js.get("score", 0))
        score = max(0.0, min(1.0, score))
        fb = str(js.get("short_feedback", "")).strip()
        state["last_llm_feedback"] = fb
        return score
    except Exception:
        # On any parsing/API issue, fall back to keyword grading
        return None

# ---- physical vs intelligence confusion detector (for better hints)
_PHYSICAL_TERMS = {
    "touch","grab","hold","lift","move","walk","see with eyes","smell","taste",
    "hands","arms","body","robot","hardware","touch things","physically"
}
def mentions_physical_confusion(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in _PHYSICAL_TERMS)

# --- Helper: detect if a message looks like an answer to the current quick-check ---
def is_answer_like(state: Dict[str, Any], text: str) -> bool:
    """Light check: does this look like an answer to the current quick-check?"""
    if not text:
        return False
    t = (text or "").strip().lower()
    # tolerate common misspellings/partial words
    typo_map = {
        "predicton": "prediction", "prediciton": "prediction",
        "predicition": "prediction", "predicit": "prediction",
        "predicitn": "prediction", "pridiction": "prediction",
        "trainning": "training", "tranning": "training",
    }
    if t in typo_map:
        return typo_map[t]
    # partial stems
    if "predic" in t:
        return "prediction"
    if "train" in t:
        return "training"
    if "data" in t or "dataset" in t or "examples" in t:
        return "data"
    # Ignore nav commands
    if t in NON_ANSWERS:
        return False

    # Treat uncertainty as answer-like so we can give a hint
    if t in {"idk", "i don't know", "i dont know", "unsure", "no idea"}:
        return True

    # Keyword scoring against the current subtopic’s rubric
    try:
        m = COURSE[state["module"]]
        sub = m["subtopics"][state["subtopic"]]["title"]
        rubric = SUBTOPIC_RUBRICS.get(sub, {})
        score = _keyword_score(t, rubric.get("correct_any", []))
        return score >= 0.30
    except Exception:
        return False


# New function: grade_answer
def grade_answer(state: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """
    Normalize the learner answer, score it vs. rubric (and optionally LLM),
    update mastery + level, and return a dict with score/mastery/etc.
    """
    m = COURSE[state["module"]]
    sub = m["subtopics"][state["subtopic"]]["title"]
    rubric = SUBTOPIC_RUBRICS.get(sub, {})

    # --- Normalize synonyms
    answer_norm = (answer or "").lower()
    answer_norm = answer_norm.replace("cant", "can't").replace("wont", "won't")
    import re as _more
    answer_norm = _more.sub(r"\b(can't|cant|cannot)\s+adapt(ive)?\b", "not adaptive", answer_norm)
    answer_norm = answer_norm.replace("less adaptive", "not adaptive").replace("not as adaptive", "not adaptive")
    answer_norm = answer_norm.replace("adaptable", "adaptive").replace("flexible", "versatile")
    answer_norm = answer_norm.replace("isnt", "isn't").replace("isn’t", "isn't")
    for pat in ["isn't adaptive", "isn't as adaptive", "not as adaptive", "isnt adaptive", "isnt as adaptive"]:
        answer_norm = answer_norm.replace(pat, "not adaptive")
    answer_norm = (answer_norm
                   .replace("specific job", "single job")
                   .replace("one job", "single job")
                   .replace("built for a specific job", "single job")
                   .replace("built of a specific job", "single job")
                   .replace("generalist", "general")
                   .replace("versatility", "versatile"))

    # --- Hybrid grading
    llm_score = grade_with_llm(state, answer_norm)
    kw_score = _keyword_score(answer_norm, rubric.get("correct_any", []))
    
        # Accept example-style answers for Definition & Types (e.g., chatbots, spam filters, code assistants, recommenders)
    if sub == "Definition & Types of AI":
        example_terms = (
            "chatbot","chat bot","assistant","copilot","code","coding","github",
            "recommender","recommendation","spotify","netflix","spam","filter",
            "autocorrect","translate","translator","vision","x-ray","speech",
            "voice","siri","alexa","maps","gps","autopilot","photos","camera"
        )
        if any(term in answer_norm for term in example_terms):
            kw_score = max(kw_score, 0.75)
        # Heuristic: concrete high-risk contexts count as correct
    if sub == "Capabilities, limits, and misconceptions":
        context_hits = any(w in answer_norm for w in [
            "legal", "law", "attorney", "court",
            "medical", "doctor", "clinic", "hospital",
            "news", "journalism", "article", "report",
            "school", "assignment", "homework", "essay",
            "statistics", "stats", "number", "finance", "bank", "trading"
        ])
        if context_hits:
            kw_score = max(kw_score, 0.8)
    if llm_score is not None:
        score = max(kw_score, min(kw_score + 0.25, (kw_score + llm_score) / 2))
    else:
        score = kw_score

    # --- Misconceptions
    for rx, flag in rubric.get("misconceptions", []):
        if rx.search(answer or ""):
            state["misconceptions"].add(flag)

    # --- Adaptivity mention
    adapt_flag = mentions_adaptivity(answer_norm)

    # --- Mastery update (EMA)
    key = (state["module"], state["subtopic"])
    prior = state["mastery"].get(key, 0.3)
    alpha = 0.6
    new_mastery = (1 - alpha) * prior + alpha * score
    state["mastery"][key] = new_mastery
    if new_mastery >= 0.85:
        state["level"] = "advanced"
    elif new_mastery <= 0.35:
        state["level"] = "beginner"
    else:
        state["level"] = "standard"

    result = {
        "subtopic": sub,
        "score": score,
        "mastery": new_mastery,
        "level": state["level"],
        "adapt": adapt_flag,
    }
    # Add richer feedback without mentioning readings/citations.
    # Keep it short, specific, and tied to the current subtopic.
    if score >= 0.65:
        base = f"Nice — '{answer}' works."
    elif score >= 0.3:
        base = f"'{answer}' is close — let's tighten it."
    else:
        base = f"I see your thought: '{answer}'. Try this framing."

    if sub == "Definition & Types of AI":
        add = "Key idea: narrow AI is a **specialist** (one job); humans are **general** learners who adapt to many jobs."
        tip = "One example of narrow AI is a spam filter or recommender — built for a single task."
    elif sub == "How AI Works: data, algorithms, predictions":
        add = "The simple pipeline is **data → training → prediction**."
        tip = "Pick one step to zoom in on next: **data**, **training**, or **prediction**."
    elif sub == "Capabilities, limits, and misconceptions":
        add = "Today’s models can be brittle outside their training data and may **hallucinate**."
        tip = "Name one real setting where a hallucination would cause harm."
    else:
        add = "Good direction — keep linking your idea to the specific concept we’re on."
        tip = ""

    explanation = base + "\n\n" + add
    if tip:
        explanation += f"\nHint: {tip}"
    result['feedback'] = explanation
    state['last_result'] = result
    return result

def _ack_text(sub: str, answer: str, strength: str = "strong", adapt_flag: bool = False) -> str:
    # Provide a one-sentence deeper explanation from the rubric for every strong/affirmative feedback
    # For each subtopic, fetch the first rubric group and use it in the explanation
    rubric = SUBTOPIC_RUBRICS.get(sub, {})
    key_idea = ', '.join(next(iter(rubric.get('correct_any', [['specialist vs general']]))))
    if sub == "Definition & Types of AI":
        if strength == "strong":
            if adapt_flag:
                return ("Yes — exactly. Narrow AI is a **specialist** built for a single job, "
                        "whereas humans are **general** learners who can adapt across many jobs. "
                        "That difference — specialist vs. generalist — is the key idea we’ll keep using.\n\n"
                        f"Deeper: The core distinction is that narrow AI is optimized for one task and cannot generalize, while humans adapt and learn broadly.")
            else:
                return ("Yes — that works. You named a valid difference. We’ll build on it as we go.\n\n"
                        f"Deeper: The main idea is that narrow AI = specialist, humans = generalist and adaptive.")
        else:
            if adapt_flag:
                return ("Right idea — you’re pointing at the **specialist vs. generalist** gap. "
                        "Let’s lock that in and keep going.\n\n"
                        f"Deeper: Specialist AI can’t transfer its skills, but humans can tackle many new problems.")
            else:
                return ("Good — you’re on the right track. We’ll build on this as we continue.\n\n"
                        f"Deeper: The big difference is being able to adapt (humans) vs. doing one job (AI).")
        # For other subtopics, say one natural key idea instead of a token list
    if sub == "How AI Works: data, algorithms, predictions":
        key_line = "data → training → prediction"
    elif sub == "Capabilities, limits, and misconceptions":
        key_line = "strengths vs. limits (e.g., bias, hallucinations)"
    else:
        key_line = key_idea

    return (
        ("Nice — that’s right. Let’s build on it.\n\nDeeper: The key idea here is " + key_line + ".")
        if strength == "strong"
        else ("You’re close — let’s tighten it.\n\nDeeper: The key idea is " + key_line + ".")
    )

def is_confused(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(phrase in t for phrase in [
        "i'm confused","im confused","i dont understand","i don't understand",
        "what do you mean","what do you mean?","not sure what that means","idk what that means",
        "i don't know what that question is asking","i dont know what that question is asking",
        "unclear","confusing","what?","what","huh","huh?"
    ])
def wants_recap(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(kw in t for kw in [
        "explain again","say that again","repeat that","repeat please","recap",
        "can you repeat","can you explain that again","what did you just say",
        "go back","remind me","review that","explain what you just said"
    ])
def wants_example(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(kw in t for kw in ["example", "show me", "for instance", "sample", "demo"])

def wants_simpler(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(kw in t for kw in ["simpler", "simplify", "easier", "in simpler terms", "basic explanation"])

# --- Global helpers (moved from inner scope to avoid NameError) ---
def wants_more(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(kw in t for kw in [
        "more", "explain more", "go deeper", "deeper",
        "in more detail", "details", "more details",
        "expand", "tell me more", "elaborate",
        "give me more", "give me more information",
        "can you explain more", "explain that more"
    ])


def has_question(text: str) -> bool:
    """Detect if the learner wrote a question we should answer before advancing."""
    t = (text or "").strip()
    low = t.lower()
    if "?" in t:
        return True
    q_starters = (
        "what", "why", "how", "when", "where", "which", "who",
        "can you", "could you", "do you", "does it", "explain",
        "clarify", "help me understand"
    )
    return any(low.startswith(s) or f" {s} " in low for s in q_starters)


def answer_learner_question(state: Dict[str, Any], text: str) -> Dict[str, str]:
    """
    Give a concise answer tied to the CURRENT subtopic and pause until learner confirms.
    (This was previously defined inside the chat route; making it global prevents
    NameError crashes when questions are asked outside that inner scope.)
    """
    m = COURSE[state["module"]]
    sub = m["subtopics"][state["subtopic"]]["title"]
    try:
        if CLIENT:
            query = f"{m['title']} — {sub}. Question: {text}"
            sources = KB.search(query, k=4)
            context = "\n\n".join(sources)
            system = (
                "You are a helpful AI literacy tutor. Answer the learner's question clearly in 2 short paragraphs max. "
                "Use plain language and one concrete example. Stay within the CURRENT subtopic only. No citations."
            )
            messages = [
                {"role": "system", "content": system + ("\n\n[EXCERPTS]\n" + context if context else "")},
                {"role": "user", "content": text}
            ]
            resp = CLIENT.chat.completions.create(model=CHAT_MODEL, messages=messages, max_tokens=260)
            answer = _soft_wrap(resp.choices[0].message.content.strip(), max_len=420)
        else:
            base = state.get("last_teach_text") or f"We’re in {m['title']} — focusing on {sub}."
            answer = base + "\n\nHere’s the short answer: " + _soft_wrap(text, max_len=180)
    except Exception:
        answer = "Here’s a concise answer based on what we just covered."

    state["awaiting_q_confirm"] = True
    state["last_question"] = text
    return _reply(state, _naturalize(answer + "\n\nDid that answer your question? (yes/no)"))

import re as __re_modnames

def _sanitize_module_mentions(text: str, current_mod: int) -> str:
    """If the model names the wrong module, replace it with the current one."""
    current_name = FRIENDLY_MODULE_NAMES.get(current_mod, "").strip()
    if not text or not current_name:
        return text
    cur_low = current_name.lower()
    # Replace any mention of other module names with the current one
    for other in FRIENDLY_MODULE_NAMES.values():
        olow = other.lower()
        if olow != cur_low:
            text = __re_modnames.sub(olow, current_name, text, flags=__re_modnames.I)
    # Normalize “we’re in the module about … —” phrasing
    text = __re_modnames.sub(
        r"we\s*'?re in the module about [^—\n]+ —",
        f"We’re in {current_name} —",
        text,
        flags=__re_modnames.I
    )
    return text

# --- Orientation deduplication helper ---
import re as ___re_orient
def _dedupe_orientation(text: str, current_mod: int) -> str:
    """Keep only the first line that looks like an orientation ("We're in …").
    Also sanitize any wrong module names inside that one line."""
    if not text:
        return text
    lines = [ln for ln in text.splitlines()]
    out = []
    seen = False
    for ln in lines:
        if ___re_orient.search(r"we['’]re\s+in\s", ln, flags=___re_orient.I):
            if seen:
                # skip duplicate orientation lines
                continue
            ln = _sanitize_module_mentions(ln, current_mod)
            seen = True
        out.append(ln)
    return "\n".join(out)

def _extract_focus_step(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    # Flexible phrasing like "explain data please"
    if "explain" in t and "data" in t:
        return "data"
    if "explain" in t and "training" in t:
        return "training"
    if "explain" in t and ("predict" in t or "prediction" in t):
        return "prediction"
    # Single-word or synonymous mentions
    if "training" in t or "train" in t:
        return "training"
    if "prediction" in t or "predict" in t or "inference" in t:
        return "prediction"
    if "data" in t or "dataset" in t or "examples" in t:
        return "data"
    if t in {"data", "training", "prediction"}:
        return t
    return None

MINI_LESSONS_HOW = {
    "data": (
        "Let’s zoom in on **data**. Models only learn from the examples they see, so coverage matters. "
        "Good datasets are representative (typical + edge cases), labeled consistently, and split into **train / validation / test** to check generalization. "
        "Watch for **bias** (who’s missing?), **leakage** (answers hidden in inputs), and **quality** (noisy labels). "
        "Sanity checks: shuffle, visualize class balance, run a tiny baseline to set a floor."
    ),
    "training": (
        "Let’s zoom in on **training**. The model adjusts millions of weights to better match labels. "
        "Track **loss** on train/validation; if train falls but validation rises, that’s **overfitting**. "
        "Fixes: regularization, augmentation, and **early stopping**. Tune hyperparameters and compare runs with clear metrics."
    ),
    "prediction": (
        "Let’s zoom in on **prediction**. After training, the model applies learned patterns to new inputs. "
        "Evaluate accuracy and **calibration** (Does 90% confidence mean ~90% correct?), latency, and failures on out-of-distribution data. "
        "For high-stakes settings, add thresholds, human review, and drift monitoring."
    ),
}
def _deepen_how(step: str) -> str:
    if step == "data":
        return ("Deeper example: build a tiny dataset of 100 emails labeled spam/not-spam. "
                "Hold out 20 for testing, 20 for validation, and check class balance. "
                "Why hold out data? To estimate real-world performance and catch leakage.")
    if step == "training":
        return ("Deeper example: plot train/val loss over epochs. "
                "If val loss bottoms out then rises, stop training (early stopping) and try augmentation or regularization.")
    return ("Deeper example: set a confidence threshold so low-confidence predictions are flagged for human review. "
            "Track accuracy and calibration each week to catch drift.")

def start_overview(state: Dict[str, Any]) -> str:
    state["stage"] = "overview"
    m = COURSE[state["module"]]
    state["module_turns"] = 0
    state["how_steps_done"] = set()
    line = FRIENDLY_UNIT_LINES.get(state["module"], m["title"])
    msg = f"{line}\n\n{m['overview']}\n\nWhen you’re ready, type 'ready' and we’ll begin."
    state.setdefault("history", []).append({"role": "assistant", "content": msg})
    return msg

def start_teaching(state: Dict[str, Any]) -> str:
    state["stage"] = "teaching"
    m = COURSE[state["module"]]
    t = m["subtopics"][state["subtopic"]]["title"]
    logging.info(f"Teaching: module={state['module']} subtopic={state['subtopic']} - {t}")
    state["awaiting_check"] = False
    state["last_subtopic"] = t
    # Only show an orientation on the FIRST turn of this subtopic
    state["subtopic_turns"] = 0
    teach = teach_response(state, f"Start teaching subtopic: {t}")
    out = _naturalize(teach)
    state.setdefault("history", []).append({"role": "assistant", "content": out})
    return out

QUIZZES: Dict[int, List[Dict[str, Any]]] = {
    0: [  # Module 1: Understanding AI
        {
            "q": "In one short sentence, contrast **narrow AI** and **human intelligence**.",
            "ok": [["narrow","single","job"],["single","task"],["specialist"],["human","general"],["adapt","general"]]
        },
        {
            "q": "Name one part of the simple pipeline: data → training → prediction.",
            "ok": [["data"],["training"],["prediction"]]
        },
        {
            "q": "Give one real-world risk if an AI **hallucinates** (makes facts up).",
            "ok": [["medical"],["legal"],["news"],["school"],["statistics"],["finance"]]
        }
    ],
}

def start_quiz(state: Dict[str, Any]) -> str:
    state["stage"] = "quiz"
    state["quiz_idx"] = 0
    return f"Quick check before we leave {COURSE[state['module']]['title']} — {QUIZZES[state['module']][0]['q']}"

def grade_quiz_answer(state: Dict[str, Any], msg: str) -> bool:
    i = state.get("quiz_idx", 0)
    item = QUIZZES[state["module"]][i]
    return _keyword_score((msg or "").lower(), item["ok"]) >= 0.4

def next_subtopic(state: Dict[str, Any]) -> Optional[str]:
    m = COURSE[state["module"]]
    if state["subtopic"] + 1 < len(m["subtopics"]):
        state["subtopic"] += 1
        state["retry_count"] = 0
        out = start_teaching(state)
        return out
    else:
        # End of subtopics for this module — but make sure we’ve taught enough
        goal = int(state.get("module_turn_goal", 15))
        turns = int(state.get("module_turns", 0))
        if turns < goal:
            state["retry_count"] = 0
            state["awaiting_check"] = True
            extra = teach_response(state, "Continue this subtopic with new examples and a quick check; do not advance.")
            return extra
        if state["module"] in QUIZZES:
            return start_quiz(state)
        state["stage"] = "await_next"
        msg = "That wraps this module’s subtopics. When you’re ready, type 'next' to move to the next module."
        state.setdefault("history", []).append({"role": "assistant", "content": msg})
        return msg

# Review helpers
def _mastery_key(state: Dict[str, Any]):
    return (state["module"], state["subtopic"])
def _is_mastered(state: Dict[str, Any], threshold: float = 0.7) -> bool:
    key = _mastery_key(state)
    return state.get("mastery", {}).get(key, 0.0) >= threshold
def _enqueue_review(state: Dict[str, Any], mod_idx: int, sub_idx: int):
    item = (mod_idx, sub_idx)
    rq = state.setdefault("review_queue", [])
    if item not in rq:
        rq.append(item)
def _start_review(state: Dict[str, Any]) -> str:
    rq = state.get("review_queue", [])
    for i, (m_idx, s_idx) in enumerate(rq):
        if m_idx == state["module"]:
            rq.pop(i)
            state["subtopic"] = s_idx
            state["stage"] = "review"
            state["awaiting_check"] = True
            state["retry_count"] = 0
            recap = (
                "Quick review: Narrow AI is a specialist (one job); humans are general (many jobs). "
                "In one short phrase, state a difference between them."
            ) if COURSE[state["module"]]["subtopics"][s_idx]["title"] == "Definition & Types of AI" else (
                "Quick review: pick one to explain in a sentence — data, training, or prediction."
            )
            return _naturalize(recap)
    state["stage"] = "await_next"
    return "That wraps this module. Type 'next' to continue to the next module."

def _profile_from_prior(state: Dict[str, Any]) -> str:
    answers = state.get("prior_answers") or []
    ans = [a.lower() for a in answers if a]
    if not ans:
        return "[no prior answers]"
    took_ai_before = ("yes" in ans[0]) if len(ans) >= 1 else False
    thinks_ai_feels = ("yes" in ans[1]) if len(ans) >= 2 else False
    named_limit = (len(ans) >= 3 and len(ans[2]) > 1)
    flags = []
    flags.append("studied_ai_before" if took_ai_before else "new_to_ai")
    if thinks_ai_feels:
        flags.append("misconception: thinks AI can think/feel like a human")
    if not named_limit:
        flags.append("unsure_about_limits")
    return "; ".join(flags)

# ----------------- Model-backed teaching with retrieval -----------------

def _bump_teach_turn(state: Dict[str, Any]):
    try:
        state["module_turns"] = int(state.get("module_turns", 0)) + 1
    except Exception:
        state["module_turns"] = 1

def teach_response(state: Dict[str, Any], user_msg: str) -> str:
    m = COURSE[state["module"]]
    sub = m["subtopics"][state["subtopic"]]["title"]
    learner_profile = _profile_from_prior(state)

    if len((user_msg or "").strip()) <= 1 and is_ack(user_msg):
        follow = next_check_question(state)
        state["awaiting_check"] = True
        state["last_subtopic"] = sub
        state["last_result"] = None
        return _naturalize(f"Got it. Quick check: {follow}")

    if is_ack(user_msg):
        follow = next_check_question(state)
        state["awaiting_check"] = True
        state["last_subtopic"] = sub
        state["last_result"] = None
        return _naturalize(f"Great — quick check: {follow}")

    if (user_msg or "").strip().lower() in ["idk", "i don't know", "i dont know", "not sure", "no idea", "unsure"]:
        state["awaiting_check"] = True
        state["last_subtopic"] = sub
        state["last_result"] = None
        hint = (
            "That’s okay — let’s work through it. Think of **narrow AI** as a specialist (like a spam filter that only sorts emails). "
            "Human intelligence is general — we can learn many different tasks. So: Narrow AI = one job; Humans = many jobs. "
            "Based on that, what is *one difference* between them?"
        )
        return _naturalize(hint)

    HALLUCINATION_PROMPT = (
        "AI sometimes **hallucinates**—it confidently makes things up when it lacks the right data. "
        "Pick one real-life place where that could cause trouble: **medical report**, **legal advice**, "
        "**news summary**, or **school assignment**. (Type one of those or your own example.)"
    )
    fallback_content: Dict[str, str] = {
        "Definition & Types of AI": (
            "AI are computer systems that handle tasks linked to human intelligence — recognizing patterns, making predictions, and using language. "
            "There are a few common types. **Narrow AI** is built for a single job (a spam filter or chess engine). **Generative AI** creates new content like text or images. "
            "Today’s systems are not human-level general intelligence.\n\n"
            "Quick check: in your own words, name one way **narrow AI** differs from **human intelligence** (one short phrase is enough)."
        ),
        "How AI Works: data, algorithms, predictions": (
            "Modern AI learns patterns from **data** using algorithms, then uses those patterns to make **predictions**. "
            "A simple path is: **data → training → prediction**.\n\n"
            "For example, a vision model trained on labeled X-rays learns features linked to pneumonia and flags likely cases on new images.\n\n"
            "Let’s go deeper. Pick **one** to start: type `data`, `training`, or `prediction`. We’ll cover the others next."
        ),
        "Capabilities, limits, and misconceptions": (
            "AI is strong at speed and consistency on well-defined tasks, but it does not understand or have intentions. "
            "Limits include brittleness outside training data, **bias**, and **hallucinations** (making things up).\n\n"
            "AI sometimes hallucinates—it confidently makes things up when it lacks data (like inventing fake statistics).\n\n"
            + HALLUCINATION_PROMPT
        ),
        "Prompt basics and anatomy (role, task, context)": (
            "A prompt is the instruction you give an AI. Good prompts set a **role**, state the **task** clearly, and give the right **context/constraints**. "
            "Example: “You are a teacher. Summarize this article in five bullet points for grade 11.”\n\n"
            "Want to try drafting a quick role-task-context prompt?"
        ),
        "Iterative refinement and examples": (
            "Treat prompting like iteration: draft → test → refine. Show an example when helpful, ask the model to show steps, then tighten instructions. \n\n"
            "What output are you trying to improve right now?"
        ),
        "Common pitfalls and how to debug prompts": (
            "Typical issues are vague goals, missing context, or packing too many steps into one instruction. Be specific about inputs/outputs, split multi-step work, and set constraints like length or format.\n\n"
            "Paste a prompt you’ve used and we’ll tighten it."
        ),
        "AI tools in education, healthcare, and business": (
            "Applied AI tools wrap models into workflows for specific domains such as education (feedback/writing support), healthcare (triage and imaging), and business (support and forecasting).\n\n"
            "Which domain is most relevant to you?"
        ),
        "Choosing the right tool: capabilities vs. needs": (
            "Pick tools by matching capabilities to the task and constraints: data sensitivity and privacy, accuracy needs and failure cost, and how it will integrate and be maintained.\n\n"
            "What is one requirement your use case can’t compromise on?"
        ),
        "Basic AI pipeline: data → training → deployment": (
            "Turning a model into a product usually involves data collection/labeling, training and evaluation, then deployment with monitoring and updates.\n\n"
            "Which stage would you like a deeper example on?"
        ),
        "Bias and fairness": (
            "Bias can arise from imbalanced data, labeling choices, or proxy features. Check performance by subgroup and use mitigation steps such as re-weighting or constraints, plus regular audits.\n\n"
            "How would you test a model for subgroup gaps?"
        ),
        "Privacy and data protection": (
            "Protect personal data by minimizing what you collect, controlling access, encrypting storage, and using anonymization or on-device options when possible.\n\n"
            "What data in your projects would need masking?"
        ),
        "Responsible use and governance": (
            "Responsible AI combines human oversight, incident response, and clear documentation of limits. In high-stakes cases, keep a human in the loop.\n\n"
            "Where should human review sit in your workflow?"
        ),
    }

    if not CLIENT:
        state["awaiting_check"] = True
        state["last_subtopic"] = sub
        state["last_result"] = None
        text = fallback_content.get(sub, "Let’s unpack this topic step by step. What would you like clarified?")
        # Always end with ONE banked question aligned to grading.
        text = text.rsplit('?', 1)[0].strip()
        text += "\n\nQuick check: " + next_check_question(state)
        text = _naturalize(text)
        state["last_teach_text"] = text
        _bump_teach_turn(state)
        return text

    try:
        query = f"{m['title']} — {sub}. {user_msg}"
        sources = KB.search(query, k=6)
        context = "\n\n".join(sources)
        # Pull short dialogue history to keep responses contextual (last 6 turns)
        short_hist = []
        try:
            for h in state.get("history", [])[-6:]:
                role = h.get("role", "user")
                content = h.get("content", "")
                if content:
                    short_hist.append({"role": role, "content": content})
        except Exception:
            short_hist = []
        system = (
            "You are an AI literacy tutor. " + DESIGN_RULES +
            "\nLearner profile: " + (_profile_from_prior(state) or "[none]") +
            f"\nRules: Write in plain sentences (no headings). Keep paragraphs short. Use a friendly, human tone tuned for level = {state.get('level','standard')}. "
            "Stay strictly within the CURRENT module and subtopic. Do NOT introduce other modules (e.g., prompting) unless the flow has advanced. "
            "Teach one idea at a time and end each turn with ONE check-for-understanding question. If you include an orientation sentence like 'We’re in … — focusing on …', do it only on the FIRST reply of a subtopic."
            "\nUse these course/knowledge excerpts when helpful (text only):\n[EXCERPTS]\n" + context
        )
        messages = [{"role": "system", "content": system}]
        if short_hist:
            messages.extend(short_hist)
        messages.append({"role": "user", "content":
            f"Teach the subtopic now: {sub} for college students at level={state.get('level','standard')}. Write one short paragraph (60–100 words) in a friendly, human tone. End with exactly ONE specific question that fits this paragraph, then stop. Keep it strictly inside the current module/subtopic. Avoid headings/bullets and avoid multiple questions."
        })
        resp = CLIENT.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=320,
        )
        state["awaiting_check"] = True
        state["last_subtopic"] = sub
        state["last_result"] = None
        text = resp.choices[0].message.content.strip()

        # If we've already sent one turn for this subtopic, strip any leading orientation line
        turns = int(state.get("subtopic_turns", 0))
        if turns > 0:
            text = re.sub(r"^\s*we\s*’?\s*re in .*?\.\s*", "", text, flags=re.I)
        text = re.sub(r"^\s*we\s*'?\s*re in .*?\.\s*", "", text, flags=re.I)

        # Strip any model-authored question at the end (take everything before the last '?')
        parts_q = text.split('?')
        if len(parts_q) > 1:
            text = '?'.join(parts_q[:-1]).strip()

        # Remove any lines that look like citations or reading callouts.
        clean_lines = []
        for ln in text.splitlines():
            s = ln.strip().lower()
            if s.startswith("from the readings") or s.startswith("from the course readings"):
                continue
            clean_lines.append(ln)
        text = '\n'.join(clean_lines).strip()

        # Always end with ONE banked question aligned to grading.
        bank_q = next_check_question(state)
        text = (text + ("\n\n" if text else "")) + f"Quick check: {bank_q}"

        # Fix any mistaken module name mentions (e.g., says "Applied AI" while in Foundational)
        text = _sanitize_module_mentions(text, state["module"])
        # Remove duplicate orientation lines, keeping only the first and sanitizing it.
        text = _dedupe_orientation(text, state["module"])

        # Normalize paragraph spacing
        text = _soft_wrap(text, max_len=420)

        # If the model repeats verbatim, vary the follow-up
        last = state.get("last_teach_text")
        if last and last == text:
            alt_check = next_check_question(state)
            text = text + f"\n\nLet’s try a different angle. {alt_check}"

        # Naturalize + memory crumb for future recaps
        text = _naturalize(text)
        state["last_teach_text"] = text
        state["subtopic_turns"] = int(state.get("subtopic_turns", 0)) + 1
        # Keep a very short recap history so we can say what we just covered
        notes = state.setdefault("notes", [])
        first_two = ' '.join(text.split('. ')[:2]).strip()
        if first_two:
            notes.append({'module': state['module'], 'subtopic': sub, 'summary': first_two})
        _bump_teach_turn(state)
        return text
    except Exception as e:
        logging.warning(f"Model error; using fallback. {e}")
        state["last_result"] = None
        txt = _naturalize(fallback_content.get(sub, "Let’s unpack this topic step by step. What would you like clarified?"))
        state.setdefault("history", []).append({"role": "assistant", "content": txt})
        _bump_teach_turn(state)
        return txt

# ----------------- Routes -----------------
@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "frontend" / "index.html")

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/chat")
async def chat(body: ChatIn):
    # --- Conversation state guard: only one question at a time ---
    if hasattr(body, 'conversation_history'):
        last_user_msg = next((msg for msg in reversed(getattr(body, 'conversation_history', [])) if msg['role'] == 'user'), None)
        last_assistant_msg = next((msg for msg in reversed(getattr(body, 'conversation_history', [])) if msg['role'] == 'assistant'), None)
        if last_assistant_msg and not last_user_msg:
            return {"message": "Please answer the previous question before continuing."}
    sid = (body.session_id or "").strip() or "default"
    if (sid == "default" or sid.lower() == "new") and sessions:
        try:
            sid = next(iter(sessions.keys()))
        except Exception:
            sid = "default"
    state = get_state(sid)
    logging.info(f"[chat] sid={sid} stage={state.get('stage')} module={state.get('module')} subtopic={state.get('subtopic')}")
    msg = (body.message or "").strip().lower()
    if body.message:
        state.setdefault("history", []).append({"role": "user", "content": body.message})
        
    # --- Early guard: if a quick-check is pending, grade normal answers immediately ---
    # BUT: if we’re in "How AI Works..." and the user typed a step (data/training/prediction),
    # handle the step-chooser BEFORE grading.
    current_subtopic = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
    step_choice = _extract_focus_step(body.message or "")
    if (
        state.get("awaiting_check")
        and current_subtopic == "How AI Works: data, algorithms, predictions"
        and step_choice is not None
    ):
        state["how_step"] = step_choice
        state["how_step_pending"] = True
        state["awaiting_check"] = True
        lesson = MINI_LESSONS_HOW[step_choice] + "\n\nDoes that make sense? (yes/no or type 'more' for a deeper example)"
        return {"message": lesson}

    if state.get("awaiting_check") and not has_cmd(msg, "next") and not has_question(body.message or "") and not is_non_answer(msg):
        result = grade_answer(state, body.message or "")
        score = result["score"]; sub = result["subtopic"]; lvl = result["level"]
        feedback = result.get("feedback", "")
        if score >= 0.65:
            state["awaiting_check"] = False; state["retry_count"] = 0
            state["last_result"] = {"subtopic": sub, "score": score, "mastery": result["mastery"], "level": lvl}
            progressed = next_subtopic(state)
            ack = _ack_text(sub, body.message or "", strength="strong", adapt_flag=result.get("adapt", False))
            fb = state.pop("last_llm_feedback", "")
            extra = ("\n\n" + feedback) if feedback else ""
            if fb:
                extra += "\n\n" + fb
            return {"message": f"{ack}{extra}\n\n{progressed}" if progressed else f"{ack}{extra} Type 'next' to continue, or ask for a tougher example."}
        elif score >= 0.3:
            state["retry_count"] = state.get("retry_count", 0) + 1
            if state["retry_count"] >= 2:
                _enqueue_review(state, state["module"], state["subtopic"])
                state["awaiting_check"] = False; state["retry_count"] = 0
                progressed = next_subtopic(state)
                ack = _ack_text(sub, body.message or "", strength="light", adapt_flag=result.get("adapt", False))
                fb = state.pop("last_llm_feedback", "")
                extra = ("\n\n" + feedback) if feedback else ""
                if fb:
                    extra += "\n\n" + fb
                return {"message": f"{ack}{extra}\n\n{progressed}" if progressed else f"{ack}{extra} Type 'next' to continue, or ask for a tougher example."}
            follow = next_check_question(state)
            return {"message": f"Close! You’re on the right track for **{sub}**. Want to try again? {feedback}\n{follow}"}
        else:
            state["retry_count"] = state.get("retry_count", 0) + 1
            hint = "Think **specialist vs. generalist** — narrow AI does one job; humans can learn many. Keep it to one short sentence."
            follow = next_check_question(state)
            if state.get("retry_count", 0) >= 2:
                _enqueue_review(state, state["module"], state["subtopic"])
                state["awaiting_check"] = False; state["retry_count"] = 0
                out = next_subtopic(state)
                return {"message": f"We’ll revisit this later if needed. For now, let’s keep moving. {feedback} " + (out or "")}
            else:
                return {"message": f"No worries — let’s make it simpler. {hint} {feedback} {follow}"}

    # If we are waiting for confirmation that a question was answered, handle that first
    if state.get("awaiting_q_confirm"):
        yn = _is_yes_no(msg)
        if yn is True or is_ack(msg):
            state["awaiting_q_confirm"] = False
            return {"message": "Great — let’s continue. If more questions pop up, just ask. Type 'next' or reply to the last question."}
        elif yn is False:
            return {"message": "No worries — what part is still unclear?"}
        if has_question(body.message or ""):
            return answer_learner_question(state, body.message or "")
        return {"message": "Did my last explanation answer your question? (yes/no)"}
    
    if state.get("stage") == "prior" and msg in NON_ANSWERS:
        pass
    # Global: learner asks for a recap/“explain again” at any time
    if wants_recap(body.message or ""):
        notes = state.get("notes", [])
        if notes:
            last = notes[-1]
            recent = last.get("summary") or state.get("last_teach_text") or "We were discussing the current subtopic."
            return {"message": _naturalize(f"Quick recap: {recent}\n\nWhat part should I clarify or give an example for?")}
        else:
            return {"message": "We haven’t covered much yet — say 'begin' to start, or ask anything and I’ll explain."}
        
    # intro
    if state["stage"] == "intro":
        if "begin" in msg:
            return {"message": start_overview(state)}
        return {"message": OPENING_MESSAGE}

    # overview
    if state["stage"] == "overview":
        if "ready" in msg:
            state["stage"] = "prior"
            state["prior_idx"] = 0
            q = COURSE[state["module"]]["prior"][0]
            return {"message": f"Quick warm-up — {q}"}
        return {"message": "When you’re ready to begin this unit, type 'ready'.\nThen we’ll do three quick questions to see where you’re at."}

    # prior
    if state["stage"] == "prior":
        m = COURSE[state["module"]]
        # Ignore navigation words here; we need a real answer to each prior item
        if is_non_answer(msg):
            q = m["prior"][state["prior_idx"]]
            return {"message": f"Quick warm-up — {q}\nPlease answer with **yes/no** or a short phrase (2–5 words)."}
        i = state["prior_idx"]
        q = m["prior"][i]
        raw = (body.message or "").strip()
        low = raw.lower()
        if is_non_answer(low):
            hint = "Please answer this exact question — type **yes/no** or a **short phrase** in your own words."
            return {"message": f"Quick warm-up — {q}\n{hint}"}
        if i in (0, 1):
            yn = _is_yes_no(low)
            if yn is None:
                state["prior_tries"][i] += 1
                hint = "Please answer yes or no (you can type 'y' or 'n')."
                return {"message": f"Quick warm-up — {q}\n{hint}"}
            state.setdefault("prior_answers", []).append("yes" if yn else "no")
        else:
            if not _valid_short_answer(raw, min_words=2):
                state["prior_tries"][i] += 1
                if state["prior_tries"][i] >= 2 and low in {"idk", "i don't know", "i dont know", "unsure", "no idea"}:
                    state.setdefault("prior_answers", []).append("idk")
                else:
                    hint = "Give a short phrase (2–5 words). Example: 'read emotions' or 'make moral choices'."
                    return {"message": f"Quick warm-up — {q}\n{hint}"}
            else:
                state.setdefault("prior_answers", []).append(raw)
        if i < len(m["prior"]) - 1:
            state["prior_idx"] += 1
            return {"message": m["prior"][state["prior_idx"]]}
        state["stage"] = "learn"
        return {"message": "Thanks! Type 'learn' to start this module’s lesson."}

    # move into teaching
    if state["stage"] == "learn":
        if "learn" in msg or "start" in msg:
            state["subtopic"] = 0
            return {"message": start_teaching(state)}
        return {"message": "Type 'learn' when you’re ready to start this module’s lesson."}

    # active teaching
    if state["stage"] == "teaching":
        # Do not allow 'next' if a quick-check is pending unless skipping is disabled
        if has_cmd(msg, "next"):
            if state.get("awaiting_check") and REQUIRE_ANSWER_BEFORE_NEXT:
                follow = next_check_question(state)
                return {"message": f"Quick check first — give one short answer (or type 'idk' for a hint). {follow}"}
            state["awaiting_check"] = False
            out = next_subtopic(state)
            return {"message": out}

        current_sub = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
    # If the learner asked a question, answer it before moving on
    if has_question(body.message or "") and not has_cmd(msg, "next"):
        resp = answer_learner_question(state, body.message or "")
        # Only allow one question at a time in the assistant's reply
        answer = resp.get("message", "")
        if answer.count("?") > 1:
            answer = answer.split("?")[0] + "?"
        return {"message": answer}
    
    # Learner asks for more depth on the current point
    if wants_more(body.message or ""):
        sub_t = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
        if sub_t == "How AI Works: data, algorithms, predictions" and state.get("how_step"):
            step = state.get("how_step")
            deeper = _deepen_how(step)
            msg = deeper + "\n\nDoes that help? (yes/no) You can also type 'next' to move on."
            if msg.count("?") > 1:
                msg = msg.split("?")[0] + "?"
            return _reply(state, msg)
        # Ask the tutor to extend THIS subtopic without moving on
        state["awaiting_check"] = True
        extended = teach_response(state, "Please go deeper on the same subtopic without moving on. Add a new question at the end.")
        if extended.count("?") > 1:
            extended = extended.split("?")[0] + "?"
        return _reply(state, extended)
    "Do NOT mention readings, papers, citations, or references; never write 'From the readings'. "
    # Early handler for 'How AI Works' choice (runs even if no 'more' was requested)
    if current_sub == "How AI Works: data, algorithms, predictions":
        sel = _extract_focus_step(body.message or "")
        if sel:
            state["how_step"] = sel
            state["how_step_pending"] = True
            state["awaiting_check"] = True
            lesson = MINI_LESSONS_HOW[sel] + "\n\nDoes that make sense? (yes/no or type 'more' for a deeper example)"
            # Limit to one question
            if lesson.count("?") > 1:
                lesson = lesson.split("?")[0] + "?"
            return _reply(state, lesson)

        # If we lost the awaiting flag but user answered, grade anyway
        if not state.get("awaiting_check") and is_answer_like(state, body.message or ""):
            result = grade_answer(state, body.message or "")
            score = result["score"]; sub = result["subtopic"]; lvl = result["level"]
            feedback = result.get("feedback", "")
            if score >= 0.65:
                state["awaiting_check"] = False; state["retry_count"] = 0
                state["last_result"] = {"subtopic": sub, "score": score, "mastery": result["mastery"], "level": lvl}
                progressed = next_subtopic(state)
                ack = _ack_text(sub, body.message or "", strength="strong", adapt_flag=result.get("adapt", False))
                if "statistic" in (body.message or "").lower() or "stats" in (body.message or "").lower():
                    ack += "\n\nGood call-out — statistics can be risky if the model makes numbers up."
                fb = state.pop("last_llm_feedback", "")
                # Always append explanation feedback
                extra = "\n\n" + feedback if feedback else ""
                if fb:
                    extra += "\n\n" + fb
                if progressed:
                    return {"message": f"{ack}{extra}\n\n{progressed}"}
                else:
                    return {"message": f"{ack}{extra} Type 'next' to continue, or ask for a tougher example."}
            elif score >= 0.3:
                state["retry_count"] = state.get("retry_count", 0) + 1
                if state["retry_count"] >= 2:  # keep momentum
                    _enqueue_review(state, state["module"], state["subtopic"])
                    state["awaiting_check"] = False; state["retry_count"] = 0
                    progressed = next_subtopic(state)
                    ack = _ack_text(sub, body.message or "", strength="light", adapt_flag=result.get("adapt", False))
                    fb = state.pop("last_llm_feedback", "")
                    extra = "\n\n" + feedback if feedback else ""
                    if fb:
                        extra += "\n\n" + fb
                    if progressed:
                        return {"message": f"{ack}{extra}\n\n{progressed}"}
                    else:
                        return {"message": f"{ack}{extra} Type 'next' to continue, or ask for a tougher example."}
                follow = next_check_question(state)
                return {"message": f"Close! You’re on the right track for **{sub}**. Focus on the key distinction we just discussed. Want to try again? {feedback}\n{follow}"}
            else:
                state["retry_count"] = state.get("retry_count", 0) + 1
                if mentions_physical_confusion(body.message or ""):
                    hint = ("Careful — that’s about **robots/physical hardware**. Here we’re comparing kinds of **intelligence**. "
                            "Think *specialist vs. general*: narrow AI does one **single job**; humans can learn **many jobs**.")
                else:
                    hint = "Think **specialist vs. generalist** — narrow AI does one job; humans can learn many. Keep it to one short sentence."
                follow = next_check_question(state)
                if state.get("retry_count", 0) >= 2:
                    _enqueue_review(state, state["module"], state["subtopic"])
                    state["awaiting_check"] = False; state["retry_count"] = 0
                    out = next_subtopic(state)
                    return {"message": f"We’ll revisit this later if needed. For now, let’s keep moving. {feedback} " + (out or "")}
                else:
                    return {"message": f"No worries — let’s make it simpler. {hint} {feedback} {follow}"}

        # start a check if user typed something that's not a command
        if not state.get("awaiting_check"):
            if (body.message or "").strip() and not any(has_cmd(msg, w) for w in ("next","begin","ready","learn")):
                state["awaiting_check"] = True
                state["last_subtopic"] = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
                state["retry_count"] = state.get("retry_count", 0)
                hint = "I’m looking for a one-sentence idea in your own words. If you’re unsure, type 'idk' for a quick hint, or 'next' to skip."
                follow = next_check_question(state)
                msg_out = f"That doesn’t quite answer the check yet. {hint} {follow}"
                if msg_out.count("?") > 1:
                    msg_out = msg_out.split("?")[0] + "?"
                return {"message": msg_out}

        # waiting for a check
        if state.get("awaiting_check"):
            if has_question(body.message or ""):
                resp = answer_learner_question(state, body.message or "")
                answer = resp.get("message", "")
                if answer.count("?") > 1:
                    answer = answer.split("?")[0] + "?"
                return {"message": answer}
            if wants_recap(body.message or ""):
                recap = state.get("last_teach_text") or "We were discussing the current subtopic."
                msg_out = _naturalize("Quick recap of the last point:\n" + recap + "\n\nWhat part should I clarify further?")
                if msg_out.count("?") > 1:
                    msg_out = msg_out.split("?")[0] + "?"
                return _reply(state, msg_out)

            if is_confused(body.message):
                # Rephrase with a shorter statement plus a targeted question
                sub_t = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
                if sub_t == "How AI Works: data, algorithms, predictions":
                    msg_out = "No problem — simple version: AI learns from **data**, adjusts during **training**, and applies patterns in **prediction**. Which part should I slow down on — data, training, or prediction?"
                    if msg_out.count("?") > 1:
                        msg_out = msg_out.split("?")[0] + "?"
                    return _reply(state, msg_out)
                follow = next_check_question(state)
                msg_out = f"No problem — I’ll rephrase and keep it short. {follow}"
                if msg_out.count("?") > 1:
                    msg_out = msg_out.split("?")[0] + "?"
                return _reply(state, msg_out)

            # If learner asks for an example or simpler phrasing, respond before grading
            if wants_example(body.message or ""):
                sub_t = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
                ex = "Here’s a quick example: "
                try:
                    hl = kb_highlights(f"example — {sub_t}", k=1)
                    if hl:
                        ex += hl[0]
                    else:
                        ex += "If a model trained on balanced emails flags spam, holding out a test set checks real performance."
                except Exception:
                    ex += "If a model trained on balanced emails flags spam, holding out a test set checks real performance."
                follow = next_check_question(state)
                msg_out = _naturalize(f"{ex}\n\nDoes that help? {follow}")
                if msg_out.count("?") > 1:
                    msg_out = msg_out.split("?")[0] + "?"
                return _reply(state, msg_out)

            if wants_simpler(body.message or ""):
                sub_t = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
                simpler = "In simple terms: "
                if sub_t == "Definition & Types of AI":
                    simpler += "narrow AI is a specialist; humans are general learners who can switch tasks."
                elif sub_t == "How AI Works: data, algorithms, predictions":
                    simpler += "models learn patterns from examples (training) and use them to make predictions."
                else:
                    simpler += "AI is powerful but limited; we use checks to avoid errors and bias."
                follow = next_check_question(state)
                msg_out = _naturalize(f"{simpler} {follow}")
                if msg_out.count("?") > 1:
                    msg_out = msg_out.split("?")[0] + "?"
                return _reply(state, msg_out)
            


        if current_sub == "How AI Works: data, algorithms, predictions":
            # If we are mid-step and waiting for a yes/no (or 'more'), handle that first
            if state.get("how_step_pending"):
                tmsg = (body.message or "").strip().lower()
                yn_local = _is_yes_no(tmsg)
                if yn_local is None and is_ack(body.message or ""):
                    yn_local = True
                if yn_local is False or any(k in tmsg for k in ("more", "detail", "example", "deeper")):
                    step = state.get("how_step", "data")
                    deeper = _deepen_how(step)
                    return _reply(state, deeper + "\n\nDoes that make sense? (yes/no)  You can also type 'next' to move on.")
                if yn_local is True:
                    state["how_step_pending"] = False
                    # Mark the just-covered step as done
                    done = state.get("how_step")
                    if done:
                        hs = state.setdefault("how_steps_done", set())
                        hs.add(done)
                    # Teach the next remaining step automatically, if any
                    remaining = [s for s in ("data","training","prediction") if s not in state.get("how_steps_done", set())]
                    if remaining:
                        nxt = remaining[0]
                        state["how_step"] = nxt
                        state["how_step_pending"] = True
                        lesson = MINI_LESSONS_HOW[nxt] + "\n\nDoes that make sense? (yes/no or type 'more' for a deeper example)"
                        
                        return _reply(state, lesson)
                    # Otherwise proceed to the next subtopic
                    state["awaiting_check"] = False
                    out = next_subtopic(state)
                    return _reply(state, "Great — let’s continue.\n\n" + (out or "Type 'next' to continue."))

        if state.get("awaiting_check"):
            # 1) Handle meta-requests first (questions/recap/example/simpler)
            if has_question(body.message or ""):
                return answer_learner_question(state, body.message or "")

            if wants_recap(body.message or ""):
                recap = state.get("last_teach_text") or "We were discussing the current subtopic."
                return _reply(state, _naturalize("Quick recap: " + recap + "\n\nWhat part should I clarify further?"))

            if is_confused(body.message):
                sub_t = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
                if sub_t == "How AI Works: data, algorithms, predictions":
                    return _reply(state, "No problem — simple version: AI learns from **data**, adjusts during **training**, and applies patterns in **prediction**. Which part should I slow down on — data, training, or prediction?")
                follow = next_check_question(state)
                return _reply(state, f"No problem — I’ll rephrase and keep it short. {follow}")

            if wants_example(body.message or ""):
                sub_t = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
                ex = "Here’s a quick example: "
                try:
                    hl = kb_highlights(f"example — {sub_t}", k=1)
                    ex += hl[0] if hl else "If a model trained on balanced emails flags spam, holding out a test set checks real performance."
                except Exception:
                    ex += "If a model trained on balanced emails flags spam, holding out a test set checks real performance."
                follow = next_check_question(state)
                return _reply(state, _naturalize(f"{ex}\n\nDoes that help? {follow}"))

            if wants_simpler(body.message or ""):
                sub_t = COURSE[state['module']]["subtopics"][state["subtopic"]]["title"]
                simpler = "In simple terms: "
                if sub_t == "Definition & Types of AI":
                    simpler += "narrow AI is a specialist; humans are general learners who can switch tasks."
                elif sub_t == "How AI Works: data, algorithms, predictions":
                    simpler += "models learn patterns from examples (training) and use them to make predictions."
                else:
                    simpler += "AI is powerful but limited; we use checks to avoid errors and bias."
                follow = next_check_question(state)
                return _reply(state, _naturalize(f"{simpler} {follow}"))

            # 2) Block navigation/empty messages until a real answer arrives
            if is_non_answer(msg):
                follow = next_check_question(state)
                return _reply(state, f"Let's answer this quick one first. One short phrase is enough. {follow}")

            # 3) Grade the learner’s answer (handles any normal sentence)
            result = grade_answer(state, body.message or "")
            score = result["score"]; sub = result["subtopic"]; lvl = result["level"]
            feedback = result.get("feedback", "")

            if score >= 0.65:
                state["awaiting_check"] = False; state["retry_count"] = 0
                state["last_result"] = {"subtopic": sub, "score": score, "mastery": result["mastery"], "level": lvl}
                progressed = next_subtopic(state)
                ack = _ack_text(sub, body.message or "", strength="strong", adapt_flag=result.get("adapt", False))
                if "statistic" in (body.message or "").lower() or "stats" in (body.message or "").lower():
                    ack += "\n\nGood call-out — statistics can be risky if the model makes numbers up."
                fb = state.pop("last_llm_feedback", "")
                extra = ("\n\n" + feedback) if feedback else ""
                if fb:
                    extra += "\n\n" + fb
                return _reply(state, f"{ack}{extra}\n\n{progressed}" if progressed else f"{ack}{extra} Type 'next' to continue, or ask for a tougher example.")

            elif score >= 0.3:
                state["retry_count"] = state.get("retry_count", 0) + 1
                if state["retry_count"] >= 2:
                    _enqueue_review(state, state["module"], state["subtopic"])
                    state["awaiting_check"] = False; state["retry_count"] = 0
                    progressed = next_subtopic(state)
                    ack = _ack_text(sub, body.message or "", strength="light", adapt_flag=result.get("adapt", False))
                    fb = state.pop("last_llm_feedback", "")
                    extra = ("\n\n" + feedback) if feedback else ""
                    if fb:
                        extra += "\n\n" + fb
                    return _reply(state, f"{ack}{extra}\n\n{progressed}" if progressed else f"{ack}{extra} Type 'next' to continue, or ask for a tougher example.")
                follow = next_check_question(state)
                return _reply(state, f"Close! You’re on the right track for **{sub}**. Focus on the key distinction we just discussed. Want to try again? {feedback}\n{follow}")

            else:
                state["retry_count"] = state.get("retry_count", 0) + 1
                if mentions_physical_confusion(body.message or ""):
                    hint = ("Careful — that’s about **robots/physical hardware**. Here we’re comparing kinds of **intelligence**. Think *specialist vs. general*: narrow AI does one **single job**; humans can learn **many jobs**.")
                else:
                    hint = "Think **specialist vs. generalist** — narrow AI does one job; humans can learn many. Keep it to one short sentence."
                follow = next_check_question(state)
                if state.get("retry_count", 0) >= 2:
                    _enqueue_review(state, state["module"], state["subtopic"])
                    state["awaiting_check"] = False; state["retry_count"] = 0
                    out = next_subtopic(state)
                    return _reply(state, f"We’ll revisit this later if needed. For now, let’s keep moving. {feedback} " + (out or ""))
                else:
                    return _reply(state, f"No worries — let’s make it simpler. {hint} {feedback} {follow}")

        if is_claim_answer(msg):
            last = state.get("last_result")
            if last and last.get("score", 0.0) >= 0.9:
                return {"message": f"Yep — you already nailed **{last['subtopic']}**. Type 'next' to continue, or ask for a tougher example."}
            elif last and last.get("score", 0.0) >= 0.4:
                follow = next_check_question(state)
                return {"message": f"You're close on **{last['subtopic']}** — want to tighten it once more? {follow}"}
            else:
                hint = "Think **specialist vs. generalist** — narrow AI does one job; humans can learn many."
                return {"message": f"I hear you. Let’s try one more tiny nudge. {hint} {next_check_question(state)}"}

        if len((body.message or "").strip()) <= 3:
            if state.get("awaiting_check"):
                follow = next_check_question(state)
                return {"message": f"Just one quick phrase is enough (or 'idk' for a hint). {follow}"}
            else:
                follow = next_check_question(state)
                state["awaiting_check"] = True
                state["last_subtopic"] = COURSE[state["module"]]["subtopics"][state["subtopic"]]["title"]
                state["last_result"] = None
                return {"message": f"Got it. Quick check before we move on: {follow}"}

        if _is_mastered(state):
            return {"message": "Looks good here — type 'next' to continue, or ask for a tougher follow-up."}

        return {"message": teach_response(state, body.message)}
    if state.get("stage") == "quiz":
        if has_question(body.message or ""):
            return answer_learner_question(state, body.message or "")
        if is_non_answer(msg):
            return {"message": QUIZZES[state["module"]][state.get("quiz_idx", 0)]["q"]}
        correct = grade_quiz_answer(state, body.message or "")
        if correct:
            state["quiz_idx"] += 1
            if state["quiz_idx"] >= len(QUIZZES[state["module"]]):
                state["stage"] = "await_next"
                return {"message": "Nice — end-of-module check passed. Type 'next' to move to the next module."}
            return {"message": "Good — next one: " + QUIZZES[state["module"]][state["quiz_idx"]]["q"]}
        else:
            return {"message": "Close. One sentence in your own words is enough. Try again: " + QUIZZES[state["module"]][state.get("quiz_idx", 0)]["q"]}
    
    if state["stage"] == "await_next":
        if "next" in msg:
            rq = [item for item in state.get("review_queue", []) if item[0] == state["module"]]
            if rq:
                return {"message": _start_review(state)}
            state["module"] += 1
            state["retry_count"] = 0
            state["subtopic"] = 0
            if state["module"] >= len(COURSE):
                state["stage"] = "done"
                return {"message": "You’ve completed all modules. Thanks for participating!"}
            return {"message": start_overview(state)}
        return {"message": "Type 'next' to continue to the next module."}

    if state.get("stage") == "review":
        if state.get("awaiting_check"):
            result = grade_answer(state, body.message or "")
            if result["score"] >= 0.7:
                state["awaiting_check"] = False
                rq = [item for item in state.get("review_queue", []) if item[0] == state["module"]]
                if rq:
                    return {"message": _start_review(state)}
                state["stage"] = "await_next"
                return {"message": "Nice — review passed. Type 'next' to proceed to the next module."}
            else:
                return {"message": "Hint: Narrow AI = one job; humans = many jobs. Try a one-phrase difference."}
        return {"message": _start_review(state)}

    if state["stage"] == "done":
        return {"message": "Course complete. Type 'restart' to start over."}

    if msg == "restart":
        sessions[sid] = {"id": sid, "stage": "intro", "module": 0, "prior_idx": 0, "subtopic": 0}
        return {"message": OPENING_MESSAGE}

    return {"message": "I didn’t catch that. Type 'begin' to start, 'ready' after the overview, 'learn' to teach, 'next' to advance, or 'restart' to start over."}

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Chatbot backend running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)