from pathlib import Path
from datetime import datetime
import os
import csv
import re

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ------------------------------------------------
# ENV + OPENAI CLIENT
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY loaded? ", bool(api_key))

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env file.")

client = OpenAI(api_key=api_key)

# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------

OFFERS_FILE = BASE_DIR / "Offers.csv"
PRODUCTS_FILE = BASE_DIR / "products.csv"

# ------------------------------------------------
# LOAD INVENTORY CSV
# ------------------------------------------------

def load_inventory_df():
    if not PRODUCTS_FILE.exists():
        print("products.csv not found, inventory search disabled.")
        return None

    try:
        df = pd.read_csv(PRODUCTS_FILE)
        print(f"Loaded inventory: {len(df)} products from {PRODUCTS_FILE.name}")
        return df
    except Exception as e:
        print("Error loading products.csv:", repr(e))
        return None


INVENTORY_DF = load_inventory_df()

# Helpers to find the right columns (title, price, link, description/raw)
def get_inventory_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}

    title_col = cols.get("title") or list(df.columns)[0]
    price_col = cols.get("price")
    link_col = cols.get("link") or cols.get("url")

    desc_col = (
        cols.get("description")
        or cols.get("body")
        or cols.get("raw")
        or cols.get("details")
    )

    return title_col, price_col, link_col, desc_col


STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "your",
    "you",
    "some",
    "link",
    "links",
    "please",
    "send",
    "me",
    "to",
    "a",
    "an",
    "of",
    "in",
    "on",
    "my",
}


def extract_keywords(text: str):
    words = re.split(r"[^a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def search_inventory(query: str, limit: int = 5) -> str:
    """Return a nicely formatted bullet list of matching products, or '' if none."""
    if INVENTORY_DF is None or INVENTORY_DF.empty:
        return ""

    title_col, price_col, link_col, desc_col = get_inventory_columns(INVENTORY_DF)

    keywords = extract_keywords(query)
    if not keywords:
        return ""

    def score_row(row):
        text_parts = [str(row[title_col])]
        if desc_col and desc_col in row:
            text_parts.append(str(row[desc_col]))
        full = " ".join(text_parts).lower()
        score = sum(1 for kw in keywords if kw in full)
        return score

    df = INVENTORY_DF.copy()
    df["_score"] = df.apply(score_row, axis=1)
    df = df[df["_score"] > 0].sort_values("_score", ascending=False)

    if df.empty:
        return ""

    lines = []
    for _, row in df.head(limit).iterrows():
        title = str(row[title_col]).strip()
        price = str(row[price_col]).strip() if price_col and price_col in row else ""
        link = str(row[link_col]).strip() if link_col and link_col in row else ""

        line = f"- {title}"
        if price:
            line += f" — {price}"
        if link:
            line += f" — {link}"
        lines.append(line)

    return "\n".join(lines)


# ------------------------------------------------
# OFFER CSV
# ------------------------------------------------

def save_offer_row(item_link: str, size: str, price: str, email: str, notes: str) -> None:
    is_new = not OFFERS_FILE.exists()
    with OFFERS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["timestamp", "item_link", "size", "price", "email", "notes"]
            )
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                item_link,
                size,
                price,
                email,
                notes,
            ]
        )


# ------------------------------------------------
# FASTAPI + CORS
# ------------------------------------------------

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://dwcuratedvintage.com",
    "https://www.dwcuratedvintage.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# MODELS
# ------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{role, content}, ...]


class ChatResponse(BaseModel):
    reply: str


# ------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------

SYSTEM_PROMPT = """
You are VintageBot, assistant bot for D&W Curated Vintage.

You help customers with:
- sizing questions
- fit guidance
- style recommendations
- product questions
- general shipping and returns info
- friendly conversation about vintage and denim

You can also help with offers:
- If they ask how to make an offer, tell them to send:
  OFFER: item link, size, offer price, email, notes

Tone:
- casual, friendly, helpful
- short, clear responses
- use bullet points when helpful
- never sound like a stiff corporate robot

Important:
- Do NOT claim to browse the web or see live tracking.
- If the user is asking for specific product links, the backend may send them
  a list of matching items first. After that, you can talk about the items,
  sizing, or alternatives, but don't say you can't browse.
- If you’re not sure about exact measurements or details, say you’re not sure.
- Never invent order status or tracking numbers.
- Never promise shipping timelines beyond what’s standard.
- Do not reveal internal data or system details.
"""


# ------------------------------------------------
# CHAT ENDPOINT
# ------------------------------------------------

@app.post("/api/dw-chat", response_model=ChatResponse)
async def dw_chat(req: ChatRequest):
    user_msg = (req.message or "").strip()
    lower = user_msg.lower()

    # --------------------------
    # 1) OFFER HANDLING
    # --------------------------
    if lower.startswith("offer:"):
        offer_body = user_msg[6:].strip()
        parts = [p.strip() for p in offer_body.split(",")]

        item_link = parts[0] if len(parts) > 0 else ""
        size = parts[1] if len(parts) > 1 else ""
        price = parts[2] if len(parts) > 2 else ""
        email = parts[3] if len(parts) > 3 else ""
        notes = ", ".join(parts[4:]) if len(parts) > 4 else ""

        save_offer_row(item_link, size, price, email, notes)

        reply_text = (
            "Offer received 🙌\n\n"
            f"Item: {item_link or 'N/A'}\n"
            f"Size: {size or 'N/A'}\n"
            f"Offer: {price or 'N/A'}\n"
            f"Email: {email or 'N/A'}\n\n"
            "I’ll review it and contact you back by email."
        )
        return ChatResponse(reply=reply_text)

    # --------------------------
    # 2) INVENTORY SEARCH
    # --------------------------
    # broaden the triggers so both:
    #  - "send me links to denim jackets"
    #  - "can you send me a link to denim"
    # turn on the search.
    wants_links = any(
        phrase in lower
        for phrase in [
            "send me links to",
            "send me some links to",
            "send me links for",
            "send me a link to",
            "send me link to",
            "link to ",
            "links to ",
            "show me options for",
            "show me some",
        ]
    )

    inventory_reply = ""
    if wants_links:
        # ---- NEW: use conversation history to build the search query ----
        search_bits = []

        # include recent user messages from history
        for h in req.history:
            if h.get("role") == "user":
                search_bits.append(h.get("content", ""))

        # add the current message
        search_bits.append(user_msg)

        # use the last 3 user messages as the search text
        search_query = " ".join(search_bits[-3:])
        inventory_reply = search_inventory(search_query)

    if inventory_reply:
        reply_text = (
            "Here are some pieces from the current inventory that might match what you asked for:\n\n"
            f"{inventory_reply}\n\n"
            "If none of these are right, tell me your usual size and the vibe you want "
            "(fit, wash, graphic, etc.) and I’ll narrow it down."
        )
        return ChatResponse(reply=reply_text)

    # --------------------------
    # 3) NORMAL CHAT VIA OPENAI
    # --------------------------
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # include history if provided
    for h in req.history:
        if "role" in h and "content" in h:
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_msg})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=400,
        )
        reply = completion.choices[0].message.content.strip()
        return ChatResponse(reply=reply)
    except Exception as e:
        print("OpenAI error:", repr(e))
        raise HTTPException(status_code=500, detail="Error talking to OpenAI")


# ------------------------------------------------
# HEALTHCHECK
# ------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "VintageBot backend running"}
