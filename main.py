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


# --------------------------
# ENV + OPENAI CLIENT
# --------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY loaded? ", bool(api_key))

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env file.")

client = OpenAI(api_key=api_key)


# --------------------------
# FILE PATHS
# --------------------------

PRODUCTS_FILE = BASE_DIR / "products.csv"
OFFERS_FILE = BASE_DIR / "Offers.csv"


# --------------------------
# FASTAPI APP + CORS
# --------------------------

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


# --------------------------
# Pydantic MODELS
# --------------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# --------------------------
# SYSTEM PROMPT
# --------------------------

SYSTEM_PROMPT = """
You are VintageBot, the personal shopping and sizing assistant for D&W Curated Vintage.

You help with:
- sizing and fit questions
- style suggestions
- what to pair with what
- general product questions
- basic shipping/returns info (only in general terms, never exact tracking)

Tone:
- casual, friendly, and concise
- talk like a helpful shop assistant, not a corporate robot
- use bullet points when it makes things clearer

Important rules:
- Do NOT claim to browse the live website.
- Do NOT invent or guess tracking numbers, order status, or exact shipping dates.
- If you don’t know something, say you’re not sure instead of making it up.
- If the user clearly wants links to inventory, it’s okay that another system has already picked the pieces.
"""


# --------------------------
# INVENTORY SEARCH
# --------------------------

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "your",
    "shirt", "tee", "tshirt", "t-shirt", "top",
    "vintage", "single", "stitch", "print", "graphic",
    "size", "mens", "womens", "unisex"
}


def tokenize(text: str):
    """Simple tokenizer: lowercase words, strip junk, drop tiny/boring words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def search_inventory(query: str) -> str | None:
    """Return a formatted string of matching items, or None if nothing good."""
    if not PRODUCTS_FILE.exists():
        print("products.csv not found; skipping inventory search.")
        return None

    try:
        df = pd.read_csv(PRODUCTS_FILE)
    except Exception as e:
        print("Error reading products.csv:", e)
        return None

    query_words = tokenize(query)
    if not query_words:
        return None

    matches: list[tuple[int, dict]] = []

    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        desc = str(row.get("description", ""))
        text_words = tokenize(f"{title} {desc}")

        if not text_words:
            continue

        # score = number of query words that appear in this item's text
        score = sum(1 for w in query_words if w in text_words)

        # require at least 2 matches so we don't return junk for vague queries
        if score >= 2:
            matches.append(
                (
                    score,
                    {
                        "title": title or "Unknown Item",
                        "price": str(row.get("price", "")).strip(),
                        "link": str(row.get("link", "")).strip(),
                    },
                )
            )

    if not matches:
        return None

    # best matches first
    matches.sort(key=lambda x: x[0], reverse=True)
    top = matches[:5]

    lines = [
        "Here are some pieces from the current inventory that might match what you asked for:\n"
    ]
    for score, item in top:
        title = item["title"]
        price = item["price"]
        link = item["link"] or "#"
        price_str = f" — {price}" if price else ""
        lines.append(f"- {title}{price_str}\n{link}\n")

    return "\n".join(lines)


# --------------------------
# OFFER CSV HELPERS
# --------------------------

def save_offer_row(item_link: str, size: str, price: str, email: str, notes: str) -> None:
    """Append an offer to Offers.csv (create file with header if missing)."""
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


def handle_offer_message(text: str) -> str:
    """
    Parse an OFFER: message and save to CSV.
    Expected format:
      OFFER: item link, size, offer price, email, notes
    """
    offer_body = text[6:].strip()  # remove "OFFER:" prefix
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
    return reply_text


# --------------------------
# OPENAI CHAT HELPER
# --------------------------

def call_openai(user_msg: str) -> str:
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=400,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", repr(e))
        raise HTTPException(status_code=500, detail="Error talking to OpenAI")


# --------------------------
# API ENDPOINTS
# --------------------------

@app.post("/", response_model=ChatResponse)
async def vintagebot_chat(req: ChatRequest):
    """
    Main chat endpoint used by the Squarespace widget.

    Request body: { "message": "..." }
    Response body: { "reply": "..." }
    """
    text = (req.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    # 1) OFFER HANDLING
    if text.upper().startswith("OFFER:"):
        reply = handle_offer_message(text)
        return ChatResponse(reply=reply)

    # 2) INVENTORY SEARCH
    inventory_reply = search_inventory(text)
    if inventory_reply:
        return ChatResponse(reply=inventory_reply)

    # 3) FALL BACK TO OPENAI
    reply = call_openai(text)
    return ChatResponse(reply=reply)


@app.get("/")
async def root():
    """Simple health check."""
    return {"status": "ok", "message": "VintageBot backend running"}
