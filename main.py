from pathlib import Path
from datetime import datetime
import os
import csv

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# --------------------------
# ENV + BASE DIR
# --------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment")

client = OpenAI(api_key=api_key)

# --------------------------
# FASTAPI + CORS
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
# Pydantic models
# --------------------------

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # not used yet, but kept for future


class ChatResponse(BaseModel):
    reply: str


# --------------------------
# Offers CSV
# --------------------------

OFFERS_FILE = BASE_DIR / "Offers.csv"


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


# --------------------------
# Inventory CSV (products.csv)
# --------------------------

PRODUCTS_FILE = BASE_DIR / "products.csv"
PRODUCT_ROWS: list[dict] = []


def load_products() -> None:
    """Load products.csv into memory as a list of dicts."""
    global PRODUCT_ROWS
    if not PRODUCTS_FILE.exists():
        print("products.csv not found, inventory search disabled.")
        PRODUCT_ROWS = []
        return

    with PRODUCTS_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        PRODUCT_ROWS = list(reader)

    print(f"Loaded {len(PRODUCT_ROWS)} products from products.csv")


load_products()


def search_inventory(query: str, max_results: int = 6) -> list[dict]:
    """Very simple keyword search over title/description/tags."""
    if not query or not PRODUCT_ROWS:
        return []

    q = query.lower()
    terms = [t for t in q.split() if len(t) > 2]
    if not terms:
        return []

    scored: list[tuple[int, dict]] = []

    for row in PRODUCT_ROWS:
        haystack = " ".join(
            [
                row.get("title", ""),
                row.get("description", ""),
                row.get("tags", ""),
            ]
        ).lower()

        score = sum(1 for t in terms if t in haystack)
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:max_results]]


def format_inventory_results(rows: list[dict]) -> str:
    """Turn matching rows into a markdown list with links."""
    lines = []
    for row in rows:
        title = row.get("title", "Untitled item").strip()
        price = (row.get("price") or "").strip()
        link = (row.get("link") or row.get("url") or "").strip()

        line = f"- [{title}]({link})" if link else f"- {title}"
        if price:
            line += f" — {price}"
        lines.append(line)

    return "\n".join(lines)


# --------------------------
# System prompt
# --------------------------

SYSTEM_PROMPT = """
You are VintageBot, assistant bot for D&W Curated Vintage.

You help customers with:
- sizing and fit questions (including conversions between men's & women's sizing)
- style recommendations and outfit ideas
- questions about specific pieces (fabric, vibe, how to wear it)
- basic info about shipping, returns, and offers

Tone:
- casual, friendly, short
- no corporate speak
- use bullet points when helpful
- if the user sounds frustrated, acknowledge it and keep things simple

Inventory rules:
- You sometimes receive a short list of matching inventory items in markdown list form.
- If items are provided, use them FIRST when suggesting links.
- If nothing in the snippet fits, say so honestly and give general advice or ask for more details.
- Never claim to see the full site or live stock; you only know what the user tells you and what is in the snippet.

Shipping / tracking rules:
- You cannot see live tracking or order status.
- You can give general expectations (e.g., 'most orders ship within X days' if the user has told you),
  but you must never invent detailed tracking updates.

Offers:
- Users can send offers in this exact format:
  OFFER: item link, size, offer price, email, notes
- When the user sends an 'OFFER:' message, the backend saves it and sends a confirmation message.
- Do NOT ask them to re-send offers in a different format.
"""


# --------------------------
# Main chat endpoint
# --------------------------

@app.post("/api/dw-chat", response_model=ChatResponse)
async def dw_chat(req: ChatRequest):
    text = (req.message or "").strip()
    if not text:
        return ChatResponse(reply="Tell me what you’re looking for and I’ll help you shop the vintage.")

    # ----------------------
    # OFFER HANDLING
    # ----------------------
    if text.upper().startswith("OFFER:"):
        offer_body = text[6:].strip()  # remove "OFFER:"
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

    # ----------------------
    # INVENTORY SEARCH
    # ----------------------
    matches = search_inventory(text)
    if matches:
        formatted = format_inventory_results(matches)
        reply = (
            "Here are some pieces from the current inventory that might match what you asked for:\n\n"
            f"{formatted}\n\n"
            "If none of these are right, tell me your usual size and the vibe (fit, wash, graphic, etc.) "
            "and I’ll narrow it down."
        )
        return ChatResponse(reply=reply)

    # ----------------------
    # FALL BACK TO OPENAI
    # ----------------------
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=400,
        )
        reply_text = completion.choices[0].message.content.strip()
        return ChatResponse(reply=reply_text)
    except Exception as e:
        print("OpenAI error:", repr(e))
        raise HTTPException(status_code=500, detail="Error talking to OpenAI")


# --------------------------
# Health check root
# --------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "VintageBot backend running"}
