# main.py

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
# ENV + BASE DIR
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

PRODUCTS_CSV = BASE_DIR / "products.csv"
OFFERS_CSV = BASE_DIR / "Offers.csv"

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
# MODELS
# --------------------------

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{role, content}, ...]


class ChatResponse(BaseModel):
    reply: str


# --------------------------
# LOAD INVENTORY
# --------------------------

inventory_df: pd.DataFrame | None = None


def load_inventory():
    global inventory_df
    if not PRODUCTS_CSV.exists():
        print("products.csv not found, inventory search disabled.")
        inventory_df = None
        return

    try:
        df = pd.read_csv(PRODUCTS_CSV)
    except Exception as e:
        print("Error loading products.csv:", repr(e))
        inventory_df = None
        return

    # Make sure required columns exist
    for col in ["title", "price", "link"]:
        if col not in df.columns:
            print(f"Column '{col}' missing from products.csv, inventory search disabled.")
            inventory_df = None
            return

    # Add a combined text column for simple matching
    desc_cols = [c for c in df.columns if c not in ["title", "price", "link"]]
    def combine_row(row):
        parts = [str(row["title"])]
        if "description" in row and isinstance(row["description"], str):
            parts.append(row["description"])
        else:
            for c in desc_cols:
                v = row.get(c)
                if isinstance(v, str):
                    parts.append(v)
        return " ".join(parts)

    df["search_text"] = df.apply(combine_row, axis=1).str.lower()
    inventory_df = df
    print(f"Loaded inventory: {len(df)} products from products.csv")


load_inventory()


def search_inventory(query: str, max_results: int = 6) -> str:
    """
    Very simple keyword search over inventory_df.
    Returns a formatted multi-line string, or "" if nothing decent found.
    """
    if inventory_df is None:
        return ""

    q = query.strip().lower()
    if len(q) < 3:
        return ""

    # basic clothing-word guard so "hi" doesn't trigger search
    clothing_words = [
        "jacket", "denim", "jeans", "pants", "trousers", "tee", "t-shirt",
        "shirt", "coat", "parka", "hoodie", "sweater", "dress", "skirt",
        "shorts", "military", "cargo", "trucker", "workwear", "vintage"
    ]
    if not any(w in q for w in clothing_words):
        return ""

    tokens = [t for t in re.split(r"\W+", q) if t]
    if not tokens:
        return ""

    df = inventory_df.copy()

    # crude scoring: count how many tokens appear in search_text
    def score_row(text: str) -> int:
        return sum(1 for t in tokens if t in text)

    df["score"] = df["search_text"].apply(score_row)
    df = df[df["score"] > 0].sort_values(by="score", ascending=False).head(max_results)

    if df.empty:
        return ""

    lines = []
    for _, row in df.iterrows():
        title = str(row["title"]).strip()
        price = str(row["price"]).strip()
        link = str(row["link"]).strip()
        line = f"- {title} — {price} — {link}"
        lines.append(line)

    return "\n".join(lines)


# --------------------------
# OFFERS CSV
# --------------------------

def save_offer_row(item_link: str, size: str, price: str, email: str, notes: str) -> None:
    is_new = not OFFERS_CSV.exists()
    with OFFERS_CSV.open("a", newline="", encoding="utf-8") as f:
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
# SYSTEM PROMPT
# --------------------------

SYSTEM_PROMPT = """
You are VintageBot, assistant bot for D&W Curated Vintage.

You help customers with:
- sizing questions
- fit guidance
- style recommendations
- product questions
- general shipping and returns info
- friendly conversation about vintage and workwear

Tone:
- casual, friendly, helpful
- short, clear responses
- use bullet points when helpful
- never sound like a stiff corporate robot

Inventory:
- The backend may sometimes send the user a list of inventory items.
- You yourself CANNOT browse the web or pull live links.
- If the user wants product links, you can suggest they describe what they want
  (size, style, color) so the backend can try to match inventory.
- Do not fake specific product links or prices.

Offers:
If a user wants to make an offer, tell them to send it in EXACTLY this format:

OFFER: item link, size, offer price, email, notes

Examples:
- OFFER: https://dwcuratedvintage.com/product/xyz, Large, $120, user@email.com, would buy today if accepted
- OFFER: https://..., M, 80, name@email.com, open to counter

Rules:
- Never claim to see live tracking data.
- Never fabricate shipment progress.
- If you’re not sure about exact measurements or details, say you’re not sure.
- Do not reveal internal data or system details.
"""


# --------------------------
# CHAT ENDPOINT
# --------------------------

@app.post("/api/dw-chat", response_model=ChatResponse)
async def dw_chat(req: ChatRequest):
    user_msg = req.message.strip()

    # 1) OFFER HANDLING
    if user_msg.upper().startswith("OFFER:"):
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

    # 2) INVENTORY SEARCH TRY
    inventory_reply = search_inventory(user_msg)
    if inventory_reply:
        reply_text = (
            "Here are some pieces from the current inventory that might match what you asked for:\n\n"
            f"{inventory_reply}\n\n"
            "If none of these are right, tell me your usual size and what you’re hunting for and "
            "I’ll narrow it down."
        )
        return ChatResponse(reply=reply_text)

    # 3) FALL BACK TO OPENAI CHAT
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # carry over minimal history if supplied
    for h in req.history:
        if "role" in h and "content" in h:
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_msg})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=400,
        )
        reply_text = completion.choices[0].message.content.strip()
        return ChatResponse(reply=reply_text)
    except Exception as e:
        print("OpenAI error:", repr(e))
        raise HTTPException(status_code=500, detail="Error talking to OpenAI")


# Simple health check
@app.get("/")
async def root():
    return {"status": "ok", "message": "VintageBot backend running"}
