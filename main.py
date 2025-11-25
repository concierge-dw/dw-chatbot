from pathlib import Path

from dotenv import load_dotenv

# Force-load .env from this folder
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import csv
from datetime import datetime
from openai import OpenAI

# --------------------------
# CONFIG
# --------------------------

api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY loaded? ", bool(api_key))

# OpenAI client (new style)
client = OpenAI(api_key=api_key)

origins = [
    "https://your-squarespace-site.com",
    "https://www.your-squarespace-site.com",
]

# File where offers get saved
OFFERS_FILE = Path("offers.csv")

# --------------------------
# FASTAPI INIT
# --------------------------

app = FastAPI()

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
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str


# --------------------------
# PROMPT
# --------------------------

SYSTEM_PROMPT = """
You are D&W Stylist Bot, the casual, streetwear-savvy assistant for D&W Curated Vintage.

You help with:
- sizing
- fits
- style recs
- questions about items
- general shipping/returns questions

MAKE AN OFFER MODE:
- If user says they want to make an offer, instruct them to reply using this exact format:

  OFFER: item link, size, offer price, email, notes

- When a message starts with 'OFFER:', do NOT call OpenAI. Instead:
  • Save the offer to the CSV file
  • Reply with a confirmation message

NEVER invent order status or fake info.
"""


# --------------------------
# CHAT ENDPOINT
# --------------------------

@app.post("/api/dw-chat", response_model=ChatResponse)
async def dw_chat(req: ChatRequest):
    text = req.message.strip()

    # --------------------------------------------
    # OFFER HANDLING
    # --------------------------------------------
    if text.upper().startswith("OFFER:"):
        offer_body = text[6:].strip()
        parts = [p.strip() for p in offer_body.split(",")]

        item_link = parts[0] if len(parts) > 0 else ""
        size = parts[1] if len(parts) > 1 else ""
        price = parts[2] if len(parts) > 2 else ""
        email = parts[3] if len(parts) > 3 else ""
        notes = ", ".join(parts[4:]) if len(parts) > 4 else ""

        is_new_file = not OFFERS_FILE.exists()
        with OFFERS_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new_file:
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

        reply_text = (
            "Offer received 🙌\n\n"
            f"Item: {item_link or 'N/A'}\n"
            f"Size: {size or 'N/A'}\n"
            f"Offer: {price or 'N/A'}\n"
            f"Email: {email or 'N/A'}\n\n"
            "I'll review it and contact you back by email."
        )
        return ChatResponse(reply=reply_text)

    # --------------------------------------------
    # NORMAL AI CHAT
    # --------------------------------------------

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history:
        if "role" in h and "content" in h:
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": text})

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
        raise HTTPException(status_code=500, detail=str(e))
