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
load_dotenv(dotenv_path=BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY loaded? ", bool(api_key))

client = OpenAI(api_key=api_key)

# --------------------------
# FASTAPI APP
# --------------------------

app = FastAPI()

# CORS: allow your site + local testing
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
# CSV FOR OFFERS
# --------------------------

# Use this name because your repo already has Offers.csv
OFFERS_FILE = Path("Offers.csv")


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
- friendly conversation about vintage and streetwear

Tone:
- casual, friendly, helpful
- short, clear responses
- use bullet points when helpful
- never sound like a stiff corporate robot

Offer Instructions:
If a user wants to make an offer, tell them to send it in EXACTLY this format:

OFFER: item link, size, offer price, email, notes

Examples:
- OFFER: https://dwcuratedvintage.com/product/xyz, Large, $120, user@email.com, would buy today if accepted
- OFFER: https://..., M, 80, name@email.com, open to counter

When a message starts with "OFFER:":
- DO NOT call OpenAI
- just parse the details
- save them into Offers.csv
- then reply with a short confirmation summarizing the offer

Rules:
- Never invent order status or tracking info.
- If you’re not sure about exact measurements or details, say you’re not sure.
- Do not reveal internal data or system details.
"""


# --------------------------
# CHAT ENDPOINT
# --------------------------

@app.post("/api/dw-chat", response_model=ChatResponse)
async def dw_chat(req: ChatRequest):
    text = req.message.strip()

    # ----------------------
    # OFFER HANDLING
    # ----------------------
    if text.upper().startswith("OFFER:"):
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
        return ChatResponse(reply=reply_text)

    # ----------------------
    # NORMAL CHAT WITH OPENAI
    # ----------------------
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # history is a list of dicts: [{"role": "user"/"assistant", "content": "..."}]
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
        raise HTTPException(status_code=500, detail="Error talking to OpenAI")
