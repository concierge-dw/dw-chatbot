from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow Squarespace frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load product data
PRODUCTS_FILE = "products.csv"
products_df = None

if os.path.exists(PRODUCTS_FILE):
    try:
        products_df = pd.read_csv(PRODUCTS_FILE)
        print(f"✅ Loaded {len(products_df)} products from products.csv")
    except Exception as e:
        print("❌ Error loading products.csv:", e)
else:
    print("❌ products.csv not found!")

class ChatRequest(BaseModel):
    message: str

def search_inventory(query):
    if products_df is None:
        return None

    query = query.lower()
    matches = products_df[
        products_df["title"].str.lower().str.contains(query, na=False)
    ]

    if matches.empty:
        return None

    results = []
    for _, row in matches.iterrows():
        title = row["title"]
        link = row["link"]
        price = row.get("price", "")

        results.append(f"🔗 {title} — {price}\n{link}")

    return "\n\n".join(results)

@app.post("/")
async def chat(req: ChatRequest):
    user_msg = req.message

    # Try product matching first
    inventory_reply = search_inventory(user_msg)

    if inventory_reply:
        return {"reply": f"Here are items matching your request:\n\n{inventory_reply}"}

    # If no match, fall back to AI
    completion = client.responses.create(
        model="gpt-4.1-mini",
        input=f"You are VintageBot, a helpful vintage clothing assistant.\nUser: {user_msg}"
    )

    reply = completion.output[0].content[0].text
    return {"reply": reply}

@app.get("/")
def root():
    return {"status": "ok", "message": "VintageBot backend running"}
