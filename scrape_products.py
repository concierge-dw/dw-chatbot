# scrape_products.py
import csv
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.dwcuratedvintage.com"
START_URL = BASE_URL + "/shop"
OUTPUT_FILE = "products.csv"


def get_product_links() -> list[str]:
    """
    Grab all product links from the main /shop page.
    Squarespace product URLs look like: /shop/p/akira-1988-vintage-single-stitch-tee
    """
    print(f"Fetching product links from {START_URL} ...")
    resp = requests.get(START_URL, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    links: list[str] = []

    # any <a href="/shop/p/...">
    for a in soup.select('a[href^="/shop/p/"]'):
        href = a.get("href")
        if not href:
            continue
        if href not in links:
            links.append(href)

    print(f"Found {len(links)} product links")
    return links


def scrape_product_page(relative_url: str) -> dict:
    """
    Scrape ONE product page and return basic info.
    """
    full_url = urljoin(BASE_URL, relative_url)
    print(f"Scraping {full_url} ...")

    resp = requests.get(full_url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # ---- title ----
    title_el = soup.select_one("h1.product-title") or soup.select_one("h1")
    title = title_el.get_text(strip=True) if title_el else "Unknown Item"

    # ---- price ----
    price_el = (
        soup.select_one("#main-product-price")
        or soup.select_one(".sqs-money-native")
        or soup.select_one(".product-price")
    )
    price = price_el.get_text(strip=True) if price_el else "Unknown Price"

    # ---- description ----
    desc_el = soup.select_one(".product-description") or soup.select_one(".sqs-html-content")
    description = desc_el.get_text("\n", strip=True) if desc_el else ""

    # crude “size/measurements” slice from description so the bot can show something
    size_bits = []
    for line in description.splitlines():
        lower = line.lower()
        if any(
            kw in lower
            for kw in ["measurements", "pit to pit", "pit-to-pit", "waist", "length", "inseam"]
        ):
            size_bits.append(line.strip())
    size_text = " ".join(size_bits)

    return {
        "title": title,
        "price": price,
        "size": size_text,
        "url": full_url,
        "description": description,
    }


def main():
    links = get_product_links()
    rows = []

    for rel in links:
        try:
            rows.append(scrape_product_page(rel))
        except Exception as e:
            print(f"!! Error scraping {rel}: {e}")

    if not rows:
        print("No products scraped. Nothing to write.")
        return

    fieldnames = ["title", "price", "size", "url", "description"]
    print(f"Writing {len(rows)} products to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()
