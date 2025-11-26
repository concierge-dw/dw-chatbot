import requests
from bs4 import BeautifulSoup
import csv

BASE_URL = "https://www.dwcuratedvintage.com"
START_URL = BASE_URL + "/shop"

def get_product_links():
    print("Fetching product links...")
    response = requests.get(START_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a in soup.select("a[href*='/product/']"):
        href = a.get("href")
        if href and href not in links:
            links.append(href)

    print(f"Found {len(links)} product links")
    return links


def scrape_product_page(url):
    full_url = BASE_URL + url
    print(f"Scraping: {full_url}")

    response = requests.get(full_url)
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.find("h1")
    title = title.text.strip() if title else "Unknown Item"

    price = soup.find("span", class_="sqs-money-native")
    price = price.text.strip() if price else "Unknown Price"

    desc_block = soup.find("div", class_="ProductItem-details-excerpt")
    description = desc_block.text.strip() if desc_block else ""

    return {
        "name": title,
        "price": price,
        "description": description,
        "link": full_url
    }


def main():
    product_links = get_product_links()
    products = []

    for link in product_links:
        data = scrape_product_page(link)
        products.append(data)

    with open("products.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "price", "description", "link"])
        writer.writeheader()
        writer.writerows(products)

    print(f"\n✅ Wrote {len(products)} products to products.csv")


if __name__ == "__main__":
    main()

