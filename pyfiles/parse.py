"""
parse.py
------------------
this takes the food json and calls gemini to rate each food on 5 flavor things
(sweet, sour, bitter, spicy, dry) and spits out a csv for the KNN part 

to run it:
    pip install google-genai pydantic python-dotenv
    python parse.py --input foods.json --output flavor_ratings.csv

NOTE: have realized google has a request limit of 10-15 per MINUTE and 20-50 per day... this probably isnt feasible without spending money

most of the credit of this file goes to gemini with some tweaking and bug fixing manually done
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# using flash bc its fast and good with data
MODEL = "gemini-2.5-flash"
FLAVOR_PROFILES = ["sweet", "sour", "bitter", "spicy", "dry"]
RATE_LIMIT_DELAY = 7  # had to add this after getting rate limited lol - HAD TO INCREASE TO 7 SECONDS BECAUSE OF STUPID LIMITS
load_dotenv()

# ---------------------------------------------------------------------------
# Prompt + schema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a culinary flavor analysis assistant.
When given a food item, you rate it on exactly 5 flavor profiles on a scale of 0.0 to 5.0.
Base your ratings on the name, description, AND ingredients provided.
All values must be floats between 0.0 and 5.0, using one decimal place.
"""


# pydantic makes sure gemini gives back the right format
class FlavorRating(BaseModel):
    sweet: float = Field(description="presence of sweetness (sugar, honey, fruit, etc.)")
    sour: float = Field(description="presence of acidity or tartness")
    bitter: float = Field(description="presence of bitterness (coffee, dark greens, etc.)")
    spicy: float = Field(description="presence of heat or spice (pepper, chili, etc.)")
    dry: float = Field(description="dryness / absence of moisture or sauce (crackers, toast, etc.)")


def build_user_prompt(item: dict) -> str:
    # just concatenate fields that exist for the food item
    parts = [f"Food name: {item.get('name', 'Unknown')}"]
    if item.get("description"):
        parts.append(f"Description: {item['description']}")
    if item.get("ingredients"):
        parts.append(f"Ingredients: {item['ingredients']}")
    if item.get("category"):
        parts.append(f"Category: {item['category']}")
    if item.get("diet_tags"):
        parts.append(f"Diet tags: {', '.join(item['diet_tags'])}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# call the api
# ---------------------------------------------------------------------------


def get_flavor_ratings(client: genai.Client, item: dict, retries: int = 3) -> dict | None:
    prompt = build_user_prompt(item)

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,  # keep it deterministic
                    response_mime_type="application/json",
                    response_schema=FlavorRating,
                ),
            )

            ratings = json.loads(response.text)

            # clamp everything to 0-5 just in case gemini goes rogue
            result = {}
            for flavor in FLAVOR_PROFILES:
                val = float(ratings.get(flavor, 0.0))
                result[flavor] = round(max(0.0, min(5.0, val)), 1)

            return result

        except json.JSONDecodeError as e:
            print(f"  [attempt {attempt}] couldn't parse json for '{item.get('name')}': {e}")
        except Exception as e:
            print(f"  [attempt {attempt}] something broke for '{item.get('name')}': {e}")
            if attempt < retries:
                time.sleep(2**attempt)  # exponential backoff

    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def load_food_data(path: str) -> list[dict]:
    # handles both regular json arrays and newline-delimited json
    text = Path(path).read_text(encoding="utf-16")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # fallback: try wrapping in brackets if its comma separated objects
    try:
        data = json.loads(f"[{text.rstrip().rstrip(',')}]")
        return data
    except json.JSONDecodeError as e:
        print(f"ERROR: couldn't parse '{path}': {e}")
        sys.exit(1)


def write_csv(results: list[dict], output_path: str):
    fieldnames = ["name", "category"] + FLAVOR_PROFILES
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\ndone! saved {len(results)} rows to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="rate food flavor profiles using gemini")
    parser.add_argument("--input", required=True, help="input json file")
    parser.add_argument("--output", default="flavor_ratings.csv", help="output csv path")
    parser.add_argument("--limit", type=int, default=None, help="only process this many items (useful for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: need GEMINI_API_KEY in your .env file")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    foods = load_food_data(args.input)
    if args.limit:
        foods = foods[: args.limit]

    print(f"processing {len(foods)} foods...\n")

    results = []
    failed = []

    for i, item in enumerate(foods, 1):
        name = item.get("name", f"item_{i}")
        print(f"[{i}/{len(foods)}] {name}")

        ratings = get_flavor_ratings(client, item)

        if ratings:
            row = {"name": name, "category": item.get("category", ""), **ratings}
            results.append(row)
            print(
                f"        sweet={ratings['sweet']} sour={ratings['sour']} "
                f"bitter={ratings['bitter']} spicy={ratings['spicy']} dry={ratings['dry']}"
            )
        else:
            print(f"  skipping '{name}', all retries failed")
            failed.append(name)

        time.sleep(RATE_LIMIT_DELAY)

    write_csv(results, args.output)

    if failed:
        print(f"\n{len(failed)} item(s) failed: {failed}")


if __name__ == "__main__":
    main()