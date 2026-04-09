import csv
import json
import sys
from pathlib import Path

INPUT_PATH = Path(r"C:\Users\lulu\OneDrive\Desktop\AI\project\OhillData")
OUTPUT_PATH = Path(r"C:\Users\lulu\OneDrive\Desktop\AI\project\GeneratedData")
FLAVOR_PROFILES = ["sweet", "salty", "savory", "spicy", "sour", "rich"]


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


def filter_categories(foods: list[dict]) -> list[dict]:
    filtered = []

    for item in foods:
        category = item.get("category", "").strip()
        filtered.append(
            {
                "name": item.get("name", ""),
                "category": category,
                "description": item.get("description", ""),
                "ingredients": item.get("ingredients", ""),
                "diet_tags": ", ".join(item.get("diet_tags", [])),
                "allergens": ", ".join(item.get("diet_tags", [])),
            }
        )

    return filtered


def write_csv(results: list[dict], output_path: str):
    fieldnames = [
        "name",
        "category",
        "description",
        "ingredients",
        "diet_tags",
        "allergens",
    ] + FLAVOR_PROFILES
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\ndone! saved {len(results)} rows to '{output_path}'")


def main(args):
    # if not args or len(args) < 1:
    #     print("Invalid number of arguments")
    #     return
    # foods = load_food_data(args[0])
    # # print(foods)
    # processed = filter_categories(foods)
    # # print(processed)
    # output_file = OUTPUT_PATH / args[1]
    # write_csv(processed, output_file)\
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    json_files = list(INPUT_PATH.glob("*.json"))

    if not json_files:
        print("No JSON files found.")
        return

    for json_file in json_files:
        print(f"Processing {json_file.name}...")

        foods = load_food_data(json_file)
        processed = filter_categories(foods)

        output_csv = OUTPUT_PATH / f"{json_file.stem}.csv"
        write_csv(processed, output_csv)



if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])
