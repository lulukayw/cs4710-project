import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import glob
import random

FLAVOR_KEYS = ["sweet", "salty", "savory", "spicy", "sour", "rich"]
DATA_PATH = "./CategorizedData"
PERIOD_ORDER = {"Breakfast": 0, "Lunch": 1, "Dinner": 2}


class Meal:
    def __init__(self, id, name, flavors, category, date, period):
        self.id = id
        self.name = name
        self.flavors = flavors  # {"sweet": 2, ...}
        self.category = category
        self.date = date
        self.period = period
        self.flavor_vector = self._build_vector()

    def _build_vector(self):
        return [self.flavors[k] for k in FLAVOR_KEYS]


class MealDatabase:
    def __init__(self):
        self.meals: dict[str, Meal] = {}
        self.scaler = None

    def add_meal(self, meal_dict):
        meal = Meal(
            id=meal_dict["id"],
            name=meal_dict["name"],
            flavors=meal_dict["flavors"],
            category=meal_dict.get("category", ""),
            date=meal_dict.get("date", ""),
            period=meal_dict.get("period", ""),
        )
        self.meals[meal.id] = meal
        return meal

    def bulk_load(self, meal_list):
        for m in meal_list:
            self.add_meal(m)
        self._normalize()

    def _normalize(self):
        meals_list = list(self.meals.values())
        normalized, self.scaler = normalize_meal_vectors(meals_list)
        for meal in normalized:
            self.meals[meal.id].flavor_vector = meal.flavor_vector

    def get(self, meal_id):
        return self.meals.get(meal_id)

    def all_meals(self):
        return list(self.meals.values())

    def filter(self, date=None, period=None, categories=None) -> list[Meal]:
        results = self.all_meals()
        if categories:
            results = [m for m in results if m.category in categories]
        if date:
            results = [m for m in results if m.date == date]
        if period:
            results = [m for m in results if m.period == period]
        return results


class UserProfile:
    def __init__(self, db: MealDatabase):
        self.db = db
        self.raw_inputs = {}  # the 0-5 values the user typed
        self.flavor_vector = None  # normalized vector (same scale as meal vectors)
        self.history = []  # list of {"meal_id", "rating"} dicts

    def input_preferences(self):
        print("\n" + "=" * 50)
        print("Rate your flavor preferences (0 = dislike, 5 = love)")
        print("=" * 50)

        for key in FLAVOR_KEYS:
            while True:
                try:
                    val = float(input(f"  {key.capitalize():10s} (0-5): "))
                    if 0 <= val <= 5:
                        self.raw_inputs[key] = val
                        break
                    else:
                        print("    ⚠ Please enter a number between 0 and 5.")
                except ValueError:
                    print("    ⚠ Invalid input. Enter a number like 3 or 4.5.")

        self._normalize_inputs()
        self._display_profile()

    def _normalize_inputs(self):
        raw_vector = np.array(
            [[self.raw_inputs[k] for k in FLAVOR_KEYS]]
        )  # shape (1, 6)
        self.flavor_vector = self.db.scaler.transform(raw_vector)[0].tolist()

    def _display_profile(self):
        print("\nYour Flavor Profile:")
        print(f"  {'Flavor':<12} {'Raw (0-5)':>10}  {'Normalized':>12}")
        print("  " + "-" * 36)
        for i, key in enumerate(FLAVOR_KEYS):
            raw = self.raw_inputs[key]
            norm = self.flavor_vector[i]
            bar = "█" * int(raw) + "░" * (5 - int(raw))
            print(f"  {key.capitalize():<12} {bar}  {raw:>4.1f}  →  {norm:.3f}")
        print()

    def update_from_history(self):
        if not self.history:
            return

        last = self.history[-1]
        meal = self.db.get(last["meal_id"])
        if not meal:
            return
        profile = np.array(self.flavor_vector)
        direction = np.array(meal.flavor_vector) - profile
        profile += (0.1 if last["rating"] == 1 else -0.1) * direction
        self.flavor_vector = np.clip(profile, 0, 1).tolist()

    def add_to_history(self, meal_id: str, rating: float):
        if not (rating == 0 or rating == 1):
            raise ValueError("Rating must be 0 (dislike) or 1 (like)")
        self.history.append({"meal_id": meal_id, "rating": rating})
        self.update_from_history()


def normalize_meal_vectors(meals):
    """
    Fits a MinMaxScaler on all meal vectors and stores the normalized
    version back on each meal. Returns meals + scaler (save the scaler
    — you'll need it to normalize the user profile too).
    """
    raw_matrix = np.array([m.flavor_vector for m in meals])  # shape: (N, 6)

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_matrix = scaler.fit_transform(raw_matrix)

    for meal, norm_vec in zip(meals, normalized_matrix):
        meal.flavor_vector = norm_vec.tolist()  # overwrite with normalized version

    return meals, scaler


def load_csv_files_to_db(db, file_paths):
    for path in file_paths:
        filename = os.path.basename(path)

        # extract date + period
        base = filename.replace("_filled.csv", "")
        date, period = base.split("-", 3)[:3], base.split("-", 3)[3]
        date = "-".join(date)  # reconstruct date

        df = pd.read_csv(path)

        for i, row in df.iterrows():
            meal_dict = {
                "id": f"{date}_{period}_{i}",
                "name": row["name"],
                "category": row.get("category", ""),
                "date": date,
                "period": period,
                "flavors": {
                    key: float(row[key]) if pd.notna(row[key]) else 0.0
                    for key in FLAVOR_KEYS
                },
            }

            db.add_meal(meal_dict)

    db._normalize()


def score_meals(user_profile, meals):
    results = []
    user_vec = np.array(user_profile.flavor_vector)

    for meal in meals:
        meal_vec = np.array(meal.flavor_vector)

        # Euclidean distance (lower is better)
        euc = np.linalg.norm(user_vec - meal_vec)

        # Cosine similarity (higher is better)
        cos = cosine_similarity([user_vec], [meal_vec])[0][0]

        results.append({"meal": meal, "euclidean": euc, "cosine": cos})

    # Normalize Euclidean → similarity
    max_euc = max(r["euclidean"] for r in results) or 1

    for r in results:
        r["euc_score"] = 1 - (r["euclidean"] / max_euc)

    # Hybrid score
    for r in results:
        r["hybrid"] = 0.5 * r["euc_score"] + 0.5 * r["cosine"]

    return sorted(results, key=lambda x: x["hybrid"], reverse=True)


def get_all_dates_periods(db):
    pairs = {(m.date, m.period) for m in db.all_meals()}

    def sort_key(pair):
        date, period = pair
        return (date, PERIOD_ORDER.get(period, 99))

    return sorted(list(pairs), key=sort_key)


def recommend_with_feedback(db, profile, date, period, categories, epsilon=0.1):
    menu = db.filter(date, period, categories)
    seen_ids = {h["meal_id"] for h in profile.history}
    menu = [m for m in menu if m.id not in seen_ids]

    while True:
        print("\n" + "=" * 50)

        scores = score_meals(profile, menu)

        if random.random() < epsilon:
            print("Exploring: shuffling recommendations...")
            explore_pool = scores[7:15] if len(scores) >= 20 else scores
            top5 = random.sample(explore_pool, min(5, len(explore_pool)))

        top5 = scores[:5]

        print("\nTop 5 Recommendations:")
        print("-" * 50)

        for i, r in enumerate(top5):
            meal = r["meal"]
            print(f"{i+1}. {meal.name} ({meal.category})")
            print(f"   Score: {r['hybrid']:.3f}")

        while True:
            choice = input("\nWhich did you eat? (1-5, or q to quit): ").strip()

            if choice.lower() == "q":
                print("Exiting recommendation loop.")
                return

            if choice in {"1", "2", "3", "4", "5"}:
                idx = int(choice) - 1
                if idx < len(top5):
                    selected = top5[idx]["meal"]
                    break

            print("Please enter a number 1–5 or q.")

        while True:
            feedback = input("Did you like it? (1 = yes, 0 = no): ").strip()
            if feedback in {"0", "1"}:
                rating = int(feedback)
                break
            print("Please enter 1 or 0.")

        profile.add_to_history(selected.id, rating)

        print(f"\nYou selected: {selected.name}")
        print("Profile updated based on your feedback.")


db = MealDatabase()
print("=" * 50)
files = glob.glob(f"{DATA_PATH}/*_filled.csv")
load_csv_files_to_db(db, files)
print("Loaded DB")
print("=" * 50)

profile = UserProfile(db)
profile.input_preferences()

date_filter = "4-08-2026"
category_filter = [
    "Omelet/Quiche/Frittata/Strata",
    "Breakfast",
    "Breakfast Meat",
    "Potato",
    "Pancakes/Waffles/French Toast",
    "Biscuit",
    'Traditional 16"',
    "Burger",
    "Traditional Sandwich",
    "Artisan",
    "Club",
    "Salad Entrée",
    "Roast/Braise",
    "Stir-Fry",
    "Casserole",
    "Burrito/Chimichanga",
    "Taco",
    "Hot Dog/Corn Dog",
    "Cream/Bisque/Puree",
    "BEEF ENTREE",
    "POULTRY ENTREE",
    "Pasta/Noodle (Entree)",
    "Rice",
    "Grains",
    "RICE/GRAINS/LEGUMES",
    "Grilled",
]
period_filter = "Lunch"

date_period_pairs = get_all_dates_periods(db)[-3:]

for date_filter, period_filter in date_period_pairs:
    print("\n" + "=" * 60)
    print(f"Now recommending for {date_filter} | {period_filter}")
    print("=" * 60)

    recommend_with_feedback(
        db,
        profile,
        date_filter,
        period_filter,
        category_filter,
    )

profile._display_profile()
