"""All parsing and feature building for the phone price project.

The only function other code needs is build_features(df). It takes the raw
CSV (as read by pandas) and returns a new DataFrame with clean numeric
columns added. It does NOT drop any rows - the cleaning notebook decides
which rows to drop, so every drop is visible and counted there.
"""
import re

import numpy as np
import pandas as pd

# Numeric features used by every model (see README / notebooks)
NUMERIC_FEATURES = [
    "log2_ram", "log2_storage", "battery_mah", "screen_in", "main_back_mp",
    "front_mp", "n_back_cameras", "weight_g", "year",
]
# Models use chip_group, not chip_family (see chip_group_from for why)
CATEGORICAL_FEATURES = ["brand", "chip_group"]
TARGET = "log_price"

# A number followed by "MP", e.g. "50MP", "12.2MP", "48 MP"
MP_PATTERN = r"(\d+(?:\.\d+)?)\s*MP"
# Storage written in the model name, e.g. "128GB", "1TB"
STORAGE_PATTERN = r"(\d+)\s*(GB|TB)\b"


def first_number(text):
    """Return the first number in a string (commas removed), or NaN."""
    if pd.isna(text):
        return np.nan
    match = re.search(r"\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(match.group()) if match else np.nan


def parse_price(text):
    """'INR 79,999' -> 79999.  Also handles Indian commas like '1,49,999'."""
    if pd.isna(text):
        return np.nan
    digits = re.sub(r"[^\d]", "", str(text))
    return float(digits) if digits else np.nan


def parse_ram(text):
    """'8GB' -> 8.  '8GB / 12GB' (several variants) -> 8, the base variant."""
    numbers = re.findall(r"(\d+(?:\.\d+)?)\s*GB", str(text))
    return min(float(n) for n in numbers) if numbers else np.nan


def camera_megapixels(text):
    """All megapixel numbers in a camera string, e.g. '50MP + 8MP' -> [50, 8]."""
    return [float(n) for n in re.findall(MP_PATTERN, str(text), flags=re.IGNORECASE)]


def largest_mp(text):
    """The main camera is taken to be the one with the most megapixels."""
    mps = camera_megapixels(text)
    return max(mps) if mps else np.nan


def count_cameras(text):
    """Number of cameras = number of 'xxMP' parts in the string."""
    n = len(camera_megapixels(text))
    return n if n > 0 else np.nan


def storage_from_name(name):
    """'iPhone 16 128GB' -> 128,  'iPad Pro 1TB' -> 1024,  no storage -> NaN."""
    matches = re.findall(STORAGE_PATTERN, str(name), flags=re.IGNORECASE)
    if not matches:
        return np.nan
    size, unit = matches[-1]
    return float(size) * (1024 if unit.upper() == "TB" else 1)


def model_group_from(brand, name):
    """Model name with the storage part removed, so that all storage
    variants of one phone share a group. Used ONLY for the train/test split."""
    base = re.sub(STORAGE_PATTERN, "", str(name), flags=re.IGNORECASE)
    base = re.sub(r"\s+", " ", base).strip().lower()
    return f"{brand.lower()} | {base}"


def chip_family_from(processor):
    """Map a processor string to a small set of chip families.
    The order of the checks matters (most specific first)."""
    p = str(processor)
    if re.search(r"\bA\d+[A-Z]?\s*(Bionic|Pro)", p):
        return "apple_a"
    if "Tensor" in p:
        return "tensor"
    if "Kirin" in p:
        return "kirin"
    if "Exynos" in p:
        return "exynos"
    if re.search(r"Unisoc|Spreadtrum", p):
        return "unisoc"
    if re.search(r"Helio|MediaTek G\d", p):
        return "helio"
    if re.search(r"Dimensity [89]\d{3}", p):
        return "dimensity_high"
    if "Dimensity" in p:
        return "dimensity_other"
    # Flagship Snapdragon: "8 Gen x", "8+ Gen x", "8s Gen x", "8 Elite",
    # and the older 3-digit flagships 835 ... 888.
    if re.search(r"Snapdragon 8(\+|s)? (Gen|Elite)|Snapdragon 8\d\d", p):
        return "snapdragon_8"
    if re.search(r"Snapdragon|MSM", p):
        return "snapdragon_other"
    return "other"


def chip_group_from(chip_family):
    """The chip category used by the MODELS.

    Same as chip_family, except Apple A-series chips are grouped with flagship
    Snapdragon 8 chips. Reason: only Apple uses Apple chips, so an 'apple_a' chip
    column would be identical to the 'Apple' brand column. A linear model cannot
    tell two identical columns apart (XᵀX has a zero eigenvalue), so the Apple
    brand premium could not be estimated. Grouping assumes an Apple chip is worth
    about the same as a flagship Snapdragon chip of the same year - both are the
    top chip of their generation.
    """
    if chip_family in ("apple_a", "snapdragon_8"):
        return "flagship_sd8_or_apple"
    return chip_family


def build_features(df):
    """Raw dataset -> DataFrame with parsed numeric columns and features.

    Keeps the original columns (so parsing can be checked) and adds:
    price_inr, ram_gb, storage_gb, weight_g, battery_mah, screen_in,
    front_mp, main_back_mp, n_back_cameras, year, brand, chip_family,
    chip_group, model_group, is_tablet, log2_ram, log2_storage, log_price.
    """
    out = df.copy()

    out["price_inr"] = out["Launched Price (India)"].apply(parse_price)
    out["ram_gb"] = out["RAM"].apply(parse_ram)
    out["storage_gb"] = out["Model Name"].apply(storage_from_name)
    out["weight_g"] = out["Mobile Weight"].apply(first_number)
    out["battery_mah"] = out["Battery Capacity"].apply(first_number)
    out["screen_in"] = out["Screen Size"].apply(first_number)  # main screen for foldables
    out["front_mp"] = out["Front Camera"].apply(largest_mp)
    out["main_back_mp"] = out["Back Camera"].apply(largest_mp)
    out["n_back_cameras"] = out["Back Camera"].apply(count_cameras)
    out["year"] = out["Launched Year"].astype(int)

    # Brand: strip spaces and merge different spellings of the same brand
    brand_fixes = {"Poco": "POCO"}
    out["brand"] = out["Company Name"].str.strip().replace(brand_fixes)

    out["chip_family"] = out["Processor"].apply(chip_family_from)
    out["chip_group"] = out["chip_family"].apply(chip_group_from)
    out["model_group"] = [model_group_from(b, n) for b, n in zip(out["brand"], out["Model Name"])]
    out["is_tablet"] = out["Model Name"].str.contains(r"pad|tab", case=False)

    # Transformed features. log2 makes "doubling" one unit of change.
    out["log2_ram"] = np.log2(out["ram_gb"])
    out["log2_storage"] = np.log2(out["storage_gb"])
    out["log_price"] = np.log(out["price_inr"])
    return out
