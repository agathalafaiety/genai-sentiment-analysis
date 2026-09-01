"""Configuracao central do projeto."""

from pathlib import Path

RANDOM_SEED = 42
LABELS = ("negative", "neutral", "positive")
RATING_TO_LABEL = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "B2W-Reviews01.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

B2W_REVISION = "4639429ec698d7821fc99a0bc665fa213d9fcd5a"
B2W_URL = (
    "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/"
    f"{B2W_REVISION}/B2W-Reviews01.csv"
)
B2W_LICENSE = "CC BY-NC-SA 4.0"

TRANSFORMER_MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
TRANSFORMER_REVISION = "cf991100d706c13c0a080c097134c05b7f436c45"

LOCAL_GENAI_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_GENAI_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
