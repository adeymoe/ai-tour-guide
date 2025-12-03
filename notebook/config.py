# ==========================================
# AI TOUR GUIDE - CONFIGURATION
# Central configuration file for all settings
# ==========================================
from dotenv import load_dotenv
import os

load_dotenv()

# ==========================================
# CITY DATABASE
# ==========================================

AVAILABLE_CITIES = {
    "Amsterdam, Netherlands": (52.3676, 4.9041),
    "Paris, France": (48.8566, 2.3522),
    "Barcelona, Spain": (41.3851, 2.1734),
    "Rome, Italy": (41.9028, 12.4964),
    "London, United Kingdom": (51.5074, -0.1278)
}

# ==========================================
# CACHE SETTINGS
# ==========================================

CACHE_DIR = "city_cache"

# ==========================================
# TOUR PLANNING PARAMETERS
# ==========================================

# Walking speed (km/h)
WALKING_SPEED_KMH = 5.0

# Time spent at each attraction (minutes)
VISIT_TIME_PER_ATTRACTION_MIN = 15

# Buffer time as percentage of total time
BUFFER_TIME_PERCENT = 0.15

# Maximum walking distance per leg (km)
MAX_WALK_LEG_KM = 3.0

# Minimum attraction score to consider
MIN_ATTRACTION_SCORE = 0.3

# Maximum attractions per tour
MAX_ATTRACTIONS_PER_TOUR = 15

# ==========================================
# A* ALGORITHM PARAMETERS
# ==========================================

# Weight for attraction scoring (higher = prefer popular attractions)
ATTRACTION_WEIGHT = 1.5

# Heuristic weight for A* (higher = faster but less optimal)
HEURISTIC_WEIGHT = 1.2

# ==========================================
# VISUALIZATION SETTINGS
# ==========================================

# Default map zoom level
DEFAULT_MAP_ZOOM = 13

# Map tile style
MAP_TILES = "OpenStreetMap"

# Output file for static maps
MAP_OUTPUT_FILE = "tour_map.html"

# ==========================================
# LLM CONFIGURATION
# ==========================================

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LLM model
LLM_MODEL = "gemini-2.5-flash"