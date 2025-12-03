# ==========================================
# AI TOUR GUIDE – TIME-BASED TOUR PLANNER
# User specifies: Start + Destination + Available Time
# System generates: Optimal route maximizing attractions
# Using OSMNX + Weighted A* Search + Greedy Selection
# ==========================================

import osmnx as ox
import networkx as nx
import heapq
import folium
import matplotlib.pyplot as plt
from shapely.geometry import Point
import math
import pickle
import os

# Import shared configuration
from config import *

print("✓ Path planning module loaded successfully.")
print(f"OSMnx version: {ox.__version__}")

try:
    import sklearn
    print(f"✓ scikit-learn is ready! Version: {sklearn.__version__}")
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠ Warning: scikit-learn not found. Using fallback method for nearest node search.")
    SKLEARN_AVAILABLE = False


# ==========================================
# SECTION 1: DATA LOADING (WITH CACHING)
# ==========================================

def load_city_data(city_name):
    """
    Load street network and tourist attractions with caching support.
    First checks cache, falls back to download if needed.
    """
    
    # Setup cache paths
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = city_name.replace(", ", "_").replace(" ", "_")
    graph_cache = os.path.join(CACHE_DIR, f"{safe_name}_graph.pkl")
    pois_cache = os.path.join(CACHE_DIR, f"{safe_name}_pois.pkl")
    
    # Try loading from cache first
    if os.path.exists(graph_cache) and os.path.exists(pois_cache):
        print(f"\n📦 Loading {city_name} from cache...")
        try:
            with open(graph_cache, 'rb') as f:
                G = pickle.load(f)
            with open(pois_cache, 'rb') as f:
                pois = pickle.load(f)
            
            print(f"✓ Loaded from cache in ~2 seconds! ⚡")
            print(f"✓ Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
            print(f"✓ POIs: {len(pois)} attractions")
            
            return G, pois
            
        except Exception as e:
            print(f"⚠ Cache load failed ({e}). Downloading fresh data...")
    
    # Cache miss - download from OpenStreetMap
    print(f"\n🌍 Downloading {city_name} data (not in cache)...")
    print("⏳ This will take 30-60 seconds, but will be cached for next time...")

    try:
        # Load the street network for walking routes
        G = ox.graph_from_place(city_name, network_type="walk", simplify=True)
        G = nx.Graph(G)

        # Fetch tourist attractions (filtered for performance)
        tags = {"tourism": ["museum", "attraction", "viewpoint", "gallery", "artwork"]}
        pois = ox.features_from_place(city_name, tags=tags)

        if pois.empty:
            print("⚠ No attractions found for this city.")
        else:
            # Clean POI data - keep only named attractions
            if "name" in pois.columns:
                pois = pois[pois["name"].notna()].copy()

        # Save to cache for next time
        print(f"💾 Saving to cache...")
        with open(graph_cache, 'wb') as f:
            pickle.dump(G, f)
        with open(pois_cache, 'wb') as f:
            pickle.dump(pois, f)

        print(f"✓ City graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        print(f"✓ Found {len(pois)} POIs (cached for next time)")

        return G, pois

    except Exception as e:
        print(f"❌ Error loading city data: {e}")
        return None, None


# ==========================================
# SECTION 2: HELPER FUNCTIONS
# ==========================================

def haversine(coord1, coord2):
    """Calculate distance (in km) between two coordinates (lat, lon)."""
    lon1, lat1, lon2, lat2 = map(math.radians, [coord1[1], coord1[0], coord2[1], coord2[0]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c  # Earth radius in km


def nearest_node_fallback(G, lon, lat):
    """
    Fallback method to find nearest node without scikit-learn.
    Uses simple distance calculation to all nodes.
    """
    min_dist = float('inf')
    nearest = None
    
    for node, data in G.nodes(data=True):
        node_lat = data['y']
        node_lon = data['x']
        dist = haversine((lat, lon), (node_lat, node_lon))
        
        if dist < min_dist:
            min_dist = dist
            nearest = node
    
    return nearest


def nearest_node(G, pois, name):
    """Find nearest graph node to a given attraction name (case-insensitive)."""
    if pois.empty or "name" not in pois.columns:
        print("⚠ No named POIs available.")
        return None

    matching_pois = pois[pois["name"].str.lower() == name.lower()]

    if matching_pois.empty:
        print(f"⚠ Attraction '{name}' not found. Please choose from available list.")
        return None

    poi = matching_pois.iloc[0]

    # Handle different geometry types
    if hasattr(poi.geometry, "centroid"):
        point = poi.geometry.centroid
    else:
        point = poi.geometry

    # Use scikit-learn method if available, otherwise use fallback
    if SKLEARN_AVAILABLE:
        try:
            return ox.distance.nearest_nodes(G, point.x, point.y)
        except:
            return nearest_node_fallback(G, point.x, point.y)
    else:
        return nearest_node_fallback(G, point.x, point.y)


def get_node_coords(G, node):
    """Get coordinates of a graph node."""
    return (G.nodes[node]["y"], G.nodes[node]["x"])  # (lat, lon)


def get_poi_coords(pois, name):
    """Get coordinates of a POI by name."""
    if pois.empty or "name" not in pois.columns:
        return None
    
    matching_pois = pois[pois["name"].str.lower() == name.lower()]
    if matching_pois.empty:
        return None
    
    poi = matching_pois.iloc[0]
    if hasattr(poi.geometry, "centroid"):
        point = poi.geometry.centroid
    else:
        point = poi.geometry
    
    return (point.y, point.x)  # (lat, lon)


def compute_attraction_score(G, node, pois, radius_km=0.4):
    """Assign attraction score based on proximity to attractions."""
    if pois.empty:
        return 0

    node_lat = G.nodes[node]["y"]
    node_lon = G.nodes[node]["x"]
    node_point = Point(node_lon, node_lat)

    nearby_count = 0
    for _, poi in pois.iterrows():
        if hasattr(poi.geometry, "centroid"):
            poi_point = poi.geometry.centroid
        else:
            poi_point = poi.geometry

        distance_deg = node_point.distance(poi_point)
        distance_km = distance_deg * 111  # approx conversion

        if distance_km < radius_km:
            nearby_count += 1

    return nearby_count


def estimate_time_minutes(distance_km, speed_kmh=WALKING_SPEED_KMH):
    """Estimate time in minutes given distance in km and speed in km/h."""
    if speed_kmh <= 0:
        return None
    return (distance_km / speed_kmh) * 60


def format_minutes(minutes):
    """Convert minutes to hours and minutes."""
    if minutes is None:
        return None, None
    h = int(minutes // 60)
    m = int(minutes % 60)
    return h, m


# ==========================================
# SECTION 3: SCENIC A* ALGORITHM
# ==========================================

def scenic_a_star(
    G,
    start,
    goal,
    pois,
    attraction_weight=ATTRACTION_WEIGHT,
    max_iterations=30000,
    max_distance_km=MAX_WALK_LEG_KM,
):
    """
    Modified A* algorithm balancing distance and attraction.
    Higher attraction_weight = more scenic routes.
    """
    pq = []
    heapq.heappush(pq, (0, start, [start], 0.0, 0))
    visited = set()

    goal_lat = G.nodes[goal]["y"]
    goal_lon = G.nodes[goal]["x"]

    iterations = 0

    while pq and iterations < max_iterations:
        iterations += 1

        cost, node, path, total_dist, total_attr = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        # Found goal
        if node == goal:
            return path, total_dist, total_attr

        # Hard cutoff for overlong legs
        if total_dist > max_distance_km:
            continue

        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue

            edge_data = G[node][neighbor]

            if isinstance(edge_data, dict):
                distance = edge_data.get("length", 100) / 1000.0  # m → km
            else:
                node_lat = G.nodes[node]["y"]
                node_lon = G.nodes[node]["x"]
                neighbor_lat = G.nodes[neighbor]["y"]
                neighbor_lon = G.nodes[neighbor]["x"]
                distance = haversine((node_lat, node_lon), (neighbor_lat, neighbor_lon))

            new_dist = total_dist + distance

            # Skip if too long
            if new_dist > max_distance_km:
                continue

            # Attraction scoring (periodic for performance)
            if iterations % 10 == 0:
                attraction = compute_attraction_score(G, neighbor, pois, radius_km=0.4)
            else:
                attraction = 0

            new_attr = total_attr + attraction

            # Heuristic: straight-line distance to goal
            neighbor_lat = G.nodes[neighbor]["y"]
            neighbor_lon = G.nodes[neighbor]["x"]
            h = haversine((neighbor_lat, neighbor_lon), (goal_lat, goal_lon))

            new_cost = new_dist - (attraction_weight * new_attr) + HEURISTIC_WEIGHT * h

            heapq.heappush(pq, (new_cost, neighbor, path + [neighbor], new_dist, new_attr))

    return None, None, None


def direct_route(G, start, goal):
    """
    Calculate direct route using standard A* (no scenic weighting).
    Used as baseline comparison.
    """
    try:
        path = nx.astar_path(G, start, goal, heuristic=lambda u, v: haversine(
            (G.nodes[u]["y"], G.nodes[u]["x"]),
            (G.nodes[v]["y"], G.nodes[v]["x"])
        ), weight='length')
        
        # Calculate distance
        total_dist = 0.0
        for i in range(len(path) - 1):
            edge_data = G[path[i]][path[i+1]]
            if isinstance(edge_data, dict):
                total_dist += edge_data.get("length", 100) / 1000.0
            else:
                coord1 = get_node_coords(G, path[i])
                coord2 = get_node_coords(G, path[i+1])
                total_dist += haversine(coord1, coord2)
        
        return path, total_dist
    except:
        return None, None


# ==========================================
# SECTION 4: TIME-BASED TOUR PLANNING
# ==========================================

def score_attraction(G, pois, poi_name, start_node, end_node):
    """
    Score an attraction based on:
    - Proximity to attractions (density)
    - Position along start-to-end corridor
    - Tourism type (museums > artwork)
    """
    poi_node = nearest_node(G, pois, poi_name)
    if poi_node is None:
        return 0
    
    # Base score: attraction density
    density_score = compute_attraction_score(G, poi_node, pois, radius_km=0.5)
    
    # Corridor score: prefer attractions between start and end
    start_coords = get_node_coords(G, start_node)
    end_coords = get_node_coords(G, end_node)
    poi_coords = get_node_coords(G, poi_node)
    
    # Distance from start and end
    dist_from_start = haversine(start_coords, poi_coords)
    dist_from_end = haversine(poi_coords, end_coords)
    direct_dist = haversine(start_coords, end_coords)
    
    # Check if attraction is roughly along the path (not a huge detour)
    # If dist_from_start + dist_from_end ≈ direct_dist, it's on the path
    detour = (dist_from_start + dist_from_end) - direct_dist
    
    if detour < 0.5:  # Very close to direct path
        corridor_score = 3.0
    elif detour < 1.0:  # Reasonable detour
        corridor_score = 2.0
    elif detour < 2.0:  # Moderate detour
        corridor_score = 1.0
    else:  # Too far off path
        corridor_score = 0.3
    
    # Type bonus
    matching_pois = pois[pois["name"].str.lower() == poi_name.lower()]
    if not matching_pois.empty:
        poi_data = matching_pois.iloc[0]
        tourism_type = poi_data.get("tourism", "")
        
        type_bonus = 1.0
        if tourism_type in ["museum", "gallery"]:
            type_bonus = 1.5
        elif tourism_type in ["attraction", "viewpoint"]:
            type_bonus = 1.3
    else:
        type_bonus = 1.0
    
    # Combined score
    total_score = (density_score * 1.5 + corridor_score * 3.0) * type_bonus
    
    return total_score


def select_attractions_for_time_budget(G, pois, start_name, end_name, available_time_minutes):
    """
    Select attractions that fit within the time budget between start and end.
    Uses greedy algorithm with scoring.
    Returns list of attraction names in visit order.
    """
    start_node = nearest_node(G, pois, start_name)
    end_node = nearest_node(G, pois, end_name)
    
    if start_node is None or end_node is None:
        return []
    
    # Calculate direct route time (baseline)
    direct_path, direct_dist = direct_route(G, start_node, end_node)
    if direct_path is None:
        print("⚠ No path found between start and destination")
        return []
    
    direct_time = estimate_time_minutes(direct_dist)
    
    print(f"\n📏 Direct route: {direct_dist:.2f} km (~{int(direct_time)} min)")
    
    # Calculate effective touring time (excluding buffer and direct travel)
    effective_time = available_time_minutes * (1 - BUFFER_TIME_PERCENT)
    available_for_detours = effective_time - direct_time
    
    if available_for_detours < VISIT_TIME_PER_ATTRACTION_MIN:
        print(f"⚠ Not enough time for detours. Need at least {int(direct_time + VISIT_TIME_PER_ATTRACTION_MIN)} minutes.")
        return [start_name, end_name]
    
    print(f"⏱️ Time available for attractions: {int(available_for_detours)} min")
    
    # Get all available attractions (excluding start and end)
    if pois.empty or "name" not in pois.columns:
        return [start_name, end_name]
    
    all_attractions = pois["name"].dropna().unique().tolist()
    all_attractions = [a for a in all_attractions 
                      if a.lower() != start_name.lower() and a.lower() != end_name.lower()]
    
    # Score all attractions
    print(f"\n🎯 Scoring {len(all_attractions)} attractions...")
    scored_attractions = []
    for attr_name in all_attractions:
        score = score_attraction(G, pois, attr_name, start_node, end_node)
        if score >= MIN_ATTRACTION_SCORE:
            scored_attractions.append((attr_name, score))
    
    # Sort by score (descending)
    scored_attractions.sort(key=lambda x: -x[1])
    
    print(f"✓ Found {len(scored_attractions)} high-quality attractions along route")
    
    # Greedy selection: pick attractions until time runs out
    selected = [start_name]
    current_node = start_node
    time_used = 0
    
    for attr_name, score in scored_attractions:
        if len(selected) >= MAX_ATTRACTIONS_PER_TOUR:
            break
        
        # Get node for this attraction
        attr_node = nearest_node(G, pois, attr_name)
        if attr_node is None:
            continue
        
        # Calculate travel time to this attraction
        current_coords = get_node_coords(G, current_node)
        attr_coords = get_node_coords(G, attr_node)
        straight_dist = haversine(current_coords, attr_coords)
        
        # Skip if too far
        if straight_dist > MAX_WALK_LEG_KM:
            continue
        
        # Estimate actual walking distance (assume 1.3x straight line)
        estimated_walk_dist = straight_dist * 1.3
        travel_time = estimate_time_minutes(estimated_walk_dist)
        
        # Total time for this stop
        stop_time = travel_time + VISIT_TIME_PER_ATTRACTION_MIN
        
        # Check if we have time
        if time_used + stop_time > available_for_detours:
            continue
        
        # Add to tour
        selected.append(attr_name)
        current_node = attr_node
        time_used += stop_time
    
    # Add destination
    selected.append(end_name)
    
    return selected


def plan_time_based_tour(G, pois, start_name, end_name, available_time_minutes):
    """
    Plan a complete tour based on available time from start to destination.
    Returns: path, total_distance, total_time, itinerary, direct_comparison, stop_nodes
    """
    print(f"\n🕐 Planning tour for {available_time_minutes} minutes...")
    
    # Get start and end nodes
    start_node = nearest_node(G, pois, start_name)
    end_node = nearest_node(G, pois, end_name)
    
    if start_node is None or end_node is None:
        return None, None, None, None, None, None
    
    # Calculate direct route (baseline)
    print(f"\n📍 Calculating direct route (baseline)...")
    direct_path, direct_dist = direct_route(G, start_node, end_node)
    direct_time = estimate_time_minutes(direct_dist) if direct_dist else None
    
    # Select attractions
    selected_attractions = select_attractions_for_time_budget(
        G, pois, start_name, end_name, available_time_minutes
    )
    
    if len(selected_attractions) < 2:
        print("❌ Not enough time for a meaningful tour")
        return None, None, None, None, None, None
    
    print(f"\n📍 Selected {len(selected_attractions)} stops:")
    for i, attr in enumerate(selected_attractions, 1):
        print(f"  {i}. {attr}")
    
    # Route between attractions
    print(f"\n🗺️ Calculating scenic routes...\n")
    
    full_path = []
    total_dist = 0.0
    total_time = 0.0
    total_attr = 0
    itinerary = []
    stop_nodes = []  # Store nodes for each stop
    
    for i in range(len(selected_attractions) - 1):
        leg_start_name = selected_attractions[i]
        leg_end_name = selected_attractions[i + 1]
        
        leg_start_node = nearest_node(G, pois, leg_start_name)
        leg_end_node = nearest_node(G, pois, leg_end_name)
        
        if leg_start_node is None or leg_end_node is None:
            print(f"⚠ Skipping leg {leg_start_name} → {leg_end_name}")
            continue
        
        # Store stop nodes
        if i == 0:
            stop_nodes.append((leg_start_name, leg_start_node))
        stop_nodes.append((leg_end_name, leg_end_node))
        
        print(f"  Leg {i+1}: {leg_start_name} → {leg_end_name}")
        
        # Calculate scenic route
        leg_path, leg_dist, leg_attr = scenic_a_star(
            G,
            leg_start_node,
            leg_end_node,
            pois,
            attraction_weight=ATTRACTION_WEIGHT,
            max_iterations=30000,
            max_distance_km=MAX_WALK_LEG_KM,
        )
        
        if leg_path is None:
            # Fallback: straight line estimate
            start_coords = get_node_coords(G, leg_start_node)
            end_coords = get_node_coords(G, leg_end_node)
            leg_dist = haversine(start_coords, end_coords)
            leg_attr = 0
            print(f"    ⚠ Using straight-line estimate: {leg_dist:.2f} km")
        else:
            print(f"    ✓ Route: {leg_dist:.2f} km | {leg_attr} scenic points")
        
        # Calculate time
        walk_time = estimate_time_minutes(leg_dist)
        
        # Add visit time (except for final destination)
        if i < len(selected_attractions) - 2:
            visit_time = VISIT_TIME_PER_ATTRACTION_MIN
        else:
            visit_time = 0  # No visit time at final destination
        
        leg_time = walk_time + visit_time
        
        # Add to totals
        if leg_path and full_path:
            full_path.extend(leg_path[1:])
        elif leg_path:
            full_path.extend(leg_path)
        
        total_dist += leg_dist
        total_time += leg_time
        total_attr += leg_attr
        
        itinerary.append((leg_start_name, leg_end_name, leg_dist, walk_time, visit_time, leg_attr))
    
    if not itinerary:
        return None, None, None, None, None, None
    
    # Prepare comparison data
    direct_comparison = {
        'path': direct_path,
        'distance': direct_dist,
        'time': direct_time
    }
    
    return full_path, total_dist, total_time, itinerary, direct_comparison, stop_nodes


# ==========================================
# SECTION 5: VISUALIZATION
# ==========================================

def visualize_tour_interactive(
    G,
    scenic_path,
    direct_path,
    city_name,
    start_name,
    end_name,
    scenic_dist,
    scenic_time,
    direct_dist,
    direct_time,
    stop_nodes,
    map_filename=MAP_OUTPUT_FILE,
):
    """
    Interactive map using folium (Leaflet).
    - Zoom & pan like Google Maps
    - Markers with labels for each stop
    - Scenic path and direct path drawn
    """

    print("\n📍 Generating interactive map (folium)...")

    if scenic_path and len(scenic_path) > 0:
        # Center map on the middle of scenic path
        mid_idx = len(scenic_path) // 2
        center_node = scenic_path[mid_idx]
    elif direct_path and len(direct_path) > 0:
        mid_idx = len(direct_path) // 2
        center_node = direct_path[mid_idx]
    else:
        # Fallback: first stop
        if stop_nodes:
            center_node = stop_nodes[0][1]
        else:
            print("⚠ No path or stops to visualize.")
            return

    center_lat = G.nodes[center_node]["y"]
    center_lon = G.nodes[center_node]["x"]

    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=DEFAULT_MAP_ZOOM, tiles=MAP_TILES)

    # --- Plot direct route (blue dashed) ---
    if direct_path and len(direct_path) > 1:
        direct_coords = [
            (G.nodes[n]["y"], G.nodes[n]["x"]) for n in direct_path
        ]
        folium.PolyLine(
            direct_coords,
            color="blue",
            weight=3,
            opacity=0.6,
            dash_array="5, 10",
            tooltip="Direct route",
        ).add_to(m)

    # --- Plot scenic route (red solid) ---
    if scenic_path and len(scenic_path) > 1:
        scenic_coords = [
            (G.nodes[n]["y"], G.nodes[n]["x"]) for n in scenic_path
        ]
        folium.PolyLine(
            scenic_coords,
            color="red",
            weight=4,
            opacity=0.9,
            tooltip="Scenic route",
        ).add_to(m)

    # --- Plot stops as markers with popups ---
    for i, (stop_name, stop_node) in enumerate(stop_nodes):
        lat = G.nodes[stop_node]["y"]
        lon = G.nodes[stop_node]["x"]

        if i == 0:
            icon_color = "green"
            prefix = "Start"
        elif i == len(stop_nodes) - 1:
            icon_color = "red"
            prefix = "Destination"
        else:
            icon_color = "orange"
            prefix = f"Stop {i}"

        popup_html = f"<b>{prefix}</b><br>{stop_name}"
        tooltip = f"{prefix}: {stop_name}"

        folium.Marker(
            [lat, lon],
            popup=popup_html,
            tooltip=tooltip,
            icon=folium.Icon(color=icon_color, icon="info-sign"),
        ).add_to(m)

    # Save map
    m.save(map_filename)
    print(f"\n✅ Interactive map saved as: {map_filename}")
    print("   → Open this file in your browser to zoom, pan, and inspect attractions.")
    
    return map_filename


# ==========================================
# STANDALONE EXECUTION
# ==========================================

def run_standalone():
    """Standalone execution for path planning only."""
    print("\n" + "=" * 70)
    print("   AI TOUR GUIDE - TIME-BASED TOUR PLANNER")
    print("   🎯 Start → Destination with optimal attractions")
    print("=" * 70)

    # City selection
    print("\nAvailable cities:")
    for i, c in enumerate(AVAILABLE_CITIES, 1):
        print(f"  {i}. {c}")

    choice = input("\nSelect a city by number (1-5): ").strip()

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(AVAILABLE_CITIES):
            print("❌ Invalid selection.")
            return None
        city_name = AVAILABLE_CITIES[idx]
    except ValueError:
        print("❌ Please enter a valid number.")
        return None

    # Load city data
    G, pois = load_city_data(city_name)

    if G is None or pois is None:
        print("❌ Failed to load city data.")
        return None

    # Show sample attractions
    if not pois.empty and "name" in pois.columns:
        print("\n" + "=" * 70)
        print("Available Attractions (sample):")
        print("=" * 70)
        sample_names = sorted(pois["name"].dropna().unique()[:30])
        for i, name in enumerate(sample_names, 1):
            print(f"  {i:2d}. {name}")
        if len(pois) > 30:
            print(f"  ... and {len(pois) - 30} more attractions")
        print("=" * 70)
    else:
        print("\n⚠ No named attractions found for this city.")
        return None

    # Get starting point
    start_name = input("\n📍 Enter your STARTING point (attraction name): ").strip()
    
    if not start_name:
        print("❌ Starting point is required.")
        return None
    
    # Verify starting point exists
    if nearest_node(G, pois, start_name) is None:
        print("❌ Starting point not found. Please choose from the list above.")
        return None

    # Get destination
    end_name = input("🎯 Enter your DESTINATION (attraction name): ").strip()
    
    if not end_name:
        print("❌ Destination is required.")
        return None
    
    # Verify destination exists
    if nearest_node(G, pois, end_name) is None:
        print("❌ Destination not found. Please choose from the list above.")
        return None
    
    # Check if start and end are the same
    if start_name.lower() == end_name.lower():
        print("❌ Start and destination must be different.")
        return None

    # Get available time
    print("\n⏱️ How much time do you have?")
    print("  Examples: 1 hour, 2 hours, 3 hours, 4 hours")
    
    time_input = input("Enter time in hours (e.g., 2 or 2.5): ").strip()
    
    try:
        time_hours = float(time_input)
        if time_hours <= 0 or time_hours > 12:
            print("❌ Please enter a time between 0.5 and 12 hours.")
            return None
        time_minutes = time_hours * 60
    except ValueError:
        print("❌ Invalid time format.")
        return None

    # Plan the tour
    print(f"\n🔍 Planning optimal tour for {time_hours} hours...")
    print(f"⚙️ Parameters:")
    print(f"   - Walking speed: {WALKING_SPEED_KMH} km/h")
    print(f"   - Visit time per attraction: {VISIT_TIME_PER_ATTRACTION_MIN} min")
    print(f"   - Buffer time: {int(BUFFER_TIME_PERCENT * 100)}%")
    
    path, total_dist, total_time, itinerary, direct_comparison, stop_nodes = plan_time_based_tour(
        G, pois, start_name, end_name, time_minutes
    )

    if path is None or not itinerary:
        print("\n❌ Could not plan a tour with the given constraints.")
        print("💡 Try: increasing available time or choosing closer attractions.")
        return None

    # Display results
    print("\n" + "=" * 70)
    print("✓ TOUR SUCCESSFULLY PLANNED!")
    print("=" * 70)

    # Show direct route comparison
    if direct_comparison and direct_comparison['distance']:
        print("\n📊 ROUTE COMPARISON:\n")
        d_h, d_m = format_minutes(direct_comparison['time'])
        print(f"🔵 Direct Route (baseline):")
        print(f"   Distance: {direct_comparison['distance']:.2f} km")
        print(f"   Time: ~{d_h}h {d_m}m")
        print(f"   Stops: 2 (start + destination only)")
        print()

    print("\n📋 SCENIC ROUTE ITINERARY:\n")
    
    for i, (start, end, dist, walk_time, visit_time, attr) in enumerate(itinerary, 1):
        walk_h, walk_m = format_minutes(walk_time)
        
        print(f"Leg {i}: {start} → {end}")
        print(f"  🚶 Walk: {dist:.2f} km (~{walk_h}h {walk_m}m)")
        
        if visit_time > 0:
            visit_h, visit_m = format_minutes(visit_time)
            print(f"  🎨 Visit: ~{visit_h}h {visit_m}m")
        
        if attr > 0:
            print(f"  ⭐ Scenic points along route: {attr}")
        
        print()

    # Summary
    print("-" * 70)
    total_h, total_m = format_minutes(total_time)
    buffer_time = total_time * (BUFFER_TIME_PERCENT / (1 - BUFFER_TIME_PERCENT))
    buffer_h, buffer_m = format_minutes(buffer_time)
    final_time = total_time + buffer_time
    final_h, final_m = format_minutes(final_time)
    
    print(f"📊 Scenic Route Summary:")
    print(f"   Total Distance: {total_dist:.2f} km")
    print(f"   Number of Stops: {len(itinerary) + 1}")
    print(f"   Active Time: {total_h}h {total_m}m (walking + visiting)")
    print(f"   Buffer Time: {buffer_h}h {buffer_m}m (breaks, photos, etc.)")
    print(f"   Total Time: {final_h}h {final_m}m")
    print(f"   Time Budget: {int(time_minutes / 60)}h {int(time_minutes % 60)}m")
    
    if final_time <= time_minutes:
        print(f"   ✓ Tour fits within your time budget!")
    else:
        print(f"   ⚠ Tour slightly exceeds budget (consider it a guideline)")
    
    # Show improvement over direct route
    if direct_comparison and direct_comparison['distance']:
        extra_dist = total_dist - direct_comparison['distance']
        extra_time = final_time - direct_comparison['time']
        extra_stops = len(itinerary) - 1
        
        print(f"\n🎯 Scenic Route Benefits:")
        print(f"   Extra distance: +{extra_dist:.2f} km ({(extra_dist/direct_comparison['distance']*100):.1f}%)")
        print(f"   Extra time: +{int(extra_time)} min")
        print(f"   Extra attractions: +{extra_stops} stops")
        print(f"   💡 Worth it for a richer experience!")
    
    print("=" * 70)

    # Visualization
    num_stops = len(set([leg[0] for leg in itinerary] + [itinerary[-1][1]]))
    visualize_tour_interactive(
        G,
        path,
        direct_comparison['path'] if direct_comparison else None,
        city_name,
        start_name,
        end_name,
        total_dist,
        final_time,
        direct_comparison['distance'] if direct_comparison else None,
        direct_comparison['time'] if direct_comparison else None,
        stop_nodes,
    )
    
    # Return tour data for potential integration
    return {
        'city_name': city_name,
        'G': G,
        'pois': pois,
        'itinerary': itinerary,
        'stop_nodes': stop_nodes,
        'total_dist': total_dist,
        'total_time': final_time,
    }


if __name__ == "__main__":
    run_standalone()
