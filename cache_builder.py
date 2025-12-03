# cache_builder.py
import osmnx as ox
import networkx as nx
import pickle
import os

CACHE_DIR = "city_cache"
AVAILABLE_CITIES = [
    "Paris, France",
    "London, UK",
    "Rome, Italy",
    "Barcelona, Spain",
    "Amsterdam, Netherlands",
]

def build_cache_for_all_cities():
    """Download and cache all cities at once."""
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print("=" * 70)
    print("BUILDING CITY CACHE FOR ALL CITIES")
    print("=" * 70)
    print(f"This will download data for {len(AVAILABLE_CITIES)} cities.")
    print("⏳ Estimated time: 3-5 minutes total\n")
    
    for i, city_name in enumerate(AVAILABLE_CITIES, 1):
        print(f"\n[{i}/{len(AVAILABLE_CITIES)}] Processing {city_name}...")
        
        safe_name = city_name.replace(", ", "_").replace(" ", "_")
        graph_cache = os.path.join(CACHE_DIR, f"{safe_name}_graph.pkl")
        pois_cache = os.path.join(CACHE_DIR, f"{safe_name}_pois.pkl")
        
        # Skip if already cached
        if os.path.exists(graph_cache) and os.path.exists(pois_cache):
            print(f"  ✓ Already cached, skipping...")
            continue
        
        try:
            # Download graph
            print(f"  📥 Downloading street network...")
            G = ox.graph_from_place(city_name, network_type="walk", simplify=True)
            G = nx.Graph(G)
            
            # Download POIs (filtered for speed)
            print(f"  📥 Downloading attractions...")
            tags = {"tourism": ["museum", "attraction", "viewpoint"]}
            pois = ox.features_from_place(city_name, tags=tags)
            
            if not pois.empty and "name" in pois.columns:
                pois = pois[pois["name"].notna()].copy()
            
            # Save to cache
            print(f"  💾 Saving to cache...")
            with open(graph_cache, 'wb') as f:
                pickle.dump(G, f)
            with open(pois_cache, 'wb') as f:
                pickle.dump(pois, f)
            
            # Get file sizes
            graph_size = os.path.getsize(graph_cache) / (1024 * 1024)  # MB
            pois_size = os.path.getsize(pois_cache) / (1024 * 1024)    # MB
            
            print(f"  ✓ Cached successfully!")
            print(f"    - Graph: {len(G.nodes)} nodes ({graph_size:.1f} MB)")
            print(f"    - POIs: {len(pois)} attractions ({pois_size:.1f} MB)")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("✓ CACHE BUILD COMPLETE!")
    print("=" * 70)
    
    # Show total cache size
    total_size = 0
    for file in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, file)
        total_size += os.path.getsize(file_path)
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"\n📦 Total cache size: {total_size_mb:.1f} MB")
    print(f"📁 Cache location: {os.path.abspath(CACHE_DIR)}/")
    print(f"\n💡 You can now copy the '{CACHE_DIR}' folder to any computer!")
    print(f"   Your main script will load instantly from these files.\n")

if __name__ == "__main__":
    build_cache_for_all_cities()