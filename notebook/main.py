# ==========================================
# MAIN INTEGRATION SCRIPT
# Combines Path Planning + LLM Tour Guide
# ==========================================

import sys
from path_planning import (
    load_city_data,
    plan_time_based_tour,
    visualize_tour_interactive,
    nearest_node,
    format_minutes,
)
from llm_tour_guide import interactive_tour_guide
from config import AVAILABLE_CITIES, GEMINI_API_KEY

def main():
    """
    Main integration: Path planning + LLM tour guide.
    """
    print("\n" + "=" * 70)
    print("   🌍 AI TOUR GUIDE SYSTEM")
    print("   Path Planning + AI Tour Narrator")
    print("=" * 70)
    
    # Check API key
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  WARNING: Gemini API key not set!")
        print("The LLM tour guide will use fallback responses.")
        print("\nTo enable full AI features:")
        print("1. Get key from: https://makersuite.google.com/app/apikey")
        print("2. Set in config.py or environment variable: GEMINI_API_KEY")
        print()
        
        proceed = input("Continue anyway? (yes/no): ").strip().lower()
        if proceed not in ['yes', 'y']:
            print("👋 Goodbye!")
            return
    
    # ==========================================
    # STEP 1: CITY SELECTION
    # ==========================================
    
    print("\n" + "=" * 70)
    print("STEP 1: SELECT CITY")
    print("=" * 70)
    print("\nAvailable cities:")
    for i, c in enumerate(AVAILABLE_CITIES, 1):
        print(f"  {i}. {c}")

    choice = input("\nSelect a city by number (1-5): ").strip()

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(AVAILABLE_CITIES):
            print("❌ Invalid selection.")
            return
        city_name = AVAILABLE_CITIES[idx]
    except ValueError:
        print("❌ Please enter a valid number.")
        return

    # Load city data
    print(f"\n🌍 Loading {city_name}...")
    G, pois = load_city_data(city_name)

    if G is None or pois is None:
        print("❌ Failed to load city data.")
        return

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
        return

    # ==========================================
    # STEP 2: TOUR PARAMETERS
    # ==========================================
    
    print("\n" + "=" * 70)
    print("STEP 2: TOUR PARAMETERS")
    print("=" * 70)
    
    # Get starting point
    start_name = input("\n📍 Enter your STARTING point (attraction name): ").strip()
    
    if not start_name:
        print("❌ Starting point is required.")
        return
    
    # Verify starting point exists
    if nearest_node(G, pois, start_name) is None:
        print("❌ Starting point not found. Please choose from the list above.")
        return

    # Get destination
    end_name = input("🎯 Enter your DESTINATION (attraction name): ").strip()
    
    if not end_name:
        print("❌ Destination is required.")
        return
    
    # Verify destination exists
    if nearest_node(G, pois, end_name) is None:
        print("❌ Destination not found. Please choose from the list above.")
        return
    
    # Check if start and end are the same
    if start_name.lower() == end_name.lower():
        print("❌ Start and destination must be different.")
        return

    # Get available time
    print("\n⏱️ How much time do you have?")
    print("  Examples: 1, 2, 3, 4 hours")
    
    time_input = input("Enter time in hours (e.g., 2 or 2.5): ").strip()
    
    try:
        time_hours = float(time_input)
        if time_hours <= 0 or time_hours > 12:
            print("❌ Please enter a time between 0.5 and 12 hours.")
            return
        time_minutes = time_hours * 60
    except ValueError:
        print("❌ Invalid time format.")
        return

    # ==========================================
    # STEP 3: PLAN TOUR
    # ==========================================
    
    print("\n" + "=" * 70)
    print("STEP 3: PLANNING YOUR TOUR")
    print("=" * 70)
    
    path, total_dist, total_time, itinerary, direct_comparison, stop_nodes = plan_time_based_tour(
        G, pois, start_name, end_name, time_minutes
    )

    if path is None or not itinerary:
        print("\n❌ Could not plan a tour with the given constraints.")
        print("💡 Try: increasing available time or choosing closer attractions.")
        return

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
    from config import BUFFER_TIME_PERCENT
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

    # ==========================================
    # STEP 4: GENERATE MAP
    # ==========================================
    
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING INTERACTIVE MAP")
    print("=" * 70)
    
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

    # ==========================================
    # STEP 5: LLM TOUR GUIDE (OPTIONAL)
    # ==========================================
    
    print("\n" + "=" * 70)
    print("STEP 5: AI TOUR GUIDE")
    print("=" * 70)
    
    print("\n🎭 Would you like an interactive AI tour guide?")
    print("   The AI will provide rich descriptions, historical context,")
    print("   and answer questions about each attraction along your route.")
    print()
    
    use_guide = input("Start AI tour guide? (yes/no): ").strip().lower()
    
    if use_guide in ['yes', 'y', 'yeah', 'sure', 'ok']:
        print("\n🎉 Starting AI tour guide experience...\n")
        interactive_tour_guide(city_name, itinerary, pois)
    else:
        print("\n✅ Tour planning complete!")
        print(f"📍 Your interactive map has been saved as 'tour_map.html'")
        print("   Open it in your browser to explore your route!")
        print("\n👋 Enjoy your tour!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Tour planning interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
