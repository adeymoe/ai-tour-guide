# ==========================================
# AI TOUR GUIDE - STREAMLIT WEB INTERFACE
# ==========================================

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime
import time

# Import our modules (assumes these exist in the project)
from path_planning import (
    load_city_data,
    plan_time_based_tour,
    nearest_node,
    format_minutes,
)
from llm_tour_guide import TourGuideAgent, AttractionDatabase
from config import AVAILABLE_CITIES, GEMINI_API_KEY

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="AI Tour Guide",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS STYLING (Voyage Explorer Theme)
# =========================================

st.markdown("""
<style>
/* Voyage Explorer Theme */

/* Background and text */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #343A40; /* Deep Slate */
    background-color: #F8F9FA; /* Cloud White */
    margin: 0;
    padding: 0;
}

/* Hero section */
.hero {
    background: #FFFFFF;
    padding: 2.5rem 2rem;
    border-radius: 14px;
    border: 1px solid #E9ECEF;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgb(0 123 255 / 0.15);
    text-align: center;
}

.hero h1 {
    font-weight: 800;
    font-size: 2.8rem;
    color: #007BFF; /* Discovery Blue */
    margin-bottom: 0.3rem;
}

.hero p {
    font-size: 1.15rem;
    color: #6C757D; /* Misty Grey */
    margin-top: 0;
}

/* Loading badge */
.loading-badge {
    display: inline-block;
    background: #007BFF; /* Discovery Blue */
    color: white;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 6px rgb(0 123 255 / 0.3);
}

/* Stats boxes */
.stats-box {
    background-color: #FFFFFF;
    padding: 1.2rem 1rem;
    border-radius: 14px;
    border: 1px solid #E9ECEF;
    text-align: center;
    box-shadow: 0 2px 8px rgb(0 123 255 / 0.1);
    color: #343A40;
    font-weight: 700;
    user-select: none;
}

.stats-box h3 {
    font-size: 1.8rem;
    margin: 0 0 0.3rem 0;
}

.stats-box p {
    font-size: 1rem;
    color: #6C757D;
    margin: 0;
}

/* Stop card */
.stop-card {
    background-color: #FFFFFF;
    padding: 1.3rem 1.5rem;
    border-radius: 14px;
    border: 1px solid #E9ECEF;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 14px rgb(255 193 7 / 0.15);
    color: #343A40;
}

.stop-card div:first-child {
    font-size: 0.95rem;
    color: #FFC107; /* Horizon Gold */
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}

.stop-card div:last-child {
    font-size: 1.5rem;
    font-weight: 800;
}

/* Chat messages */
.chat-message {
    background: #FFFFFF;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    border: 1px solid #E9ECEF;
    box-shadow: 0 2px 8px rgb(0 123 255 / 0.1);
    color: #343A40;
    font-size: 1rem;
    line-height: 1.5;
}

.chat-message.user {
    background: #E9F2FF;
    border-left: 5px solid #007BFF; /* Discovery Blue */
}

.chat-message.guide {
    background: #FFF8E1;
    border-left: 5px solid #FFC107; /* Horizon Gold */
}

/* Section titles */
.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: white;
    margin-bottom: 1rem;
    border-bottom: 3px solid #007BFF;
    padding-bottom: 0.3rem;
}

/* Buttons */
.stButton>button {
    width: 100%;
    background-color: #007BFF; /* Discovery Blue */
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 0.75rem 1rem;
    border-radius: 999px;
    border: none;
    box-shadow: 0 6px 20px rgb(0 123 255 / 0.4);
    transition: background-color 0.3s ease, box-shadow 0.3s ease;
}

.stButton>button:hover {
    background-color: #0056b3;
    box-shadow: 0 8px 24px rgb(0 86 179 / 0.6);
    transform: translateY(-2px);
}

/* Input text */
input[type="text"], .stTextInput>div>input {
    border-radius: 12px;
    border: 1.5px solid #CED4DA;
    padding: 0.5rem 0.75rem;
    font-size: 1rem;
    color: white;
    transition: border-color 0.3s ease;
}

input[type="text"]:focus, .stTextInput>div>input:focus {
    border-color: #007BFF;
    outline: none;
    box-shadow: 0 0 6px rgb(0 123 255 / 0.5);
}

/* Progress indicator */
.progress-indicator {
    margin-bottom: 1rem;
    user-select: none;
}

.progress-step {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    margin-right: 0.5rem;
    font-size: 0.9rem;
    font-weight: 700;
    cursor: default;
    user-select: none;
}

.progress-step.completed {
    background-color: #28a745;
    color: white;
}

.progress-step.current {
    background-color: #007BFF;
    color: white;
}

.progress-step.pending {
    background-color: #E9ECEF;
    color: #6C757D;
}

/* Tooltip styling */
.tooltip {
    color: #6C757D;
    font-size: 0.9rem;
}

/* Scrollbar for conversation */
div[data-testid="stVerticalBlock"] > div[role="list"] {
    max-height: 400px;
    overflow-y: auto;
    padding-right: 10px;
}

/* Responsive tweaks */
@media (max-width: 768px) {
    .stats-box h3 {
        font-size: 1.4rem;
    }
    .stop-card div:last-child {
        font-size: 1.2rem;
    }
    .section-title {
        font-size: 1.3rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

# Initialize all session states
session_keys = {
    'tour_planned': False,
    'current_stop': 0,
    'show_tips': {},
    'user_questions': {},
    'conversation_history': [],
    'attractions_cache': {},
    'current_city': None,
    'city_data': None,
    'G': None,
    'pois': None,
    'city_name': None,
    'itinerary': None,
    'route_nodes': None,
    'stops_info': None,
    'total_distance': 0,
    'total_time': 0,
    'direct_route_info': None,
    'guide': None,
    'stops': None
}

for key, default_value in session_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ==========================================
# CACHED DATA LOADING
# ==========================================

@st.cache_data(show_spinner=False)
def cached_load_city_data(city_name):
    """Cached version of city data loading"""
    return load_city_data(city_name)

def load_city_attractions(city_name):
    """Load and cache attractions for a city with session persistence"""
    # Return cached if available in session
    if city_name in st.session_state.attractions_cache:
        return st.session_state.attractions_cache[city_name]

    try:
        result = cached_load_city_data(city_name)

        # Validate return format
        if not isinstance(result, tuple) or len(result) != 2:
            st.error("Unexpected return format from load_city_data (expected (G, pois)).")
            return None

        G, pois = result

        # Validate types
        if pois is None:
            st.error("No POIs returned for this city.")
            return None

        # Handle GeoDataFrame appropriately
        try:
            pois_df = pois
            if hasattr(pois_df, "empty"):
                if pois_df.empty:
                    st.error("No points of interest found for this city.")
                    return None
            else:
                pois_df = pd.DataFrame(pois)
                if pois_df.shape[0] == 0:
                    st.error("No points of interest found for this city.")
                    return None
        except Exception:
            pois_df = pd.DataFrame(pois)
            if pois_df.shape[0] == 0:
                st.error("No points of interest found for this city.")
                return None

        # Extract names
        name_cols = [c for c in pois_df.columns if c.lower() == 'name']
        if not name_cols:
            if 'display_name' in pois_df.columns:
                raw_names = pois_df['display_name'].astype(str).fillna('')
            elif 'tags' in pois_df.columns:
                def _extract_from_tags(v):
                    try:
                        if isinstance(v, dict):
                            return v.get('name') or v.get('official_name') or v.get('alt_name') or str(v)
                        return str(v)
                    except Exception:
                        return str(v)
                raw_names = pois_df['tags'].apply(_extract_from_tags).astype(str).fillna('')
            else:
                if 'osm_id' in pois_df.columns and 'geometry' in pois_df.columns:
                    raw_names = pois_df['osm_id'].astype(str)
                else:
                    raw_names = pois_df.iloc[:, 0].astype(str).fillna('')
        else:
            raw_names = pois_df[name_cols[0]].astype(str).fillna('')

        # Clean and dedupe names
        attractions = sorted({n.strip() for n in raw_names if n and n.strip()})

        # Cache the data in session
        st.session_state.attractions_cache[city_name] = {
            'G': G,
            'pois': pois_df,
            'attractions': attractions
        }

        return st.session_state.attractions_cache[city_name]

    except Exception as e:
        st.error(f"Error loading city data: {str(e)}")
        return None

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_tour_map(G, route_nodes, stops_info, city_name, highlight_stop=None):
    """Create an interactive Folium map with the tour route and highlighted current stop."""
    if not route_nodes:
        try:
            any_node = next(iter(G.nodes))
            center_lat = G.nodes[any_node].get('y')
            center_lon = G.nodes[any_node].get('x')
        except Exception:
            center_lat, center_lon = 0, 0
    else:
        center_node = route_nodes[len(route_nodes)//2]
        center_lat = G.nodes[center_node].get('y')
        center_lon = G.nodes[center_node].get('x')

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')

    try:
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route_nodes]
        folium.PolyLine(route_coords, color='#007BFF', weight=5, opacity=0.8, popup='Tour Route').add_to(m)
    except Exception:
        pass

    for idx, (stop_name, node_id) in enumerate(stops_info, 1):
        try:
            lat = G.nodes[node_id]['y']
            lon = G.nodes[node_id]['x']
        except Exception:
            continue

        is_highlighted = (highlight_stop is not None and idx - 1 == highlight_stop)

        if is_highlighted:
            color = 'orange'
            icon = 'star'
        elif idx == 1:
            color = 'green'
            icon = 'play'
        elif idx == len(stops_info):
            color = 'red'
            icon = 'stop'
        else:
            color = 'blue'
            icon = 'info-sign'

        marker = folium.Marker(
            location=[lat, lon],
            popup=f"<b>Stop {idx}</b><br>{stop_name}",
            tooltip=(f"{'⭐ CURRENT: ' if is_highlighted else ''}{idx}. {stop_name}"),
            icon=folium.Icon(color=color, icon=icon)
        )
        marker.add_to(m)

        if is_highlighted:
            folium.Circle(location=[lat, lon], radius=100, color='#FFC107', fill=True,
                          fillColor='#FFC107', fillOpacity=0.2, weight=3).add_to(m)

    return m

def add_chat_message(role, message):
    """Add a message to the conversation history."""
    st.session_state.conversation_history.append({
        'role': role,
        'message': message,
        'timestamp': datetime.now()
    })

def reset_tour():
    """Reset the tour to start over."""
    st.session_state.tour_planned = False
    st.session_state.current_stop = 0
    st.session_state.show_tips = {}
    st.session_state.user_questions = {}
    st.session_state.conversation_history = []

# ==========================================
# MAIN APP
# ==========================================

def main():
    # HERO SECTION
    st.markdown("""
    <div class="hero">
      <div style="text-align:center">
        <h1>Discover Cities with AI</h1>
        <p>Personalized city tours optimized for your time and interests.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # SIDEBAR - TOUR PLANNING
    # ==========================================
    with st.sidebar:
        st.header("🎯 Plan Your Tour")

        # City selection
        city_name = st.selectbox(
            "🌍 Choose a city",
            options=list(AVAILABLE_CITIES.keys()),
            help="Where would you like to explore?",
            key="city_selector"
        )

        # Load attractions for the selected city (only once per session)
        if city_name != st.session_state.current_city:
            st.session_state.current_city = city_name
            with st.spinner(f"🔄 Loading {city_name} attractions..."):
                city_data = load_city_attractions(city_name)
                if city_data:
                    st.success(f"✅ Loaded {len(city_data['attractions'])} attractions!")
                    time.sleep(0.4)
                    # Store in session state
                    st.session_state.city_data = city_data

        # Access cached data safely
        city_data = st.session_state.attractions_cache.get(city_name)
        if city_data and isinstance(city_data, dict) and 'attractions' in city_data:
            attractions_list = city_data['attractions']
            st.markdown(f'<div class="loading-badge">✓ {len(attractions_list)} attractions loaded</div>', unsafe_allow_html=True)

            # Starting point - searchable selectbox
            st.markdown("**📍 Starting point**")
            start_location = st.selectbox(
                "Select or search starting point:",
                options=[""] + attractions_list,
                format_func=lambda x: "Type to search..." if x == "" else x,
                key="start_select",
                help="Click and type to search"
            )

            # Destination - searchable selectbox
            st.markdown("**🏁 Final stop**")
            end_location = st.selectbox(
                "Select or search destination:",
                options=[""] + attractions_list,
                format_func=lambda x: "Type to search..." if x == "" else x,
                key="end_select",
                help="Click and type to search"
            )

        else:
            st.warning("⏳ Loading city data, please wait...")
            start_location = None
            end_location = None

        # Available time
        available_time = st.slider(
            "⏱ Total time available (minutes)",
            min_value=30,
            max_value=480,
            value=180,
            step=15,
            help="Includes walking and visit time",
            key="time_slider"
        )

        st.markdown("---")

        # Plan Tour Button
        if st.button("🚀 Plan My Tour", type="primary", key="plan_button"):
            if not start_location or not end_location or start_location == "" or end_location == "":
                st.error("⚠️ Please select both starting point and destination!")
            elif start_location == end_location:
                st.error("⚠️ Starting point and destination must be different!")
            else:
                with st.spinner(f"🔍 Planning your tour in {city_name}..."):
                    try:
                        # Use cached data
                        city_data = st.session_state.attractions_cache.get(city_name)
                        if not city_data:
                            st.error("City data not loaded. Please wait and try again.")
                        else:
                            G = city_data['G']
                            pois = city_data['pois']

                            # Plan tour
                            result = plan_time_based_tour(
                                G, pois,
                                start_location, end_location,
                                available_time
                            )

                            if not result:
                                st.error("❌ Could not plan tour. Check inputs or try different time.")
                            else:
                                # Unpack result
                                route_nodes, total_distance, total_time, itinerary, direct_route_info, stops_info = result

                                # Save to session
                                st.session_state.G = G
                                st.session_state.pois = pois
                                st.session_state.city_name = city_name
                                st.session_state.itinerary = itinerary
                                st.session_state.route_nodes = route_nodes
                                st.session_state.stops_info = stops_info
                                st.session_state.total_distance = total_distance
                                st.session_state.total_time = total_time
                                st.session_state.direct_route_info = direct_route_info

                                # Initialize tour guide
                                st.session_state.guide = TourGuideAgent(city_name, itinerary, pois)
                                st.session_state.stops = st.session_state.guide.get_tour_stops()

                                st.session_state.tour_planned = True
                                st.session_state.current_stop = 0
                                st.session_state.conversation_history = []

                                # Welcome message
                                welcome = st.session_state.guide.generate_welcome_message()
                                add_chat_message('guide', welcome)

                                st.success("✅ Tour planned successfully!")
                                time.sleep(0.4)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # Reset button
        if st.session_state.tour_planned:
            st.markdown("---")
            if st.button("🔁 Plan New Tour", key="reset_button"):
                reset_tour()

    # ==========================================
    # MAIN CONTENT AREA
    # ==========================================
    if not st.session_state.tour_planned:
        # Welcome & instructions
        st.markdown("<div class='section-title'>Why use AI Tour Guide?</div>", unsafe_allow_html=True)
        st.markdown("""
            - Smart planning that fits your available time.
            - Conversational AI guide with tips and directions.
            - Interactive maps showing stops and route.
        """)
        st.markdown("---")
        st.markdown("<div class='section-title'>Available Cities</div>", unsafe_allow_html=True)
        city_cols = st.columns(len(AVAILABLE_CITIES))
        for idx, (cname, coords) in enumerate(AVAILABLE_CITIES.items()):
            with city_cols[idx]:
                st.markdown(f"**{cname.split(',')[0]}**")
                st.caption(f"{coords[0]:.2f}°N, {coords[1]:.2f}°E")
        st.markdown("---")
        st.markdown("<div class='section-title'>How to use</div>", unsafe_allow_html=True)
        st.markdown("Select a city, pick a starting and ending attraction, set your available time and click 'Plan My Tour'.")

    else:
        # Tour planned: left = map + controls, right = conversation + details
        map_col, chat_col = st.columns([1.2, 1])

        with map_col:
            st.markdown("<div class='section-title'>🗺️ Your Tour Map</div>", unsafe_allow_html=True)

            # ============
            # Controls ABOVE the map (Get Tips / Next / Previous / Ask)
            # ============
            current_idx = st.session_state.current_stop
            stops = st.session_state.stops
            current_stop_name = stops[current_idx]

            control_container = st.container()
            with control_container:
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button("💡 Get Tips", key=f"tips_btn_{current_idx}"):
                        with st.spinner("Getting tips..."):
                            try:
                                tips = st.session_state.guide.get_visitor_tips(current_stop_name)
                                add_chat_message('user', f"Can you give me tips for {current_stop_name}?")
                                add_chat_message('guide', tips)
                            except Exception as e:
                                add_chat_message('guide', f"Sorry, couldn't fetch tips: {str(e)}")

                with bcol2:
                    if current_idx < len(stops) - 1:
                        if st.button("➡️ Next Stop", key=f"next_btn_{current_idx}", type="primary"):
                            try:
                                directions = None
                                for leg_start, leg_end, dist, walk_time, visit_time, attr in st.session_state.itinerary:
                                    if leg_start.lower() == current_stop_name.lower():
                                        directions = st.session_state.guide.get_walking_directions(leg_start, leg_end, dist, walk_time)
                                        add_chat_message('user', "I'm ready to go to the next stop!")
                                        add_chat_message('guide', f"Great! Let's head to **{leg_end}**.\n\n🚶 **Distance:** {dist:.2f} km\n⏱️ **Time:** ~{int(walk_time)} minutes\n\n{directions}")
                                        break
                                st.session_state.current_stop += 1
                            except Exception as e:
                                add_chat_message('guide', f"Could not compute next leg: {str(e)}")
                    else:
                        st.success("🎉 You've completed the tour!")

                # Previous Stop (full-width)
                if current_idx > 0:
                    if st.button("⬅️ Previous Stop", key=f"prev_btn_{current_idx}"):
                        st.session_state.current_stop -= 1
                        add_chat_message('user', "Let me go back to the previous stop.")
                        add_chat_message('guide', f"No problem! Going back to **{stops[current_idx-1]}**.")

                # Ask your guide anything input
                question = st.text_input(
                    "❓ Ask your guide anything",
                    placeholder="e.g., What are the opening hours?",
                    key=f"question_input_{current_idx}"
                )
                if question:
                    with st.spinner("Thinking..."):
                        try:
                            answer = st.session_state.guide.answer_question(question, current_stop_name)
                        except Exception as e:
                            answer = f"Sorry, couldn't retrieve an answer: {str(e)}"
                        add_chat_message('user', question)
                        add_chat_message('guide', answer)

            # Now render the map below the controls
            tour_map = create_tour_map(
                st.session_state.G,
                st.session_state.route_nodes,
                st.session_state.stops_info,
                st.session_state.city_name,
                highlight_stop=st.session_state.current_stop
            )

            st_folium(tour_map, width=700, height=700, key=f"map_{st.session_state.current_stop}")

            # Stats
            st.markdown("---")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.markdown(f"""
                    <div class="stats-box">
                        <h3>📍 {len(st.session_state.stops)}</h3>
                        <p>Total Stops</p>
                    </div>
                """, unsafe_allow_html=True)
            with stats_col2:
                st.markdown(f"""
                    <div class="stats-box">
                        <h3>🚶 {st.session_state.total_distance:.2f} km</h3>
                        <p>Total Distance</p>
                    </div>
                """, unsafe_allow_html=True)
            with stats_col3:
                st.markdown(f"""
                    <div class="stats-box">
                        <h3>⏰ {format_minutes(st.session_state.total_time)}</h3>
                        <p>Total Time</p>
                    </div>
                """, unsafe_allow_html=True)

        with chat_col:
            st.markdown("<div class='section-title'>💬 Your AI Guide</div>", unsafe_allow_html=True)

            # Progress indicator
            current_idx = st.session_state.current_stop
            stops = st.session_state.stops

            progress_html = '<div style="margin-bottom:0.6rem">'
            for i in range(len(stops)):
                if i < current_idx:
                    progress_html += f'<span style="margin-right:6px;padding:0.5rem 1rem;border-radius:999px;background:#28a745;color:white;font-weight:700;">✓ Stop {i+1}</span>'
                elif i == current_idx:
                    progress_html += f'<span style="margin-right:6px;padding:0.5rem 1rem;border-radius:999px;background:#007BFF;color:white;font-weight:700;">📍 Stop {i+1}</span>'
                else:
                    progress_html += f'<span style="margin-right:6px;padding:0.5rem 1rem;border-radius:999px;background:#E9ECEF;color:#6C757D;font-weight:700;">Stop {i+1}</span>'
            progress_html += '</div>'
            st.markdown(progress_html, unsafe_allow_html=True)

            # Conversation history
            st.markdown("---")
            for msg in st.session_state.conversation_history:
                if msg['role'] == 'guide':
                    st.markdown(f"""
                        <div class="chat-message guide">
                            <div style="font-weight:700;">Guide</div>
                            <div>{msg['message']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message user">
                            <div style="font-weight:700;">You</div>
                            <div>{msg['message']}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Current stop details on the right
            current_stop_name = stops[current_idx]
            st.markdown(f"""
                <div class="stop-card">
                    <div style="font-size:0.95rem;color:#FFC107;font-weight:700;letter-spacing:0.05em;">Stop {current_idx + 1}/{len(stops)}</div>
                    <div style="font-size:1.5rem;font-weight:800;margin-top:0.2rem;">{current_stop_name}</div>
                </div>
            """, unsafe_allow_html=True)

            with st.spinner("Loading attraction details..."):
                try:
                    description = st.session_state.guide.describe_attraction(current_stop_name)
                except Exception as e:
                    description = f"Description not available: {str(e)}"

            st.markdown(f'<div style="white-space:pre-wrap;color:white;">{description}</div>', unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()