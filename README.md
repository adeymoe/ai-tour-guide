# 🌿 AI Tour Guide (Test Version)

AI Tour Guide is an ongoing project showcasing a test version of an AI-powered city tour planner and conversational guide. It combines advanced path planning algorithms with large language model (LLM) integration and real-time map data to deliver personalized, time-optimized city tours. The system leverages Google Gemini models for natural language interactions and OpenStreetMap data for geographic context.

---

## How It's Made:

**Tech stack:** Python, Streamlit, Folium, NetworkX, GeoPandas, Google Gemini API

- **Frontend:**
  - Streamlit for rapid interactive web UI development
  - Folium for dynamic, interactive map visualizations embedded in the app
  - Custom CSS implementing the "Voyage Explorer" theme for a modern, accessible user experience
  - Session state management to persist user inputs, loaded data, and conversation history across page reloads
  - Searchable dropdowns for city attractions with explicit load triggers to optimize performance

- **Backend / Core Logic:**
  - Path planning algorithms including weighted A* search for multi-stop route optimization under time constraints
  - Hybrid routing combining walking and public transport options
  - Real-time geographic data loading and caching using GeoPandas and OpenStreetMap extracts
  - Tour itinerary generation with distance, time, and attraction scoring
  - Conversational AI agent powered by Google Gemini for contextual tour guidance, tips, and Q&A

- **State & Performance:**
  - Extensive use of Streamlit's `session_state` to maintain UI state and avoid unnecessary reloads
  - Caching of city data and attraction lists with `st.cache_data` to reduce latency and API calls
  - Explicit user control over data loading to prevent automatic heavy operations on UI changes

---

## Project Overview:

- **Personalized Tour Planning:**  
  Users select a city, load its attractions on demand, and specify start/end points plus available time. The app computes an optimized route maximizing points of interest within constraints.

- **Conversational AI Guide:**  
  An LLM-powered assistant provides dynamic tips, walking directions, and answers user questions about attractions during the tour.

- **Interactive Map Visualization:**  
  Folium maps display the planned route with markers for each stop, highlighting the current location and providing intuitive navigation cues.

- **State Persistence & UX:**  
  The app preserves all user selections, loaded data, and conversation history even if the page is refreshed or revisited, ensuring a seamless experience.

---

## Current Status:

- This is a **work-in-progress test version** demonstrating core functionalities.
- The UI and backend logic are actively evolving with ongoing improvements planned.
- Some features (e.g., public transport integration, advanced itinerary customization) are under development.
- The app is not yet production-ready but serves as a technical prototype and proof of concept.

---

## Future Enhancements:

- Full integration of multi-modal transport options with real-time schedules
- Enhanced LLM prompt engineering for richer, context-aware tour conversations
- User authentication and personalized tour history tracking
- Mobile-friendly responsive design and offline caching capabilities
- Deployment pipeline for scalable cloud hosting and API management

---

## Running Locally:

```bash
# 1. Clone the repository
git clone https://github.com/adeymoe/ai-tour-guide.git
cd ai-tour-guide

# 2. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables (GEMINI_API_KEY)

# 5. Run the Streamlit app
streamlit run app.py