# ==========================================
# LLM TOUR GUIDE MODULE
# AI-powered tour narrator with rich descriptions
# Uses Google Gemini for natural language generation
# ==========================================

import google.generativeai as genai
from config import GEMINI_API_KEY, LLM_MODEL
import sys

print("✓ LLM tour guide module loaded successfully.")

# ==========================================
# SECTION 1: ATTRACTION DATABASE
# ==========================================

class AttractionDatabase:
    """
    Manages attraction information and provides context to the LLM.
    """
    
    def __init__(self, city_name, pois):
        self.city_name = city_name
        self.pois = pois
        self.attraction_cache = {}
    
    def get_attraction_info(self, attraction_name):
        """
        Get basic information about an attraction from OSM data.
        """
        if attraction_name in self.attraction_cache:
            return self.attraction_cache[attraction_name]
        
        if self.pois.empty or "name" not in self.pois.columns:
            return None
        
        matching_pois = self.pois[self.pois["name"].str.lower() == attraction_name.lower()]
        
        if matching_pois.empty:
            return None
        
        poi = matching_pois.iloc[0]
        
        info = {
            'name': attraction_name,
            'type': poi.get('tourism', 'attraction'),
            'city': self.city_name,
        }
        
        # Add optional fields if available
        if 'addr:street' in poi:
            info['street'] = poi['addr:street']
        if 'website' in poi:
            info['website'] = poi['website']
        if 'opening_hours' in poi:
            info['hours'] = poi['opening_hours']
        
        self.attraction_cache[attraction_name] = info
        return info


# ==========================================
# SECTION 2: TOUR GUIDE AGENT
# ==========================================

class TourGuideAgent:
    """
    AI-powered tour guide using Google Gemini.
    Provides rich descriptions, historical context, and answers questions.
    """
    
    def __init__(self, city_name, itinerary, pois):
        self.city_name = city_name
        self.itinerary = itinerary
        self.pois = pois
        self.database = AttractionDatabase(city_name, pois)
        self.current_stop = 0
        self.conversation_history = []
        
        # Initialize Gemini
        self.llm_available = False
        if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(LLM_MODEL)
                self.llm_available = True
                print("✓ Gemini AI initialized successfully!")
            except Exception as e:
                print(f"⚠ Warning: Could not initialize Gemini: {e}")
                print("Using fallback responses...")
        else:
            print("⚠ No API key found. Using fallback responses...")
    
    def get_tour_stops(self):
        """Extract unique stops from itinerary."""
        stops = []
        seen = set()
        
        for leg_start, leg_end, _, _, _, _ in self.itinerary:
            if leg_start.lower() not in seen:
                stops.append(leg_start)
                seen.add(leg_start.lower())
        
        # Add final destination
        if self.itinerary:
            final_dest = self.itinerary[-1][1]
            if final_dest.lower() not in seen:
                stops.append(final_dest)
        
        return stops
    
    def generate_welcome_message(self):
        """Generate welcome message for the tour."""
        stops = self.get_tour_stops()
        
        if not self.llm_available:
            return self._fallback_welcome(stops)
        
        prompt = f"""You are an enthusiastic tour guide in {self.city_name}. 
Generate a warm, engaging welcome message (2-3 sentences) for tourists about to start a walking tour.

The tour includes these stops:
{', '.join(stops)}

Be friendly, exciting, and mention 1-2 highlights they'll see. Keep it concise."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return self._fallback_welcome(stops)
    
    def _fallback_welcome(self, stops):
        """Fallback welcome message when LLM is unavailable."""
        return f"""🎉 Welcome to your {self.city_name} walking tour! 

Today we'll explore {len(stops)} incredible attractions, from {stops[0]} to {stops[-1]}. 
Get ready for an unforgettable journey through history, culture, and beauty!"""
    
    def describe_attraction(self, attraction_name):
        """Generate rich description of an attraction."""
        info = self.database.get_attraction_info(attraction_name)
        
        if not self.llm_available:
            return self._fallback_description(attraction_name, info)
        
        attraction_type = info['type'] if info else 'attraction'
        
        prompt = f"""You are a knowledgeable tour guide in {self.city_name}.

Provide a rich, engaging description (3-4 paragraphs) of: {attraction_name}

Include:
1. What it is and why it's significant
2. Historical background or interesting facts
3. What visitors can see/experience there
4. One unique or surprising detail

Type: {attraction_type}
City: {self.city_name}

Be enthusiastic but informative. Write in second person ("you'll see..."). Keep it conversational."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return self._fallback_description(attraction_name, info)
    
    def _fallback_description(self, attraction_name, info):
        """Fallback description when LLM is unavailable."""
        attraction_type = info['type'] if info else 'attraction'
        
        return f"""🏛️ {attraction_name}

This is a notable {attraction_type} in {self.city_name}. It's a popular destination 
for visitors interested in the cultural and historical heritage of the area.

The site offers unique insights into the local architecture and traditions. 
Many tourists find it to be a highlight of their visit to {self.city_name}.

Take your time to explore and appreciate the details that make this place special!"""
    
    def get_walking_directions(self, from_attraction, to_attraction, distance_km, walk_time_min):
        """Generate walking directions between attractions."""
        if not self.llm_available:
            return self._fallback_directions(from_attraction, to_attraction, distance_km, walk_time_min)
        
        prompt = f"""You are a tour guide giving walking directions in {self.city_name}.

From: {from_attraction}
To: {to_attraction}
Distance: {distance_km:.2f} km
Time: ~{int(walk_time_min)} minutes

Provide brief, friendly walking directions (2-3 sentences). Include:
- General direction (north, south, etc.)
- What to look for along the way
- Encouragement

Be conversational and helpful."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return self._fallback_directions(from_attraction, to_attraction, distance_km, walk_time_min)
    
    def _fallback_directions(self, from_attraction, to_attraction, distance_km, walk_time_min):
        """Fallback directions when LLM is unavailable."""
        return f"""🚶 Walking from {from_attraction} to {to_attraction}

Distance: {distance_km:.2f} km (~{int(walk_time_min)} minutes)

Follow the scenic route marked on your map. Take your time and enjoy the 
architecture and street life along the way. You'll arrive at {to_attraction} soon!"""
    
    def answer_question(self, question, current_attraction):
        """Answer user questions about attractions or the tour."""
        if not self.llm_available:
            return self._fallback_answer(question, current_attraction)
        
        prompt = f"""You are a knowledgeable tour guide in {self.city_name}.

Current location: {current_attraction}

Tourist question: {question}

Provide a helpful, accurate answer (2-3 sentences). If you don't know specific details, 
give general helpful information. Be friendly and conversational."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return self._fallback_answer(question, current_attraction)
    
    def _fallback_answer(self, question, current_attraction):
        """Fallback answer when LLM is unavailable."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['photo', 'picture', 'camera']):
            return f"📸 Great question! The best photo spots at {current_attraction} are usually near the main entrance or facade. Early morning or late afternoon light works best!"
        
        elif any(word in question_lower for word in ['time', 'long', 'hours']):
            return f"⏰ Most visitors spend 15-30 minutes at {current_attraction}. Take your time to fully appreciate it!"
        
        elif any(word in question_lower for word in ['ticket', 'price', 'cost', 'fee']):
            return f"💰 For current ticket prices and booking information for {current_attraction}, I recommend checking their official website or asking at the entrance."
        
        elif any(word in question_lower for word in ['food', 'eat', 'restaurant', 'cafe']):
            return f"🍽️ There are usually cafes and restaurants near {current_attraction}. Look for local spots just off the main tourist areas for better value!"
        
        else:
            return f"That's an interesting question about {current_attraction}! For detailed information, I recommend checking with the information desk or official guides at the site."
    
    def get_visitor_tips(self, attraction_name):
        """Generate practical visitor tips."""
        if not self.llm_available:
            return self._fallback_tips(attraction_name)
        
        prompt = f"""You are a tour guide providing practical tips for visiting {attraction_name} in {self.city_name}.

Provide 4-5 bullet-point tips including:
- Best time to visit
- What to bring/wear
- Photography tips
- Common mistakes to avoid
- Insider recommendations

Be specific and practical. Format as bullet points."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return self._fallback_tips(attraction_name)
    
    def _fallback_tips(self, attraction_name):
        """Fallback tips when LLM is unavailable."""
        return f"""💡 VISITOR TIPS: {attraction_name}

• Best time: Early morning or late afternoon for fewer crowds
• Bring: Water, comfortable shoes, camera
• Photography: Check if flash photography is allowed inside
• Tickets: Book online in advance when possible
• Insider tip: Take time to explore the surrounding area too!"""


# ==========================================
# SECTION 3: INTERACTIVE TOUR GUIDE
# ==========================================

def interactive_tour_guide(city_name, itinerary, pois):
    """
    Main interactive tour guide experience.
    Walks users through each stop with rich descriptions and Q&A.
    """
    
    print("\n" + "=" * 70)
    print("   🎭 AI TOUR GUIDE - INTERACTIVE EXPERIENCE")
    print(f"   Exploring {city_name}")
    print("=" * 70)
    
    # Initialize tour guide
    guide = TourGuideAgent(city_name, itinerary, pois)
    stops = guide.get_tour_stops()
    
    # Welcome message
    print("\n" + guide.generate_welcome_message())
    print("\n" + "=" * 70)
    
    # Tour through each stop
    for stop_num, stop_name in enumerate(stops, 1):
        print(f"\n📍 STOP {stop_num}/{len(stops)}: {stop_name}")
        print("-" * 70)
        
        # Describe attraction
        description = guide.describe_attraction(stop_name)
        print(f"\n{description}\n")
        
        # Interactive Q&A for this stop
        print("-" * 70)
        print("🎤 COMMANDS:")
        print("   • Type a question to learn more")
        print("   • 'tips' - Get visitor tips")
        print("   • 'next' - Continue to next stop")
        print("   • 'quit' - End tour")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\n🎤 You: ").strip().lower()
                
                if not user_input:
                    continue
                
                if user_input in ['next', 'continue', 'move on', 'go']:
                    break
                
                elif user_input in ['quit', 'exit', 'stop', 'end']:
                    print("\n👋 Thanks for touring with us! Enjoy the rest of your visit!")
                    return
                
                elif user_input in ['tips', 'tip', 'advice', 'help']:
                    tips = guide.get_visitor_tips(stop_name)
                    print(f"\n{tips}\n")
                
                else:
                    # Answer question
                    answer = guide.answer_question(user_input, stop_name)
                    print(f"\n🗣️ Guide: {answer}\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Tour interrupted. Goodbye!")
                return
        
        # Walking directions to next stop (if not last stop)
        if stop_num < len(stops):
            # Find the leg in itinerary
            for leg_start, leg_end, dist, walk_time, visit_time, attr in itinerary:
                if leg_start.lower() == stop_name.lower():
                    print(f"\n🚶 WALKING DIRECTIONS:")
                    print("-" * 70)
                    directions = guide.get_walking_directions(
                        leg_start, leg_end, dist, walk_time
                    )
                    print(f"{directions}\n")
                    break
    
    # Tour complete
    print("\n" + "=" * 70)
    print("   🎉 TOUR COMPLETE!")
    print("=" * 70)
    print(f"\n✨ Thank you for exploring {city_name} with us!")
    print("We hope you enjoyed your tour and discovered amazing places!")
    print("\n📸 Don't forget to share your photos and memories!")
    print("👋 Safe travels!\n")


# ==========================================
# STANDALONE EXECUTION (FOR TESTING)
# ==========================================

if __name__ == "__main__":
    print("\n⚠️  This module is designed to be imported by main.py")
    print("To test the tour guide, run: python main.py")
    print("\nFor standalone testing, you need to provide:")
    print("  - city_name: str")
    print("  - itinerary: list of tuples")
    print("  - pois: pandas DataFrame")
