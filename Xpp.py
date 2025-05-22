import streamlit as st
from einstein import EinsteinPersona
from kafka import KafkaPersona
from holmes import HolmesPersona
from white import WhitePersona
from custom_persona import CustomPersona
import time
import random
from gtts import gTTS  # For text-to-speech
import io
import base64

# Add pyttsx3 as an alternative (works offline)
try:
    import pyttsx3
    pyttsx3_available = True
except ImportError:
    pyttsx3_available = False

st.set_page_config(
    page_title="RAG Personas Chat",
    page_icon="🤖",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
    .persona-header {
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .user-message {
        background-color: #E8F4F9;
        margin-left: 20%;
        margin-right: 5%;
        color: #2c3e50;
    }
    .assistant-message {
        background-color: #F0F0F0;
        margin-right: 20%;
        margin-left: 5%;
        color: #34495e;
    }
    .user-message b {
        color: #2980b9;
    }
    .assistant-message b {
        color: #16a085;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 10px;
    }
    .message-content {
        flex-grow: 1;
    }
    .audio-player {
        margin-top: 5px;
        width: 100%;
    }
    .play-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 5px 10px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 12px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Persona definitions with voice characteristics
PERSONAS = {
    "Albert Einstein": {
        "class": EinsteinPersona,
        "emoji": "🧠",
        "avatar": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/220px-Albert_Einstein_Head.jpg",
        "description": "Theoretical physicist and creator of the theory of relativity.",
        "voice": {
            "speed": 150,  # words per minute
            "pitch": 110,  # Hz (slightly higher than average male voice)
            "accent": "en"  # English accent
        }
    },
    "Franz Kafka": {
        "class": KafkaPersona,
        "emoji": "📚",
        "avatar": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Franz_Kafka_1917.jpg/220px-Franz_Kafka_1917.jpg",
        "description": "Existentialist writer known for 'The Metamorphosis' and 'The Trial'.",
        "voice": {
            "speed": 120,
            "pitch": 100,
            "accent": "de"  # German accent
        }
    },
    "Sherlock Holmes": {
        "class": HolmesPersona,
        "emoji": "🔍",
        "avatar": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Sherlock_Holmes_Portrait_Paget.jpg/220px-Sherlock_Holmes_Portrait_Paget.jpg",
        "description": "Fictional detective famed for logical reasoning and keen observation.",
        "voice": {
            "speed": 180,  # Fast, analytical speech
            "pitch": 105,
            "accent": "en-uk"  # British accent
        }
    },
    "Walter White": {
        "class": WhitePersona,
        "emoji": "🧪",
        "avatar": "https://upload.wikimedia.org/wikipedia/en/0/03/Walter_White_S5B.png",
        "description": "Chemistry teacher turned drug kingpin in Breaking Bad.",
        "voice": {
            "speed": 130,
            "pitch": 95,  # Lower pitch for intensity
            "accent": "en-us"  # American accent
        }
    }
}

# Initialize session state
if "personas_initialized" not in st.session_state:
    st.session_state.personas_initialized = False
    st.session_state.personas = {}
    st.session_state.chats = {name: [] for name in PERSONAS.keys()}
    st.session_state.custom_personas = {}
    st.session_state.show_add_persona = False
    st.session_state.voice_enabled = True  # Voice feature enabled by default
    st.session_state.audio_cache = {}  # Cache for audio files

# Sidebar
st.sidebar.title("🎭 Choose Persona")
all_personas = list(PERSONAS.keys()) + list(st.session_state.custom_personas.keys())
selected_persona = st.sidebar.selectbox("Select a character to talk to:", all_personas)

# Voice settings
st.sidebar.markdown("### 🔊 Voice Settings")
voice_enabled = st.sidebar.checkbox("Enable voice responses", value=st.session_state.voice_enabled)
if voice_enabled != st.session_state.voice_enabled:
    st.session_state.voice_enabled = voice_enabled
    st.rerun()

if st.sidebar.button("➕ Add Persona"):
    st.session_state.show_add_persona = not st.session_state.show_add_persona

# Persona adder form
if st.session_state.show_add_persona:
    st.sidebar.markdown("### 🔍 Add a New Persona from Wikipedia")
    with st.sidebar.form("add_persona_form"):
        persona_name = st.text_input("Name:", placeholder="e.g., Naguib Mahfouz")
        persona_description = st.text_area("Description:", placeholder="Egyptian novelist and Nobel Laureate.")
        persona_characteristics = st.text_area("Traits (one per line):", 
                                               placeholder="- Formal literary tone\n- References to Egyptian culture")
        emoji_choices = ["📚", "✍️", "🎭", "🎬", "🎨", "🎤", "🧠", "👑", "🧪", "🔭", "🧬", "🖋️"]
        persona_emoji = st.selectbox("Choose an emoji:", emoji_choices)
        
        # Voice settings for custom persona
        st.markdown("### 🎙️ Voice Characteristics")
        voice_speed = st.slider("Speech speed (words per minute)", 80, 200, 150)
        voice_pitch = st.slider("Pitch (relative)", 80, 120, 100)
        voice_accent = st.selectbox("Accent", ["en", "en-uk", "en-us", "de", "fr", "es", "it"])
        
        submit_persona = st.form_submit_button("Add Persona")

        if submit_persona and persona_name and persona_description and persona_characteristics:
            formatted_characteristics = "\n".join([
                f"- {char.strip('-').strip()}" 
                for char in persona_characteristics.split("\n") if char.strip()
            ])
            from custom_persona import create_prompt_template
            prompt_template = create_prompt_template(persona_name, formatted_characteristics)
            custom_persona = CustomPersona(persona_name, persona_description, prompt_template)
            status = st.empty()
            status.markdown("⏳ Creating index from Wikipedia...")

            result = custom_persona.create_index_from_wikipedia()
            if "تم إنشاء فهرس بنجاح" in result:
                avatar_url = f"https://ui-avatars.com/api/?name={persona_name.replace(' ', '+')}&background=random&color=fff&size=128"
                st.session_state.custom_personas[persona_name] = {
                    "class": custom_persona,
                    "emoji": persona_emoji,
                    "avatar": avatar_url,
                    "description": persona_description,
                    "custom": True,
                    "voice": {
                        "speed": voice_speed,
                        "pitch": voice_pitch,
                        "accent": voice_accent
                    }
                }
                st.session_state.chats[persona_name] = []
                status.success(f"✅ {persona_name} added successfully!")
                st.session_state.show_add_persona = False
                st.rerun()
            else:
                status.error(f"❌ Failed to add persona: {result}")

# Get current persona info
if selected_persona in PERSONAS:
    persona_info = PERSONAS[selected_persona]
    is_custom = False
else:
    persona_info = st.session_state.custom_personas[selected_persona]
    is_custom = True

# Text-to-speech function
def text_to_speech(text, persona_name, persona_info):
    # Check cache first
    cache_key = f"{persona_name}_{hash(text)}"
    if cache_key in st.session_state.audio_cache:
        return st.session_state.audio_cache[cache_key]
    
    try:
        if pyttsx3_available:
            # Use pyttsx3 for offline voice generation with more control
            engine = pyttsx3.init()
            
            # Set voice properties based on persona
            voice_props = persona_info.get("voice", {})
            engine.setProperty('rate', voice_props.get("speed", 150))
            engine.setProperty('pitch', voice_props.get("pitch", 100) / 100)
            
            # Try to set accent if available
            voices = engine.getProperty('voices')
            if "en-uk" in str(voice_props.get("accent", "")) and any("english" in v.id.lower() for v in voices):
                engine.setProperty('voice', [v.id for v in voices if "english" in v.id.lower()][0])
            elif "en-us" in str(voice_props.get("accent", "")) and any("english" in v.id.lower() for v in voices):
                engine.setProperty('voice', [v.id for v in voices if "english" in v.id.lower()][0])
            
            # Save to memory
            engine.save_to_file(text, f"temp_{persona_name}.mp3")
            engine.runAndWait()
            
            with open(f"temp_{persona_name}.mp3", "rb") as f:
                audio_bytes = f.read()
            
            # Encode as base64 for HTML audio player
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_html = f"""
            <audio controls class="audio-player">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            st.session_state.audio_cache[cache_key] = audio_html
            return audio_html
        else:
            # Fall back to gTTS
            tts = gTTS(text=text, lang=persona_info.get("voice", {}).get("accent", "en"), slow=False)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            
            # Encode as base64 for HTML audio player
            audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
            audio_html = f"""
            <audio controls class="audio-player">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            st.session_state.audio_cache[cache_key] = audio_html
            return audio_html
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return ""

# Main area
st.title("🤖 RAG Persona Simulator")
st.caption("Chat with famous personas from history, literature, or fiction")

col1, col2 = st.columns([1, 3])
with col1:
    st.image(persona_info["avatar"], width=150)
with col2:
    st.markdown(f"## {persona_info['emoji']} {selected_persona}")
    st.markdown(persona_info["description"])
    if is_custom:
        st.caption("Custom persona added from Wikipedia")

# Conversation history
st.markdown("### 💬 Conversation")
chat_container = st.container()

with chat_container:
    for message in st.session_state.chats[selected_persona]:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-content">
                    <b>You:</b> {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            audio_html = ""
            if st.session_state.voice_enabled and "content" in message:
                audio_html = text_to_speech(message["content"], selected_persona, persona_info)
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <img src="{persona_info['avatar']}" class="avatar">
                <div class="message-content">
                    <b>{selected_persona}:</b> {message["content"]}
                    {audio_html if st.session_state.voice_enabled else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Message input form
status_placeholder = st.empty()
with st.form("message_form", clear_on_submit=True):
    user_input = st.text_input(f"Ask {selected_persona} something:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.chats[selected_persona].append({"role": "user", "content": user_input})

    with chat_container:
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-content">
                <b>You:</b> {user_input}
            </div>
        </div>
        """, unsafe_allow_html=True)

    status_placeholder.markdown("Thinking...")

    if not st.session_state.personas_initialized:
        status_placeholder.markdown("Loading personas for the first time...")
        st.session_state.personas = {name: info["class"]() for name, info in PERSONAS.items()}
        st.session_state.personas_initialized = True

    try:
        persona_instance = (
            st.session_state.personas[selected_persona]
            if selected_persona in PERSONAS
            else st.session_state.custom_personas[selected_persona]["class"]
        )
        response = persona_instance.ask(user_input)
        st.session_state.chats[selected_persona].append({"role": "assistant", "content": response})

        with chat_container:
            audio_html = ""
            if st.session_state.voice_enabled:
                audio_html = text_to_speech(response, selected_persona, persona_info)
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <img src="{persona_info['avatar']}" class="avatar">
                <div class="message-content">
                    <b>{selected_persona}:</b> {response}
                    {audio_html if st.session_state.voice_enabled else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

    status_placeholder.empty()

# Clear chat button
if st.button("🗑️ Clear Conversation"):
    st.session_state.chats[selected_persona] = []
    st.session_state.audio_cache = {}  # Clear audio cache
    st.rerun()

# App Info
st.markdown("---")
st.markdown("### ℹ️ About This App")
st.markdown("""
This app uses **Retrieval-Augmented Generation (RAG)** to simulate conversations with famous personas:
- Powered by LLaMA 3.2 for responses
- Uses FAISS vector indexing for persona knowledge
- Employs `all-MiniLM` for text embedding
- Supports adding custom personas temporarily via Wikipedia
- Features **voice responses** with persona-specific characteristics
""")