# app.py
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import json
import chromadb
from chromadb.utils import embedding_functions
import re
from datetime import datetime, timedelta

# ----------------------------
# 1️⃣ Load env and initialize Groq
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please add GROQ_API_KEY to your .env file")

client = Groq(api_key=GROQ_API_KEY)

# ----------------------------
# 2️⃣ Streamlit UI setup
# ----------------------------
st.set_page_config(page_title="🎬 CineBook ", page_icon="🎟️")
st.title("🎬 CineBook — AI Movie Ticket & Policy Assistant")

st.markdown("""
### 🍿 Welcome to CineBook AI Assistant  
Ask about movie showtimes, get genre-based suggestions, or ask policy questions (refunds, privacy, guidelines).

**CineBook Theaters (Chennai):**                        
1️⃣ CineWorld Downtown   
2️⃣ StarMax Cinemas  
3️⃣ Galaxy Cinemas  
4️⃣ SilverLine Theaters               
5️⃣ Regal Movie Hub
""")
st.divider()

DATA_FOLDER = "data"

# ----------------------------
# 3️⃣ Load JSON files & preprocess into natural language facts
# ----------------------------
def load_all_data(data_folder=DATA_FOLDER):
    movies_map = {}
    theaters_map = {}
    shows_list = []
    extra_texts = []

    # Read movies.json
    movies_path = os.path.join(data_folder, "movies.json")
    if os.path.exists(movies_path):
        with open(movies_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for m in data.get("movies", []):
                # Accept either movie_id or id naming
                mid = m.get("movie_id") or m.get("id")
                movies_map[mid] = {
                    "title": m.get("title"),
                    "language": m.get("language"),
                    "genre": m.get("genre")
                }

    # Read theaters.json
    theaters_path = os.path.join(data_folder, "theaters.json")
    if os.path.exists(theaters_path):
        with open(theaters_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for t in data.get("theaters", []):
                tid = t.get("theater_id") or t.get("id")
                theaters_map[tid] = {
                    "name": t.get("name"),
                    "location": t.get("location")
                }

    # Read shows.json
    shows_path = os.path.join(data_folder, "shows.json")
    if os.path.exists(shows_path):
        with open(shows_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for s in data.get("shows", []):
                shows_list.append({
                    "show_id": s.get("show_id"),
                    "theater_id": s.get("theater_id"),
                    "movie_id": s.get("movie_id"),
                    # accept 'time' or 'timings' arrays
                    "time": s.get("time") or (", ".join(s.get("timings", [])) if s.get("timings") else None)
                })

    # Flatten other JSONs (policies etc.) into readable sentences
    def flatten(obj, parent=""):
        parts = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{parent} {k}".strip()
                parts.extend(flatten(v, key))
        elif isinstance(obj, list):
            for item in obj:
                parts.extend(flatten(item, parent))
        else:
            parts.append(f"{parent}: {obj}")
        return parts

    # Iterate all files in data folder and collect textual content for RAG
    for filename in os.listdir(data_folder):
        path = os.path.join(data_folder, filename)
        if not filename.endswith(".json"):
            continue
        # skip movies/theaters/shows (they are processed above)
        if filename in ("movies.json", "theaters.json", "shows.json"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                flat = flatten(data)
                extra_texts.append(". ".join(flat))
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Build human-readable fact sentences for each show
    fact_sentences = []
    for s in shows_list:
        tid = s.get("theater_id")
        mid = s.get("movie_id")
        time = s.get("time") or "time not specified"
        theater = theaters_map.get(tid, {"name": f"Unknown ({tid})", "location": "Unknown"})
        movie = movies_map.get(mid, {"title": f"Unknown ({mid})", "language": "Unknown", "genre": "Unknown"})
        sentence = (
            f"The movie '{movie['title']}' ({movie['language']}, {movie['genre']}) "
            f"is showing at {theater['name']} in {theater['location']} at {time}."
        )
        fact_sentences.append(sentence)

    # Also include simple movie facts (title -> genre/lang) for retrieval
    for mid, info in movies_map.items():
        fact_sentences.append(f"The movie '{info['title']}' is a {info.get('genre','Unknown')} film in {info.get('language','Unknown')}.")

    # add theater descriptions too
    for tid, info in theaters_map.items():
        fact_sentences.append(f"{info['name']} is located in {info['location']} and is part of CineBook's 5 theaters.")

    # Combine everything to return
    corpus = fact_sentences + extra_texts
    return {
        "movies": movies_map,
        "theaters": theaters_map,
        "shows": shows_list,
        "corpus": corpus,
        "extra_texts": extra_texts
    }

DATA = load_all_data(DATA_FOLDER)

# ----------------------------
# 4️⃣ Build / reuse Chroma collection with embeddings
# ----------------------------
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

# Refresh collection to pick up new/updated JSONs each run
try:
    chroma_client.delete_collection("movie_agent")
except Exception:
    pass

collection = chroma_client.create_collection(name="movie_agent", embedding_function=embedding_fn)

# Add documents
collection.add(
    documents=DATA["corpus"],
    ids=[f"doc_{i}" for i in range(len(DATA["corpus"]))]
)

# ----------------------------
# 5️⃣ Utility: recommendation (genre/language), direct data lookup
# ----------------------------
def recommend_by_genre_or_language(user_text):
    text = user_text.lower()
    found = []
    # genres and languages extracted from movies data
    for s in DATA["shows"]:
        mid = s.get("movie_id")
        m = DATA["movies"].get(mid, {})
        genre = (m.get("genre") or "").lower()
        language = (m.get("language") or "").lower()
        title = m.get("title") or "Unknown"
        theater_info = DATA["theaters"].get(s.get("theater_id"), {})
        theater_name = theater_info.get("name", f"Unknown ({s.get('theater_id')})")
        show_time = s.get("time") or "time not specified"

        # match genre keywords
        # check if any genre word appears in user text
        if any(g in text for g in re.split(r"[\/,\s]+", genre)) and any(g in text for g in re.split(r"[\/,\s]+", genre)):
            found.append({
                "title": title, "language": m.get("language"), "genre": m.get("genre"),
                "theater": theater_name, "time": show_time
            })
        # match language
        if language and language in text:
            found.append({
                "title": title, "language": m.get("language"), "genre": m.get("genre"),
                "theater": theater_name, "time": show_time
            })

    # Deduplicate by title + theater + time
    uniq = []
    seen = set()
    for f in found:
        key = (f["title"], f["theater"], f["time"])
        if key not in seen:
            seen.add(key)
            uniq.append(f)

    if not uniq:
        return None

    # Format response
    lines = []
    for u in uniq:
        lines.append(f"🎬 {u['title']} ({u['language']}, {u['genre']}) — {u['theater']} at {u['time']}")
    return "\n".join(lines)

def list_all_shows():
    """Return a readable list of all shows in all theaters."""
    if not shows_data:
        return "No show data available."

    # Group shows by theater
    theater_groups = {}
    for s in shows_data:
        theater = s["theater"]
        if theater not in theater_groups:
            theater_groups[theater] = []
        theater_groups[theater].append(
            f"'{s['movie']}' ({s['language']}, {s['genre']}) - {s['time']}"
        )

    # Format neatly
    response = "🎬 **Here are all the shows currently playing in CineBook theaters:**\n\n"
    for theater, entries in theater_groups.items():
        response += f"🏢 **{theater}:**\n" + "\n".join(f"• {entry}" for entry in entries) + "\n\n"
    return response.strip()


# ----------------------------
# 6️⃣ Refund policy handler (returns summary; if showtime present, calculate eligibility)
# ----------------------------
def parse_time_string(time_str):
    """Parse times like '06:00 PM' -> datetime.time; returns None if fails"""
    try:
        return datetime.strptime(time_str.strip(), "%I:%M %p")
    except Exception:
        # try H:M (24h)
        try:
            return datetime.strptime(time_str.strip(), "%H:%M")
        except Exception:
            return None

def handle_refund_query(user_text):
    ut = user_text.lower()
    if not any(k in ut for k in ("refund", "cancel", "cancellation", "eligible", "refunds")):
        return None

    # load refund_policy.json
    refund_path = os.path.join(DATA_FOLDER, "refund_policy.json")
    if not os.path.exists(refund_path):
        return "Sorry — refund policy data not found."

    with open(refund_path, "r", encoding="utf-8") as f:
        rdata = json.load(f)

    # Try to extract main policy pieces (we expect structure with "policies" array)
    policies = rdata.get("policies") or rdata.get("refund_policy") or rdata.get("policies", [])
    # Build human friendly summary from the file content
    summary_lines = []
    # If the JSON is the format you provided earlier:
    for p in rdata.get("policies", []):
        pid = p.get("policy_id") or p.get("policyId")
        title = p.get("policy_title") or p.get("policy_title", "")
        desc = p.get("description", "")
        details = p.get("details", {})
        # summarize only key ones
        if "CANCELLATION" in (pid or "") or "CANCELLATION" in (title or "").upper():
            cutoff = details.get("eligibility_cut_off") or details.get("eligibility_cutoff") or details.get("eligibility_cut_off")
            summary_lines.append(f"🔹 Cancellation Window: {cutoff or 'Not specified'}")
        if "FEES" in (pid or "") or "FEES" in (title or "").upper():
            proc = details.get("processing_time") or details.get("processing_time") or details.get("processing_time")
            fee = details.get("service_fee_status") or details.get("cancellation_fee")
            summary_lines.append(f"🔹 Fees & Processing: {fee or 'See policy'}. Processing time: {proc or 'Not specified'}.")
        if "NO_SHOW" in (pid or "") or "NO-SHOW" in (title or "").upper() or "NO_SHOW" in (title or "").upper():
            summary_lines.append("🔹 No-show: Tickets forfeited; no refund.")
        if "COMPANY_CANCELLATION" in (pid or "") or "CANCELLATION" in (title or "").upper() and "provider" in p.get("description","").lower():
            summary_lines.append("🔹 If provider cancels, full refund (including service fees) will be issued automatically.")

    # fallback generic summary using known template in your file (if present)
    if not summary_lines:
        # try extract from first policies list if matches your format
        for p in rdata.get("policies", []):
            tpl = p.get("agent_response_template")
            if tpl:
                summary_lines.append(tpl)

    # Base reply
    reply = "🎟️ **CineBook Refund & Cancellation Policy Summary**\n\n"
    if summary_lines:
        reply += "\n".join(summary_lines)
    else:
        reply += "Standard rules apply. Please see refund_policy.json for details."

    # If user mentions relative words like "yesterday" and "tomorrow" AND a show time, attempt to calculate
    # Look for a time string in user_text (e.g., "6:00 PM" or "18:00")
    time_match = re.search(r'(\d{1,2}:\d{2}\s?(?:AM|PM|am|pm))', user_text)
    if not time_match:
        # also search shows data for likely referenced movie name — if user named a movie, we can try to get its showtime(s)
        # Extract movie titles and check if any appear in user_text
        for mid, m in DATA["movies"].items():
            title = m.get("title","").lower()
            if title and title in user_text.lower():
                # find shows for this movie
                times = []
                theaters = []
                for s in DATA["shows"]:
                    if s.get("movie_id") == mid:
                        times.append(s.get("time"))
                        th = DATA["theaters"].get(s.get("theater_id"), {}).get("name")
                        theaters.append((th, s.get("time")))
                if times:
                    reply += "\n\nFound showtimes for that movie:\n"
                    for th, tm in theaters:
                        reply += f"- {th} at {tm}\n"
                    # If only one time, compute eligibility
                    if len(times) == 1:
                        time_match = re.search(r'(\d{1,2}:\d{2}\s?(?:AM|PM|am|pm))', times[0] or "")
                break

    # if we have a matched time, do a calculation
    if time_match:
        time_str = time_match.group(1)
        parsed = parse_time_string(time_str)
        if parsed:
            # build a full datetime for the next occurrence of that time (assume 'movie tomorrow' if user says tomorrow)
            # check for 'tomorrow' or 'today' in user text
            now = datetime.now()
            day_offset = 0
            if "tomorrow" in user_text.lower():
                day_offset = 1
            elif "today" in user_text.lower():
                day_offset = 0
            elif "yesterday" in user_text.lower():
                # weird: booked yesterday for tomorrow scenario is usually day_offset=1 for the event
                day_offset = 1

            movie_dt = datetime(
                year=now.year,
                month=now.month,
                day=now.day,
                hour=parsed.hour,
                minute=parsed.minute
            ) + timedelta(days=day_offset)

            cutoff_dt = movie_dt - timedelta(hours=2)
            # compute human readable remaining time
            if datetime.now() <= cutoff_dt:
                remain = cutoff_dt - datetime.now()
                minutes = int(remain.total_seconds()//60)
                reply += f"\n\n✅ Based on the showtime {time_str} on {movie_dt.strftime('%Y-%m-%d')}, you are eligible to cancel up to {cutoff_dt.strftime('%I:%M %p')} (2 hours before). You have approximately {minutes} minutes remaining to cancel."
            else:
                reply += f"\n\n❌ Based on the showtime {time_str} on {movie_dt.strftime('%Y-%m-%d')}, you're past the 2-hour cancellation cutoff and are not eligible for a refund."
        else:
            reply += "\n\n(Note: I found a time but could not parse it for exact calculation.)"

    # always append agent template from file if available for clarity
    # try to append agent_response_template from RF_001_CANCELLATION_WINDOW
    for pol in rdata.get("policies", []):
        if pol.get("policy_id", "").startswith("RF_001"):
            if pol.get("agent_response_template"):
                reply += f"\n\nTemplate: {pol.get('agent_response_template')}"
            break

    return reply

# ----------------------------
# 7️⃣ RAG retrieval function
# ----------------------------
def retrieve_context(user_query, top_k=20):
    results = collection.query(query_texts=[user_query], n_results=top_k)
    docs = results["documents"][0] if results and "documents" in results else []
    return "\n".join(docs) if docs else ""

# ----------------------------
# 8️⃣ Chat session init
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about movies, refunds, or policies...")

# ----------------------------
# 9️⃣ Main hybrid logic:
#     1) refund handler
#     2) genre/language recommender
#     3) RAG + LLM fallback (strict)
# ----------------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 1️⃣ Refund/cancellation policy handler
    # Step 1️⃣: Handle "list all shows" or similar queries directly
    if any(phrase in user_input.lower() for phrase in [
        "all shows",
        "show all theaters",
        "every theater",
        "all movies",
        "list all shows",
        "all showtimes",
        "all screenings",
        "list of shows",
        "everything playing"
    ]):
        reply = list_all_shows()


    else:
        policy_reply = handle_refund_query(user_input)
        if policy_reply:
            reply = policy_reply
        else:
        # 2️⃣ Genre/language quick recommender
            rec = recommend_by_genre_or_language(user_input)
            if rec:
                reply = rec
            else:
            # 3️⃣ RAG fallback
                context = retrieve_context(user_input)
                if not context.strip():
                    reply = "I'm sorry, I can only provide information about CineBook's 5 theaters, their movies, and related policies."
                else:
                    system_prompt = f"""
                    You are an AI support agent for a fictional movie ticket booking app called CineBook.

                    IMPORTANT RULES:
                    - Only use the context provided below to answer.
                    - CineBook has only 5 theaters: CineWorld Downtown, StarMax Cinemas, Galaxy Cinemas, SilverLine Theaters, and Regal Movie Hub.
                    - Do NOT mention or provide information about real-world theaters outside this dataset.
                    - If the question is outside CineBook data, say: "I'm sorry, I can only provide information about CineBook theaters and their shows."
                    - Be concise, natural, and mention movie names, theater names, and showtimes clearly if present in context.

                    CONTEXT:
                    {context}
                    """

                    chat = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *st.session_state.messages
                        ]
                    )
                    reply = chat.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)

    # show retrieved context for transparency (if any)
    with st.expander("📄 Retrieved Context (for debugging/explainability)"):
        st.write(retrieve_context(user_input))
