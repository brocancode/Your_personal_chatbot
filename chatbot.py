import streamlit as st
import random
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
# ----------------------------

# Load intents
with open('intents.json') as file:
    intents_data = json.load(file)

# Prepare training data
training_sentences = []
training_labels = []
classes = []

for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Vectorize the training sentences
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_sentences)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, training_labels)

# ----------------------------
# Streamlit UI configuration
st.set_page_config(page_title="Honey Bun Chatbot", page_icon="🤖", layout="wide")

# Full-page background image
st.markdown(
    f"""
    <style>
    /* Full-page background */
    .stApp {{
        background: url("background.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
    }}

    /* Make main content area transparent */
    .stApp > .main {{
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
    }}

    /* Chat container */
    .chat-container {{
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        background-color: rgba(255,255,255,0.6);
        border-radius: 10px;
        margin-bottom: 10px;
    }}

    /* Input box and button */
    .stTextInput>div>div>input {{
        border-radius: 20px;
        padding: 10px;
        font-size: 16px;
    }}
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-size: 16px;
    }}

    /* Markdown text */
    .stMarkdown {{
        font-family: 'Helvetica', sans-serif;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hey there love!🌸🌼🌺🌷🪷🌻🪻")
st.subheader("Ask me a question!😙")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Functions
def predict_intent(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

def get_response(intent):
    for i in intents_data['intents']:
        if i['tag'] == intent:
            response = random.choice(i['responses'])
            gif = random.choice(i['gifs']) if 'gifs' in i else None
            return response, gif
    return "Sorry, I didn't understand that.", None

# ----------------------------
# Display chat history inside a scrollable container
chat_placeholder = st.container()
with chat_placeholder:
    for msg in st.session_state.messages:
        sender, content = msg["sender"], msg["content"]
        if sender == "You":
            st.markdown(f"**{sender}:** {content}")
        else:
            if content.endswith(('.gif', '.png', '.jpg')):
                st.image(content, width=250)
            else:
                st.markdown(f"**{sender}:** {content}")

# ----------------------------
# User input at the bottom
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        # Add user message
        st.session_state.messages.append({"sender": "You", "content": user_input})

        # Predict intent
        intent = predict_intent(user_input)
        response, gif = get_response(intent)

        # Add bot response
        st.session_state.messages.append({"sender": "Bob", "content": response})
        if gif:
            st.session_state.messages.append({"sender": "Bob", "content": gif})

        # Clear input and rerun
        st.session_state.user_input = ""

st.text_input("You: ", key="user_input", on_change=handle_input)