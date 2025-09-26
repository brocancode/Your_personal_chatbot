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
st.set_page_config(page_title="Personalized Chatbot", page_icon="🤖", layout="wide")

# Full-page background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("background.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom chat UI styling
st.markdown("""
    <style>
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stMarkdown {
            font-family: 'Helvetica', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌟 Personalized Chatbot")
st.subheader("Chat with me and get text + GIF responses!")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Functions
def predict_intent(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

def get_response(intent):
    for i in intents_data['intents']:
        if i['tag'] == intent:
            response = random.choice(i['responses'])
            gif = random.choice(i['gifs'])
            return response, gif
    return "Sorry, I didn't understand that.", None
# ----------------------------

# ----------------------------
# User input
user_input = st.text_input("You: ", "")

if user_input:
    intent = predict_intent(user_input)
    response, gif = get_response(intent)
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", response))
    if gif:
        st.session_state.messages.append(("Bot", gif))

# Display chat history
for sender, message in st.session_state.messages:
    if sender == "You":
        st.markdown(f"**{sender}:** {message}")
    else:
        if message.endswith(('.gif', '.png', '.jpg')):
            st.image(message, width=250)
        else:
            st.markdown(f"**{sender}:** {message}")
# ----------------------------