import streamlit as st
from transformers import pipeline

# --- Configuration ---
# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="Gistify - Text Summarizer App", # Updated page title
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Model Loading ---
@st.cache_resource
def load_summarizer_model():
    """
    Loads the pre-trained summarization model.
    Using @st.cache_resource to cache the model,
    preventing it from reloading every time the app re-runs.
    We'll use 'sshleifer/distilbart-cnn-12-6' as it's a good balance
    between performance and model size for general summarization.
    """
    st.info("Loading summarization model... This might take a moment.")
    try:
        # Initialize the summarization pipeline
        # Using 'sshleifer/distilbart-cnn-12-6' which is a distilled BART model
        # suitable for summarization.
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        st.success("Model loaded successfully!")
        return summarizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop() # Stop the app if model loading fails

# Load the summarizer model when the app starts
summarizer = load_summarizer_model()

# --- Streamlit UI ---
st.title("✍️ Gistify - Simple Text Summarizer") # Updated app title
st.markdown(
    """
    This app uses a pre-trained AI model (`distilbart-cnn-12-6`) to
    generate concise summaries of your text.
    """
)

# Text input area
input_text = st.text_area(
    "Enter your text here:",
    height=250,
    placeholder="Paste your long text here to get a summary...",
    help="The longer the text, the longer it might take to summarize."
)

# Summarization button
if st.button("Summarize Text", type="primary"):
    if input_text:
        # Display a spinner while processing
        with st.spinner("Summarizing..."):
            try:
                # Perform summarization
                # min_length and max_length can be adjusted based on desired summary length
                summary = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
                # Extract the summary text
                summarized_text = summary[0]['summary_text']

                st.subheader("Summary:")
                st.success(summarized_text)

            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
                st.warning("Please ensure your input text is valid and try again.")
    else:
        st.warning("Please enter some text to summarize.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Khan Faisal** with ❤️ using Streamlit and Hugging Face Transformers.")
st.markdown("Connect with me: [Portfolio](https://khanfaisal.netlify.app) | [GitHub](https://github.com/khanfaisal79960) | [Linkedin](https://www.linkedin.com/khanfaisal79960) | [Medium](https://medium.com/@khanfaisal79960) | [Instagram](https://instagram.com/mr._perfect_1004)") # Placeholder for social links
