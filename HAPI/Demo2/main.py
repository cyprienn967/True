from flask import Flask, render_template
from flask_socketio import SocketIO
import eventlet
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")

# Predefined texts for the demo
TEXT_LLAMMA7B = "This is the output from Llama7b without modifications."
TEXT_HAPI = "This is the output using Llama7b with HAPI, which significantly enhances accuracy."

# Words to highlight
RED_WORDS = {"modifications", "without"}  # Example of less accurate terms
GREEN_WORDS = {"HAPI", "enhances", "accuracy"}  # Example of improved terms

def highlight_text(text):
    words = text.split(" ")
    highlighted_words = []
    
    for word in words:
        if word in RED_WORDS:
            highlighted_words.append(f'<span class="red">{word}</span>')
        elif word in GREEN_WORDS:
            highlighted_words.append(f'<span class="green">{word}</span>')
        else:
            highlighted_words.append(word)
    
    return " ".join(highlighted_words)

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("generate")
def handle_generate(data):
    prompt = data.get("prompt", "")
    
    # Split text into chunks for real-time streaming
    words_llama = highlight_text(TEXT_LLAMMA7B).split(" ")
    words_hapi = highlight_text(TEXT_HAPI).split(" ")

    generated_llama = ""
    generated_hapi = ""

    time.sleep(0.5)  # Small delay before generation starts

    for i in range(max(len(words_llama), len(words_hapi))):
        eventlet.sleep(0.2)  # Delay for streaming effect

        if i < len(words_llama):
            generated_llama += words_llama[i] + " "
        if i < len(words_hapi):
            generated_hapi += words_hapi[i] + " "

        # Emit updates to frontend
        socketio.emit("update_output", {"llama": generated_llama, "hapi": generated_hapi})

if __name__ == "__main__":
    socketio.run(app, debug=True)
