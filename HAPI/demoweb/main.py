from flask import Flask, render_template
from flask_socketio import SocketIO
import eventlet
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")

TEXT_LLAMMA7B = "This is the output from Llama7b without modifications."
TEXT_HAPI = "This is the output using Llama7b with HAPI, which significantly enhances accuracy."
TEXT_FLAGGED = "Flagged tokens: <span class='red'>[inaccurate]</span>, <span class='red'>[biased]</span>, <span class='red'>[hallucination]</span>"

RED_WORDS = {"modifications", "without"}
GREEN_WORDS = {"HAPI", "enhances", "accuracy"}

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

    words_llama = highlight_text(TEXT_LLAMMA7B).split(" ")
    words_hapi = highlight_text(TEXT_HAPI).split(" ")
    words_flagged = TEXT_FLAGGED.split(" ")

    generated_llama = ""
    generated_hapi = ""
    generated_flagged = ""

    time.sleep(0.5)

    for i in range(max(len(words_llama), len(words_hapi), len(words_flagged))):
        eventlet.sleep(0.2)

        if i < len(words_llama):
            generated_llama += words_llama[i] + " "
        if i < len(words_hapi):
            generated_hapi += words_hapi[i] + " "
        if i < len(words_flagged):
            generated_flagged += words_flagged[i] + " "

        socketio.emit("update_output", {
            "llama": generated_llama,
            "hapi": generated_hapi,
            "flagged": generated_flagged
        })

if __name__ == "__main__":
    socketio.run(app, debug=True)
