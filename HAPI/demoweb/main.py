from flask import Flask, render_template
from flask_socketio import SocketIO
import eventlet
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")

# WORDS TO BE HIGHLIGHTED
RED_WORDS = {"replace", "treatment", "without", "cure for HIV", "eliminates the virus", 
             "tested in human trials", "first major battle", "forced Iraq to shift strategy", 
             "battle between Saudi and Kuwaiti forces", "successfully tested"}

GREEN_WORDS = {"still undergoing clinical trials", "guide RNA (gRNA)", "no definitive cure for HIV", "experimental phases", "undetectable", "key engagement", "highlighted vulnerabilities", "U.S. coalition forces"}

ORANGE_WORDS = {"Cas9 enzyme", "genetic disorders", "clinical trials", "gene-editing"}

def highlight_text(text, mode="llama"):
    """Wraps words in <span> tags to apply red, green, or orange highlighting in CSS.
    
    - `mode="llama"` highlights **red** & **orange** (for Baseline Output).
    - `mode="hapi"` highlights **green** & **orange** (for HAPI Output).
    """
    
    # Apply red highlighting only for Llama7b output
    if mode == "llama":
        for word in RED_WORDS:
            text = text.replace(word, f"<span class='red'>{word}</span>")

    # Apply green highlighting only for HAPI output
    if mode == "hapi":
        for word in GREEN_WORDS:
            text = text.replace(word, f"<span class='green'>{word}</span>")

    # Apply orange highlighting for BOTH outputs
    for word in ORANGE_WORDS:
        text = text.replace(word, f"<span class='orange'>{word}</span>")

    return text

# PREDEFINED RESPONSES WITH HIGHLIGHTED TEXT
responses = {
    "CRISPR": {
        "llama": highlight_text("CRISPR is a revolutionary gene-editing tool that allows scientists to modify DNA with high precision. "
                                "It works by using RNA to direct the Cas9 enzyme to a specific location in the genome, where it makes a cut. "
                                "Scientists can then modify or replace defective genes. Researchers have used CRISPR to treat sickle cell disease, cystic fibrosis, "
                                "and some cancers.", mode="llama"),
        
        "hapi": highlight_text("CRISPR is a gene-editing technique that enables precise modifications in DNA. "
                               "It utilizes a guide RNA (gRNA) to locate a targeted sequence and the Cas9 enzyme to cut at that point. "
                               "This allows for potential correction of genetic disorders like sickle cell disease and hereditary blindness. "
                               "Although research is progressing, CRISPR therapies are still undergoing clinical trials and are not yet widely available for human treatment.", 
                               mode="hapi"),
        
        "flagged": highlight_text("CRISPR, guide RNA, Cas9 enzyme, treat sickle cell disease")
    },
    "HIV": {
        "llama": highlight_text("Yes, in 2025 scientists have developed a functional cure for HIV using gene-editing technology and monoclonal antibodies. "
                                "The treatment, which permanently eliminates the virus from the body, has been successfully tested in human trials and is now being distributed worldwide.", mode="llama"),
        
        "hapi": highlight_text("As of 2025, there is no definitive cure for HIV. However, significant advancements have been made in antiretroviral therapy (ART), "
                               "which allows people with HIV to live long and healthy lives with undetectable viral loads. "
                               "Researchers are exploring promising treatments, such as gene-editing and monoclonal antibodies, but these approaches remain in experimental phases.", 
                               mode="hapi"),
        
        "flagged": highlight_text("cure for HIV, eliminates the virus, tested in human trials")
    },
    "battle": {
        "llama": highlight_text("The Battle of Khafji was the first major battle of the Gulf War, taking place in January 1991. "
                                "It was a battle between Saudi and Kuwaiti forces on one side and Iraqi forces on the other. "
                                "The battle was fought over the city of Khafji, which was captured by Iraq in the early stages of the war. "
                                "It was significant because it marked the first coalition victory, forcing Iraq to shift its war strategy.", mode="llama"),
        
        "hapi": highlight_text("The Battle of Khafji, fought in January 1991, was a key engagement in the Gulf War. "
                               "Iraqi forces launched an offensive, capturing the city, but were repelled by Saudi, Kuwaiti, and U.S. coalition forces after intense combat. "
                               "The battle demonstrated Iraq’s willingness to go on the offensive despite aerial bombardment and served as a test of Saudi Arabia’s military capabilities. "
                               "Though a tactical victory for the coalition, it highlighted vulnerabilities in coordination and preparedness on both sides.", 
                               mode="hapi"),
        
        "flagged": highlight_text("first major battle, forced Iraq to shift strategy, battle between Saudi and Kuwaiti forces")
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("generate")
def handle_generate(data):
    prompt = data.get("prompt", "").lower()

    # Identify the correct response set
    if "crispr" in prompt:
        response_set = responses["CRISPR"]
    elif "hiv" in prompt:
        response_set = responses["HIV"]
    elif "battle" in prompt:
        response_set = responses["battle"]
    else:
        response_set = {"llama": "No relevant information found.", "hapi": "No relevant information found.", "flagged": "None"}

    # Splitting text for streaming effect
    words_llama = response_set["llama"].split(" ")
    words_hapi = response_set["hapi"].split(" ")
    words_flagged = response_set["flagged"].split(" ")

    generated_llama, generated_hapi, generated_flagged = "", "", ""

    time.sleep(0.2)  # Smaller delay before starting

    for i in range(max(len(words_llama), len(words_hapi), len(words_flagged))):
        eventlet.sleep(0.05)  # Smoother and faster streaming

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
