document.addEventListener("DOMContentLoaded", function () {
    var socket = io();
    var toggleSwitch = document.getElementById("toggle-switch");

    document.getElementById("prompt").addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            let prompt = this.value.trim();
            if (!prompt) return;
            this.value = "";

            document.getElementById("llama-output").innerHTML = "";
            document.getElementById("hapi-output").innerHTML = "";
            document.getElementById("flagged-tokens").innerHTML = "";

            setTimeout(() => {
                socket.emit("generate", { prompt: prompt });
            }, 500);
        }
    });

    socket.on("update_output", function (data) {
        document.getElementById("llama-output").innerHTML = data.llama;
        document.getElementById("hapi-output").innerHTML = data.hapi;
        document.getElementById("flagged-tokens").innerHTML = data.flagged;
    });

    toggleSwitch.addEventListener("change", function () {
        let isFlaggedView = toggleSwitch.checked;
        document.getElementById("hapi-output").style.display = isFlaggedView ? "none" : "block";
        document.getElementById("flagged-tokens").style.display = isFlaggedView ? "block" : "none";
    });
});
