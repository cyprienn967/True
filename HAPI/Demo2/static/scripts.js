document.addEventListener("DOMContentLoaded", function () {
    var socket = io();

    document.getElementById("prompt").addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            let prompt = this.value;
            this.value = ""; // Clear input

            document.getElementById("llama-output").innerHTML = "";
            document.getElementById("hapi-output").innerHTML = "";

            setTimeout(() => {
                socket.emit("generate", { prompt: prompt });
            }, 500); // Delay before sending the request
        }
    });

    socket.on("update_output", function (data) {
        document.getElementById("llama-output").innerHTML = data.llama;
        document.getElementById("hapi-output").innerHTML = data.hapi;
    });
});
