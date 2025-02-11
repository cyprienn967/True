document.addEventListener("DOMContentLoaded", async function () {
    // Run the splash sequence first
    await runSplashSequence();
  
    // Now show the original UI (exactly as provided)
    document.getElementById("main-ui").style.display = "block";
    document.getElementById("input-ui").style.display = "block";
  
    var socket = io();
    var isFlaggedView = false; // Toggle state
  
    document.getElementById("prompt").addEventListener("keypress", function (event) {
      if (event.key === "Enter") {
        let prompt = this.value.trim();
        if (!prompt) return;
        this.value = ""; // Clear input field
  
        document.getElementById("llama-output").innerHTML = "";
        document.getElementById("hapi-output").innerHTML = "";
        document.getElementById("flagged-tokens").innerHTML = "";
  
        setTimeout(() => {
          socket.emit("generate", { prompt: prompt });
        }, 500); // Small delay
      }
    });
  
    socket.on("update_output", function (data) {
      document.getElementById("llama-output").innerHTML = data.llama;
      document.getElementById("hapi-output").innerHTML = data.hapi;
      document.getElementById("flagged-tokens").innerHTML = data.flagged; // New flagged tokens output
    });
  
    // Toggle between HAPI output and flagged tokens (using the original element ID)
    document.getElementById("toggle-btn").addEventListener("click", function () {
      isFlaggedView = !isFlaggedView;
  
      if (isFlaggedView) {
        document.getElementById("hapi-output").style.display = "none";
        document.getElementById("flagged-tokens").style.display = "block";
      } else {
        document.getElementById("hapi-output").style.display = "block";
        document.getElementById("flagged-tokens").style.display = "none";
      }
    });
  });
  
  // Function to animate the splash sequence
  async function runSplashSequence() {
    const splashScreen = document.getElementById("splash-screen");
    const splashText = document.getElementById("splash-text");
    const loadingScreen = document.getElementById("loading-screen");
  
    // Multiline code snippet to mimic a terminal
    const codeSnippet = `import demoweb
  client = demoweb.Client(api_key='YOUR_API_KEY')
  response = client.generate(prompt='Hello, World!')
  print(response)`;
  
    const lines = codeSnippet.split("\n");
    splashText.innerHTML = "";
  
    // For each line, create a div and animate letter by letter.
    for (let li = 0; li < lines.length; li++) {
      let lineText = lines[li];
      let lineDiv = document.createElement("div");
      lineDiv.className = "splash-line";
      splashText.appendChild(lineDiv);
      for (let ci = 0; ci < lineText.length; ci++) {
        lineDiv.innerHTML += lineText[ci];
        await new Promise(resolve => setTimeout(resolve, 50)); // Delay per character
      }
      // Slight pause before the next line begins
      await new Promise(resolve => setTimeout(resolve, 200));
    }
  
    // Wait for the Enter key (without displaying any prompt)
    await new Promise(resolve => {
      function onKeyDown(event) {
        if (event.key === "Enter") {
          document.removeEventListener("keydown", onKeyDown);
          resolve();
        }
      }
      document.addEventListener("keydown", onKeyDown);
    });
  
    // Fade out the splash screen
    splashScreen.style.transition = "opacity 0.5s";
    splashScreen.style.opacity = 0;
    await new Promise(resolve => setTimeout(resolve, 500));
    splashScreen.style.display = "none";
  
    // Show the loading screen with spinner for one second
    loadingScreen.style.display = "flex";
    await new Promise(resolve => setTimeout(resolve, 1000));
    loadingScreen.style.display = "none";
  }
  