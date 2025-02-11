document.addEventListener("DOMContentLoaded", async function () {
    // Run the splash sequence first
    await runSplashSequence();
  
    // Now show the original UI (EXACTLY as provided)
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
  
  // Function to animate the splash sequence with manually colored code
  async function runSplashSequence() {
    const splashScreen = document.getElementById("splash-screen");
    const splashText = document.getElementById("splash-text");
    const loadingScreen = document.getElementById("loading-screen");
  
    // Plain text version of your code snippet (with blank lines between each code line)
    const plainLines = [
      "from HAPI import hapi",
      "",
      "api = hapi(api_key=\"your_hapi_key\")",
      "",
      "response = api.generate(model, prompt)",
      "",
      "print(response.text)"
    ];
  
    // Manually colored HTML version for each line:
    const highlightedLines = [
      // Line 1
      '<span style="color:#FFFF00;">from</span> <span style="color:#ADD8E6;">HAPI</span> <span style="color:#FFFF00;">import</span> <span style="color:#ADD8E6;">hapi</span>',
      // Blank line
      '',
      // Line 2
      '<span style="color:#ADD8E6;">api </span><span style="color:#00FF00;">=</span><span style="color:#ADD8E6;"> hapi(api_key</span><span style="color:#00FF00;">=</span><span style="color:#FFC0CB;">"your_hapi_key"</span><span style="color:#ADD8E6;">)</span>',
      // Blank line
      '',
      // Line 3
      '<span style="color:#ADD8E6;">response </span><span style="color:#00FF00;">=</span><span style="color:#ADD8E6;"> api.</span><span style="color:#FFC0CB;">generate</span><span style="color:#ADD8E6;">(model, prompt)</span>',
      // Blank line
      '',
      // Line 4
      '<span style="color:#00FF00;">print</span><span style="color:#ADD8E6;">(response.text)</span>'
    ];
  
    splashText.innerHTML = "";
  
    // For each line, create a fixed-height div and animate letter by letter.
    for (let li = 0; li < plainLines.length; li++) {
      let lineText = plainLines[li];
      let lineDiv = document.createElement("div");
      lineDiv.className = "splash-line";
      // Start with empty text for smooth typing.
      lineDiv.textContent = "";
      splashText.appendChild(lineDiv);
      let currentText = "";
      for (let ci = 0; ci < lineText.length; ci++) {
        currentText += lineText[ci];
        lineDiv.textContent = currentText;
        await new Promise(resolve => setTimeout(resolve, 50)); // Delay per character
      }
      // Once the line is complete, replace it with the manually colored HTML.
      lineDiv.innerHTML = highlightedLines[li];
      await new Promise(resolve => setTimeout(resolve, 200)); // Pause before next line
    }
  
    // Wait for the Enter key (with no visible prompt)
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
  