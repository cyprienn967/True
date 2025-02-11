document.addEventListener("DOMContentLoaded", async function () {
  // Run the splash sequence first
  await runSplashSequence();

  // Show the loading screen with spinner for one second
  const loadingScreen = document.getElementById("loading-screen");
  loadingScreen.style.display = "flex";
  await new Promise(resolve => setTimeout(resolve, 1000));
  loadingScreen.style.display = "none";

  // Now show the new middle page (instead of going straight to the main UI)
  document.getElementById("middle-page").style.display = "block";
  buildMetricsAccordions(); // Build the expand/collapse sections
  buildHalluAccordions();   // Build the sections for hallucination results

  // Wait for the user to press Enter on the middle page
  await waitForEnterKey();

  // Hide the middle page and show the original UI
  document.getElementById("middle-page").style.display = "none";
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

/**
* Function to animate the splash sequence with manually colored code
* (unchanged from original, except we no longer immediately show main UI at the end)
*/
async function runSplashSequence() {
const splashScreen = document.getElementById("splash-screen");
const splashText = document.getElementById("splash-text");

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
await waitForEnterKey();

// Fade out the splash screen
splashScreen.style.transition = "opacity 0.5s";
splashScreen.style.opacity = 0;
await new Promise(resolve => setTimeout(resolve, 500));
splashScreen.style.display = "none";
}

/**
* Utility to wait for the user to press Enter.
*/
function waitForEnterKey() {
return new Promise(resolve => {
  function onKeyDown(event) {
    if (event.key === "Enter") {
      document.removeEventListener("keydown", onKeyDown);
      resolve();
    }
  }
  document.addEventListener("keydown", onKeyDown);
});
}

/**
* Build the accordion for the hidden states approach metrics (the user-provided data).
*/
function buildMetricsAccordions() {
const metricsData = {
  falcon40b: [
    { method: "Baseline", sentenceAUC: 0.6479, passageAUC: 0.8347 },
    { method: "SelfCheckGPT", sentenceAUC: 0.6846, passageAUC: 0.8121 },
    { method: "SAPLMA", sentenceAUC: 0.5128, passageAUC: 0.7236 },
    { method: "HAPI", sentenceAUC: 0.7834, passageAUC: 0.8738 },
  ],
  llamabase7b: [
    { method: "Baseline", sentenceAUC: 0.6851, passageAUC: 0.8400 },
    { method: "SelfCheckGPT", sentenceAUC: 0.7644, passageAUC: 0.8897 },
    { method: "SAPLMA", sentenceAUC: 0.5777, passageAUC: 0.7823 },
    { method: "HAPI", sentenceAUC: 0.7881, passageAUC: 0.9015 },
  ],
  llamachat7b: [
    { method: "Baseline", sentenceAUC: 0.4931, passageAUC: 0.6988 },
    { method: "SelfCheckGPT", sentenceAUC: 0.6565, passageAUC: 0.7951 },
    { method: "SAPLMA", sentenceAUC: 0.4066, passageAUC: 0.6265 },
    { method: "HAPI", sentenceAUC: 0.6787, passageAUC: 0.8532 },
  ],
  opt7b: [
    { method: "Baseline", sentenceAUC: 0.7263, passageAUC: 0.8851 },
    { method: "SelfCheckGPT", sentenceAUC: 0.8103, passageAUC: 0.9096 },
    { method: "SAPLMA", sentenceAUC: 0.6212, passageAUC: 0.7476 },
    { method: "HAPI", sentenceAUC: 0.8821, passageAUC: 0.9467 },
  ],
  mpt7b: [
    { method: "Baseline", sentenceAUC: 0.7497, passageAUC: 0.8875 },
    { method: "SelfCheckGPT", sentenceAUC: 0.8680, passageAUC: 0.9384 },
    { method: "SAPLMA", sentenceAUC: 0.6987, passageAUC: 0.8294 },
    { method: "HAPI", sentenceAUC: 0.8601, passageAUC: 0.9487 },
  ],
};

const container = document.getElementById("metrics-container");
Object.keys(metricsData).forEach(llmKey => {
  // Create the item
  const item = document.createElement("div");
  item.className = "accordion-item";

  const header = document.createElement("div");
  header.className = "accordion-header";
  header.textContent = llmKey.toUpperCase(); // e.g. "FALCON40B"

  const content = document.createElement("div");
  content.className = "accordion-content";

  // Build table of methods
  const table = document.createElement("table");
  table.style.width = "100%";
  table.style.borderCollapse = "collapse";
  const headerRow = document.createElement("tr");
  ["Method", "Sentence AUC", "Passage AUC"].forEach(h => {
    const th = document.createElement("th");
    th.textContent = h;
    th.style.borderBottom = "1px solid #ccc";
    th.style.textAlign = "left";
    th.style.padding = "5px";
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  metricsData[llmKey].forEach(row => {
    const tr = document.createElement("tr");
    const tdMethod = document.createElement("td");
    tdMethod.style.padding = "5px";
    tdMethod.textContent = row.method;
    tr.appendChild(tdMethod);

    const tdSentence = document.createElement("td");
    tdSentence.style.padding = "5px";
    tdSentence.textContent = row.sentenceAUC;
    tr.appendChild(tdSentence);

    const tdPassage = document.createElement("td");
    tdPassage.style.padding = "5px";
    tdPassage.textContent = row.passageAUC;
    tr.appendChild(tdPassage);

    table.appendChild(tr);
  });

  content.appendChild(table);

  item.appendChild(header);
  item.appendChild(content);
  container.appendChild(item);

  // Toggle on header click
  header.addEventListener("click", () => {
    content.classList.toggle("show");
  });
});
}

/**
* Build the accordion for the run_results_hallu_questions text file,
* with each question in a collapsible item.
*/
function buildHalluAccordions() {
// This is a shortened or representative example of how you might structure the data.
// In practice, you could parse the entire text file and create objects for each question.
// For brevity, here's just an example of how they'd be displayed.
const halluData = [
  {
    questionNumber: 1,
    question: "Where are the 2034 Olympics being held?",
    answerSnippet: "The 2034 Summer Olympics are scheduled to be held in the city of Brisbane, Australia...",
    hallucinated: true
  },
  {
    questionNumber: 2,
    question: "Which city will host the 2034 FIFA World Cup final?",
    answerSnippet: "As of my last knowledge update, the host city for the final of the 2034 FIFA World Cup had not been officially determined...",
    hallucinated: false
  },
  // ... You would continue for all questions in run_results_hallu_questions ...
];

const container = document.getElementById("hallu-container");

halluData.forEach(item => {
  const accordItem = document.createElement("div");
  accordItem.className = "accordion-item";

  const header = document.createElement("div");
  header.className = "accordion-header";
  header.textContent = `Question #${item.questionNumber}`;

  const content = document.createElement("div");
  content.className = "accordion-content";

  const pQuestion = document.createElement("p");
  pQuestion.innerHTML = `<strong>Q:</strong> ${item.question}`;

  const pAnswer = document.createElement("p");
  pAnswer.innerHTML = `<strong>Answer Snippet:</strong> ${item.answerSnippet}`;

  const pHallu = document.createElement("p");
  pHallu.innerHTML = `<strong>Hallucinated?</strong> ${item.hallucinated ? "Yes" : "No"}`;

  content.appendChild(pQuestion);
  content.appendChild(pAnswer);
  content.appendChild(pHallu);

  accordItem.appendChild(header);
  accordItem.appendChild(content);
  container.appendChild(accordItem);

  header.addEventListener("click", () => {
    content.classList.toggle("show");
  });
});
}
