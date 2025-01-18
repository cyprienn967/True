const express = require("express");
const sqlite3 = require("better-sqlite3");
const bodyParser = require("body-parser");
const path = require("path");

const app = express();
app.use(bodyParser.json());

const cors = require("cors");
app.use(cors());


// Set up SQLite database
const dbPath = path.resolve(__dirname, "user_data.db");
const db = new sqlite3(dbPath);

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    api_key TEXT
  )
`);

// Endpoint to fetch the API key for a user
app.get("/api/get_key", (req, res) => {
  const { username } = req.query;

  if (!username) {
    return res.status(400).json({ error: "Username is required" });
  }

  try {
    const result = db.prepare("SELECT api_key FROM users WHERE username = ?").get(username);
    res.json({ api_key: result ? result.api_key : null });
  } catch (error) {
    console.error("Error fetching API key:", error);
    res.status(500).json({ error: "Failed to fetch API key" });
  }
});

// Endpoint to store an API key for a user
app.post("/api/store_key", (req, res) => {
  const { username, api_key } = req.body;

  if (!username || !api_key) {
    return res.status(400).json({ error: "Username and API key are required" });
  }

  try {
    db.prepare("INSERT OR REPLACE INTO users (username, api_key) VALUES (?, ?)").run(username, api_key);
    res.json({ message: "API key stored successfully" });
  } catch (error) {
    console.error("Error storing API key:", error);
    res.status(500).json({ error: "Failed to store API key" });
  }
});

// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on http://127.0.0.1:${PORT}`);
});
