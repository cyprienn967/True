const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const fs = require('fs');

const app = express(); // Define the app variable here
const PORT = 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Dummy credentials for authentication
const credentials = {
  admin: 'password123',
  user1: 'user1password',
  user2: 'user2password',
};

// Load metrics and API keys dynamically
const loadUserData = () => {
  const data = fs.readFileSync('./metrics.json', 'utf8');
  return JSON.parse(data);
};

const saveUserData = (data) => {
  fs.writeFileSync('./metrics.json', JSON.stringify(data, null, 2));
};

// Route for sign-in
app.post('/api/signin', (req, res) => {
  const { username, password } = req.body;

  if (credentials[username] && credentials[username] === password) {
    const userData = loadUserData();

    if (userData[username]) {
      res.json({ success: true, userData: userData[username] });
    } else {
      res.status(404).json({ success: false, message: 'User not found' });
    }
  } else {
    res.status(401).json({ success: false, message: 'Invalid username or password' });
  }
});

// Route to update API key
app.post('/api/update-api-key', (req, res) => {
  const { username, apiKey } = req.body;

  if (!username || !apiKey) {
    return res.status(400).json({ success: false, message: 'Missing username or API key' });
  }

  const userData = loadUserData();

  if (userData[username]) {
    userData[username].apiKey = apiKey; // Update the API key
    saveUserData(userData); // Save the updated data
    res.json({ success: true, message: 'API key updated successfully', apiKey });
  } else {
    res.status(404).json({ success: false, message: 'User not found' });
  }
});

// Default route for testing
app.get('/', (req, res) => {
  res.send('Server is running. Access the API endpoints!');
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
