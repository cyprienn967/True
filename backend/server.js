const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const fs = require('fs');

const app = express();
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

// Route for sign-in
app.post('/api/signin', (req, res) => {
  const { username, password } = req.body;

  // Validate credentials
  if (credentials[username] && credentials[username] === password) {
    // Read metrics dynamically from the JSON file
    fs.readFile('./metrics.json', 'utf8', (err, data) => {
      if (err) {
        console.error('Error reading metrics file:', err);
        return res.status(500).json({ success: false, message: 'Server error' });
      }

      const metrics = JSON.parse(data);

      if (metrics[username]) {
        res.json({ success: true, userData: metrics[username] });
      } else {
        res.status(404).json({ success: false, message: 'User metrics not found' });
      }
    });
  } else {
    res.status(401).json({ success: false, message: 'Invalid username or password' });
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
