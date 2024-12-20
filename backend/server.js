// const express = require('express');
// const cors = require('cors');

// const app = express();

// // Middleware
// app.use(cors());
// app.use(express.json());

// // POST endpoint to receive the API key
// app.post('/api/submit-key', (req, res) => {
//   const { apiKey } = req.body;
//   console.log("Received API key:", apiKey);
//   // In the future, you would run benchmarking code here.
//   return res.json({ message: 'hello yes it worked' });
// });

// const PORT = 5000;
// app.listen(PORT, () => {
//   console.log(`Server is running on port ${PORT}`);
// });

// Install dependencies: npm install express body-parser cors
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const PORT = 5000;

// Use middleware
app.use(cors());
app.use(bodyParser.json());

// Hardcoded credentials (for demonstration purposes only)
const validCredentials = {
  username: 'admin',
  password: 'password123',
};

// Route for sign-in
app.post('/api/signin', (req, res) => {
  const { username, password } = req.body;

  if (username === validCredentials.username && password === validCredentials.password) {
    res.json({ success: true, message: 'Sign-in successful!' });
  } else {
    res.status(401).json({ success: false, message: 'Invalid username or password' });
  }
});

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
