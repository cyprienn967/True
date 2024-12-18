const express = require('express');
const cors = require('cors');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// POST endpoint to receive the API key
app.post('/api/submit-key', (req, res) => {
  const { apiKey } = req.body;
  console.log("Received API key:", apiKey);
  // In the future, you would run benchmarking code here.
  return res.json({ message: 'hello yes it worked' });
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
