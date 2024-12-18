import React, { useState } from 'react';

function App() {
  const [apiKey, setApiKey] = useState('');
  const [responseMessage, setResponseMessage] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await fetch('http://localhost:5000/api/submit-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ apiKey: apiKey })
      });
      
      const data = await response.json();
      setResponseMessage(data.message);
    } catch (error) {
      console.error('Error:', error);
      setResponseMessage('There was an error');
    }
  };

  return (
    <div style={{ margin: '50px' }}>
      <h1>Enter Your LLM API Key</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Paste your API key here"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          style={{ width: '300px', padding: '5px' }}
        />
        <button type="submit" style={{ marginLeft: '10px' }}>
          Submit
        </button>
      </form>
      {responseMessage && (
        <div style={{ marginTop: '20px', color: 'green' }}>
          <strong>{responseMessage}</strong>
        </div>
      )}
    </div>
  );
}

export default App;

