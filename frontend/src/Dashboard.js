import React, { useEffect, useState } from 'react';

function Dashboard() {
  const [userData, setUserData] = useState(null);
  const [apiKey, setApiKey] = useState('');
  const [message, setMessage] = useState('');
  const [username, setUsername] = useState('');

  useEffect(() => {
    const storedUserData = localStorage.getItem('userData');
    if (storedUserData) {
      const parsedData = JSON.parse(storedUserData);
      setUserData(parsedData);
      setApiKey(parsedData.apiKey || ''); // Set the initial API key
      setUsername(parsedData.username); // Extract and store the username
    }
  }, []);

  const handleUpdateApiKey = async () => {
    if (!userData) return;

    try {
      const response = await fetch('http://localhost:5000/api/update-api-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, apiKey }), // Use the username from localStorage
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('API key updated successfully!');
        setUserData({ ...userData, apiKey: data.apiKey });
      } else {
        setMessage(data.message || 'Failed to update API key.');
      }
    } catch (error) {
      setMessage('An error occurred. Please try again.');
    }
  };

  if (!userData) {
    return <p>Loading...</p>;
  }

  return (
    <div style={{ textAlign: 'center', padding: '50px' }}>
      <h1>Welcome to {userData.company}'s Dashboard</h1>
      <h2>Metrics</h2>
      <ul>
        {userData.metrics && userData.metrics.length > 0 ? (
          userData.metrics.map((metric, index) => (
            <li key={index}>Metric {index + 1}: {metric}</li>
          ))
        ) : (
          <p>No metrics available.</p>
        )}
      </ul>

      <h2>API Key</h2>
      <input
        type="text"
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
        placeholder="Enter your API key"
        style={{ padding: '10px', width: '300px', marginBottom: '10px' }}
      />
      <br />
      <button
        onClick={handleUpdateApiKey}
        style={{ padding: '10px 20px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '5px' }}
      >
        Update API Key
      </button>
      <p>{message}</p>
      <h3>Stored API Key: {userData.apiKey || 'No API key saved yet.'}</h3>
    </div>
  );
}

export default Dashboard;
