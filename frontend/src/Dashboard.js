import React, { useEffect, useState } from 'react';

function Dashboard() {
  const [userData, setUserData] = useState(null);
  const [apiKey, setApiKey] = useState('');
  const [message, setMessage] = useState('');
  const [average, setAverage] = useState(null);

  useEffect(() => {
    const storedUserData = localStorage.getItem('userData');
    if (storedUserData) {
      const parsedData = JSON.parse(storedUserData);
      setUserData(parsedData);
      setApiKey(parsedData.apiKey || '');
    }
  }, []);

  const handleRunTest = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/run-test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ apiKey }),
      });

      const data = await response.json();

      if (response.ok) {
        setAverage(data.average);
        setMessage('Test completed successfully!');
      } else {
        setMessage(data.message || 'Failed to run test.');
      }
    } catch (error) {
      setMessage('An error occurred. Please try again.');
    }
  };

  return (
    <div style={{ textAlign: 'center', padding: '50px' }}>
      <h1>Welcome to {userData?.company}'s Dashboard</h1>
      <h2>Metrics</h2>
      <ul>
        {userData?.metrics?.length > 0 ? (
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
        style={{ padding: '10px 20px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '5px' }}
        onClick={handleRunTest}
      >
        Test
      </button>
      <p>{message}</p>
      {average !== null && <h3>Metric: Average: {average.toFixed(2)} words</h3>}
    </div>
  );
}

export default Dashboard;
