import React, { useEffect, useState } from 'react';

function Dashboard() {
  const [userData, setUserData] = useState(null);

  useEffect(() => {
    // Retrieve user data from localStorage
    const storedUserData = localStorage.getItem('userData');
    if (storedUserData) {
      setUserData(JSON.parse(storedUserData));
    }
  }, []);

  if (!userData) {
    return <p>Loading...</p>;
  }

  return (
    <div style={{ textAlign: 'center', padding: '50px' }}>
      <h1>Welcome to {userData.company}'s Dashboard</h1>
      <h2>Metrics</h2>
      <ul>
        {userData.metrics.map((metric, index) => (
          <li key={index}>Metric {index + 1}: {metric}</li>
        ))}
      </ul>
    </div>
  );
}

export default Dashboard;
