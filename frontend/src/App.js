import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './MainPage';
import SignIn from './SignIn';
import Dashboard from './Dashboard'; // Import the new component

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/dashboard" element={<Dashboard />} /> {/* New route */}
      </Routes>
    </Router>
  );
}

export default App;
 

