import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './MainPage';
import SignIn from './SignIn';

function App() {
  return (
    <Router>
      <Routes>
        {/* Main page route */}
        <Route path="/" element={<MainPage />} />

        {/* Sign-in page route */}
        <Route path="/signin" element={<SignIn />} />
      </Routes>
    </Router>
  );
}

export default App;
 
