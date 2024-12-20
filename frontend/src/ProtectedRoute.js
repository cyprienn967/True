import React from 'react';
import { Navigate } from 'react-router-dom';

function ProtectedRoute({ children }) {
  const userData = localStorage.getItem('userData');
  return userData ? children : <Navigate to="/signin" />;
}

export default ProtectedRoute;
