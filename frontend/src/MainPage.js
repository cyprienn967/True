// This file contains the main page of the application. It displays the header, main section, and a form to submit an API key to detect bias in AI models. The form is submitted to the backend server, which then sends a request to the OpenAI API to analyze the model's responses for bias. The results are displayed on the page once the analysis is complete.    
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function MainPage() {
    const [apiKey, setApiKey] = useState('');
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);
  
    const handleSubmit = async (event) => {
      event.preventDefault();
      setLoading(true);
      setResponse(null);
  
      try {
        const res = await fetch('http://localhost:5000/api/detect-bias', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ apiKey }),
        });
        const data = await res.json();
        setResponse(data);
      } catch (error) {
        console.error('Error:', error);
        setResponse({ error: 'There was an issue with the request.' });
      } finally {
        setLoading(false);
      }
    };
  
    const navigate = useNavigate();
  
    return (

      <div style={{ fontFamily: 'Arial, sans-serif' }}>
        {/* Header Section */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '20px 40px',
            borderBottom: '1px solid #ddd',
          }}
        >
          {/* Logo */}
          <div style={{ fontSize: '24px', fontWeight: 'bold' }}>True</div>
  
          {/* Navigation */}
          <div style={{ display: 'flex', gap: '20px', fontSize: '18px' }}>
            <a href="#pricing" style={{ textDecoration: 'none', color: 'black' }}>Pricing</a>
            <a href="#features" style={{ textDecoration: 'none', color: 'black' }}>Features</a>
          </div>
  
          {/* Get Started Button */}
          <button
            onClick={() => navigate('/SignIn')}
            style={{
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              padding: '10px 20px',
              cursor: 'pointer',
              fontSize: '16px',
            }}
          >
            Sign in 
          </button>
        </div>
  
        {/* Main Section */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            height: '80vh',
            textAlign: 'center',
            padding: '0 20px',
          }}
        >
          {/* Main Text */}
          <h1 style={{ fontSize: '48px', fontWeight: 'bold', marginBottom: '20px' }}>
            The future of AI safety is here
          </h1>
          <p style={{ fontSize: '20px', color: 'grey', marginBottom: '40px' }}>
            Deploy, test, and build safe AI faster with True
          </p>
  
          {/* Buttons */}
          <div style={{ display: 'flex', gap: '20px', marginBottom: '30px' }}>
            <button
              style={{
                backgroundColor: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                padding: '10px 20px',
                cursor: 'pointer',
                fontSize: '16px',
              }}
            >
              Book a Demo
            </button>
  
            <button
              style={{
                backgroundColor: 'white',
                color: '#007bff',
                border: '2px solid #007bff',
                borderRadius: '5px',
                padding: '10px 20px',
                cursor: 'pointer',
                fontSize: '16px',
              }}
            >
              Explore Features
            </button>
          </div>
  
          {/* Textbox */}
          <form onSubmit={handleSubmit} style={{ width: '300px' }}>
            <input
              type="text"
              placeholder="Paste LLM key here"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '5px',
                border: '1px solid #ccc',
                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
                fontSize: '16px',
              }}
            />
            <button
              type="submit"
              style={{
                display: 'none', // Hide submit button, user presses Enter to submit
              }}
            >
              Submit
            </button>
          </form>
  
          {loading && <p>Detecting bias, please wait...</p>}
  
          {response && response.analysis && (
            <div style={{ marginTop: '20px' }}>
              <h2>Analysis Results</h2>
              <p>Total Prompts: {response.analysis.total_prompts}</p>
              <p>Bias Indicators: {response.analysis.bias_indicators}</p>
              <p>Bias Percentage: {response.analysis.bias_percentage.toFixed(2)}%</p>
  
              <h3>Responses</h3>
              <ul>
                {response.results.map((result, index) => (
                  <li key={index}>
                    <strong>Prompt:</strong> {result.prompt}
                    <br />
                    <strong>Response:</strong> {result.response}
                  </li>
                ))}
              </ul>
            </div>
          )}
  
          {response && response.error && (
            <p style={{ color: 'red' }}>{response.error}</p>
          )}
        </div>
      </div>
    );
  }

export default MainPage;
// import React from 'react';
// import { useNavigate } from 'react-router-dom';

// function MainPage() {
//   const navigate = useNavigate();

//   return (
//     <div style={{ textAlign: 'center', padding: '50px' }}>
//       <h1>Welcome to True</h1>
//       <p>Your gateway to AI safety.</p>
      // <button
      //   onClick={() => navigate('/signin')}
      //   style={{
      //     backgroundColor: '#007bff',
      //     color: 'white',
      //     padding: '10px 20px',
      //     border: 'none',
      //     borderRadius: '5px',
      //     cursor: 'pointer',
      //   }}
      // >
      //   Sign In
      // </button>
//     </div>
//   );
// }

// export default MainPage;
