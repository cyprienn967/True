"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";

export default function Dashboard() {
  const searchParams = useSearchParams();
  const username = searchParams.get("username"); // Get the username from the query params

  const [apiKey, setApiKey] = useState(""); // API key from the input field
  const [savedKey, setSavedKey] = useState<string | null>(null); // Saved API key from the backend

  // Fetch stored API key when the component loads
  useEffect(() => {
    if (username) {
      fetch(`http://127.0.0.1:5000/api/get_key?username=${username}`)
        .then((response) => response.json())
        .then((data) => {
          const key = data.api_key || null; // If no API key exists, set to null
          setSavedKey(key); // Update the savedKey state
          setApiKey(key || ""); // Pre-fill the input field with the saved key if it exists
        })
        .catch((error) => console.error("Error fetching API key:", error));
    }
  }, [username]);

  const handleSave = () => {
    if (username) {
      fetch("http://127.0.0.1:5000/api/store_key", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, api_key: apiKey }),
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.message) {
            setSavedKey(apiKey); // Update the displayed saved key
          }
        })
        .catch((error) => console.error("Error saving API key:", error));
    }
  };

  return (
    <div className="w-full min-h-screen flex items-center justify-center">
      <div className="bg-gray-50 dark:bg-neutral-950 w-full max-w-md p-8 rounded-lg shadow-md">
        <div className="text-center">
          <h2 className="text-2xl font-bold leading-9 tracking-tight text-black dark:text-white">
            Dashboard for {username}
          </h2>
        </div>

        <div className="mt-6 space-y-4">
          {/* Input for API Key */}
          <div>
            <label
              htmlFor="apiKey"
              className="block text-sm font-medium leading-6 text-neutral-700 dark:text-neutral-400"
            >
              API Key
            </label>
            <input
              id="apiKey"
              type="text"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key here..."
              className="block w-full bg-white dark:bg-neutral-900 px-4 rounded-md border-0 py-1.5 shadow-input text-black placeholder:text-gray-400 focus:ring-2 focus:ring-neutral-400 focus:outline-none sm:text-sm sm:leading-6 dark:text-white"
            />
          </div>

          {/* Save Button */}
          <button
            onClick={handleSave}
            className="bg-black hover:bg-black/90 text-white text-sm transition font-medium duration-200 rounded-full px-4 py-2 flex items-center justify-center w-full dark:text-black dark:bg-white dark:hover:bg-neutral-100"
          >
            Save API Key
          </button>

          {/* Display Saved API Key */}
          <div className="mt-4">
            <p className="text-sm text-neutral-700 dark:text-neutral-400">
              Saved API Key:
            </p>
            <p className="text-black dark:text-white">
              {savedKey ? savedKey : "Nothing saved yet"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
