"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";

const categories = [
  { id: "profile", name: "My Profile" },
  { id: "bias", name: "Bias and Fairness" },
  { id: "hallucinations", name: "Hallucinations" },
  { id: "jailbreaking", name: "Jailbreaking" },
  { id: "cot", name: "CoT" },
];

export default function Dashboard() {
  const searchParams = useSearchParams();
  const username = searchParams.get("username");

  const [activeCategory, setActiveCategory] = useState("profile");
  const [apiKey, setApiKey] = useState("");
  const [savedKey, setSavedKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch stored API key when the component loads
  useEffect(() => {
    if (username) {
      console.log("Fetching API key for username:", username); // Debugging log
      fetch(`http://127.0.0.1:5000/api/get_key?username=${username}`)
        .then((response) => {
          if (!response.ok) {
            throw new Error("Failed to fetch API key");
          }
          return response.json();
        })
        .then((data) => {
          const key = data.api_key || null;
          console.log("Fetched API key:", key); // Debugging log
          setSavedKey(key);
          setApiKey(key || "");
        })
        .catch((error) => {
          console.error("Error fetching API key:", error);
          setError("Unable to fetch API key. Please try again later.");
        });
    }
  }, [username]);

  const handleSave = () => {
    setError(null); // Reset error state
    if (username) {
      console.log("Saving API key:", apiKey, "for username:", username); // Debugging log
      fetch("http://127.0.0.1:5000/api/store_key", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, api_key: apiKey }),
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error("Failed to save API key");
          }
          return response.json();
        })
        .then((data) => {
          console.log("Response from server:", data); // Debugging log
          if (data.message) {
            setSavedKey(apiKey);
          }
        })
        .catch((error) => {
          console.error("Error saving API key:", error);
          setError("Unable to save API key. Please try again.");
        });
    } else {
      console.error("Username is missing!"); // Debugging log
      setError("No username provided. Unable to save API key.");
    }
  };

  return (
    <div className="w-full min-h-screen flex" style={{ paddingTop: "80px" }}>
      {/* Sidebar */}
      <div className="w-1/4 bg-gray-100 dark:bg-neutral-900 px-4 py-6">
        <h3 className="text-lg font-bold text-black dark:text-white mb-4">
          Dashboard
        </h3>
        <ul className="space-y-4">
          {categories.map((category) => (
            <li key={category.id}>
              <button
                onClick={() => setActiveCategory(category.id)}
                className={`w-full text-left px-4 py-2 rounded-lg transition-colors ${
                  activeCategory === category.id
                    ? "bg-black text-white dark:bg-white dark:text-black"
                    : "text-neutral-700 dark:text-neutral-400 hover:bg-gray-200 dark:hover:bg-neutral-800"
                }`}
              >
                {category.name}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Main Content */}
      <div className="w-3/4 bg-gray-50 dark:bg-neutral-950 p-8">
        {/* My Profile Section */}
        {activeCategory === "profile" && (
          <div>
            <h2 className="text-2xl font-bold text-black dark:text-white mb-6">
              My Profile
            </h2>
            <div className="mt-6 space-y-4">
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
              <button
                onClick={handleSave}
                className="bg-black hover:bg-black/90 text-white text-sm transition font-medium duration-200 rounded-full px-4 py-2 flex items-center justify-center w-full dark:text-black dark:bg-white dark:hover:bg-neutral-100"
              >
                Save API Key
              </button>
              {error && (
                <p className="mt-4 text-red-500 text-sm">
                  {error}
                </p>
              )}
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
        )}

        {/* Bias and Fairness Section */}
        {/* {activeCategory === "bias" && (
          <div>
            <h2 className="text-2xl font-bold text-black dark:text-white mb-6">
              Bias and Fairness
            </h2>
            <p className="text-neutral-700 dark:text-neutral-400 mb-4">
              Below are some visualizations from the bias analysis.
            </p>
            <div className="space-y-6">
              <div>
                <h3 className="font-semibold">Spider Plot</h3>
                <img
                  src="/bias/bias_spider_plot.png"
                  alt="Bias Spider Plot"
                  className="mt-2 border rounded max-w-full"
                />
              </div>
              <div>
                <h3 className="font-semibold">Heatmap</h3>
                <img
                  src="/bias/bias_heatmap.png"
                  alt="Bias Heatmap"
                  className="mt-2 border rounded max-w-full"
                />
              </div>
              <div>
                <h3 className="font-semibold">Toxicity Pie Chart</h3>
                <img
                  src="/bias/toxicity_pie_chart.png"
                  alt="Toxicity Pie Chart"
                  className="mt-2 border rounded max-w-full"
                />
              </div>
            </div>
          </div>
        )} */}
        {/* Bias and Fairness Section */}
        {activeCategory === "bias" && (
  <div>
    <h2 className="text-2xl font-bold text-black dark:text-white mb-6">
      Bias and Fairness
    </h2>
    <p className="text-neutral-700 dark:text-neutral-400 mb-4">
      Below are some visualizations from the bias analysis.
    </p>
    <div className="space-y-6">
      <div>
        <h3 className="font-semibold">Spider Plot</h3>
        <img
          src="/bias/bias_spider_plot.png"
          alt="Bias Spider Plot"
          className="mt-2 border rounded max-w-full w-1/2 transition-transform duration-300 hover:scale-105"
        />
      </div>
      <div>
        <h3 className="font-semibold">Heatmap</h3>
        <img
          src="/bias/bias_heatmap.png"
          alt="Bias Heatmap"
          className="mt-2 border rounded max-w-full w-1/2 transition-transform duration-300 hover:scale-105"
        />
      </div>
      <div>
        <h3 className="font-semibold">Toxicity Pie Chart</h3>
        <img
          src="/bias/toxicity_pie_chart.png"
          alt="Toxicity Pie Chart"
          className="mt-2 border rounded max-w-full w-1/2 transition-transform duration-300 hover:scale-105"
        />
      </div>
    </div>
  </div>
)}


        {/* Other Sections (Hallucinations, Jailbreaking, CoT) */}
        {activeCategory !== "profile" && activeCategory !== "bias" && (
          <div>
            <h2 className="text-2xl font-bold text-black dark:text-white mb-6">
              {categories.find((c) => c.id === activeCategory)?.name}
            </h2>
            <p className="text-neutral-700 dark:text-neutral-400">
              Content for{" "}
              {categories.find((c) => c.id === activeCategory)?.name} will go here.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
