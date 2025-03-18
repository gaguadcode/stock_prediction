import React, { useState } from "react";
import axios from "axios";

function App() {
  const [userInput, setUserInput] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const runWorkflow = async () => {
    setLoading(true);
    setResponse(null);
    
    try {
      const res = await axios.post("http://127.0.0.1:8000/run-workflow", {
        user_input: userInput
      });
      setResponse(res.data);
    } catch (error) {
      console.error("Error:", error);
      setResponse({ error: "Failed to connect to backend" });
    }
    
    setLoading(false);
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>📊 AI Workflow Runner</h1>

      <textarea
        rows="3"
        cols="50"
        placeholder="Enter your query..."
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
      />
      <br />
      <button onClick={runWorkflow} disabled={loading}>
        {loading ? "Processing..." : "Run Workflow"}
      </button>

      {response && (
        <div style={{ marginTop: "20px", textAlign: "left", maxWidth: "600px", margin: "auto" }}>
          <h3>Workflow Output:</h3>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
