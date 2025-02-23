import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useState, useEffect } from "react";
import VideoFeed from "./components/VideoFeed";
import TextDisplay from "./components/TextDisplay";
import Controls from "./components/Controls";
import { connectToRecognitionService, disconnectRecognitionService } from "./services/aslRecognitionService";
import Learn from "./pages/Learn";
import Contact from "./pages/Contact";
import Navigation from "./components/Navigation";

export default function App() {
  const [recognizedText, setRecognizedText] = useState("");

  useEffect(() => {
    connectToRecognitionService(setRecognizedText);
    return () => disconnectRecognitionService();
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Navigation /> {/* Include the navigation bar */}
        <Routes>
          <Route
            path="/"
            element={
              <div className="flex flex-col items-center justify-center p-4">
                <h1 className="text-3xl font-bold text-blue-600 mb-4">ASL Recognition App</h1>
                <VideoFeed />
                <TextDisplay recognizedText={recognizedText} />
                <Controls setRecognizedText={setRecognizedText} />
              </div>
            }
          />
          <Route path="/learn" element={<Learn />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </div>
    </Router>
  );
}
