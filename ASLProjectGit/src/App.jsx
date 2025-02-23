import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { Auth0Provider, useAuth0 } from "@auth0/auth0-react";
import VideoFeed from "./components/VideoFeed";
import TextDisplay from "./components/TextDisplay";
import Controls from "./components/Controls";
import { connectToRecognitionService, disconnectRecognitionService } from "./services/aslRecognitionService";
import Learn from "./pages/Learn";
import Contact from "./pages/Contact";
import Navigation from "./components/Navigation";

export default function App() {
  const [recognizedText, setRecognizedText] = useState("");
  const { isAuthenticated } = useAuth0();  // Use the Auth0 hook to check authentication status

  useEffect(() => {
    connectToRecognitionService(setRecognizedText);
    return () => disconnectRecognitionService();
  }, []);

  return (
    <Auth0Provider
      domain="dev-vfqqan6a4x0sdp5k.us.auth0.com"
      clientId="aWhm1zmiQuijDOiG9Tay5g4HsfkWUFP5"
      authorizationParams={{
        redirect_uri: "http://localhost:5174/",
        scope: "openid profile email"
      }}
    >
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
            
            {/* Protect /dashboard route */}
            <Route
              path="/dashboard"
              element={isAuthenticated ? <Dashboard /> : <Navigate to="/" />}
            />
          </Routes>
        </div>
      </Router>
    </Auth0Provider>
  );
}
