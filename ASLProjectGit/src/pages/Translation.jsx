import { useState } from "react";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import SSlogo from "../assets/SSlogo.png";

export default function Translation() {
  const [recognizedText, setRecognizedText] = useState("Waiting for ASL input...");
  const [isStreaming, setIsStreaming] = useState(true); // Default to streaming

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
      
      {/* Logo */}
      <img 
        src={SSlogo} 
        alt="SignSense Logo" 
        className="fixed top-6 left-6 w-20 h-20 z-50 opacity-90 hover:scale-110 transition-transform duration-300" 
      />

      {/* Page Title */}
      <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-green-400 mb-8 drop-shadow-lg">
        ASL Translation
      </h1>

      {/* Card Container */}
      <Card className="w-full max-w-4xl p-8 bg-gray-800 shadow-2xl rounded-3xl border border-gray-700">
        <CardContent className="flex flex-col items-center space-y-6">

          {/* Camera Feed */}
          {isStreaming ? (
            <div className="w-full max-w-3xl">
              <img 
                src="http://localhost:5000/video_feed" 
                alt="ASL Camera Feed" 
                className="w-full h-[450px] object-cover rounded-xl shadow-lg border-4 border-blue-500 hover:shadow-blue-500 transition-all duration-300"
              />
            </div>
          ) : (
            <div className="w-full h-[450px] bg-gray-700 flex items-center justify-center rounded-xl border-4 border-gray-600">
              <p className="text-gray-400 text-lg font-medium">Camera Off</p>
            </div>
          )}

          {/* Recognized Text Display */}
          <div className="w-full p-6 bg-gray-900 rounded-xl shadow-md text-center border border-gray-700">
            <h2 className="text-xl font-semibold text-gray-300">Recognized Text:</h2>
            <p className="text-blue-400 mt-2 min-h-[60px] text-3xl font-bold">
              {recognizedText}
            </p>
          </div>

          {/* Action Buttons */}
          <div className="mt-6 flex space-x-6">
            <Button 
              onClick={() => setRecognizedText(prevText => prevText.split(" ").slice(0, -1).join(" "))} 
              className="px-6 py-3 rounded-xl font-semibold text-lg transition-all bg-yellow-500 hover:bg-yellow-400 hover:shadow-yellow-400 shadow-md"
            >
              Delete Word
            </Button>
            
            <Button 
              onClick={() => setRecognizedText("Waiting for ASL input...")} 
              className="px-6 py-3 rounded-xl font-semibold text-lg transition-all bg-red-500 hover:bg-red-400 hover:shadow-red-400 shadow-md"
            >
              Clear Text
            </Button>

            <Button 
              onClick={() => alert("ASL input completed!")} 
              className="px-6 py-3 rounded-xl font-semibold text-lg transition-all bg-green-500 hover:bg-green-400 hover:shadow-green-400 shadow-md"
            >
              Done
            </Button>
          </div>

          {/* Toggle Camera */}
          <Button 
            onClick={() => setIsStreaming(!isStreaming)} 
            className="mt-6 px-6 py-3 rounded-xl font-semibold text-lg transition-all bg-purple-500 hover:bg-purple-400 hover:shadow-purple-400 shadow-md"
          >
            {isStreaming ? "Turn Off Camera" : "Turn On Camera"}
          </Button>

        </CardContent>
      </Card>
    </div>
  );
}