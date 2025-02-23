console.log("ASLRecognition Component is rendering...");

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import SSlogo from "@/assets/SSlogo.png";
import axios from "axios";

export default function ASLRecognition() {
  const [recognizedText, setRecognizedText] = useState("Waiting for ASL input...");
  const [isStreaming, setIsStreaming] = useState(false);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-6 relative">
      <img src={SSlogo} alt="SignSense Logo" className="fixed top-4 left-4 w-16 h-16 z-50" />
      <h1 className="text-4xl font-bold text-blue-700 mb-6">ASL Recognition for Medical Use</h1>
      <Card className="w-full max-w-3xl p-6 bg-white shadow-xl rounded-2xl">
        <CardContent className="flex flex-col items-center">
          
          {/* Streamed Video */}
          {isStreaming ? (
            <img src="http://localhost:5000/video_feed" alt="ASL Camera Feed" className="w-full max-h-96 rounded-lg shadow-lg border-2 border-gray-300" />
          ) : (
            <div className="w-full max-h-96 rounded-lg shadow-lg border-2 border-gray-300 flex items-center justify-center">
              <p className="text-gray-500">Camera Off</p>
            </div>
          )}

          <div className="mt-4 w-full p-4 bg-gray-100 rounded-lg shadow-md text-center border border-gray-200">
            <h2 className="text-lg font-semibold text-gray-800">Recognized Text:</h2>
            <p className="text-gray-700 mt-2 min-h-[50px] text-xl font-medium">{recognizedText}</p>
          </div>

          <div className="mt-4 flex space-x-4">
            <Button onClick={() => setIsStreaming(!isStreaming)} className="bg-blue-500 text-white hover:bg-blue-600 px-4 py-2 rounded-lg">
              {isStreaming ? "Stop Recognition" : "Start Recognition"}
            </Button>
            <Button onClick={() => setRecognizedText("Waiting for ASL input...")} className="bg-red-500 text-white hover:bg-red-600 px-4 py-2 rounded-lg">
              Clear Text
            </Button>
          </div>
          <p className="text-red-500 text-3xl font-bold">Tailwind is working!</p>
        </CardContent>
      </Card>
    </div>
  );
}
