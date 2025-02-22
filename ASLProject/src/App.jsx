import { useState, useEffect } from "react";
import VideoFeed from "./components/VideoFeed";
import TextDisplay from "./components/TextDisplay";
import Controls from "./components/Controls";
import { connectToRecognitionService, disconnectRecognitionService } from "./services/aslRecognitionService";

export default function App() {
  const [recognizedText, setRecognizedText] = useState("");

  useEffect(() => {
    connectToRecognitionService(setRecognizedText);
    return () => disconnectRecognitionService();
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold text-blue-600 mb-4">ASL Recognition App</h1>
      <VideoFeed />
      <TextDisplay recognizedText={recognizedText} />
      <Controls setRecognizedText={setRecognizedText} />
    </div>
  );
}
