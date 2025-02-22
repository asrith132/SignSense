import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function ASLRecognition() {
  const videoRef = useRef(null);
  const [recognizedText, setRecognizedText] = useState("");

  useEffect(() => {
    async function startVideo() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    }
    startVideo();
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-6">
      <h1 className="text-4xl font-bold text-blue-700 mb-6">ASL Recognition for Medical Use</h1>
      <Card className="w-full max-w-3xl p-6 bg-white shadow-xl rounded-2xl">
        <CardContent className="flex flex-col items-center">
          <video ref={videoRef} autoPlay playsInline className="w-full max-h-96 rounded-lg shadow-lg border-2 border-gray-300" />
          <div className="mt-4 w-full p-4 bg-gray-100 rounded-lg shadow-md text-center border border-gray-200">
            <h2 className="text-lg font-semibold text-gray-800">Recognized Text:</h2>
            <p className="text-gray-700 mt-2 min-h-[50px] text-xl font-medium">{recognizedText || "Waiting for ASL input..."}</p>
          </div>
          <Button className="mt-6 bg-red-500 text-white hover:bg-red-600 px-4 py-2 rounded-lg" onClick={() => setRecognizedText("")}>Clear Text</Button>
        </CardContent>
      </Card>
    </div>
  );
}
