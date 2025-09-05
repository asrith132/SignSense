import { useEffect, useRef } from "react";

export default function VideoFeed() {
  const videoRef = useRef(null);

  useEffect(() => {
    async function startVideo() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          console.log("Webcam feed started.");
        }
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    }
    startVideo();
  }, []);

  return (
    <div className="flex justify-center w-full p-4">
      <div className="border-4 border-gray-500 rounded-lg p-2 shadow-lg">
        <video ref={videoRef} autoPlay playsInline className="w-full max-h-[500px] rounded-lg" />
      </div>
    </div>
  );
}