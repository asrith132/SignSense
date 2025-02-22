import { useEffect, useRef } from "react";

export default function VideoFeed() {
  const videoRef = useRef(null);

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
    <div className="flex justify-center">
      <video ref={videoRef} autoPlay playsInline className="w-full max-w-md rounded-lg shadow-lg" />
    </div>
  );
}
