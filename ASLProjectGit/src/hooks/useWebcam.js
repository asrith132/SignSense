import { useEffect, useRef } from "react";

export function useWebcam() {
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

  return videoRef;
}
