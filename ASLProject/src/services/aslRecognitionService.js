let socket;

export function connectToRecognitionService(setRecognizedText) {
  socket = new WebSocket("ws://your-backend-url"); // Replace with your backend URL

  socket.onopen = () => console.log("Connected to ASL recognition service");

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    setRecognizedText(data.recognizedText);
  };

  socket.onerror = (error) => console.error("WebSocket Error:", error);

  socket.onclose = () => console.log("Disconnected from ASL recognition service");
}

export function disconnectRecognitionService() {
  if (socket) {
    socket.close();
  }
}
