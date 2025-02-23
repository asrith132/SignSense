import { useState } from "react";

const videoData = [
  { src: "/videos/accident1.mp4", answer: "accident" },
  { src: "/videos/accident2.mp4", answer: "accident" },
  { src: "/videos/accident3.mp4", answer: "accident" },
  { src: "/videos/accident4.mp4", answer: "accident" },
  { src: "/videos/accident5.mp4", answer: "accident" },
  { src: "/videos/accident6.mp4", answer: "accident" },
  { src: "/videos/accident7.mp4", answer: "accident" },
  { src: "/videos/accident8.mp4", answer: "accident" },
  { src: "/videos/accident9.mp4", answer: "accident" },
  { src: "/videos/accident10.mp4", answer: "accident" },
  { src: "/videos/accident11.mp4", answer: "accident" },
  { src: "/videos/accident12.mp4", answer: "accident" },
  { src: "/videos/accident13.mp4", answer: "accident" },
  { src: "/videos/accident14.mp4", answer: "accident" },
  { src: "/videos/accident15.mp4", answer: "accident" },
  { src: "/videos/accident16.mp4", answer: "accident" },
  { src: "/videos/accident17.mp4", answer: "accident" },
  { src: "/videos/accident18.mp4", answer: "accident" },
  { src: "/videos/accident19.mp4", answer: "accident" },
  { src: "/videos/accident20.mp4", answer: "accident" },
  { src: "/videos/accident21.mp4", answer: "accident" },
  { src: "/videos/accident22.mp4", answer: "accident" },
  { src: "/videos/accident23.mp4", answer: "accident" },
  { src: "/videos/accident24.mp4", answer: "accident" },
  { src: "/videos/accident25.mp4", answer: "accident" },
  { src: "/videos/accident26.mp4", answer: "accident" },
  { src: "/videos/accident27.mp4", answer: "accident" },
  { src: "/videos/accident28.mp4", answer: "accident" },
  { src: "/videos/accident29.mp4", answer: "accident" },
  { src: "/videos/accident30.mp4", answer: "accident" },
  { src: "/videos/accident31.mp4", answer: "accident" },
  { src: "/videos/accident32.mp4", answer: "accident" },
  { src: "/videos/accident33.mp4", answer: "accident" },
  { src: "/videos/accident34.mp4", answer: "accident" },
  { src: "/videos/accident35.mp4", answer: "accident" },
  { src: "/videos/accident36.mp4", answer: "accident" },
  { src: "/videos/accident37.mp4", answer: "accident" },
  { src: "/videos/accident38.mp4", answer: "accident" },
  { src: "/videos/accident39.mp4", answer: "accident" },
  ...Array.from({ length: 14 }, (_, i) => ({
    src: `/videos/bath${i + 1}.mp4`,
    answer: "bath"
  })),
  ...Array.from({ length: 11 }, (_, i) => ({
    src: `/videos/breathe${i + 1}.mp4`,
    answer: "breathe"
  })),
  ...Array.from({ length: 9 }, (_, i) => ({
    src: `/videos/chest${i + 1}.mp4`,
    answer: "chest"
  })),
  ...Array.from({ length: 33 }, (_, i) => ({
    src: `/videos/cold${i + 1}.mp4`,
    answer: "cold"
  })),
  ...Array.from({ length: 21 }, (_, i) => ({
    src: `/videos/hospital${i + 1}.mp4`,
    answer: "hospital"
  })),
  ...Array.from({ length: 38 }, (_, i) => ({
    src: `/videos/hot${i}.mp4`,
    answer: "hot"
  })),
  ...Array.from({ length: 24 }, (_, i) => ({
    src: `/videos/hurt${i + 1}.mp4`,
    answer: "hurt"
  })),
  ...Array.from({ length: 11 }, (_, i) => ({
    src: `/videos/insurance${i + 1}.mp4`,
    answer: "insurance"
  })),
  ...Array.from({ length: 91 }, (_, i) => ({
    src: `/videos/none${i + 1}.mp4`,
    answer: "none"
  })),
  ...Array.from({ length: 57 }, (_, i) => ({
    src: `/videos/proceed${i + 1}.mp4`,
    answer: "proceed"
  })),
  ...Array.from({ length: 36 }, (_, i) => ({
    src: `/videos/refuse${i + 1}.mp4`,
    answer: "refuse"
  })),
  ...Array.from({ length: 45 }, (_, i) => ({
    src: `/videos/start${i + 1}.mp4`,
    answer: "start"
  })),

];


export default function Learn() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userInput, setUserInput] = useState("");
  const [streak, setStreak] = useState(0);
  const [message, setMessage] = useState("");

  const currentVideo = videoData[currentIndex];

  const handleSubmit = () => {
    if (userInput.trim().toLowerCase() === currentVideo.answer.toLowerCase()) {
      setStreak(streak + 1);
      setMessage("✅ Correct! Keep going!");
    } else {
      setStreak(0);
      setMessage(`❌ Incorrect! The correct answer was: ${currentVideo.answer}. Try again.`);
    }
  
    // Move to the next random video
    const nextIndex = Math.floor(Math.random() * videoData.length);
    setCurrentIndex(nextIndex);
    setUserInput("");
  };  

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-white p-8">
      <h1 className="text-4xl font-bold text-green-600 mb-6">Learn ASL</h1>
      <p className="text-xl mb-6">Welcome to the learning section of the ASL recognition website!</p>

      <div className="mt-8 flex flex-col items-center w-full">
        {/* Larger video player */}
        <video 
          key={currentVideo.src} 
          src={currentVideo.src} 
          controls 
          autoPlay 
          className="w-3/4 md:w-1/2 lg:w-2/3 xl:w-1/2 rounded-lg shadow-md"
        />

        {/* Larger input */}
        <input
          type="text"
          placeholder="Enter your guess..."
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          className="mt-6 p-3 border border-gray-300 rounded-md text-center text-xl w-4/5 md:w-2/3 lg:w-1/2"
        />

        {/* Larger submit button */}
        <button
          onClick={handleSubmit}
          className="mt-6 bg-green-500 text-white px-8 py-4 text-lg rounded-md hover:bg-green-600 transition"
        >
          Submit
        </button>

        {/* Message and streak with larger text */}
        <p className="mt-4 text-2xl">{message}</p>
        <p className="mt-4 font-bold text-xl text-gray-700">🔥 Streak: {streak}</p>
      </div>
    </div>
  );
}
