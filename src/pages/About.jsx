export default function About() {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-blue-600 mb-6 drop-shadow-lg">
          About SignSense
        </h1>
  
        <div className="max-w-3xl bg-gray-800 shadow-lg rounded-2xl p-6 border border-gray-700 text-center">
          <p className="text-lg text-gray-300">
            SignSense is an AI-powered **ASL Recognition System** designed to bridge communication gaps
            between **deaf and hearing individuals**. Using advanced **machine learning and computer vision**, 
            SignSense interprets **American Sign Language gestures** in real-time.
          </p>
  
          <div className="mt-6">
            <h2 className="text-xl font-semibold text-blue-400">Core Features</h2>
            <ul className="list-disc mt-3 text-gray-300 text-lg">
              <li>🔹 Real-time **ASL Gesture Recognition**</li>
              <li>🔹 Leaderboards to track **daily streaks**</li>
              <li>🔹 Interactive **learning modules**</li>
              <li>🔹 Live video **gesture tracking**</li>
            </ul>
          </div>
        </div>
  
        <p className="mt-6 text-gray-400">Empowering communication, one sign at a time! ✋🤟</p>
      </div>
    );
  }
  