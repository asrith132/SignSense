import { Link } from "react-router-dom";
import SSlogo from "../assets/SSlogo.png";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
      
      {/* Logo */}
      <img 
        src={SSlogo} 
        alt="SignSense Logo" 
        className="w-40 h-auto mb-6 opacity-90 hover:scale-110 transition-transform duration-300"
      />

      {/* Title */}
      <h1 className="text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-green-400 drop-shadow-lg">
        Welcome to SignSense
      </h1>

      {/* Navigation Button */}
      <Link 
        to="/translation" 
        className="mt-8 bg-blue-500 text-white px-6 py-3 text-lg rounded-lg hover:bg-blue-600 hover:shadow-blue-500 transition-shadow"
      >
        Start Translating
      </Link>
    </div>
  );
}
