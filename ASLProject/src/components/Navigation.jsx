import { Link } from "react-router-dom";
import SSlogo from "../assets/SSlogo.png";


export default function Navigation() {
  return (
    <nav className="fixed top-0 left-0 w-full flex items-center justify-between bg-white shadow-md px-8 py-4 z-50">
      {/* Logo on the Left */}
      <img src={SSlogo} alt="SignSense Logo" className="h-10 w-auto" />

      {/* Centered Navigation Links */}
      <div className="flex space-x-8 text-lg font-medium text-gray-800">
        <Link to="/" className="hover:text-blue-500 transition">Home</Link>
        <Link to="/learn" className="hover:text-blue-500 transition">Learn</Link>
        <Link to="/about" className="hover:text-blue-500 transition">About</Link>
        <Link to="/contact" className="hover:text-blue-500 transition">Contact</Link>
      </div>

      {/* Right-Side Icons (Optional) */}
      <div className="flex space-x-4">
        <button className="hover:text-blue-500 transition">❓</button>
        <button className="hover:text-blue-500 transition">🌍</button>
        <button className="hover:text-blue-500 transition">👤</button>
      </div>
    </nav>
  );
}
