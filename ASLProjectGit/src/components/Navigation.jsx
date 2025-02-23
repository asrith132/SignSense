import { Link } from "react-router-dom";
import SSlogo from "../assets/SSlogo.png";
import { useAuth0 } from "@auth0/auth0-react";

export default function Navigation() {
  const { loginWithRedirect, logout, user, isAuthenticated } = useAuth0();

  return (
    <nav className="fixed top-0 left-0 w-full flex items-center justify-between bg-gray-900 shadow-md px-10 py-4 z-50 border-b border-gray-700">
      {/* Logo */}
      <img
        src={SSlogo}
        alt="SignSense Logo"
        className="h-12 w-auto cursor-pointer hover:scale-110 transition-transform duration-300"
      />

      {/* Navigation Links */}
      <div className="flex space-x-10 text-lg font-semibold text-gray-300">
        <Link to="/" className="hover:text-blue-400 transition-all duration-300">
          Home
        </Link>
        <Link to="/learn" className="hover:text-blue-400 transition-all duration-300">
          Learn
        </Link>
        <Link to="/leaderboards" className="hover:text-purple-400 transition-all duration-300">
          Leaderboards
        </Link>
        <Link to="/about" className="hover:text-green-400 transition-all duration-300">
          About
        </Link>
        <Link to="/contact" className="hover:text-blue-400 transition-all duration-300">
          Contact
        </Link>
      </div>

      {/* Right-Side: Log In / Log Out */}
      <div className="flex space-x-6 items-center">
        {isAuthenticated ? (
          // If logged in, show "Sign Out" button
          <button
            onClick={() => {
              alert("You have been signed out. Please log in again!");
              logout({ returnTo: "http://localhost:5173/" });
            }}
            className="text-lg font-semibold text-blue-400 hover:text-blue-500 transition-all duration-300"
          >
            Sign Out
          </button>
        ) : (
          // Otherwise, show "Log In" button
          <button
            onClick={() => loginWithRedirect({ redirect_uri: "http://localhost:5173/" })}
            className="text-lg font-semibold text-blue-400 hover:text-blue-500 transition-all duration-300"
          >
            Log In
          </button>
        )}
      </div>
    </nav>
  );
}
