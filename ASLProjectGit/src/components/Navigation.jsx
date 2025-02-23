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
        <Link to="/" className="hover:text-blue-400 transition-all duration-300">Home</Link>
        <Link to="/learn" className="hover:text-blue-400 transition-all duration-300">Learn</Link>
        <Link to="/leaderboards" className="hover:text-purple-400 transition-all duration-300">Leaderboards</Link>
        <Link to="/about" className="hover:text-green-400 transition-all duration-300">About</Link>
        <Link to="/contact" className="hover:text-blue-400 transition-all duration-300">Contact</Link>
      </div>

      {/* Right-Side: Profile or Login */}
      <div className="flex space-x-6 items-center">
        {isAuthenticated ? (
          <div className="relative group">
            <button className="focus:outline-none">
              <img
                src={user.picture}
                alt={user.name || "Profile"}
                className="h-12 w-12 rounded-full border-2 border-blue-400 cursor-pointer hover:shadow-lg transition-shadow duration-300"
              />
            </button>
            {/* Logout Dropdown */}
            <div className="absolute right-0 mt-2 w-36 bg-gray-800 shadow-lg rounded-lg p-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-sm text-gray-300 px-2">{user?.name}</p>
              <button
                onClick={() => logout({ returnTo: "http://localhost:5174/" })}
                className="w-full text-left px-2 py-2 text-sm text-red-400 hover:bg-gray-700 rounded-lg transition"
              >
                Log Out
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => loginWithRedirect({ redirect_uri: "http://localhost:5174/" })}
            className="text-lg font-semibold text-blue-400 hover:text-blue-500 transition-all duration-300"
          >
            Log In
          </button>
        )}
      </div>
    </nav>
  );
}