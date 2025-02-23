import { Link } from "react-router-dom";
import SSlogo from "../assets/SSlogo.png";
import { useAuth0 } from "@auth0/auth0-react";

export default function Navigation() {
  const { loginWithRedirect, logout, user, isAuthenticated } = useAuth0();

  console.log("Auth0 User Data:", { isAuthenticated, user });


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

      {/* Right-Side: Profile Picture or Login Button */}
      <div className="flex space-x-4 items-center">
        {isAuthenticated ? (
          <div className="relative group">
            {/* Profile Picture */}
            <button className="focus:outline-none">
              <img
                src={user.picture}
                alt={user.name || "Profile"}
                className="h-10 w-10 rounded-full border-2 border-gray-300 cursor-pointer"
              />
            </button>
            {/* Logout Dropdown */}
            <div className="absolute right-0 mt-2 w-32 bg-white shadow-lg rounded-md p-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-sm text-gray-800 px-2">{user?.name}</p>
              <button
                onClick={() => logout({ returnTo: "http://localhost:5173/" })}
                className="w-full text-left px-2 py-1 text-sm text-red-500 hover:bg-gray-100 rounded-md"
              >
                Log Out
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => loginWithRedirect({ redirect_uri: "http://localhost:5173/" })}
            className="hover:text-blue-500 transition"
          >
            Log In
          </button>
        )}
      </div>
    </nav>
  );
}
