import { Link } from "react-router-dom";

export default function Navigation() {
  return (
    <nav className="p-4 bg-blue-500 text-white">
      <Link to="/" className="mr-4">Home</Link>
      <Link to="/learn">Learn</Link> {/* Updated the link */}
    </nav>
  );
}
