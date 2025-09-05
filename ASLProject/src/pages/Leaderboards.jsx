import { useState, useEffect } from "react";

export default function Leaderboards() {
  // Dummy leaderboard data (Replace with backend data later)
  const [leaderboard, setLeaderboard] = useState([
    { username: "sanketh.edara@gmail.com", streak: 7 },
    { username: "asrithrn@gmail.com", streak: 4},
  ]);

  useEffect(() => {
    // Future: Fetch leaderboard data from API
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
      
      {/* Page Title */}
      <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-600 mb-6 drop-shadow-lg">
        Daily Leaderboards
      </h1>

      {/* Leaderboard Table */}
      <div className="w-full max-w-3xl bg-gray-800 shadow-lg rounded-2xl p-6 border border-gray-700">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="text-gray-300 text-lg border-b border-gray-600">
              <th className="p-3">Rank</th>
              <th className="p-3">Username</th>
              <th className="p-3">Streak 🔥</th>
            </tr>
          </thead>
          <tbody>
            {leaderboard.map((user, index) => (
              <tr 
                key={index} 
                className={`text-lg font-semibold ${
                  index === 0 ? "text-yellow-400" : index === 1 ? "text-gray-300" : index === 2 ? "text-orange-300" : "text-white"
                } hover:bg-gray-700 transition-all`}
              >
                <td className="p-4">{index + 1}️⃣</td>
                <td className="p-4">{user.username}</td>
                <td className="p-4">{user.streak}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <p className="mt-6 text-gray-400">🔥 Top ASL streaks of the day 🔥</p>
    </div>
  );
}
