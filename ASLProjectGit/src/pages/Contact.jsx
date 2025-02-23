import ContactImage from "../assets/Contact.png";

export default function Contact() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white pt-20 px-6">
      <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-green-400 mb-6 drop-shadow-lg">
        Contact Us
      </h1>

      {/* Team Image */}
      <div className="w-full max-w-2xl h-96 bg-gray-800 rounded-lg flex items-center justify-center shadow-lg border border-gray-700 overflow-hidden">
        <img
          src={ContactImage}
          alt="Team Photo"
          className="object-cover w-full h-full"
        />
      </div>

      {/* Contact Information */}
      <div className="mt-6 w-full max-w-2xl p-6 bg-gray-800 rounded-xl shadow-md text-center border border-gray-700">
        <h2 className="text-2xl font-semibold text-blue-400">Our Team</h2>
        <p className="text-gray-300 mt-2">Feel free to reach out to us!</p>
        <div className="mt-4 space-y-3">
          <p className="text-gray-300">📧 <span className="font-semibold">Email:</span> team@signsense.com</p>
          <p className="text-gray-300">📍 <span className="font-semibold">Address:</span> 1301 Third Street, West Lafayette, IN</p>
          <p className="text-gray-300">📞 <span className="font-semibold">Phone:</span> +1 (630) 822 - 2757</p>
        </div>
      </div>
    </div>
  );
}
