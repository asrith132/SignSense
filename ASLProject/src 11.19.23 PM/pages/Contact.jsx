export default function Contact() {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-white p-6">
        <h1 className="text-3xl font-bold text-blue-600 mb-4">Contact Us</h1>
        
        {/* Placeholder for Team Image */}
        <div className="w-full max-w-2xl h-64 bg-gray-300 rounded-lg flex items-center justify-center">
          <span className="text-gray-700">[ Team Image Placeholder ]</span>
        </div>
  
        {/* Contact Information */}
        <div className="mt-6 w-full max-w-2xl p-4 bg-gray-100 rounded-lg shadow-md text-center border border-gray-200">
          <h2 className="text-xl font-semibold text-gray-800">Our Team</h2>
          <p className="text-gray-700 mt-2">Feel free to reach out to us!</p>
          <div className="mt-4 space-y-2">
            <p className="text-gray-800">📧 Email: team@signsense.com</p>
            <p className="text-gray-800">📍 Address: 123 ASL Street, Sign City, USA</p>
            <p className="text-gray-800">📞 Phone: +1 (123) 456-7890</p>
          </div>
        </div>
      </div>
    );
  }
  