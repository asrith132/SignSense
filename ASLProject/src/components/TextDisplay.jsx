export default function TextDisplay({ recognizedText }) {
    return (
      <div className="mt-4 w-full p-3 bg-white rounded-lg shadow-md text-center">
        <h2 className="text-lg font-semibold">Recognized Text:</h2>
        <p className="text-gray-700 mt-2 min-h-[50px]">
          {recognizedText || "Waiting for ASL input..."}
        </p>
      </div>
    );
  }
  