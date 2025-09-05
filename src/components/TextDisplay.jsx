export default function TextDisplay({ recognizedText }) {
  return (
    <div className="mt-6 w-full p-6 bg-white rounded-lg shadow-md text-center border border-gray-200 min-h-[150px]">
      <h2 className="text-lg font-semibold">Recognized Text:</h2>
      <p className="text-gray-700 mt-2 min-h-[100px] text-xl font-medium">
        {recognizedText || "Waiting for ASL input..."}
      </p>
    </div>
  );
}