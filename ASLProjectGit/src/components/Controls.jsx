import { Button } from "../components/ui/button.jsx";

export default function Controls({ setRecognizedText }) {
  return (
    <div className="mt-6 flex gap-4">
      <Button onClick={() => setRecognizedText("")} className="bg-red-500 hover:bg-red-600 px-4 py-2 rounded-lg">
        Clear Text
      </Button>
      <Button onClick={() => setRecognizedText(prev => prev.split(' ').slice(0, -1).join(' '))} className="bg-red-500 text-white font-bold px-6 py-3 rounded-lg border-2 border-red-700 shadow-md">
        Delete Word
      </Button>
      <Button onClick={() => alert('Done!')} className="bg-green-500 text-white font-bold px-6 py-3 rounded-lg border-2 border-green-700 shadow-md">
        Done
      </Button>
    </div>
  );
}