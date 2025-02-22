import { Button } from "../components/ui/button.jsx";

export default function Controls({ setRecognizedText }) {
  return (
    <div className="mt-4 flex gap-4">
      <Button onClick={() => setRecognizedText("")} className="bg-red-500 hover:bg-red-600">
        Clear Text
      </Button>
    </div>
  );
}
