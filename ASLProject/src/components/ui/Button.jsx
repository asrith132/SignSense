export function Button({ onClick, children, className = "" }) {
    return (
      <button
        onClick={onClick}
        className={`bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg ${className}`}
      >
        {children}
      </button>
    );
  }
  