export function Card({ children, className }) {
    return (
      <div className={`bg-gray-800 p-6 rounded-2xl shadow-md border border-gray-700 ${className}`}>
        {children}
      </div>
    );
  }
  
  export function CardContent({ children, className }) {
    return <div className={`p-4 ${className}`}>{children}</div>;
  }