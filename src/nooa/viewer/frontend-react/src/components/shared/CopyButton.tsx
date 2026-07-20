import { useState, useCallback, useRef, useEffect } from 'react';

interface CopyButtonProps {
  text: string;
  className?: string;
  label?: string;
  title?: string;
}

export function CopyButton({ text, className = '', label = 'Copy', title }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    }
    setCopied(true);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setCopied(false), 1500);
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      title={title}
      className={`text-xs px-2 py-1 rounded transition-colors ${
        copied ? 'bg-green-700 text-green-200' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
      } ${className}`}
    >
      {copied ? 'Copied' : label}
    </button>
  );
}
