import { useEffect, useRef, useState, useCallback } from 'react';
import hljs from 'highlight.js/lib/core';
import json from 'highlight.js/lib/languages/json';
import python from 'highlight.js/lib/languages/python';
import markdown from 'highlight.js/lib/languages/markdown';
import { CopyButton } from './CopyButton';

hljs.registerLanguage('json', json);
hljs.registerLanguage('python', python);
hljs.registerLanguage('markdown', markdown);

interface CodeBoxProps {
  code: string;
  language?: string;
  showLineNumbers?: boolean;
  className?: string;
  maxHeight?: string;
}

interface SearchMatch {
  lineIndex: number;
  start: number;
  end: number;
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightLine(
  lineText: string,
  searchTerm: string,
  lineMatches: SearchMatch[],
  currentGlobalIndex: number,
  globalOffset: number,
): string {
  if (!searchTerm || lineMatches.length === 0) return escapeHtml(lineText);

  let result = '';
  let pos = 0;

  for (let i = 0; i < lineMatches.length; i++) {
    const m = lineMatches[i];
    const globalIdx = globalOffset + i;
    const isCurrent = globalIdx === currentGlobalIndex;

    result += escapeHtml(lineText.substring(pos, m.start));
    const matchText = lineText.substring(m.start, m.end);
    const cls = isCurrent ? 'bg-yellow-400 text-gray-900' : 'bg-yellow-800 text-gray-100';
    result += `<mark class="${cls} rounded-sm px-0">${escapeHtml(matchText)}</mark>`;
    pos = m.end;
  }

  result += escapeHtml(lineText.substring(pos));
  return result;
}

export function CodeBox({
  code,
  language = 'json',
  showLineNumbers = true,
  className = '',
  maxHeight = 'none',
}: CodeBoxProps) {
  // Defensive: upstream callers occasionally pass undefined/null/non-string
  // values (e.g. a missing span attribute). Coerce to string so split()
  // and subsequent operations never crash.
  const safeCode = typeof code === 'string' ? code : code == null ? '' : String(code);
  // If the language isn't registered with highlight.js, skip highlighting
  // rather than warning/throwing. Falls back to escaped plain text.
  const highlightable = hljs.getLanguage(language) !== undefined;
  const searchInputRef = useRef<HTMLInputElement>(null);
  const searchBarRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentMatch, setCurrentMatch] = useState(0);

  useEffect(() => {
    if (searchOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [searchOpen]);

  useEffect(() => {
    if (!searchOpen) return;
    const handler = (e: MouseEvent) => {
      if (searchBarRef.current && !searchBarRef.current.contains(e.target as Node)) {
        setSearchTerm('');
        setSearchOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [searchOpen]);

  const lines = safeCode.split('\n');

  // Compute matches
  const allMatches: SearchMatch[][] = [];
  let totalMatches = 0;
  if (searchTerm) {
    const lower = searchTerm.toLowerCase();
    for (const line of lines) {
      const lineMatches: SearchMatch[] = [];
      const lowerLine = line.toLowerCase();
      let pos = 0;
      let idx = lowerLine.indexOf(lower, pos);
      while (idx !== -1) {
        lineMatches.push({
          lineIndex: allMatches.length,
          start: idx,
          end: idx + searchTerm.length,
        });
        pos = idx + searchTerm.length;
        idx = lowerLine.indexOf(lower, pos);
      }
      totalMatches += lineMatches.length;
      allMatches.push(lineMatches);
    }
  }

  const safeCurrentMatch = totalMatches > 0 ? currentMatch % totalMatches : 0;

  const goNext = useCallback(() => {
    if (totalMatches > 0) setCurrentMatch((c) => (c + 1) % totalMatches);
  }, [totalMatches]);

  const goPrev = useCallback(() => {
    if (totalMatches > 0) setCurrentMatch((c) => (c - 1 + totalMatches) % totalMatches);
  }, [totalMatches]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        if (e.shiftKey) goPrev();
        else goNext();
      } else if (e.key === 'Escape') {
        e.stopPropagation();
        setSearchTerm('');
        setSearchOpen(false);
      }
    },
    [goNext, goPrev],
  );

  // Scroll current match into view
  useEffect(() => {
    if (totalMatches > 0 && contentRef.current) {
      const mark = contentRef.current.querySelector('mark.bg-yellow-400');
      if (mark) mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [safeCurrentMatch, totalMatches, searchTerm]);

  // Highlight code and split into per-line HTML for line-by-line rendering
  const highlightedLines = useRef<string[]>([]);
  if (!searchTerm) {
    if (highlightable) {
      try {
        const result = hljs.highlight(safeCode, { language });
        highlightedLines.current = result.value.split('\n');
      } catch {
        highlightedLines.current = lines.map(escapeHtml);
      }
    } else {
      // Unknown language (e.g. "text", "yaml") — render as plain text
      highlightedLines.current = lines.map(escapeHtml);
    }
  }

  const renderSearchLine = (line: string, i: number, globalIdx: { v: number }) => {
    const lineMatches = allMatches[i] || [];
    const html = highlightLine(line, searchTerm, lineMatches, safeCurrentMatch, globalIdx.v);
    globalIdx.v += lineMatches.length;
    return html;
  };

  return (
    <div className={`relative group rounded bg-gray-900 ${className}`}>
      <div className="absolute top-2 right-2 flex items-center gap-1 z-10">
        {searchOpen ? (
          <div
            ref={searchBarRef}
            className="flex items-center gap-1 bg-gray-800 rounded border border-gray-700 px-1.5 py-0.5"
          >
            <input
              ref={searchInputRef}
              type="text"
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentMatch(0);
              }}
              onKeyDown={handleKeyDown}
              onClick={(e) => e.stopPropagation()}
              placeholder="Search..."
              className="w-24 bg-transparent text-gray-200 text-xs outline-none placeholder-gray-500"
            />
            {searchTerm && (
              <span
                className={`text-[10px] min-w-[35px] text-center ${totalMatches > 0 ? 'text-green-400' : 'text-red-400'}`}
              >
                {totalMatches > 0 ? `${safeCurrentMatch + 1}/${totalMatches}` : '0/0'}
              </span>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                goPrev();
              }}
              className="text-gray-400 hover:text-gray-200 text-[10px] px-0.5"
              title="Previous (Shift+Enter)"
            >
              &#9650;
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                goNext();
              }}
              className="text-gray-400 hover:text-gray-200 text-[10px] px-0.5"
              title="Next (Enter)"
            >
              &#9660;
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setSearchTerm('');
                setSearchOpen(false);
              }}
              className="text-gray-400 hover:text-gray-200 text-xs px-0.5"
              title="Close (Esc)"
            >
              &#215;
            </button>
          </div>
        ) : (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setSearchOpen(true);
            }}
            className="opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-gray-200 p-1"
            title="Search in code"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="w-3.5 h-3.5"
            >
              <path
                fillRule="evenodd"
                d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        )}
        <div className="opacity-0 group-hover:opacity-100 transition-opacity">
          <CopyButton text={safeCode} />
        </div>
      </div>
      <div ref={contentRef} className="overflow-auto" style={{ maxHeight }}>
        <pre className="text-sm leading-relaxed p-0 m-0">
          {(() => {
            const globalIdx = { v: 0 };
            return lines.map((line, i) => {
              const lineHtml = searchTerm
                ? renderSearchLine(line, i, globalIdx)
                : (highlightedLines.current[i] ?? escapeHtml(line));

              return (
                <div key={i} className="flex">
                  {showLineNumbers && (
                    <span className="select-none text-gray-600 text-right shrink-0 pr-3 min-w-[3ch]">
                      {i + 1}
                    </span>
                  )}
                  <span
                    className="flex-1 whitespace-pre-wrap break-words"
                    dangerouslySetInnerHTML={{ __html: lineHtml || '&nbsp;' }}
                  />
                </div>
              );
            });
          })()}
        </pre>
      </div>
    </div>
  );
}
