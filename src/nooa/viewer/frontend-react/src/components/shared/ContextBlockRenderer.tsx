import { CodeBox } from './CodeBox';

interface Segment {
  type: 'text';
  content: string;
}

interface BlockSegment {
  type: 'block';
  key: string;
  attrsStr: string;
  content: string;
}

type Part = Segment | BlockSegment;

const FRAMEWORK_KEYS = new Set([
  'system_prompt',
  'self',
  'doc',
  'instructions',
  'format_description',
]);

/**
 * Split the argument string of a Python-repr call on top-level commas,
 * respecting triple-quoted strings, single/double-quoted strings, and
 * nested parens/brackets/braces.
 */
function splitTopLevelArgs(argsStr: string): string[] {
  const args: string[] = [];
  let current = '';
  let depth = 0;
  let i = 0;

  while (i < argsStr.length) {
    // Triple-quoted strings: consume until matching closing triple-quote
    const triple = argsStr.slice(i, i + 3);
    if (triple === "'''" || triple === '"""') {
      const end = argsStr.indexOf(triple, i + 3);
      if (end !== -1) {
        current += argsStr.slice(i, end + 3);
        i = end + 3;
        continue;
      }
    }

    // Single-quoted strings: consume until unescaped closing quote
    const ch = argsStr[i];
    if ((ch === "'" || ch === '"') && depth === 0) {
      let j = i + 1;
      while (j < argsStr.length) {
        if (argsStr[j] === '\\') { j += 2; continue; }
        if (argsStr[j] === ch) break;
        j++;
      }
      current += argsStr.slice(i, j + 1);
      i = j + 1;
      continue;
    }

    if (ch === '(' || ch === '[' || ch === '{') depth++;
    else if (ch === ')' || ch === ']' || ch === '}') depth--;

    // Top-level comma: split here
    if (depth === 0 && ch === ',' && argsStr[i + 1] === ' ') {
      args.push(current);
      current = '';
      i += 2; // skip ", "
      continue;
    }

    current += ch;
    i++;
  }

  if (current.trim()) args.push(current);
  return args;
}

/**
 * Detect and reformat Python-repr-style content: ClassName(field=val, ...)
 * Returns the reformatted string, or null if content doesn't match the pattern.
 */
function formatPythonRepr(text: string): string | null {
  const trimmed = text.trim();
  // Must start with UpperCamelCase or CamelCase identifier followed by "("
  const match = /^([A-Z]\w*)\(/.exec(trimmed);
  if (!match) return null;
  // Must end with ")"
  if (!trimmed.endsWith(')')) return null;

  const className = match[1];
  const inner = trimmed.slice(className.length + 1, -1); // strip ClassName( and )
  const args = splitTopLevelArgs(inner);

  // Only reformat if there are multiple arguments — single-arg is fine as-is
  if (args.length <= 1) return null;

  return `${className}(\n${args.map((a) => `  ${a.trim()}`).join(',\n')}\n)`;
}

function parseContextBlocks(text: string): Part[] {
  const parts: Part[] = [];
  const re = /<([a-zA-Z_][a-zA-Z0-9_-]*)([^>]*)>\n([\s\S]*?)\n<\/\1>/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    const [fullMatch, key, attrsStr, content] = match;
    const before = text.slice(lastIndex, match.index);
    if (before.trim()) parts.push({ type: 'text', content: before });
    parts.push({ type: 'block', key, attrsStr, content });
    lastIndex = match.index + fullMatch.length;
  }

  const tail = text.slice(lastIndex);
  if (tail.trim()) parts.push({ type: 'text', content: tail });

  return parts;
}

interface ContextBlockRendererProps {
  content: string;
  plain?: boolean;
}

export function ContextBlockRenderer({ content, plain }: ContextBlockRendererProps) {
  if (plain) {
    return <CodeBox code={content} language="markdown" showLineNumbers={false} />;
  }

  const parts = parseContextBlocks(content);

  if (parts.length === 0 || parts.every((p) => p.type === 'text')) {
    return <CodeBox code={content} language="markdown" showLineNumbers={false} />;
  }

  return (
    <div className="space-y-1.5">
      {parts.map((part, i) => {
        if (part.type === 'text') {
          return (
            <CodeBox key={i} code={part.content.trim()} language="markdown" showLineNumbers={false} />
          );
        }

        const isFramework = FRAMEWORK_KEYS.has(part.key);
        const attrs = part.attrsStr.trim();
        const borderColor = isFramework ? 'border-gray-600' : 'border-teal-700';
        const keyColor = isFramework ? 'text-gray-400' : 'text-teal-300';

        const reformatted = formatPythonRepr(part.content);
        const displayContent = reformatted ?? part.content;
        // Use python highlighting when we reformatted (looks like a repr call),
        // otherwise keep markdown which handles prose/markdown fine.
        const language = reformatted ? 'python' : 'markdown';

        return (
          <div key={i} className={`rounded border ${borderColor} overflow-hidden`}>
            <div className={`px-3 py-1.5 flex items-start gap-2 text-xs ${isFramework ? 'bg-gray-800/60' : 'bg-teal-900/25'}`}>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={`font-mono font-semibold ${keyColor} shrink-0`}>{part.key}</span>
                  <span className="ml-auto shrink-0 text-gray-600 font-mono">
                    {part.content.split('\n').length} lines
                  </span>
                </div>
                {attrs && (
                  <div className="text-gray-500 font-mono mt-0.5 break-all">{attrs}</div>
                )}
              </div>
            </div>
            <div className="border-t border-gray-700/50">
              <CodeBox code={displayContent} language={language} showLineNumbers={false} className="rounded-none pl-3" />
            </div>
          </div>
        );
      })}
    </div>
  );
}
