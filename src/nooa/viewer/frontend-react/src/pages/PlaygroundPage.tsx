import { useSearchParams, useNavigate } from "react-router";
import { useMemo } from "react";
import { Playground } from "@/components/playground/Playground";
import type { PlaygroundRequest } from "@/components/playground/PlaygroundContext";

const STORAGE_KEY = "playground-data";

export function storePlaygroundData(data: PlaygroundRequest) {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

export function PlaygroundPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const sessionId = searchParams.get("session_id") || "";
  const returnTo = searchParams.get("return") || "";

  const request = useMemo((): PlaygroundRequest | null => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }, []);

  const handleClose = () => {
    if (returnTo) {
      navigate(returnTo);
    } else if (sessionId) {
      navigate(`/traces/view?session_id=${encodeURIComponent(sessionId)}`);
    } else {
      navigate(-1);
    }
  };

  if (!request || request.messages.length === 0) {
    return (
      <div className="max-w-[100rem] mx-auto px-4 py-6">
        <div className="text-gray-500 py-12 text-center">
          No playground data. Open a playground from an LLM call span.
        </div>
        <div className="text-center mt-4">
          <button
            onClick={handleClose}
            className="text-sm text-gray-400 hover:text-gray-200"
          >
            &#9666; Back
          </button>
        </div>
      </div>
    );
  }

  return <Playground request={request} onClose={handleClose} />;
}
