import { createContext, useContext, useCallback } from "react";
import { useNavigate, useLocation } from "react-router";
import type { PlaygroundMessage } from "@/api/playground";
import { storePlaygroundData } from "@/pages/PlaygroundPage";

export interface PlaygroundRequest {
  messages: PlaygroundMessage[];
  originalOutput: string | null;
  model: string | null;
}

const PlaygroundCtx = createContext<((req: PlaygroundRequest) => void) | null>(null);

export function usePlayground() {
  return useContext(PlaygroundCtx);
}

export function PlaygroundProvider({
  sessionId,
  children,
}: {
  sessionId: string;
  children: React.ReactNode;
}) {
  const navigate = useNavigate();
  const location = useLocation();

  const openPlayground = useCallback(
    (req: PlaygroundRequest) => {
      storePlaygroundData(req);
      const returnTo = encodeURIComponent(location.pathname + location.search);
      navigate(
        `/playground?session_id=${encodeURIComponent(sessionId)}&return=${returnTo}`,
      );
    },
    [navigate, sessionId, location],
  );

  return (
    <PlaygroundCtx.Provider value={openPlayground}>
      {children}
    </PlaygroundCtx.Provider>
  );
}
