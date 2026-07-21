import { Routes, Route, Navigate, NavLink } from 'react-router';
import { TraceList } from '@/pages/TraceList';
import { TraceDetail } from '@/pages/TraceDetail';
import { EvalExperimentList } from '@/pages/EvalExperimentList';
import { EvalExperimentDetail } from '@/pages/EvalExperimentDetail';
import { EvalTraceDetail } from '@/pages/EvalTraceDetail';
import { PlaygroundPage } from '@/pages/PlaygroundPage';
import { MemoryPage } from '@/pages/MemoryPage';
import { MemoryRecordDetail } from '@/pages/MemoryRecordDetail';

function NavBar() {
  return (
    <nav className="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-[100rem] mx-auto px-4 flex items-center h-12 gap-6">
        <span className="text-sm font-semibold text-gray-300 tracking-tight">NVIDIA OO Agents Viewer</span>
        <div className="flex gap-1">
          <NavLink
            to="/eval"
            className={({ isActive }) =>
              `px-3 py-1.5 text-sm rounded transition-colors ${
                isActive
                  ? 'bg-gray-800 text-gray-100'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
              }`
            }
          >
            Evaluations
          </NavLink>
          <NavLink
            to="/traces"
            className={({ isActive }) =>
              `px-3 py-1.5 text-sm rounded transition-colors ${
                isActive
                  ? 'bg-gray-800 text-gray-100'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
              }`
            }
          >
            Traces
          </NavLink>
          <NavLink
            to="/memory"
            className={({ isActive }) =>
              `px-3 py-1.5 text-sm rounded transition-colors ${
                isActive
                  ? 'bg-gray-800 text-gray-100'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
              }`
            }
          >
            Memory
          </NavLink>
        </div>
      </div>
    </nav>
  );
}

export function App() {
  return (
    <div className="min-h-screen">
      <NavBar />
      <Routes>
        <Route path="/" element={<Navigate to="/eval" replace />} />
        <Route path="/traces" element={<TraceList />} />
        <Route path="/traces/view" element={<TraceDetail />} />
        <Route path="/eval" element={<EvalExperimentList />} />
        <Route path="/eval/experiment/:id" element={<EvalExperimentDetail />} />
        <Route path="/eval/experiment/:id/trace/:traceId" element={<EvalTraceDetail />} />
        <Route path="/memory" element={<MemoryPage />} />
        <Route path="/memory/record" element={<MemoryRecordDetail />} />
        <Route path="/playground" element={<PlaygroundPage />} />
        <Route
          path="*"
          element={<div className="text-center py-12 text-gray-500">Page not found</div>}
        />
      </Routes>
    </div>
  );
}
