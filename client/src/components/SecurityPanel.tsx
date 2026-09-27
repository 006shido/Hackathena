import React from 'react';
import { Shield, Activity, Cpu, Sparkles, AlertTriangle, Eye, Mic, Clock } from 'lucide-react';
import { AttackMode } from '../types/attack';

interface SecurityPanelProps {
  peerAttackState?: {
    active: boolean;
    mode: AttackMode;
    faceSwap: boolean;
    voiceTransform: boolean;
  };
  isTester?: boolean;
  onClose?: () => void;
}

export const SecurityPanel: React.FC<SecurityPanelProps> = ({
  peerAttackState,
  onClose,
}) => {
  const isAttackSimulated = peerAttackState?.active;

  return (
    <aside className="w-full lg:w-80 shrink-0 flex flex-col h-full bg-slate-950/95 lg:bg-slate-950/80 border-l border-slate-800/80 p-5 overflow-y-auto">
      {/* Panel Title */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/40">
            <Shield className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-tight text-white">DeepTrace Monitoring</h2>
            <p className="text-[10px] font-mono text-slate-400">AI Defense Pipeline</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1 rounded bg-slate-900 px-2 py-0.5 text-[10px] font-mono text-cyan-400 border border-slate-800">
            <Activity className="h-3 w-3 animate-pulse" /> Standby
          </span>
          {onClose && (
            <button
              onClick={onClose}
              className="lg:hidden p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Close panel"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Main Status Callout */}
      <div className="mt-4 p-4 rounded-xl bg-slate-900/60 border border-slate-800">
        <div className="flex items-center gap-2 mb-2">
          <Clock className="h-4 w-4 text-cyan-400" />
          <span className="text-xs font-mono font-medium text-slate-300">System Status</span>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed font-mono">
          Waiting for security analysis...
        </p>
        <div className="mt-2 text-[11px] text-slate-500 bg-slate-950/70 p-2.5 rounded-lg border border-slate-800/80">
          Monitoring will appear here once the real-time AI deepfake detection engine is connected.
        </div>
      </div>

      {/* Architecture Placeholder Pipeline */}
      <div className="mt-6 flex-1 flex flex-col justify-start">
        <h3 className="text-[11px] font-mono font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
          <Cpu className="h-3.5 w-3.5 text-cyan-400" />
          <span>Planned Detection Pipeline</span>
        </h3>

        <div className="space-y-2.5 text-xs font-mono">
          {/* Node 1: Video Face Analysis */}
          <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800/80 flex items-start gap-2.5">
            <Eye className="h-4 w-4 text-slate-400 mt-0.5 shrink-0" />
            <div>
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-200">Face Analysis</span>
                <span className="text-[10px] text-slate-500">Phase 2</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Neural boundary blending & landmark consistency check
              </p>
            </div>
          </div>

          {/* Node 2: Audio Voice Analysis */}
          <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800/80 flex items-start gap-2.5">
            <Mic className="h-4 w-4 text-slate-400 mt-0.5 shrink-0" />
            <div>
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-200">Voice Analysis</span>
                <span className="text-[10px] text-slate-500">Phase 2</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Spectral flux & vocoder harmonic artifact detection
              </p>
            </div>
          </div>

          {/* Node 3: Audio-Visual Lip-Sync */}
          <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800/80 flex items-start gap-2.5">
            <Sparkles className="h-4 w-4 text-slate-400 mt-0.5 shrink-0" />
            <div>
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-200">Lip-Sync Correlation</span>
                <span className="text-[10px] text-slate-500">Phase 2</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Phoneme-viseme temporal synchronization audit
              </p>
            </div>
          </div>
        </div>

        {/* Future Risk Engine Metric Box */}
        <div className="mt-5 p-3.5 rounded-xl bg-slate-900/50 border border-slate-800/80">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-mono text-slate-400">DeepTrace Risk Engine</span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              {isAttackSimulated ? 'ANOMALY DETECTED' : 'BASELINE NORMAL'}
            </span>
          </div>

          {isAttackSimulated ? (
            <div className="flex items-start gap-2 p-2 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
              <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
              <div>
                <span className="font-semibold block font-mono text-[11px]">Stream Telemetry</span>
                <span className="text-[11px] text-rose-200">
                  Controlled simulated {peerAttackState.mode.toUpperCase()} attack injected by remote tester.
                </span>
              </div>
            </div>
          ) : (
            <p className="text-[11px] text-slate-400 font-mono">
              Continuous neural stream analysis ready for integration.
            </p>
          )}
        </div>
      </div>

      {/* Footer disclaimer */}
      <div className="pt-4 border-t border-slate-800/80 text-[10px] font-mono text-slate-500 text-center">
        DeepTrace v1.0.0 • Hackathon Prototype
      </div>
    </aside>
  );
};
