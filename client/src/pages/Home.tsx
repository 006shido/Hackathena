import React, { useState } from 'react';
import {
  Shield,
  Video,
  Lock,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronUp,
  Cpu,
  Eye,
  Mic,
  ExternalLink,
  Sparkles,
} from 'lucide-react';

interface HomePageProps {
  onNavigate: (mode?: 'signin' | 'signup') => void;
  onQuickDemo?: (role: 'user' | 'tester') => void;
}

export const HomePage: React.FC<HomePageProps> = ({
  onNavigate,
  onQuickDemo,
}) => {
  const [openFaq, setOpenFaq] = useState<number | null>(0);
  const [roomInput, setRoomInput] = useState('');

  const handleNavigate = (mode: 'signin' | 'signup' = 'signin') => {
    onNavigate(mode);
  };

  const toggleFaq = (index: number) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  const handleStartRoom = (e: React.FormEvent) => {
    e.preventDefault();
    if (roomInput.trim()) {
      const code = roomInput.trim().toUpperCase().replace(/[^A-Z0-9-]/g, '');
      const url = new URL(window.location.href);
      url.searchParams.set('room', code);
      window.history.pushState({}, '', url.toString());
    }
    handleNavigate('signup');
  };

  const faqs = [
    {
      q: 'How does DeepTrace detect deepfakes in real-time?',
      a: 'DeepTrace runs an in-browser neural pipeline using MediaPipe and WebGL. It tracks 478 3D facial landmarks to spot subtle warping, edge feathering discrepancies, and unnatural blink rates. Simultaneously, a spectral acoustic analyzer scans 48kHz audio for vocoder harmonic frequencies and synthetic voice clone signatures.',
    },
    {
      q: 'Do I need to install software, drivers, or browser extensions?',
      a: 'No. DeepTrace is 100% browser-native. It utilizes standard WebRTC, WebGL, and Web Audio APIs built into modern browsers (Chrome, Edge, Safari, Firefox). You can start or join a secure call with just a link.',
    },
    {
      q: 'Are our video calls recorded or saved on your servers?',
      a: 'Never. All media is encrypted end-to-end via DTLS-SRTP and flows directly peer-to-peer between participants. Our servers only coordinate the initial WebRTC signaling handshake — zero video or audio frames ever touch or reside on our servers.',
    },
    {
      q: 'What is the Attack Simulation mode for testers?',
      a: 'Authorized tester accounts have access to an internal sandbox to test defense thresholds. Testers can simulate AI face swaps and robotic voice transforms under controlled conditions to observe how the detection HUD responds.',
    },
    {
      q: 'Is DeepTrace free to use?',
      a: 'Yes. DeepTrace is an open-source security platform. Anyone can create rooms, invite friends, and test real-time deepfake defense capabilities for free.',
    },
  ];

  return (
    <div className="min-h-screen w-full bg-[#fbfbfd] text-zinc-900 font-sans selection:bg-zinc-900 selection:text-white flex flex-col justify-between antialiased">
      {/* 1. Minimal Header */}
      <header className="sticky top-0 z-50 w-full bg-[#fbfbfd]/90 backdrop-blur-md border-b border-zinc-200/70 transition-all">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-15 flex items-center justify-between">
          {/* Logo & Project Name */}
          <div
            className="flex items-center gap-2.5 cursor-pointer select-none group"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          >
            <div className="h-8 w-8 rounded-lg bg-zinc-900 text-white flex items-center justify-center p-1.5 shadow-xs transition-transform group-hover:scale-105">
              <img src="/logo.svg" alt="DeepTrace" className="h-full w-full invert" />
            </div>
            <div className="flex items-center">
              <span className="text-base font-semibold tracking-tight text-zinc-900">DeepTrace</span>
            </div>
          </div>

          {/* Quick Nav Links */}
          <nav className="hidden sm:flex items-center gap-6 text-xs font-medium text-zinc-600" aria-label="Page links">
            <a href="#how-it-works" className="hover:text-zinc-950 transition-colors">How it works</a>
            <a href="#features" className="hover:text-zinc-950 transition-colors">Features</a>
            <a href="#faq" className="hover:text-zinc-950 transition-colors">FAQ</a>
            <a
              href="https://github.com/abhinavaby/Hackathena"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-950 transition-colors flex items-center gap-1"
            >
              <span>GitHub</span>
              <ExternalLink className="h-3 w-3 text-zinc-400" />
            </a>
          </nav>

          {/* Action Buttons: Simple & Human */}
          <div className="flex items-center gap-2 sm:gap-2.5">
            <button
              type="button"
              onClick={() => handleNavigate('signin')}
              className="px-3.5 py-1.5 text-xs font-medium text-zinc-700 hover:text-zinc-950 hover:bg-zinc-100 rounded-lg transition-colors cursor-pointer active:scale-95"
            >
              Log in
            </button>
            <button
              type="button"
              onClick={() => handleNavigate('signup')}
              className="group px-3.5 py-1.5 text-xs font-medium text-white bg-zinc-900 hover:bg-black rounded-lg shadow-xs transition-all flex items-center gap-1.5 cursor-pointer active:scale-95 hover:shadow-md"
            >
              <span>Sign up</span>
              <ArrowRight className="h-3.5 w-3.5 text-zinc-400 group-hover:translate-x-0.5 transition-transform" />
            </button>
          </div>
        </div>
      </header>

      {/* 2. Main Hero Section */}
      <main className="flex-1">
        <section className="pt-16 pb-16 sm:pt-24 sm:pb-20 max-w-3xl mx-auto px-4 sm:px-6 text-center">
          {/* Simple, Powerful Headline (H1 for SEO) */}
          <h1 className="text-3xl sm:text-5xl lg:text-[54px] font-bold text-zinc-950 tracking-tight leading-[1.12] mb-5">
            Video calls you can actually trust.
          </h1>

          {/* Direct, Honest Description */}
          <p className="max-w-2xl mx-auto text-base sm:text-lg text-zinc-600 leading-relaxed mb-8 font-normal">
            DeepTrace protects peer-to-peer video meetings by spotting synthetic face swaps and cloned voices before they fool you. Runs entirely in your browser with zero latency and zero cloud recordings.
          </p>

          {/* Interactive Room Launcher Box */}
          <div className="max-w-md mx-auto p-2 bg-white rounded-2xl border border-zinc-200 shadow-sm mb-6">
            <form onSubmit={handleStartRoom} className="flex items-center gap-2">
              <div className="flex-1 flex items-center gap-2 pl-3">
                <Video className="h-4 w-4 text-zinc-400 shrink-0" />
                <input
                  type="text"
                  placeholder="Enter room code or name..."
                  value={roomInput}
                  onChange={(e) => setRoomInput(e.target.value)}
                  className="w-full bg-transparent text-xs sm:text-sm text-zinc-900 placeholder:text-zinc-400 focus:outline-none"
                />
              </div>
              <button
                type="submit"
                className="px-4 py-2 rounded-xl bg-zinc-900 hover:bg-black text-white text-xs sm:text-sm font-medium transition-all shrink-0 cursor-pointer active:scale-95"
              >
                Join or Start
              </button>
            </form>
          </div>

          {/* Quick Demo Shortcuts */}
          <div className="flex flex-wrap items-center justify-center gap-3 text-xs text-zinc-500">
            <span>Or try instantly:</span>
            {onQuickDemo && (
              <>
                <button
                  type="button"
                  onClick={() => onQuickDemo('user')}
                  className="px-2.5 py-1 rounded-md bg-zinc-100 hover:bg-zinc-200 text-zinc-700 font-medium transition-colors cursor-pointer active:scale-95"
                >
                  Regular User Demo
                </button>
                <button
                  type="button"
                  onClick={() => onQuickDemo('tester')}
                  className="px-2.5 py-1 rounded-md bg-zinc-100 hover:bg-zinc-200 text-zinc-700 font-medium transition-colors flex items-center gap-1 cursor-pointer active:scale-95"
                >
                  <Cpu className="h-3 w-3 text-blue-600" />
                  <span>Tester Attack Simulator</span>
                </button>
              </>
            )}
          </div>
        </section>

        {/* 3. Simple 3-Column How It Works */}
        <section id="how-it-works" className="py-14 sm:py-18 bg-white border-y border-zinc-200/80">
          <div className="max-w-4xl mx-auto px-4 sm:px-6">
            <div className="mb-10 text-center sm:text-left">
              <span className="text-[11px] font-mono text-zinc-500 uppercase tracking-widest block mb-1">
                How It Works
              </span>
              <h2 className="text-2xl font-bold text-zinc-950 tracking-tight">
                Designed to be effortless and invisible.
              </h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-5 rounded-xl border border-zinc-200/80 bg-zinc-50/50">
                <div className="font-mono text-xs font-bold text-zinc-400 mb-3">01</div>
                <h3 className="text-sm font-semibold text-zinc-900 mb-1.5">Share a simple link</h3>
                <p className="text-xs text-zinc-600 leading-relaxed">
                  Generate a 6-character room code or copy a link. Your peer joins instantly through their browser without installing anything.
                </p>
              </div>

              <div className="p-5 rounded-xl border border-zinc-200/80 bg-zinc-50/50">
                <div className="font-mono text-xs font-bold text-zinc-400 mb-3">02</div>
                <h3 className="text-sm font-semibold text-zinc-900 mb-1.5">Passive real-time check</h3>
                <p className="text-xs text-zinc-600 leading-relaxed">
                  DeepTrace's lightweight MediaPipe and WebGL models inspect facial landmark geometry and audio spectrum frequencies at 60 FPS.
                </p>
              </div>

              <div className="p-5 rounded-xl border border-zinc-200/80 bg-zinc-50/50">
                <div className="font-mono text-xs font-bold text-zinc-400 mb-3">03</div>
                <h3 className="text-sm font-semibold text-zinc-900 mb-1.5">Immediate alert on spoofing</h3>
                <p className="text-xs text-zinc-600 leading-relaxed">
                  If an identity mismatch, face boundary splice, or synthetic voice tone is detected, you receive an immediate alert banner and chime.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Core Features List */}
        <section id="features" className="py-14 sm:py-18 max-w-4xl mx-auto px-4 sm:px-6">
          <div className="mb-10 text-center sm:text-left">
            <span className="text-[11px] font-mono text-zinc-500 uppercase tracking-widest block mb-1">
              Core Capabilities
            </span>
            <h2 className="text-2xl font-bold text-zinc-950 tracking-tight">
              Built specifically for deepfake threats.
            </h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-5 rounded-xl border border-zinc-200 bg-white">
              <div className="flex items-center gap-2 mb-2">
                <Eye className="h-4 w-4 text-zinc-800" />
                <h3 className="text-sm font-semibold text-zinc-900">478-Point Facial Landmark Mesh</h3>
              </div>
              <p className="text-xs text-zinc-600 leading-relaxed">
                Tracks full cheek curvature, eyelid movement, lip motion, and nose bridge geometry to detect generative face replacement and perimeter warping.
              </p>
            </div>

            <div className="p-5 rounded-xl border border-zinc-200 bg-white">
              <div className="flex items-center gap-2 mb-2">
                <Mic className="h-4 w-4 text-zinc-800" />
                <h3 className="text-sm font-semibold text-zinc-900">Spectral Voice Clone Detection</h3>
              </div>
              <p className="text-xs text-zinc-600 leading-relaxed">
                Analyzes acoustic spectrum modulation on live audio streams, separating natural human speech harmonics from neural voice clones and robotic vocoders.
              </p>
            </div>

            <div className="p-5 rounded-xl border border-zinc-200 bg-white">
              <div className="flex items-center gap-2 mb-2">
                <Lock className="h-4 w-4 text-zinc-800" />
                <h3 className="text-sm font-semibold text-zinc-900">End-to-End WebRTC Privacy</h3>
              </div>
              <p className="text-xs text-zinc-600 leading-relaxed">
                Direct browser-to-browser media connection. No video frames or audio streams pass through or are saved on any central recording database.
              </p>
            </div>

            <div className="p-5 rounded-xl border border-zinc-200 bg-white">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="h-4 w-4 text-zinc-800" />
                <h3 className="text-sm font-semibold text-zinc-900">Security Tester Sandbox</h3>
              </div>
              <p className="text-xs text-zinc-600 leading-relaxed">
                Empowers security researchers with controlled attack simulation tools to test how different deepfake models behave against live defense detectors.
              </p>
            </div>
          </div>
        </section>

        {/* 5. Clean FAQ Section (Crucial for Google Indexing) */}
        <section id="faq" className="py-14 sm:py-18 bg-white border-t border-zinc-200/80">
          <div className="max-w-3xl mx-auto px-4 sm:px-6">
            <div className="text-center mb-10">
              <span className="text-[11px] font-mono text-zinc-500 uppercase tracking-widest block mb-1">
                Frequently Asked Questions
              </span>
              <h2 className="text-2xl font-bold text-zinc-950 tracking-tight">
                Everything you need to know
              </h2>
            </div>

            <div className="divide-y divide-zinc-200 border-y border-zinc-200">
              {faqs.map((item, idx) => {
                const isOpen = openFaq === idx;
                return (
                  <div key={idx} className="py-4">
                    <button
                      type="button"
                      onClick={() => toggleFaq(idx)}
                      className="w-full flex items-center justify-between text-left text-sm font-medium text-zinc-900 hover:text-zinc-700 transition-colors cursor-pointer py-1"
                    >
                      <span>{item.q}</span>
                      {isOpen ? (
                        <ChevronUp className="h-4 w-4 text-zinc-400 shrink-0 ml-4" />
                      ) : (
                        <ChevronDown className="h-4 w-4 text-zinc-400 shrink-0 ml-4" />
                      )}
                    </button>
                    {isOpen && (
                      <p className="pt-2 text-xs sm:text-sm text-zinc-600 leading-relaxed pr-6">
                        {item.a}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* 6. Simple Bottom Call to Action */}
        <section className="py-14 sm:py-18 bg-[#fbfbfd] text-center border-t border-zinc-200/80">
          <div className="max-w-2xl mx-auto px-4 sm:px-6">
            <h2 className="text-2xl font-bold text-zinc-950 mb-3 tracking-tight">
              Ready to verify your next video call?
            </h2>
            <p className="text-xs sm:text-sm text-zinc-600 mb-6 max-w-md mx-auto leading-relaxed">
              Open a room in seconds. No credit cards or complicated installations.
            </p>
            <div className="flex items-center justify-center gap-2.5">
              <button
                type="button"
                onClick={() => handleNavigate('signup')}
                className="group px-4 py-2.5 rounded-lg bg-zinc-900 hover:bg-black text-white text-xs sm:text-sm font-medium shadow-xs transition-all flex items-center gap-1.5 cursor-pointer active:scale-95 hover:shadow-md"
              >
                <span>Sign up free</span>
                <ArrowRight className="h-3.5 w-3.5 text-zinc-400 group-hover:translate-x-0.5 transition-transform" />
              </button>
              <button
                type="button"
                onClick={() => handleNavigate('signin')}
                className="px-4 py-2.5 rounded-lg border border-zinc-300 bg-white hover:bg-zinc-50 text-zinc-800 text-xs sm:text-sm font-medium transition-all cursor-pointer active:scale-95"
              >
                Log in
              </button>
            </div>
          </div>
        </section>
      </main>

      {/* 7. Clean Human Footer */}
      <footer className="border-t border-zinc-200 bg-white py-6">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-zinc-500">
          <div className="flex items-center gap-2">
            <span className="font-medium text-zinc-800">DeepTrace</span>
          </div>

          <div className="flex items-center gap-5">
            <button
              type="button"
              onClick={() => handleNavigate('signup')}
              className="hover:text-zinc-900 transition-colors cursor-pointer"
            >
              Sign up
            </button>
            <button
              type="button"
              onClick={() => handleNavigate('signin')}
              className="hover:text-zinc-900 transition-colors cursor-pointer"
            >
              Log in
            </button>
            <a
              href="https://github.com/abhinavaby/Hackathena"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-900 transition-colors flex items-center gap-1"
            >
              <span>GitHub</span>
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>

          <div>
            © {new Date().getFullYear()} DeepTrace.
          </div>
        </div>
      </footer>
    </div>
  );
};
