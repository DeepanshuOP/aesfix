import type { Metadata, Viewport } from "next";
import "./globals.css";
import React from "react";
import { Camera, Sparkles, BookOpen, ExternalLink, Terminal } from "lucide-react";
import { HealthPill } from "@/components/HealthPill";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1.0,
  maximumScale: 5.0,
  themeColor: "#09090b",
};

export const metadata: Metadata = {
  title: "AesFix — Closed-Loop Image Aesthetics via Measured Critique",
  description:
    "Grounded 5-aspect photo criticism, deterministic Tier-A retouching, and measured aesthetic gain with Lightroom XMP export.",
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps): React.JSX.Element {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased selection:bg-cyan-500 selection:text-black">
        {/* Persistent Research Prototype Banner */}
        <div className="w-full bg-amber-500/10 border-b border-amber-500/25 px-4 py-2 text-center text-xs font-mono text-amber-300 tracking-wide">
          Research prototype — measurements are pilot, not human-calibrated.
        </div>

        {/* Navigation Bar */}
        <header className="sticky top-0 z-50 w-full border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-md">
          <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
            {/* Brand Logo & Mission */}
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 text-black shadow-lg shadow-cyan-500/20">
                <Camera className="h-5 w-5 text-black stroke-[2.2]" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-lg font-bold tracking-tight text-white">
                    Aes<span className="text-cyan-400">Fix</span>
                  </span>
                  <span className="rounded-full bg-cyan-950/80 px-2 py-0.5 font-mono text-[10px] font-semibold text-cyan-400 border border-cyan-800/50">
                    Pilot • Closed Loop
                  </span>
                </div>
                <p className="hidden text-[11px] text-zinc-400 sm:block">
                  Aesthetic Diagnostics &amp; Measured Gain
                </p>
              </div>
            </div>

            {/* Navigation Links */}
            <nav className="hidden items-center gap-6 md:flex">
              <a
                href="/"
                className="flex items-center gap-1.5 text-xs font-medium text-zinc-300 transition-colors hover:text-cyan-400"
              >
                <Sparkles className="h-3.5 w-3.5" />
                Studio
              </a>
              <a
                href="/research"
                className="flex items-center gap-1.5 text-xs font-medium text-zinc-400 transition-colors hover:text-cyan-400"
              >
                <BookOpen className="h-3.5 w-3.5" />
                Research Notes
              </a>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1.5 text-xs font-medium text-zinc-400 transition-colors hover:text-cyan-400"
              >
                <Terminal className="h-3.5 w-3.5" />
                API Docs
              </a>
            </nav>

            {/* Gateway Status Badge & GitHub */}
            <div className="flex items-center gap-3">
              <HealthPill />
              <a
                href="https://github.com/DeepanshuOP/CritiQ"
                target="_blank"
                rel="noreferrer"
                className="rounded-lg p-2 text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-white"
                aria-label="GitHub Repository"
              >
                <ExternalLink className="h-4 w-4" />
              </a>
            </div>
          </div>
        </header>

        {/* Page View Body */}
        <main className="min-h-[calc(100vh-4rem)] studio-grid">{children}</main>
      </body>
    </html>
  );
}
