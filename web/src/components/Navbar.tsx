"use client";

import React, { useState } from "react";
import { Camera, Sparkles, BookOpen, ExternalLink, Terminal, Menu, X, HelpCircle, Sun, Moon } from "lucide-react";
import { HealthPill } from "@/components/HealthPill";

export function Navbar(): React.JSX.Element {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);
  const [isDark, setIsDark] = useState<boolean>(true);

  const toggleTheme = (): void => {
    const nextDark = !isDark;
    setIsDark(nextDark);
    if (typeof document !== "undefined") {
      if (nextDark) {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-zinc-800/80 bg-zinc-950/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand Logo */}
        <a href="/" className="flex items-center gap-3 group">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 text-black shadow-lg shadow-cyan-500/20 transition-transform group-hover:scale-105">
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
        </a>

        {/* Desktop Navigation */}
        <nav className="hidden items-center gap-6 md:flex">
          <a
            href="/"
            className="flex items-center gap-1.5 text-xs font-medium text-zinc-300 transition-colors hover:text-cyan-400"
          >
            <Sparkles className="h-3.5 w-3.5" />
            Studio
          </a>
          <a
            href="/#how-it-works"
            className="flex items-center gap-1.5 text-xs font-medium text-zinc-400 transition-colors hover:text-cyan-400"
          >
            <HelpCircle className="h-3.5 w-3.5" />
            How It Works
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

        {/* Right side controls: Health pill, Theme toggle, Mobile Menu button */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:block">
            <HealthPill />
          </div>

          {/* Theme toggle */}
          <button
            type="button"
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="rounded-lg p-2 text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-white"
          >
            {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>

          <a
            href="https://github.com/DeepanshuOP/CritiQ"
            target="_blank"
            rel="noreferrer"
            className="hidden sm:flex rounded-lg p-2 text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-white"
            aria-label="GitHub Repository"
          >
            <ExternalLink className="h-4 w-4" />
          </a>

          {/* Mobile hamburger menu toggle */}
          <button
            type="button"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden rounded-lg p-2 text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-white"
            aria-label="Toggle mobile menu"
            aria-expanded={isMobileMenuOpen}
          >
            {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Navigation */}
      {isMobileMenuOpen && (
        <div className="border-t border-zinc-800 bg-zinc-950/95 px-4 py-4 md:hidden backdrop-blur-lg">
          <nav className="flex flex-col gap-3">
            <div className="mb-2 block sm:hidden">
              <HealthPill />
            </div>
            <a
              href="/"
              onClick={() => setIsMobileMenuOpen(false)}
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-900 hover:text-cyan-400"
            >
              <Sparkles className="h-4 w-4" />
              Studio
            </a>
            <a
              href="/#how-it-works"
              onClick={() => setIsMobileMenuOpen(false)}
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 hover:bg-zinc-900 hover:text-cyan-400"
            >
              <HelpCircle className="h-4 w-4" />
              How It Works
            </a>
            <a
              href="/research"
              onClick={() => setIsMobileMenuOpen(false)}
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 hover:bg-zinc-900 hover:text-cyan-400"
            >
              <BookOpen className="h-4 w-4" />
              Research Notes
            </a>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 hover:bg-zinc-900 hover:text-cyan-400"
            >
              <Terminal className="h-4 w-4" />
              API Docs
            </a>
          </nav>
        </div>
      )}
    </header>
  );
}
