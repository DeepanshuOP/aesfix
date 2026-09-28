"use client";

import React, { useState } from "react";
import {
  Upload,
  Sparkles,
  Download,
  ShieldCheck,
  Zap,
  Sliders,
  ChevronRight,
  TrendingUp,
  AlertCircle,
  FileCode,
} from "lucide-react";
import { CompareSlider } from "@/components/CompareSlider";

// Why goal context options mirror the backend GoalProfile registry:
// Enables users to condition the judge weighting and fix vocabulary to their creative domain.
interface GoalOption {
  id: string;
  label: string;
  description: string;
}

const GOAL_OPTIONS: GoalOption[] = [
  { id: "general", label: "General", description: "Balanced PARA benchmark weighting" },
  { id: "portrait", label: "Portrait", description: "Subject clarity & flattering skin tones" },
  { id: "listing", label: "E-Commerce", description: "Clean lighting & high technical acutance" },
  { id: "thumbnail", label: "Thumbnail", description: "Vibrant contrast & punchy saturation" },
];

interface AspectItem {
  id: string;
  name: string;
  score: number;
  color: string;
  isBottleneck?: boolean;
}

export default function StudioPage(): React.JSX.Element {
  const [selectedGoal, setSelectedGoal] = useState<string>("general");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  // Mock baseline telemetry demonstrating complete closed-loop pipeline output
  const aspects: AspectItem[] = [
    { id: "composition", name: "Composition & Framing", score: 0.76, color: "bg-sky-500" },
    { id: "lighting", name: "Lighting & Dynamic Range", score: 0.48, color: "bg-amber-500", isBottleneck: true },
    { id: "color", name: "Color Harmony & Palettes", score: 0.72, color: "bg-purple-500" },
    { id: "clarity", name: "Subject Separation", score: 0.82, color: "bg-pink-500" },
    { id: "technical", name: "Technical Precision", score: 0.88, color: "bg-emerald-500" },
  ];

  const handleSimulateRun = (): void => {
    setIsProcessing(true);
    setTimeout(() => {
      setIsProcessing(false);
    }, 1200);
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Studio Hero Header */}
      <div className="mb-8 flex flex-col items-start justify-between gap-4 border-b border-zinc-800 pb-6 md:flex-row md:items-end">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full bg-cyan-950/60 px-3 py-1 text-xs font-mono font-medium text-cyan-400 border border-cyan-800/40 mb-3">
            <Zap className="h-3 w-3" />
            <span>Closed-Loop Photography AI • Empty Quadrant Solved</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            AesFix Studio Workspace
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-400">
            Diagnose photographic flaws across 5 dimensions, execute deterministic Tier-A edits in &lt;25ms,
            and mathematically verify aesthetic gain (AG@1) with identity drift gating.
          </p>
        </div>

        {/* Goal Conditioning Selector */}
        <div className="flex flex-col gap-1.5 w-full sm:w-auto">
          <label className="text-[11px] font-mono uppercase tracking-wider text-zinc-400">
            Creative Goal Context
          </label>
          <div className="grid grid-cols-2 gap-1.5 rounded-xl bg-zinc-900 p-1 border border-zinc-800 sm:flex">
            {GOAL_OPTIONS.map((g) => (
              <button
                key={g.id}
                type="button"
                onClick={() => setSelectedGoal(g.id)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  selectedGoal === g.id
                    ? "bg-cyan-500 text-black font-semibold shadow-md shadow-cyan-500/20"
                    : "text-zinc-400 hover:text-white hover:bg-zinc-800/60"
                }`}
              >
                {g.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Studio 2-Column Grid */}
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Left Column: Input, Diagnosis & ANFIS Rulebook Trace (5 Cols) */}
        <div className="flex flex-col gap-6 lg:col-span-5">
          {/* Photo Dropzone / Selector */}
          <div className="glass-card rounded-2xl p-5">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400">
                1. Input Photo
              </span>
              <span className="text-xs text-zinc-500 font-mono">Sample: landscape_01.jpg</span>
            </div>

            <div className="group relative flex aspect-[16/9] w-full cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-zinc-700 bg-zinc-950/60 p-4 transition-all hover:border-cyan-500/60 hover:bg-zinc-900/40">
              <Upload className="mb-2 h-7 w-7 text-zinc-400 transition-transform group-hover:scale-110 group-hover:text-cyan-400" />
              <p className="text-xs font-medium text-zinc-300">
                Drop your photo here or <span className="text-cyan-400 underline">browse</span>
              </p>
              <p className="text-[10px] text-zinc-500 mt-1">JPEG, PNG, DNG, RAW up to 50MB</p>
            </div>

            <button
              type="button"
              onClick={handleSimulateRun}
              disabled={isProcessing}
              className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-4 py-3 text-xs font-bold text-black shadow-lg shadow-cyan-500/20 transition-all hover:brightness-110 active:scale-[0.99] disabled:opacity-50"
            >
              <Sparkles className="h-4 w-4" />
              {isProcessing ? "Executing Closed Loop..." : "Run Grounded Diagnosis & Fix"}
            </button>
          </div>

          {/* 5-Aspect Diagnosis & ANFIS Telemetry */}
          <div className="glass-card rounded-2xl p-5">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400">
                2. Aspect Diagnosis (Takagi-Sugeno ANFIS)
              </span>
              <span className="rounded-full bg-amber-950/80 px-2 py-0.5 font-mono text-[10px] font-bold text-amber-400 border border-amber-800/40">
                Overall: B (0.68)
              </span>
            </div>

            {/* Dimensional Score Bars */}
            <div className="flex flex-col gap-3">
              {aspects.map((a) => (
                <div key={a.id} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5 text-zinc-300">
                      {a.name}
                      {a.isBottleneck && (
                        <span className="inline-flex items-center gap-0.5 rounded bg-red-950/80 px-1.5 py-0.2 font-mono text-[9px] font-bold text-red-400 border border-red-800/40">
                          <AlertCircle className="h-2.5 w-2.5" /> BOTTLENECK
                        </span>
                      )}
                    </span>
                    <span className="font-mono text-xs font-semibold text-zinc-400">
                      {(a.score * 10).toFixed(1)} / 10
                    </span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${a.color}`}
                      style={{ width: `${a.score * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Fired Rulebook Trace */}
            <div className="mt-5 rounded-xl bg-zinc-950/80 p-3.5 border border-zinc-800/80">
              <div className="flex items-center gap-1.5 text-[11px] font-mono font-semibold text-zinc-400 mb-1.5">
                <Sliders className="h-3.5 w-3.5 text-cyan-400" />
                <span>ANFIS Fired Rule Trace (#14):</span>
              </div>
              <p className="font-mono text-[11px] text-zinc-300 leading-relaxed">
                <span className="text-cyan-400 font-bold">IF</span> Lighting is{" "}
                <span className="text-amber-400">Underexposed</span>{" "}
                <span className="text-cyan-400 font-bold">AND</span> Composition is{" "}
                <span className="text-sky-400">Balanced</span>{" "}
                <span className="text-cyan-400 font-bold">THEN</span> Apply{" "}
                <span className="text-emerald-400 font-semibold">Exposure (+0.4EV) &amp; Tone Lift</span>.
              </p>
            </div>
          </div>
        </div>

        {/* Right Column: Verified Fix, Slider & XMP Exporter (7 Cols) */}
        <div className="flex flex-col gap-6 lg:col-span-7">
          {/* Main Inspection Canvas */}
          <div className="glass-card rounded-2xl p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400">
                  3. Verified Fix &amp; Identity Guard
                </span>
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-950/80 px-2 py-0.5 font-mono text-[10px] font-bold text-emerald-400 border border-emerald-800/40">
                  <ShieldCheck className="h-3 w-3" /> Identity Guard Veto: PASSED (Drift 0.08 &lt; 0.35)
                </span>
              </div>
            </div>

            {/* Before / After Interactive Scrub Slider */}
            <CompareSlider
              originalImage="/dark.jpg"
              editedImage="/bright.jpg"
              verifiedGain={0.18}
              aspectFixed="Lighting & Dynamic Range"
              fixName="Tier-A Exposure Lift (+0.35EV) & Shadows (+25)"
            />

            {/* Action Bar: Export Buttons */}
            <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <a
                href="http://localhost:8000/api/v1/export-xmp?fix_name=tone_lift"
                download="aesfix_recipe.xmp"
                className="flex items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-2.5 text-xs font-semibold text-zinc-200 border border-zinc-800 transition-all hover:bg-zinc-800 hover:text-white"
              >
                <FileCode className="h-4 w-4 text-cyan-400" />
                <span>Download Adobe Lightroom .XMP</span>
              </a>

              <button
                type="button"
                className="flex items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-2.5 text-xs font-semibold text-zinc-200 border border-zinc-800 transition-all hover:bg-zinc-800 hover:text-white"
              >
                <Download className="h-4 w-4 text-emerald-400" />
                <span>Download High-Res Fixed JPEG</span>
              </button>
            </div>
          </div>

          {/* Skill Curriculum Coach Preview (Roadmap §25 F4.2) */}
          <div className="glass-card rounded-2xl p-5 border-l-4 border-l-cyan-500">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-cyan-400" />
                <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-white">
                  Skill Curriculum Coach ("Duolingo for Photography")
                </h3>
              </div>
              <span className="font-mono text-[11px] text-zinc-400">Weekly Trajectory</span>
            </div>

            <p className="text-xs text-zinc-300 leading-relaxed">
              Your composition improved <strong className="text-cyan-400">+14.2%</strong> over your last 6 uploads.
              Your current bottleneck is <strong className="text-amber-400">Lighting &amp; Dynamic Range</strong>.
            </p>

            <div className="mt-3 flex items-center justify-between rounded-xl bg-zinc-950/70 p-3 border border-zinc-800">
              <div className="text-xs">
                <span className="text-zinc-400 font-mono text-[10px] uppercase">Active Drill</span>
                <p className="font-semibold text-zinc-200">Highlight Recovery &amp; Fill Exposure</p>
              </div>
              <button
                type="button"
                className="flex items-center gap-1 rounded-lg bg-zinc-800 px-2.5 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:text-white"
              >
                <span>View 3 Counterfactual Lessons</span>
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
