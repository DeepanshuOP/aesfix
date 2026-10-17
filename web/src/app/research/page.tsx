"use client";

import React from "react";
import { BookOpen, ShieldCheck, Sparkles, SlidersHorizontal, Scale, CheckCircle2 } from "lucide-react";

export default function ResearchPage(): React.JSX.Element {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="border-b border-zinc-800 pb-8 mb-8">
        <div className="inline-flex items-center gap-2 rounded-full bg-cyan-950/60 px-3 py-1 text-xs font-mono font-medium text-cyan-400 border border-cyan-800/40 mb-3">
          <BookOpen className="h-3.5 w-3.5" />
          <span>Research Methodology &amp; Closed-Loop Thesis</span>
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
          Aesthetic Diagnostics via Measured Edit Validation
        </h1>
        <p className="mt-3 text-sm text-zinc-400 leading-relaxed">
          Why contemporary retouching assistants fail without falsifiable edit validation, and how closed-loop measurement establishes whether photographic adjustments genuinely improve visual quality.
        </p>
      </div>

      {/* Sections */}
      <div className="flex flex-col gap-10 text-zinc-300 text-sm leading-relaxed">
        {/* Section 1 */}
        <section className="glass-card rounded-2xl p-6 border border-zinc-800">
          <div className="flex items-center gap-2.5 mb-3">
            <Scale className="h-5 w-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">
              The Critique Without Measurement Problem
            </h2>
          </div>
          <p className="text-zinc-300">
            Automated image retouching agents commonly suggest adjustments such as brightening shadows, boosting saturation, or increasing contrast. However, without a closed validation loop, these suggestions frequently degrade photographic fidelity through clipped highlights, crushed dynamic range, or unnatural color shifts.
          </p>
          <p className="mt-3 text-zinc-400">
            Under Goodhart&apos;s Law, optimizing against a single uncalibrated reward model encourages reward gaming, where an agent introduces artifacts that score high on specific aesthetic heuristics while compromising overall photographic merit.
          </p>
        </section>

        {/* Section 2 */}
        <section className="glass-card rounded-2xl p-6 border border-zinc-800">
          <div className="flex items-center gap-2.5 mb-3">
            <SlidersHorizontal className="h-5 w-5 text-amber-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">
              Deterministic Parameterized Retouching
            </h2>
          </div>
          <p className="text-zinc-300">
            Rather than hallucinating synthetic pixels through unconstrained diffusion models, AesFix executes deterministic adjustments across photographic parameters:
          </p>
          <ul className="mt-3 list-disc list-inside space-y-1.5 text-zinc-400 font-mono text-xs">
            <li>Linear-light exposure adjustments with shoulder compression to avoid harsh clipping</li>
            <li>Mid-grey anchored contrast pivoting to preserve shadow and highlight balance</li>
            <li>White balance and temperature adjustments using neutral chromaticity pivots</li>
            <li>Smooth continuous unsharp masking and noise reduction filters</li>
          </ul>
          <p className="mt-3 text-zinc-300">
            Because every adjustment is mathematically deterministic, each applied fix can be exported directly into Adobe Lightroom compatible sidecars for non-destructive inspection.
          </p>
        </section>

        {/* Section 3 */}
        <section className="glass-card rounded-2xl p-6 border border-zinc-800">
          <div className="flex items-center gap-2.5 mb-3">
            <ShieldCheck className="h-5 w-5 text-emerald-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">
              Identity Guard &amp; Structural Drift Gating
            </h2>
          </div>
          <p className="text-zinc-300">
            Before any candidate edit can be evaluated for aesthetic gain, it must pass through an Identity Guard filter. The guard monitors:
          </p>
          <ul className="mt-3 list-disc list-inside space-y-1.5 text-zinc-400 font-mono text-xs">
            <li>Structural similarity across spatial luminance distributions</li>
            <li>Color layout consistency to prevent catastrophic chromatic warping</li>
            <li>Over-processing checks against extreme tonal clipping</li>
          </ul>
          <p className="mt-3 text-zinc-300">
            If an edit introduces noticeable structural drift or compromises photographic identity, the edit is vetoed regardless of predicted aesthetic reward.
          </p>
        </section>

        {/* Section 4 */}
        <section className="glass-card rounded-2xl p-6 border border-zinc-800">
          <div className="flex items-center gap-2.5 mb-3">
            <Sparkles className="h-5 w-5 text-purple-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">
              Empirical Noise Floor Formulation
            </h2>
          </div>
          <p className="text-zinc-300">
            A central defect of existing aesthetic scorers is treating minute score variations as proof of improvement. Minor perturbations such as re-saving a file with compression or minute resizing alter model predictions even when no genuine aesthetic change occurred.
          </p>
          <p className="mt-3 text-zinc-300">
            To guard against false positives, an empirical noise floor is computed directly on the source image across standard neutral perturbations:
          </p>
          <div className="mt-3 rounded-xl bg-zinc-950 p-4 border border-zinc-800 font-mono text-xs text-cyan-300">
            noise_floor = maximum score variance observed across neutral compression and scaling transformations
          </div>
          <p className="mt-3 text-zinc-400">
            An edit is only classified as a measured improvement when the measured aesthetic gain strictly exceeds this empirical noise floor threshold.
          </p>
        </section>

        {/* Section 5 */}
        <section className="glass-card rounded-2xl p-6 border border-zinc-800">
          <div className="flex items-center gap-2.5 mb-3">
            <CheckCircle2 className="h-5 w-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">
              The Honesty of the &quot;Leave It Alone&quot; Verdict
            </h2>
          </div>
          <p className="text-zinc-300">
            A trustworthy retouching assistant must know when not to retouch. When an image is already well-exposed and balanced, forcing automated fixes introduces degradation.
          </p>
          <p className="mt-3 text-zinc-300">
            When none of the proposed candidate fixes produce a measured improvement above the noise floor without guard drift, the system emits an explicit verdict advising the photographer to leave the photograph untouched.
          </p>
        </section>
      </div>
    </div>
  );
}
