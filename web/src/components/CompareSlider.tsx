"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { Sparkles, SlidersHorizontal } from "lucide-react";

// Why typed interface is strictly declared:
// Enforces TypeScript strict mode without any type assertions.
export interface CompareSliderProps {
  originalImage: string;
  editedImage: string;
  verifiedGain?: number | null;
  aspectFixed?: string;
  fixName?: string;
}

export function CompareSlider({
  originalImage,
  editedImage,
  verifiedGain = 0.16,
  aspectFixed = "Lighting & Dynamic Range",
  fixName = "Exposure & Tone Curve Lift",
}: CompareSliderProps): React.JSX.Element {
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Why pointer coordinates are normalized relative to container bounds:
  // Ensures touch and mouse interactions track accurately on both mobile viewports and desktop monitors.
  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const clampedPct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(clampedPct);
  }, []);

  const handleTouchMove = useCallback(
    (e: TouchEvent) => {
      if (!isDragging) return;
      handleMove(e.touches[0].clientX);
    },
    [isDragging, handleMove]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;
      handleMove(e.clientX);
    },
    [isDragging, handleMove]
  );

  const handlePointerUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handlePointerUp);
      window.addEventListener("touchmove", handleTouchMove);
      window.addEventListener("touchend", handlePointerUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handlePointerUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handlePointerUp);
    };
  }, [isDragging, handleMouseMove, handleTouchMove, handlePointerUp]);

  return (
    <div className="flex flex-col gap-3">
      {/* Interactive Canvas Container */}
      <div
        ref={containerRef}
        className="relative aspect-[4/3] w-full select-none overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900 shadow-2xl"
        onMouseDown={() => setIsDragging(true)}
        onTouchStart={() => setIsDragging(true)}
      >
        {/* Edited (After) Image Layer - Full Canvas */}
        <img
          src={editedImage}
          alt="AesFix Tier-A Retouched"
          className="absolute inset-0 h-full w-full object-cover"
          draggable={false}
        />

        {/* Original (Before) Image Layer - Clipped by Slider Position */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ width: `${sliderPosition}%` }}
        >
          <img
            src={originalImage}
            alt="Original Input"
            className="absolute inset-0 h-full w-full object-cover"
            style={{
              width: containerRef.current
                ? `${containerRef.current.clientWidth}px`
                : "100%",
              maxWidth: "none",
            }}
            draggable={false}
          />
        </div>

        {/* Vertical Divider Line with Center Drag Handle */}
        <div
          className="pointer-events-none absolute bottom-0 top-0 w-0.5 bg-white shadow-[0_0_12px_rgba(255,255,255,0.8)]"
          style={{ left: `${sliderPosition}%` }}
        >
          <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 flex h-8 w-8 items-center justify-center rounded-full border-2 border-white bg-zinc-950/90 text-white shadow-xl backdrop-blur-sm">
            <SlidersHorizontal className="h-4 w-4" />
          </div>
        </div>

        {/* Before / After Badges */}
        <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/60 px-2 py-1 text-[11px] font-mono font-medium text-zinc-300 backdrop-blur-sm border border-white/10">
          BEFORE (Input)
        </div>
        <div className="pointer-events-none absolute right-3 top-3 rounded-md bg-cyan-950/80 px-2 py-1 text-[11px] font-mono font-medium text-cyan-300 backdrop-blur-sm border border-cyan-500/30">
          AFTER (Verified Fix)
        </div>

        {/* Verified Gain Badge Overlay */}
        {verifiedGain !== null && verifiedGain !== undefined && (
          <div className="pointer-events-none absolute bottom-3 left-3 flex items-center gap-1.5 rounded-lg bg-emerald-950/90 px-3 py-1.5 text-xs font-mono font-bold text-emerald-300 border border-emerald-500/40 backdrop-blur-md shadow-lg">
            <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
            <span>AG@1: +{verifiedGain.toFixed(2)} Gain Verified</span>
          </div>
        )}
      </div>

      {/* Fix Telemetry Footer */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-xs">
        <span className="text-zinc-400">
          Targeted Aspect: <strong className="text-zinc-200">{aspectFixed}</strong>
        </span>
        <span className="font-mono text-cyan-400">
          Op: <span className="text-zinc-300">{fixName}</span>
        </span>
      </div>
    </div>
  );
}
