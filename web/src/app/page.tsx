"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  Sparkles,
  ShieldCheck,
  ShieldAlert,
  AlertCircle,
  FileCode,
  CheckCircle2,
  XCircle,
  ImageIcon,
} from "lucide-react";
import { CompareSlider } from "@/components/CompareSlider";
import { HealthPill } from "@/components/HealthPill";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface GoalOption {
  id: string;
  label: string;
  description: string;
}

const GOAL_OPTIONS: GoalOption[] = [
  { id: "general", label: "General", description: "Balanced benchmark weighting" },
  { id: "portrait", label: "Portrait", description: "Subject clarity & flattering skin tones" },
  { id: "listing", label: "E-Commerce", description: "Clean lighting & high technical acutance" },
  { id: "thumbnail", label: "Thumbnail", description: "Vibrant contrast & punchy saturation" },
];

interface MeasuredFixPreview {
  fix_id: string;
  op: string;
  params: Record<string, unknown>;
  expected_aspect: string;
  reasoning: string;
  baseline_score: number;
  post_edit_score: number;
  aesthetic_gain: number;
  guard_passed: boolean;
  guard_drift: number;
  gain_badge: string;
  xmp_xml: string;
  execution_time_ms: number;
  delta: number;
  epsilon: number;
  judge_id: string;
  verdict: string;
  direction: number;
  claimed_aspect: string;
  guard_backend?: string;
}

interface JobResult {
  job_id: string;
  aspect_scores: Record<string, number>;
  critique_rationale: Record<string, string>;
  measured_fixes: MeasuredFixPreview[];
  verified_fixes?: MeasuredFixPreview[];
  best_fix: MeasuredFixPreview | null;
  uncertainty_badge: string;
  total_latency_seconds: number;
  overall_score?: number | null;
  grade_label?: string | null;
  anfis_rulebook_summary?: string | null;
  top_fired_rules?: string[];
  image_verdict: string;
  noise_floor: number;
  judge_id: string;
}

interface JobStatusPayload {
  job_id: string;
  status: string;
  result?: JobResult;
  error?: string | null;
}

interface SSEEventData {
  job_id: string;
  stage: string;
  progress_percent: number;
  message: string;
}

export default function StudioPage(): React.JSX.Element {
  const [selectedGoal, setSelectedGoal] = useState<string>("general");
  const [userCritique, setUserCritique] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [progressStage, setProgressStage] = useState<string>("");
  const [progressMessage, setProgressMessage] = useState<string>("");
  const [jobResult, setJobResult] = useState<JobResult | null>(null);
  const [selectedFixIndex, setSelectedFixIndex] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setLocalPreviewUrl(URL.createObjectURL(file));
      setJobResult(null);
      setJobId(null);
      setErrorMessage(null);
    }
  };

  const handleLoadSample = async (sampleName: string): Promise<void> => {
    try {
      setIsProcessing(true);
      setErrorMessage(null);
      const res = await fetch(`/samples/${sampleName}`);
      if (!res.ok) throw new Error(`Could not load sample ${sampleName}`);
      const blob = await res.blob();
      const file = new File([blob], sampleName, { type: "image/jpeg" });
      setSelectedFile(file);
      setLocalPreviewUrl(URL.createObjectURL(file));
      setJobResult(null);
      setJobId(null);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Error loading sample.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setLocalPreviewUrl(URL.createObjectURL(file));
      setJobResult(null);
      setJobId(null);
      setErrorMessage(null);
    }
  };

  const fetchJobResult = async (id: string): Promise<void> => {
    try {
      const res = await fetch(`${API_URL}/api/v1/pipeline/jobs/${id}`, {
        headers: { Accept: "application/json" },
      });
      if (!res.ok) {
        throw new Error(`Failed to fetch job: ${res.statusText}`);
      }
      const data: JobStatusPayload = await res.json();
      if (data.result) {
        setJobResult(data.result);
        setSelectedFixIndex(0);
      } else if (data.error) {
        setErrorMessage(data.error);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Error retrieving diagnosis result.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSubmit = async (): Promise<void> => {
    if (!selectedFile) {
      setErrorMessage("Please select or drop an image file first.");
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);
    setProgressPercent(10);
    setProgressStage("QUEUED");
    setProgressMessage("Uploading photo to diagnosis pipeline...");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      if (selectedGoal) formData.append("goal", selectedGoal);
      if (userCritique) formData.append("user_critique", userCritique);

      const submitRes = await fetch(`${API_URL}/api/v1/pipeline/submit`, {
        method: "POST",
        headers: {
          Authorization: "Bearer mock-user-demo",
        },
        body: formData,
      });

      if (!submitRes.ok) {
        const errJson = await submitRes.json().catch(() => null);
        throw new Error(errJson?.detail || `Upload failed with status ${submitRes.status}`);
      }

      const submitData = (await submitRes.json()) as { job_id: string; stream_url: string };
      const currentJobId = submitData.job_id;
      setJobId(currentJobId);

      // Connect to Server-Sent Events (SSE)
      const sseUrl = `${API_URL}${submitData.stream_url}`;
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const es = new EventSource(sseUrl);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as SSEEventData;
          setProgressPercent(parsed.progress_percent);
          setProgressStage(parsed.stage);
          setProgressMessage(parsed.message);

          if (parsed.stage === "COMPLETE" || parsed.progress_percent >= 100) {
            es.close();
            fetchJobResult(currentJobId);
          }
        } catch {
          // Non-JSON event ignored
        }
      };

      es.onerror = () => {
        es.close();
        // Fallback polling once closed
        fetchJobResult(currentJobId);
      };
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to execute pipeline.");
      setIsProcessing(false);
    }
  };

  const currentFixes = jobResult?.measured_fixes ?? jobResult?.verified_fixes ?? [];
  const activeFix = currentFixes[selectedFixIndex] ?? null;
  const originalPreviewUrl = jobId
    ? `${API_URL}/api/v1/pipeline/jobs/${jobId}/preview/original`
    : localPreviewUrl || "";
  const fixedPreviewUrl = jobId && activeFix
    ? `${API_URL}/api/v1/pipeline/jobs/${jobId}/preview/${selectedFixIndex}`
    : originalPreviewUrl;

  const isLeaveIt =
    jobResult &&
    (jobResult.image_verdict.startsWith("Leave it") ||
      !jobResult.best_fix ||
      currentFixes.length === 0 ||
      currentFixes.every((f) => f.verdict !== "measurable improvement" && f.verdict !== "measured improvement above noise floor"));

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Studio Header */}
      <div className="mb-8 flex flex-col items-start justify-between gap-4 border-b border-zinc-800 pb-6 md:flex-row md:items-end">
        <div>
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <div className="inline-flex items-center gap-2 rounded-full bg-cyan-950/60 px-3 py-1 text-xs font-mono font-medium text-cyan-400 border border-cyan-800/40">
              <Sparkles className="h-3 w-3" />
              <span>Closed-Loop Photography AI</span>
            </div>
            <HealthPill />
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            AesFix Studio Workspace
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-400">
            Diagnose photographic flaws across aesthetic dimensions, execute deterministic edits, and measure improvement above empirical noise floor ε with Lightroom XMP export.
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

      {/* Error Alert */}
      {errorMessage && (
        <div className="mb-6 flex items-center gap-2 rounded-xl bg-red-950/80 p-4 text-xs text-red-200 border border-red-800/50">
          <AlertCircle className="h-4 w-4 text-red-400 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Main Studio 2-Column Grid */}
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Left Column: Upload, Progress & Aspect Diagnosis (5 Cols) */}
        <div className="flex flex-col gap-6 lg:col-span-5">
          {/* Photo Dropzone */}
          <div className="glass-card rounded-2xl p-5">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400">
                1. Input Photo
              </span>
              {selectedFile && (
                <span className="text-xs text-cyan-400 font-mono truncate max-w-[200px]">
                  {selectedFile.name}
                </span>
              )}
            </div>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
            />

            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className="group relative flex aspect-[16/9] w-full cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-zinc-700 bg-zinc-950/60 p-4 transition-all hover:border-cyan-500/60 hover:bg-zinc-900/40"
            >
              {localPreviewUrl ? (
                <img
                  src={localPreviewUrl}
                  alt="Selected Preview"
                  className="absolute inset-0 h-full w-full object-contain rounded-xl p-1"
                />
              ) : (
                <>
                  <Upload className="mb-2 h-7 w-7 text-zinc-400 transition-transform group-hover:scale-110 group-hover:text-cyan-400" />
                  <p className="text-xs font-medium text-zinc-300">
                    Drop your photo here or <span className="text-cyan-400 underline">browse</span>
                  </p>
                  <p className="text-[10px] text-zinc-500 mt-1">JPEG, PNG up to 20MB</p>
                </>
              )}
            </div>

            {/* Quick Pilot Sample Selector */}
            <div className="mt-3">
              <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider block mb-1.5">
                Or Load A Benchmark Pilot Sample:
              </span>
              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => handleLoadSample("test.jpg")}
                  className="rounded-lg bg-zinc-900 px-2 py-1.5 text-[11px] font-mono text-zinc-300 border border-zinc-800 hover:border-cyan-500 hover:text-white transition-all text-center"
                >
                  Reference
                </button>
                <button
                  type="button"
                  onClick={() => handleLoadSample("dark.jpg")}
                  className="rounded-lg bg-zinc-900 px-2 py-1.5 text-[11px] font-mono text-zinc-300 border border-zinc-800 hover:border-amber-500 hover:text-white transition-all text-center"
                >
                  Underexposed
                </button>
                <button
                  type="button"
                  onClick={() => handleLoadSample("blur.jpg")}
                  className="rounded-lg bg-zinc-900 px-2 py-1.5 text-[11px] font-mono text-zinc-300 border border-zinc-800 hover:border-pink-500 hover:text-white transition-all text-center"
                >
                  Motion Blur
                </button>
              </div>
            </div>

            {/* User Critique Input */}
            <div className="mt-3">
              <label className="text-[11px] font-mono text-zinc-400">
                Optional Photographer Note / Critique Target:
              </label>
              <input
                type="text"
                value={userCritique}
                onChange={(e) => setUserCritique(e.target.value)}
                placeholder="e.g. lift shadow detail, correct colour cast..."
                className="mt-1 w-full rounded-lg bg-zinc-900/90 px-3 py-2 text-xs text-zinc-200 border border-zinc-800 focus:outline-none focus:border-cyan-500"
              />
            </div>

            <button
              type="button"
              onClick={handleSubmit}
              disabled={isProcessing || !selectedFile}
              className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-4 py-3 text-xs font-bold text-black shadow-lg shadow-cyan-500/20 transition-all hover:brightness-110 active:scale-[0.99] disabled:opacity-50"
            >
              <Sparkles className="h-4 w-4" />
              {isProcessing ? "Running Closed Loop..." : "Run Grounded Diagnosis & Fix"}
            </button>
          </div>

          {/* SSE Progress Indicator */}
          {isProcessing && (
            <div className="glass-card rounded-2xl p-5 border border-cyan-800/40">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-bold uppercase text-cyan-400">
                  {progressStage || "PROCESSING"}
                </span>
                <span className="font-mono text-xs text-zinc-300">{progressPercent}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <p className="mt-2 text-xs text-zinc-400 font-mono">{progressMessage}</p>
            </div>
          )}

          {/* Measured Aspect Scores from API */}
          {jobResult && (
            <div className="glass-card rounded-2xl p-5">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400">
                  2. Measured Aspect Scores
                </span>
                <span className="rounded-full bg-zinc-900 px-2 py-0.5 font-mono text-[10px] text-zinc-400 border border-zinc-800">
                  Noise Floor ε: {jobResult.noise_floor.toFixed(4)}
                </span>
              </div>

              <div className="flex flex-col gap-3">
                {Object.entries(jobResult.aspect_scores).map(([aspect, score]) => (
                  <div key={aspect} className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="capitalize text-zinc-300">
                        {aspect.replace(/_/g, " ")}
                      </span>
                      <span className="font-mono text-xs font-semibold text-zinc-400">
                        {score.toFixed(4)}
                      </span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                      <div
                        className="h-full rounded-full bg-cyan-500 transition-all duration-500"
                        style={{ width: `${Math.max(0, Math.min(100, score * 100))}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {jobResult.uncertainty_badge && (
                <div className="mt-4 rounded-xl bg-zinc-950/80 p-3 border border-zinc-800">
                  <span className="font-mono text-[11px] text-zinc-300">
                    {jobResult.uncertainty_badge}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Compare Canvas, Per-Fix Cards & XMP (7 Cols) */}
        <div className="flex flex-col gap-6 lg:col-span-7">
          {/* Main Inspection Canvas */}
          <div className="glass-card rounded-2xl p-5">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400">
                3. Inspection &amp; Edit Comparison
              </span>
              {activeFix && (
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-mono text-[10px] font-bold border ${
                    activeFix.guard_passed
                      ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/40"
                      : "bg-red-950/80 text-red-400 border-red-800/40"
                  }`}
                >
                  {activeFix.guard_passed ? (
                    <ShieldCheck className="h-3 w-3" />
                  ) : (
                    <ShieldAlert className="h-3 w-3" />
                  )}
                  Drift: {activeFix.guard_drift.toFixed(4)} ({activeFix.guard_passed ? "PASS" : "VETOED"}) • Guard: {activeFix.guard_backend || "SSIM + rgb_layout_stat"}
                </span>
              )}
            </div>

            {/* Slider or Empty State */}
            {originalPreviewUrl ? (
              <CompareSlider
                originalImage={originalPreviewUrl}
                editedImage={fixedPreviewUrl}
                gain={activeFix ? activeFix.delta : null}
                noiseFloor={jobResult?.noise_floor ?? null}
                aspectFixed={activeFix ? `${activeFix.expected_aspect} (${activeFix.direction > 0 ? "+1" : "-1"})` : "None"}
                fixName={activeFix ? activeFix.op : "Original"}
                verdict={activeFix?.verdict}
              />
            ) : (
              <div className="flex aspect-[4/3] w-full flex-col items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-950/80 text-zinc-500">
                <ImageIcon className="mb-2 h-10 w-10 text-zinc-600" />
                <p className="text-xs font-mono">Upload a photo to view before/after comparison</p>
              </div>
            )}

            {/* XMP Download Action Bar */}
            {jobId && activeFix && (
              <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
                <a
                  href={`${API_URL}/api/v1/pipeline/jobs/${jobId}/xmp?fix_index=${selectedFixIndex}`}
                  download={`aesfix_${jobId}_fix${selectedFixIndex + 1}.xmp`}
                  className="flex items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-2.5 text-xs font-semibold text-zinc-200 border border-zinc-800 transition-all hover:bg-zinc-800 hover:text-white"
                >
                  <FileCode className="h-4 w-4 text-cyan-400" />
                  <span>Download Lightroom .XMP (Fix #{selectedFixIndex + 1})</span>
                </a>

                <a
                  href={fixedPreviewUrl}
                  download={`aesfix_retouched_${jobId}.png`}
                  className="flex items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-2.5 text-xs font-semibold text-zinc-200 border border-zinc-800 transition-all hover:bg-zinc-800 hover:text-white"
                >
                  <Sparkles className="h-4 w-4 text-emerald-400" />
                  <span>Save Retouched Preview</span>
                </a>
              </div>
            )}
          </div>

          {/* "Leave It" State Card */}
          {isLeaveIt && (
            <div className="rounded-2xl border border-amber-500/40 bg-amber-950/20 p-5 backdrop-blur-md">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-mono font-bold text-amber-300 uppercase tracking-wide">
                    Verdict: Leave it alone
                  </h3>
                  <p className="mt-1 text-xs text-zinc-300 leading-relaxed">
                    No candidate photographic edit produced measurable improvement above the noise floor
                    (ε = {jobResult?.noise_floor.toFixed(4)}) without identity guard drift.
                    The photo is best left in its current state.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Per-Fix Cards */}
          {currentFixes.length > 0 && (
            <div className="glass-card rounded-2xl p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400">
                  4. Candidate Edits ({currentFixes.length})
                </span>
                <span className="text-[11px] font-mono text-zinc-500">
                  Click card to inspect preview
                </span>
              </div>

              <div className="flex flex-col gap-3">
                {currentFixes.map((fix, idx) => {
                  const isSelected = selectedFixIndex === idx;
                  const isImprovement = fix.verdict === "measurable improvement" || fix.verdict === "measured improvement above noise floor";
                  const isGuardRejected = fix.verdict === "rejected by guard";

                  return (
                    <div
                      key={fix.fix_id || idx}
                      onClick={() => setSelectedFixIndex(idx)}
                      className={`cursor-pointer rounded-xl p-4 transition-all border ${
                        isSelected
                          ? "bg-zinc-900/90 border-cyan-500/70 shadow-md shadow-cyan-500/10"
                          : "bg-zinc-950/60 border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/40"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs font-bold text-white">
                              #{idx + 1} {fix.op}
                            </span>
                            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300">
                              {fix.expected_aspect} ({fix.direction > 0 ? "+1" : "-1"})
                            </span>
                          </div>
                          <p className="mt-1 text-xs text-zinc-400 leading-snug">{fix.reasoning}</p>
                        </div>

                        <div className="text-right shrink-0">
                          <span
                            className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-mono text-[10px] font-bold border ${
                              isImprovement
                                ? "bg-emerald-950/80 text-emerald-300 border-emerald-800/40"
                                : isGuardRejected
                                ? "bg-red-950/80 text-red-300 border-red-800/40"
                                : "bg-zinc-900 text-zinc-400 border-zinc-800"
                            }`}
                          >
                            {isImprovement ? (
                              <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                            ) : isGuardRejected ? (
                              <ShieldAlert className="h-3 w-3 text-red-400" />
                            ) : (
                              <XCircle className="h-3 w-3 text-zinc-500" />
                            )}
                            {fix.verdict}
                          </span>
                        </div>
                      </div>

                      {/* Metrics strip: Δ, ε, drift, params */}
                      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-zinc-800/60 font-mono text-[11px]">
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">Δ (Gain)</span>
                          <span className={fix.delta > 0 ? "text-emerald-400 font-bold" : "text-zinc-400"}>
                            {fix.delta > 0 ? `+${fix.delta.toFixed(4)}` : fix.delta.toFixed(4)}
                          </span>
                        </div>
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">ε (Noise)</span>
                          <span className="text-zinc-300">{fix.epsilon.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">Guard ({fix.guard_backend || "SSIM + rgb_layout_stat"})</span>
                          <span className={fix.guard_passed ? "text-zinc-300" : "text-red-400"}>
                            {fix.guard_drift.toFixed(4)}
                          </span>
                        </div>
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">Parameters</span>
                          <span className="text-zinc-300 truncate block" title={JSON.stringify(fix.params)}>
                            {Object.entries(fix.params)
                              .map(([k, v]) => `${k}:${v}`)
                              .join(" ")}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* How It Works Section */}
      <section id="how-it-works" className="mt-16 border-t border-zinc-800 pt-12">
        <div className="mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-cyan-950/60 px-3 py-1 text-xs font-mono font-medium text-cyan-400 border border-cyan-800/40 mb-2">
            <span>Closed-Loop Architecture</span>
          </div>
          <h2 className="text-2xl font-extrabold tracking-tight text-white sm:text-3xl">
            How AesFix Works
          </h2>
          <p className="mt-2 text-sm text-zinc-400 max-w-3xl leading-relaxed">
            Every candidate photographic edit is generated deterministically, gated against structural drift, and evaluated against the empirical noise floor before recommending adjustments.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="glass-card rounded-2xl p-5 border border-zinc-800">
            <span className="font-mono text-xs font-bold text-cyan-400 block mb-2">STEP 1</span>
            <h3 className="text-base font-bold text-white mb-2">Aspect Criticism</h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Diagnoses composition, lighting, color, clarity, and technical precision to identify true photographic bottlenecks.
            </p>
          </div>

          <div className="glass-card rounded-2xl p-5 border border-zinc-800">
            <span className="font-mono text-xs font-bold text-amber-400 block mb-2">STEP 2</span>
            <h3 className="text-base font-bold text-white mb-2">Deterministic Edits</h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Executes continuous, parameterized operations (exposure shoulder curves, mid-grey contrast pivot, white balance) reproducible in Lightroom.
            </p>
          </div>

          <div className="glass-card rounded-2xl p-5 border border-zinc-800">
            <span className="font-mono text-xs font-bold text-emerald-400 block mb-2">STEP 3</span>
            <h3 className="text-base font-bold text-white mb-2">Identity Guard</h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Monitors structural similarity and layout embeddings to veto adjustments that compromise photographic identity or induce clipping.
            </p>
          </div>

          <div className="glass-card rounded-2xl p-5 border border-zinc-800">
            <span className="font-mono text-xs font-bold text-purple-400 block mb-2">STEP 4</span>
            <h3 className="text-base font-bold text-white mb-2">Noise Floor Gating</h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Calculates empirical perturbation noise floor ε. Fixes are only approved when measured aesthetic gain Δ strictly exceeds ε; otherwise advising to leave it alone.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
