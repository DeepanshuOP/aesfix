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
  Bot,
  SlidersHorizontal,
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

export interface AgentStepData {
  step_number: number;
  op: string;
  params: Record<string, unknown>;
  aspect: string;
  direction: number;
  rationale: string;
  status: "accepted" | "rejected";
  verdict_reason: string;
  baseline_overall_score: number;
  post_overall_score: number;
  overall_delta: number;
  noise_floor: number;
  claimed_aspect: string;
  claimed_aspect_delta: number;
  guard_drift: number;
  guard_passed: boolean;
  guard_backend: string;
  aspect_scores_before: Record<string, number>;
  aspect_scores_after: Record<string, number>;
  aspect_deltas: Record<string, number>;
  execution_duration_ms: number;
  preview_data_url?: string;
  preview_url?: string;
}

export interface AgentCompleteData {
  job_id: string;
  status: string;
  verdict: "measured improvement" | "leave it";
  initial_overall_score: number;
  final_overall_score: number;
  total_gain: number;
  noise_floor: number;
  total_steps: number;
  accepted_steps_count: number;
  rejected_steps_count: number;
  stacked_xmp: string;
  provider: string;
  model_id: string;
  final_image_preview?: string;
  final_image_url?: string;
  trajectory_url?: string;
}

export default function StudioPage(): React.JSX.Element {
  // Mode selection
  const [isAgentMode, setIsAgentMode] = useState<boolean>(false);

  // Common inputs
  const [selectedGoal, setSelectedGoal] = useState<string>("general");
  const [userCritique, setUserCritique] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);

  // Pipeline mode state
  const [jobId, setJobId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [progressStage, setProgressStage] = useState<string>("");
  const [progressMessage, setProgressMessage] = useState<string>("");
  const [jobResult, setJobResult] = useState<JobResult | null>(null);
  const [selectedFixIndex, setSelectedFixIndex] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Agent mode state
  const [agentSteps, setAgentSteps] = useState<AgentStepData[]>([]);
  const [agentComplete, setAgentComplete] = useState<AgentCompleteData | null>(null);
  const [selectedAgentStepIdx, setSelectedAgentStepIdx] = useState<number | null>(null);
  const [agentProvider, setAgentProvider] = useState<string>("gemini");
  const [agentModelId, setAgentModelId] = useState<string>("gemini-3.8-flash");

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
      setAgentSteps([]);
      setAgentComplete(null);
      setSelectedAgentStepIdx(null);
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
      setAgentSteps([]);
      setAgentComplete(null);
      setSelectedAgentStepIdx(null);
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
      setAgentSteps([]);
      setAgentComplete(null);
      setSelectedAgentStepIdx(null);
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

  const handleRunAgent = async (): Promise<void> => {
    if (!selectedFile) {
      setErrorMessage("Please select or drop an image file first.");
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);
    setAgentSteps([]);
    setAgentComplete(null);
    setSelectedAgentStepIdx(null);
    setProgressPercent(15);
    setProgressStage("AGENT_INIT");
    setProgressMessage("Starting AesFix Closed-Loop Agent...");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      if (selectedGoal) formData.append("user_goal", selectedGoal);
      formData.append("provider", agentProvider);
      formData.append("model_id", agentModelId);

      const response = await fetch(`${API_URL}/api/v1/agent/run`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok || !response.body) {
        const errJson = await response.json().catch(() => null);
        throw new Error(errJson?.detail || `Agent request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";

        for (const chunk of chunks) {
          if (!chunk.trim()) continue;
          const eventMatch = chunk.match(/^event:\s*(.+)$/m);
          const dataMatch = chunk.match(/^data:\s*(.+)$/m);

          if (eventMatch && dataMatch) {
            const evName = eventMatch[1].trim();
            const rawData = dataMatch[1].trim();

            if (evName === "init") {
              const initPayload = JSON.parse(rawData) as {
                job_id: string;
                provider: string;
                model_id: string;
              };
              setJobId(initPayload.job_id);
              setAgentProvider(initPayload.provider);
              setAgentModelId(initPayload.model_id);
              setProgressStage("AGENT_LOOP");
              setProgressPercent(30);
              setProgressMessage(`Model: ${initPayload.model_id} (${initPayload.provider}). Perceiving image...`);
            } else if (evName === "step") {
              const stepPayload = JSON.parse(rawData) as AgentStepData;
              setAgentSteps((prev) => [...prev, stepPayload]);
              const pct = Math.min(95, 30 + stepPayload.step_number * 12);
              setProgressPercent(pct);
              setProgressMessage(
                `Step ${stepPayload.step_number}: ${stepPayload.status.toUpperCase()} ${stepPayload.op} (${stepPayload.aspect})`
              );
            } else if (evName === "complete") {
              const completePayload = JSON.parse(rawData) as AgentCompleteData;
              setAgentComplete(completePayload);
              setProgressPercent(100);
              setProgressStage("COMPLETE");
              setProgressMessage(`Finished: verdict '${completePayload.verdict}'`);
              setIsProcessing(false);
            } else if (evName === "error") {
              const errPayload = JSON.parse(rawData) as { error: string };
              setErrorMessage(errPayload.error || "Agent execution failed.");
              setIsProcessing(false);
            }
          }
        }
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to run agent.");
      setIsProcessing(false);
    }
  };

  const handleRunPipeline = async (): Promise<void> => {
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
        fetchJobResult(currentJobId);
      };
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to execute pipeline.");
      setIsProcessing(false);
    }
  };

  const handleSubmit = (): void => {
    if (isAgentMode) {
      handleRunAgent();
    } else {
      handleRunPipeline();
    }
  };

  // Compute preview URLs for Pipeline Mode
  const currentFixes = jobResult?.measured_fixes ?? jobResult?.verified_fixes ?? [];
  const activeFix = currentFixes[selectedFixIndex] ?? null;
  const originalPreviewUrl = jobId
    ? `${API_URL}/api/v1/pipeline/jobs/${jobId}/preview/original`
    : localPreviewUrl || "";
  const fixedPreviewUrl = jobId && activeFix
    ? `${API_URL}/api/v1/pipeline/jobs/${jobId}/preview/${selectedFixIndex}`
    : originalPreviewUrl;

  const isPipelineLeaveIt =
    jobResult &&
    (jobResult.image_verdict.startsWith("Leave it") ||
      !jobResult.best_fix ||
      currentFixes.length === 0 ||
      currentFixes.every(
        (f) =>
          f.verdict !== "measurable improvement" &&
          f.verdict !== "measured improvement above noise floor"
      ));

  // Compute preview URLs for Agent Mode
  const activeAgentStep =
    selectedAgentStepIdx !== null && agentSteps[selectedAgentStepIdx]
      ? agentSteps[selectedAgentStepIdx]
      : null;

  const agentDisplayImage = activeAgentStep?.preview_data_url
    ? activeAgentStep.preview_data_url
    : agentComplete?.final_image_preview
    ? agentComplete.final_image_preview
    : localPreviewUrl || "";

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Studio Header & Operational Banner */}
      <div className="mb-8 flex flex-col items-start justify-between gap-4 border-b border-zinc-800 pb-6 md:flex-row md:items-end">
        <div>
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <div className="inline-flex items-center gap-2 rounded-full bg-cyan-950/60 px-3 py-1 text-xs font-mono font-medium text-cyan-400 border border-cyan-800/40">
              <Sparkles className="h-3 w-3" />
              <span>Closed-Loop Photography AI</span>
            </div>
            <HealthPill />

            {/* Agent Mode / Pipeline Mode Toggle Pill */}
            <div className="inline-flex items-center rounded-full bg-zinc-900 p-0.5 border border-zinc-800">
              <button
                type="button"
                onClick={() => setIsAgentMode(false)}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-mono font-medium transition-all ${
                  !isAgentMode
                    ? "bg-zinc-800 text-white shadow-sm"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <SlidersHorizontal className="h-3 w-3 text-cyan-400" />
                <span>Single-Pass Pipeline</span>
              </button>
              <button
                type="button"
                onClick={() => setIsAgentMode(true)}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-mono font-medium transition-all ${
                  isAgentMode
                    ? "bg-cyan-500 text-black font-semibold shadow-sm shadow-cyan-500/20"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <Bot className="h-3 w-3" />
                <span>Agent Mode (Closed Loop)</span>
              </button>
            </div>
          </div>

          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            {isAgentMode ? "AesFix Autonomous Agent" : "AesFix Studio Workspace"}
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-400">
            {isAgentMode
              ? "Multi-turn perceive-plan-act-verify agent. Proposes parametric edits, measures per-aspect aesthetic delta against noise floor ε, and commits Lightroom XMP sidecars."
              : "Diagnose photographic flaws across aesthetic dimensions, execute deterministic edits, and measure improvement above empirical noise floor ε with Lightroom XMP export."}
          </p>
        </div>

        {/* Creative Goal Context */}
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

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Left Column: Photo Upload & Parameters (5 Cols) */}
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
              <div className="grid grid-cols-2 gap-2">
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
                  Low-key night (hard case)
                </button>
                <button
                  type="button"
                  onClick={() => handleLoadSample("underexposed_synthetic.jpg")}
                  className="rounded-lg bg-zinc-900 px-2 py-1.5 text-[11px] font-mono text-zinc-300 border border-zinc-800 hover:border-blue-500 hover:text-white transition-all text-center"
                >
                  Underexposed (synthetic)
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

            {/* Critique / User Goal Note */}
            <div className="mt-3">
              <label className="text-[11px] font-mono text-zinc-400">
                {isAgentMode ? "Agent Goal Directives:" : "Optional Photographer Note / Critique Target:"}
              </label>
              <input
                type="text"
                value={userCritique}
                onChange={(e) => setUserCritique(e.target.value)}
                placeholder={
                  isAgentMode
                    ? "e.g. lift shadow detail, retain saturation, avoid color cast"
                    : "e.g. lift shadow detail, correct colour cast..."
                }
                className="mt-1 w-full rounded-lg bg-zinc-900/90 px-3 py-2 text-xs text-zinc-200 border border-zinc-800 focus:outline-none focus:border-cyan-500"
              />
            </div>

            {/* Provider and Model Selector in Agent Mode */}
            {isAgentMode && (
              <div className="mt-3 grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10px] font-mono text-zinc-400 uppercase">Provider</label>
                  <select
                    value={agentProvider}
                    onChange={(e) => setAgentProvider(e.target.value)}
                    className="mt-1 w-full rounded-lg bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-200 border border-zinc-800 focus:border-cyan-500"
                  >
                    <option value="gemini">Gemini</option>
                    <option value="openrouter">OpenRouter</option>
                    <option value="heuristic">Heuristic Fallback</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] font-mono text-zinc-400 uppercase">Model ID</label>
                  <select
                    value={agentModelId}
                    onChange={(e) => setAgentModelId(e.target.value)}
                    className="mt-1 w-full rounded-lg bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-200 border border-zinc-800 focus:border-cyan-500 font-mono"
                  >
                    <option value="gemini-3.8-flash">gemini-3.8-flash</option>
                    <option value="google/gemma-4-31b-it:free">gemma-4-31b-it:free</option>
                    <option value="heuristic">heuristic</option>
                  </select>
                </div>
              </div>
            )}

            {/* Primary Action Button */}
            <button
              type="button"
              onClick={handleSubmit}
              disabled={isProcessing || !selectedFile}
              className={`mt-4 flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-xs font-bold shadow-lg transition-all hover:brightness-110 active:scale-[0.99] disabled:opacity-50 ${
                isAgentMode
                  ? "bg-gradient-to-r from-emerald-400 to-cyan-500 text-black shadow-emerald-500/20"
                  : "bg-gradient-to-r from-cyan-500 to-blue-600 text-black shadow-cyan-500/20"
              }`}
            >
              {isAgentMode ? <Bot className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
              {isProcessing
                ? "Running Closed Loop..."
                : isAgentMode
                ? "Run AesFix Agent (Closed Loop)"
                : "Run Grounded Diagnosis & Fix"}
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

          {/* Pipeline Mode Aspect Scores */}
          {!isAgentMode && jobResult && (
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
                      <span className="capitalize text-zinc-300">{aspect.replace(/_/g, " ")}</span>
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

          {/* Agent Mode Summary Card */}
          {isAgentMode && agentComplete && (
            <div className="glass-card rounded-2xl p-5 border border-zinc-800">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-mono font-bold uppercase text-zinc-400">
                  Trajectory Outcome
                </span>
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 font-mono text-[11px] font-bold border ${
                    agentComplete.verdict === "measured improvement"
                      ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/40"
                      : "bg-amber-950/80 text-amber-400 border-amber-800/40"
                  }`}
                >
                  {agentComplete.verdict === "measured improvement" ? (
                    <CheckCircle2 className="h-3 w-3" />
                  ) : (
                    <XCircle className="h-3 w-3" />
                  )}
                  {agentComplete.verdict}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="rounded-lg bg-zinc-900/60 p-2.5 border border-zinc-800">
                  <span className="text-zinc-500 text-[10px] block">MEASURED GAIN Δ</span>
                  <span
                    className={`text-sm font-bold ${
                      agentComplete.total_gain > 0 ? "text-emerald-400" : "text-zinc-400"
                    }`}
                  >
                    {agentComplete.total_gain > 0
                      ? `+${agentComplete.total_gain.toFixed(4)}`
                      : agentComplete.total_gain.toFixed(4)}
                  </span>
                </div>
                <div className="rounded-lg bg-zinc-900/60 p-2.5 border border-zinc-800">
                  <span className="text-zinc-500 text-[10px] block">NOISE FLOOR ε</span>
                  <span className="text-sm font-bold text-zinc-300">
                    {agentComplete.noise_floor.toFixed(4)}
                  </span>
                </div>
                <div className="rounded-lg bg-zinc-900/60 p-2.5 border border-zinc-800">
                  <span className="text-zinc-500 text-[10px] block">ACCEPTED STEPS</span>
                  <span className="text-sm font-bold text-emerald-400">
                    {agentComplete.accepted_steps_count} / {agentComplete.total_steps}
                  </span>
                </div>
                <div className="rounded-lg bg-zinc-900/60 p-2.5 border border-zinc-800">
                  <span className="text-zinc-500 text-[10px] block">REJECTED STEPS</span>
                  <span className="text-sm font-bold text-zinc-400">
                    {agentComplete.rejected_steps_count}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Compare Canvas, Timeline & Fixes (7 Cols) */}
        <div className="flex flex-col gap-6 lg:col-span-7">
          {/* Main Inspection Canvas */}
          <div className="glass-card rounded-2xl p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400">
                {isAgentMode
                  ? "Agent Retouching Canvas (Before / After)"
                  : "3. Inspection & Edit Comparison"}
              </span>

              {/* Active Provider & Model Badge */}
              {isAgentMode && (
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-zinc-900 px-2.5 py-0.5 font-mono text-[10px] text-cyan-400 border border-zinc-800">
                    Provider: {agentComplete?.provider || agentProvider}
                  </span>
                  <span className="rounded-full bg-zinc-900 px-2.5 py-0.5 font-mono text-[10px] text-zinc-400 border border-zinc-800">
                    Model: {agentComplete?.model_id || agentModelId}
                  </span>
                </div>
              )}

              {!isAgentMode && activeFix && (
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
            {isAgentMode ? (
              localPreviewUrl ? (
                <CompareSlider
                  originalImage={localPreviewUrl}
                  editedImage={agentDisplayImage}
                  gain={
                    activeAgentStep
                      ? activeAgentStep.overall_delta
                      : agentComplete
                      ? agentComplete.total_gain
                      : null
                  }
                  noiseFloor={
                    activeAgentStep
                      ? activeAgentStep.noise_floor
                      : agentComplete?.noise_floor ?? null
                  }
                  aspectFixed={
                    activeAgentStep
                      ? `${activeAgentStep.aspect} (${activeAgentStep.direction > 0 ? "+1" : "-1"})`
                      : agentComplete?.verdict === "measured improvement"
                      ? "Multi-Aspect Stacked"
                      : "Original"
                  }
                  fixName={
                    activeAgentStep
                      ? `Step ${activeAgentStep.step_number}: ${activeAgentStep.op}`
                      : agentComplete
                      ? "Final Retouched Result"
                      : "Original"
                  }
                  verdict={
                    activeAgentStep
                      ? activeAgentStep.status === "accepted"
                        ? "measurable improvement"
                        : "rejected by guard"
                      : agentComplete?.verdict
                  }
                />
              ) : (
                <div className="flex aspect-[4/3] w-full flex-col items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-950/80 text-zinc-500">
                  <ImageIcon className="mb-2 h-10 w-10 text-zinc-600" />
                  <p className="text-xs font-mono">Upload a photo to view before/after comparison</p>
                </div>
              )
            ) : originalPreviewUrl ? (
              <CompareSlider
                originalImage={originalPreviewUrl}
                editedImage={fixedPreviewUrl}
                gain={activeFix ? activeFix.delta : null}
                noiseFloor={jobResult?.noise_floor ?? null}
                aspectFixed={
                  activeFix
                    ? `${activeFix.expected_aspect} (${activeFix.direction > 0 ? "+1" : "-1"})`
                    : "None"
                }
                fixName={activeFix ? activeFix.op : "Original"}
                verdict={activeFix?.verdict}
              />
            ) : (
              <div className="flex aspect-[4/3] w-full flex-col items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-950/80 text-zinc-500">
                <ImageIcon className="mb-2 h-10 w-10 text-zinc-600" />
                <p className="text-xs font-mono">Upload a photo to view before/after comparison</p>
              </div>
            )}

            {/* Action Bar: Download Lightroom XMP */}
            {isAgentMode && agentComplete?.stacked_xmp && (
              <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
                <a
                  href={`${API_URL}/api/v1/agent/jobs/${agentComplete.job_id}/xmp`}
                  download={`aesfix_agent_${agentComplete.job_id}.xmp`}
                  className="flex items-center justify-center gap-2 rounded-xl bg-emerald-950/40 px-4 py-2.5 text-xs font-semibold text-emerald-300 border border-emerald-800/40 transition-all hover:bg-emerald-900/60"
                >
                  <FileCode className="h-4 w-4 text-emerald-400" />
                  <span>Download Stacked Lightroom .XMP</span>
                </a>

                {agentComplete.final_image_url && (
                  <a
                    href={`${API_URL}${agentComplete.final_image_url}`}
                    download={`aesfix_retouched_${agentComplete.job_id}.jpg`}
                    className="flex items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-2.5 text-xs font-semibold text-zinc-200 border border-zinc-800 transition-all hover:bg-zinc-800 hover:text-white"
                  >
                    <Sparkles className="h-4 w-4 text-cyan-400" />
                    <span>Save Retouched Photograph</span>
                  </a>
                )}
              </div>
            )}

            {!isAgentMode && jobId && activeFix && (
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

          {/* Leave It State Cards */}
          {isAgentMode && agentComplete && agentComplete.verdict === "leave it" && (
            <div className="rounded-2xl border border-amber-500/40 bg-amber-950/20 p-5 backdrop-blur-md">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-mono font-bold text-amber-300 uppercase tracking-wide">
                    Verdict: Leave it alone
                  </h3>
                  <p className="mt-1 text-xs text-zinc-300 leading-relaxed">
                    The autonomous agent halted after evaluating candidate photographic adjustments.
                    No edit yielded a measured aesthetic improvement Δ exceeding the noise floor
                    (ε = {agentComplete.noise_floor.toFixed(4)}) without identity guard drift or collateral damage.
                  </p>
                </div>
              </div>
            </div>
          )}

          {!isAgentMode && isPipelineLeaveIt && (
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

          {/* AGENT MODE: Step Timeline (proposed → executed → measured → accepted/rejected + reason) */}
          {isAgentMode && agentSteps.length > 0 && (
            <div className="glass-card rounded-2xl p-5">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400">
                  Agent Step Timeline ({agentSteps.length} Steps)
                </span>
                <span className="text-[11px] font-mono text-zinc-500">
                  Click thumbnail or step to inspect
                </span>
              </div>

              <div className="flex flex-col gap-4">
                {agentSteps.map((step, idx) => {
                  const isSelected = selectedAgentStepIdx === idx;
                  const isAccepted = step.status === "accepted";

                  return (
                    <div
                      key={step.step_number}
                      onClick={() =>
                        setSelectedAgentStepIdx(selectedAgentStepIdx === idx ? null : idx)
                      }
                      className={`cursor-pointer rounded-xl p-4 transition-all border ${
                        isSelected
                          ? "bg-zinc-900/95 border-cyan-500/80 shadow-md shadow-cyan-500/10"
                          : "bg-zinc-950/60 border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/40"
                      }`}
                    >
                      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                        <div className="flex items-center gap-3">
                          {/* Step Thumbnail */}
                          {step.preview_data_url ? (
                            <img
                              src={step.preview_data_url}
                              alt={`Step ${step.step_number} preview`}
                              className="h-14 w-14 rounded-lg object-cover border border-zinc-700 shrink-0"
                            />
                          ) : (
                            <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-800 shrink-0">
                              <ImageIcon className="h-6 w-6 text-zinc-600" />
                            </div>
                          )}

                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-xs font-bold text-white">
                                Step #{step.step_number}: {step.op}
                              </span>
                              <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300">
                                {step.aspect} ({step.direction > 0 ? "+1" : "-1"})
                              </span>
                            </div>
                            <p className="mt-1 text-xs text-zinc-400 leading-snug">
                              {step.rationale}
                            </p>
                          </div>
                        </div>

                        {/* Status Pill */}
                        <div className="shrink-0 sm:text-right">
                          <span
                            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 font-mono text-[10px] font-bold border ${
                              isAccepted
                                ? "bg-emerald-950/80 text-emerald-300 border-emerald-800/40"
                                : "bg-red-950/80 text-red-300 border-red-800/40"
                            }`}
                          >
                            {isAccepted ? (
                              <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                            ) : (
                              <XCircle className="h-3 w-3 text-red-400" />
                            )}
                            {step.status.toUpperCase()}
                          </span>
                        </div>
                      </div>

                      {/* Timeline Flow: proposed → executed → measured → accepted/rejected */}
                      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-zinc-800/60 font-mono text-[11px]">
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">
                            1. PROPOSED
                          </span>
                          <span className="text-zinc-300 truncate block">
                            {Object.entries(step.params)
                              .map(([k, v]) => `${k}:${v}`)
                              .join(" ")}
                          </span>
                        </div>
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">
                            2. EXECUTED
                          </span>
                          <span className="text-zinc-300">{step.execution_duration_ms}ms</span>
                        </div>
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">
                            3. MEASURED Δ
                          </span>
                          <span
                            className={
                              step.overall_delta > 0
                                ? "text-emerald-400 font-bold"
                                : "text-zinc-400"
                            }
                          >
                            {step.overall_delta > 0
                              ? `+${step.overall_delta.toFixed(4)}`
                              : step.overall_delta.toFixed(4)}{" "}
                            (ε={step.noise_floor.toFixed(4)})
                          </span>
                        </div>
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">
                            4. GUARD DRIFT
                          </span>
                          <span
                            className={step.guard_passed ? "text-zinc-300" : "text-red-400 font-bold"}
                          >
                            {step.guard_drift.toFixed(4)} ({step.guard_passed ? "PASS" : "FAIL"})
                          </span>
                        </div>
                      </div>

                      {/* Verdict Reason Box */}
                      <div className="mt-2.5 rounded-lg bg-zinc-950/80 p-2.5 border border-zinc-800/80 text-[11px] font-mono text-zinc-400">
                        <span className="text-zinc-500 font-bold block mb-0.5">VERDICT REASON:</span>
                        <p className="text-zinc-300 leading-relaxed">{step.verdict_reason}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* PIPELINE MODE: Candidate Fix Cards */}
          {!isAgentMode && currentFixes.length > 0 && (
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
                  const isImprovement =
                    fix.verdict === "measurable improvement" ||
                    fix.verdict === "measured improvement above noise floor";
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
                          <span
                            className={
                              fix.delta > 0 ? "text-emerald-400 font-bold" : "text-zinc-400"
                            }
                          >
                            {fix.delta > 0 ? `+${fix.delta.toFixed(4)}` : fix.delta.toFixed(4)}
                          </span>
                        </div>
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">
                            ε (Noise)
                          </span>
                          <span className="text-zinc-300">{fix.epsilon.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">
                            Guard ({fix.guard_backend || "SSIM + rgb_layout_stat"})
                          </span>
                          <span className={fix.guard_passed ? "text-zinc-300" : "text-red-400"}>
                            {fix.guard_drift.toFixed(4)}
                          </span>
                        </div>
                        <div>
                          <span className="text-zinc-500 block text-[9px] uppercase">
                            Parameters
                          </span>
                          <span
                            className="text-zinc-300 truncate block"
                            title={JSON.stringify(fix.params)}
                          >
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
