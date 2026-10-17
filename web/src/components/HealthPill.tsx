"use client";

import React, { useEffect, useState } from "react";
import { Activity } from "lucide-react";

interface HealthState {
  status: "healthy" | "offline" | "checking";
  timestamp?: number;
}

export function HealthPill(): React.JSX.Element {
  const [health, setHealth] = useState<HealthState>({ status: "checking" });
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    let isMounted = true;

    async function checkHealth(): Promise<void> {
      try {
        const res = await fetch(`${apiUrl}/health`, {
          method: "GET",
          headers: { Accept: "application/json" },
        });
        if (res.ok) {
          const data = await res.json();
          if (isMounted) {
            setHealth({
              status: data.status === "healthy" ? "healthy" : "offline",
              timestamp: data.timestamp,
            });
          }
        } else {
          if (isMounted) setHealth({ status: "offline" });
        }
      } catch {
        if (isMounted) setHealth({ status: "offline" });
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [apiUrl]);

  const isHealthy = health.status === "healthy";
  const isChecking = health.status === "checking";

  return (
    <a
      href={`${apiUrl}/health`}
      target="_blank"
      rel="noreferrer"
      title={`API Endpoint: ${apiUrl}/health`}
      className="inline-flex items-center gap-2 rounded-full bg-zinc-900/90 px-3 py-1 text-xs font-mono border border-zinc-800 transition-colors hover:border-zinc-700"
    >
      <Activity className="h-3 w-3 text-zinc-400" />
      <span className="flex items-center gap-1.5">
        <span
          className={`h-2 w-2 rounded-full ${
            isChecking
              ? "bg-amber-400 animate-ping"
              : isHealthy
              ? "bg-emerald-400 animate-pulse"
              : "bg-red-500"
          }`}
        />
        <span className={isHealthy ? "text-emerald-400" : isChecking ? "text-amber-400" : "text-red-400"}>
          {isChecking ? "API: connecting" : isHealthy ? "API: healthy" : "API: offline"}
        </span>
      </span>
    </a>
  );
}
