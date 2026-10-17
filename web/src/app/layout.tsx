import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import React from "react";
import { Navbar } from "@/components/Navbar";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

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
      <body className={`${inter.variable} ${mono.variable} min-h-screen bg-zinc-950 text-zinc-100 antialiased selection:bg-cyan-500 selection:text-black font-sans`}>
        {/* Persistent Research Prototype Banner */}
        <div className="w-full bg-amber-500/10 border-b border-amber-500/25 px-4 py-2 text-center text-xs font-mono text-amber-300 tracking-wide">
          Research prototype — measurements are pilot, not human-calibrated.
        </div>

        {/* Responsive Navigation Bar with Mobile Nav */}
        <Navbar />

        {/* Page View Body */}
        <main className="min-h-[calc(100vh-4rem)] studio-grid">{children}</main>
      </body>
    </html>
  );
}
