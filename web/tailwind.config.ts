import type { Config } from "tailwindcss";

// Why custom aesthetic tokens are declared:
// AesFix utilizes a high-contrast dark visual system where aspect domains have dedicated
// color cues (lighting = amber, composition = cyan, technical = emerald, color = purple).
const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        border: "hsl(var(--border))",
        primary: {
          DEFAULT: "#06B6D4", // Cyan 500 for crisp visual accents
          foreground: "#000000",
        },
        secondary: {
          DEFAULT: "#18181B", // Zinc 900
          foreground: "#F4F4F5",
        },
        aspect: {
          composition: "#38BDF8", // Sky
          lighting: "#F59E0B",    // Amber
          color: "#A855F7",       // Purple
          clarity: "#EC4899",     // Pink
          technical: "#10B981",   // Emerald
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
