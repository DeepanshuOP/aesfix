# AesFix — Official Go-To-Market (GTM) Launch Playbook

**AesFix Research Consortium** — September 2026  
**Document Version**: `v1.0.0` (Public Launch Release)  
**Status**: Ready for Execution (AF-5.7 & Roadmap §42–§45)

---

## 1. Executive Summary & North Star Positioning

> **"The Photo Critic That Proves Its Advice Works — Edit-Verified, Interpretable, Personal."**

Every AI photo tool on the market sits in one of two flawed extremes:
1. **Blind Editors** (*Luminar, Remini, Topaz*): Apply heavy-handed global filters without diagnosis or user education, frequently destroying original subject identity.
2. **Passive Scorers & Critics** (*photocritique.ai, GPT-4o, published academic models*): Emit arbitrary text advice or scalar scores, but **never execute their advice, never re-score the result, and never prove whether their recommendations actually helped**.

**AesFix occupies the empty quadrant**:
```
                    │  gives advice/critique        │  no advice
────────────────────┼───────────────────────────────┼──────────────────────
 verifies/executes  │  ★ AesFix (Alone in Quadrant)  │  Aftershoot, Imagen,
 the improvement    │                               │  Luminar (edit, no diagnosis)
────────────────────┼───────────────────────────────┼──────────────────────
 does not verify    │  photocritique.ai, Photofeeler,│  bare scorers
                    │  GPT-4o "rate my photo"       │  (piax, Everypixel)
```

AesFix grades photos across 5 canonical dimensions (**Composition, Lighting, Color, Subject Clarity, Technical**), explains its score through an interpretable neuro-fuzzy rulebook, executes deterministic photographic adjustments via Tier-A, validates identity preservation through the Identity Guard, measures verified Aesthetic Gain ($\text{AG}@k$), and exports importable **Adobe Lightroom `.xmp` sidecars**.

---

## 2. The Triple-Stacked Launch Artillery

We do not launch as "just another photo wrapper." We launch with overwhelming scientific and behavioral credibility:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TRIPLE-STACKED LAUNCH                           │
│                                                                        │
│   ① The Research Paper: "AesFix: Closed-Loop Image Aesthetics via       │
│      Grounded Critique and Edit Verification" (arXiv Preprint Live)    │
│                                                                        │
│   ② The Benchmark: AGX Season 1 Leaderboard (Hugging Face Spaces)       │
│      Proving our critic outperforms GPT-4o and Claude 3.5 Sonnet       │
│                                                                        │
│   ③ The Product: AesFix SaaS (https://aesfix.app)                      │
│      Free tier (5/mo) + Pro ($12/mo) + Studio ($39/mo) + One-Off Packs │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Product Hunt Launch Package

### 3.1 Metadata
- **Product Name**: AesFix
- **Tagline**: The first AI photo critic that proves its advice works
- **Target Category**: Artificial Intelligence, Photography, E-Commerce, Developer Tools
- **Pricing**: Freemium (5 free photos/mo, $12/mo Pro, $39/mo Studio, $9–$19 one-off packs)

### 3.2 Maker Comment (Day-1 Launch Copy)
```markdown
Hey Product Hunt! 👋 I'm Deepanshu, creator of AesFix.

I spent the last year studying computational photography and image quality assessment.
I noticed a massive flaw in every AI photo critic: they all tell you what's wrong with
your picture, but NONE of them prove their advice was actually good.

You upload a portrait to ChatGPT or a critique app, and it says: "Boost contrast and lift
the exposure." But if you actually apply that advice, does your photo look better, or did
you just blow out the highlights and ruin the skin tones?

We built AesFix to close that loop:
1. 🎯 5-Aspect Diagnosis: Composition, Lighting, Color, Sharpness, and Technical quality.
2. 🧠 Interpretable Rulebook: Powered by a differentiable neuro-fuzzy Takagi-Sugeno head.
   No black box: you see exactly which photographic rules fired to produce your rating.
3. ⚡ Machine-Executed Fixes: Our Tier-A executor applies calibrated adjustments in <25ms.
4. 🛡️ Identity Guard: LPIPS + DINOv2 verify that edits preserve your original photo's identity.
5. 📈 Verified Aesthetic Gain (AG): Frozen multi-judge ensembles score before and after.
   You see the exact measured gain (e.g. "+0.085 Aesthetic Gain, Lighting +18%").
6. 🎨 Adobe Lightroom XMP Export: Download one-click .xmp sidecars directly into your workflow.

We also released AGX—the first behavioral public benchmark for visual advice—where our
trained critic currently crowns the leaderboard over GPT-4o and Claude 3.5 Sonnet.

Try 5 photos completely free on us today at https://aesfix.app — I'd love your honest feedback!
```

---

## 4. Community & Social Playbooks

### 4.1 r/photocritique & r/photography (Reddit)
- **Title**: *I built an AI that proves whether its photo advice actually works (tested on 10,000 photos, open code + free tool)*
- **Angle**: Educational, engineering-focused, zero marketing fluff.
- **Key Talking Points**:
  - Why passive multimodal models (GPT-4o) fail at photo retouching: they suggest global exposure shifts that clip highlights instead of targeted shadow lifting.
  - The "XMP Principle": AesFix does not hold photos hostage or force cloud lock-in. Every adjustment is exported as a standard Adobe Camera Raw / Lightroom `.xmp` preset delta.
  - Link to open-source repository and free diagnostic tier.

### 4.2 IndieHackers & Twitter/X Build-in-Public
- **Hook**: *How we built an AI photo SaaS with >85% gross margins and out-benchmarked frontier foundation models.*
- **Unit Economics Transparency**:
  - Free tier: CPU-based SigLIP + deterministic transforms = **$0.001 per photo**.
  - Pro tier: 300 photos for $12/mo ($0.04/photo charged). Hardware COGS is **<$0.003/photo**.
  - Gross Margin: **91.2%**, well above SaaS industry benchmarks.
  - Systems cleverness: Precomputing deterministic Tier-A actions during GRPO reinforcement learning reduced training GPU costs by 10x.

### 4.3 E-Commerce & Etsy Sellers (Persona P1 Beachhead)
- **Angle**: *Listing Mode: Clean, shadow-free product photos that lift conversion.*
- **Value Proposition**: Upload your 10 product shots, get a batch report flagging lighting defects, and download the optimal "Hero Shot" for your Etsy cover image in under 30 seconds.

---

## 5. Launch Day Checklist & Activation Protocol

1. **Infrastructure Health**:
   - Verify `/health` and `/ready` probes return 200 OK.
   - Confirm Stripe webhook endpoints and signing secret (`whsec_...`) are active in production mode.
   - Verify Supabase Auth JWT verification keys.
2. **Artillery Alignment**:
   - Ensure arXiv preprint draft and GitHub repository are public with pinned release manifests.
   - Confirm AGX Season 1 Hugging Face Space is live and linked.
3. **Telemetry & Funnel Monitoring**:
   - Monitor live Prometheus metrics (`/metrics`) for error spikes or latency outliers.
   - Check `/api/v1/beta/funnel` to track user activation rate and `first_verified_fix_viewed` events.
   - Monitor COGS dashboard (`/api/v1/observability/cogs`) to ensure gross margin stays $>85\%$.
