"""
╔══════════════════════════════════════════════════════════════════╗
║              S.S_AI Summarizer — Enterprise Dark UI             ║
║         Built with Streamlit + HuggingFace Transformers          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

# ══════════════════════════════════════════════════════
# 1. PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="S.S_AI Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
# 2. GLOBAL CSS — Enterprise Dark Design System
# ══════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── CSS Variables ── */
:root {
  /* Backgrounds — deep slate layers, not pure black */
  --bg-base:        #0a0e17;
  --bg-surface:     #0f1420;
  --bg-raised:      #141928;
  --bg-overlay:     #1a2133;
  --bg-hover:       rgba(255,255,255,0.04);

  /* Borders */
  --border-faint:   rgba(255,255,255,0.06);
  --border-subtle:  rgba(255,255,255,0.10);
  --border-accent:  rgba(56,189,248,0.40);

  /* Accent — single electric blue, no competing hues */
  --accent:         #38bdf8;
  --accent-dim:     rgba(56,189,248,0.15);
  --accent-glow:    rgba(56,189,248,0.20);
  --accent-dark:    #0ea5e9;

  /* Text hierarchy — four clear levels */
  --text-primary:   #f0f4fb;
  --text-secondary: #8b96ab;
  --text-muted:     #4c5668;
  --text-disabled:  #2e3647;

  /* Semantic */
  --success:        #34d399;
  --success-dim:    rgba(52,211,153,0.12);
  --error:          #f87171;
  --error-dim:      rgba(248,113,113,0.10);
  --warning:        #fbbf24;

  /* Fonts */
  --font-display:   'Outfit', sans-serif;
  --font-body:      'DM Sans', sans-serif;
  --font-mono:      'DM Mono', monospace;

  /* Radii */
  --r-sm:  6px;
  --r-md:  10px;
  --r-lg:  16px;
  --r-xl:  22px;

  /* Shadows */
  --shadow-sm:  0 1px 3px rgba(0,0,0,0.4);
  --shadow-md:  0 4px 20px rgba(0,0,0,0.5);
  --shadow-lg:  0 8px 40px rgba(0,0,0,0.6);
  --shadow-accent: 0 0 32px rgba(56,189,248,0.12);
}

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-base) !important;
  font-family: var(--font-body) !important;
  color: var(--text-primary) !important;
}

/* Remove Streamlit chrome */
#MainMenu, footer { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Main container ── */
[data-testid="stAppViewContainer"] > .main > .block-container {
  max-width: 900px !important;
  padding: 2.5rem 2rem 5rem !important;
  margin: 0 auto !important;
}

/* ══════════════════════════════════
   SIDEBAR TOGGLE — HIGHLY VISIBLE
══════════════════════════════════ */

/* The collapsed-state arrow (when sidebar is hidden) */
[data-testid="collapsedControl"] {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--r-md) !important;
  color: var(--accent) !important;
  width: 36px !important;
  height: 36px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  box-shadow: var(--shadow-sm), 0 0 12px var(--accent-glow) !important;
  transition: background 0.2s, border-color 0.2s, box-shadow 0.2s !important;
  opacity: 1 !important;
}
[data-testid="collapsedControl"]:hover {
  background: var(--bg-overlay) !important;
  border-color: var(--accent) !important;
  box-shadow: var(--shadow-md), 0 0 20px var(--accent-glow) !important;
}
[data-testid="collapsedControl"] svg {
  fill: var(--accent) !important;
  stroke: var(--accent) !important;
  color: var(--accent) !important;
  opacity: 1 !important;
}

/* Sidebar expand/collapse button inside sidebar */
[data-testid="stSidebarCollapseButton"] > button,
button[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebar"] button[kind="header"] {
  background: var(--bg-overlay) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--r-md) !important;
  color: var(--accent) !important;
  opacity: 1 !important;
  width: 34px !important;
  height: 34px !important;
  padding: 6px !important;
  transition: background 0.2s, border-color 0.2s !important;
}
[data-testid="stSidebarCollapseButton"] > button:hover,
[data-testid="stSidebar"] button[kind="header"]:hover {
  background: var(--accent-dim) !important;
  border-color: var(--accent) !important;
}
[data-testid="stSidebarCollapseButton"] > button svg,
[data-testid="stSidebar"] button[kind="header"] svg {
  fill: var(--accent) !important;
  stroke: var(--accent) !important;
  color: var(--accent) !important;
  opacity: 1 !important;
}

/* ══════════════════════════════════
   SIDEBAR
══════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border-faint) !important;
}
[data-testid="stSidebar"] * {
  font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
  font-family: var(--font-display) !important;
  font-size: 0.68rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  margin-bottom: 0.5rem !important;
  margin-top: 1.2rem !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--border-faint) !important;
  margin: 1rem 0 !important;
}

/* ══════════════════════════════════
   TEXTAREA
══════════════════════════════════ */
textarea {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--r-lg) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.96rem !important;
  line-height: 1.75 !important;
  padding: 1.1rem 1.3rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
  resize: vertical !important;
  caret-color: var(--accent) !important;
}
textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-dim), var(--shadow-accent) !important;
  outline: none !important;
  background: var(--bg-overlay) !important;
}
textarea::placeholder {
  color: var(--text-muted) !important;
  font-style: italic;
}
[data-testid="stTextArea"] label {
  display: none !important;
}

/* ══════════════════════════════════
   BUTTONS — Primary
══════════════════════════════════ */
.stButton > button {
  width: 100% !important;
  background: var(--accent) !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  color: #050a14 !important;
  font-family: var(--font-display) !important;
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em !important;
  padding: 0.85rem 1.5rem !important;
  cursor: pointer !important;
  transition: transform 0.18s, box-shadow 0.18s, filter 0.18s !important;
  box-shadow: 0 2px 12px var(--accent-glow), 0 1px 3px rgba(0,0,0,0.4) !important;
}
.stButton > button:hover {
  filter: brightness(1.08) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 28px rgba(56,189,248,0.35), 0 2px 8px rgba(0,0,0,0.4) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
  filter: brightness(0.96) !important;
}

/* Download button — ghost style */
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--r-md) !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  padding: 0.75rem 1.5rem !important;
  cursor: pointer !important;
  transition: border-color 0.2s, color 0.2s, background 0.2s !important;
  box-shadow: none !important;
  filter: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-dim) !important;
  transform: none !important;
  box-shadow: none !important;
  filter: none !important;
}

/* ══════════════════════════════════
   PROGRESS BAR
══════════════════════════════════ */
.stProgress > div {
  background: var(--bg-overlay) !important;
  border-radius: 99px !important;
  height: 3px !important;
  border: none !important;
}
.stProgress > div > div {
  background: var(--accent) !important;
  border-radius: 99px !important;
  box-shadow: 0 0 8px var(--accent-glow) !important;
}

/* ══════════════════════════════════
   SPINNER
══════════════════════════════════ */
.stSpinner > div {
  border-top-color: var(--accent) !important;
  border-right-color: transparent !important;
}

/* ══════════════════════════════════
   SLIDER
══════════════════════════════════ */
.stSlider [role="slider"] {
  background: var(--accent) !important;
  border: 2px solid var(--bg-base) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
.stSlider [data-testid="stSliderTrack"] > div {
  background: var(--accent) !important;
}

/* ══════════════════════════════════
   ANIMATIONS
══════════════════════════════════ */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
  0%   { background-position: -300% center; }
  100% { background-position:  300% center; }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.6; }
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ══════════════════════════════════
   COMPONENT CLASSES
══════════════════════════════════ */

/* ── Topbar ── */
.ss-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0 1.4rem;
  animation: fadeUp 0.5s ease both;
}
.ss-logo {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.ss-logo-mark {
  width: 32px;
  height: 32px;
  background: var(--accent);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 0.8rem;
  color: #050a14;
  letter-spacing: -0.03em;
  flex-shrink: 0;
}
.ss-logo-text {
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
}
.ss-logo-text span {
  color: var(--accent);
}
.ss-topbar-right {
  display: flex;
  align-items: center;
  gap: 1rem;
}
.ss-version-tag {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-muted);
  background: var(--bg-raised);
  border: 1px solid var(--border-faint);
  border-radius: 99px;
  padding: 0.2rem 0.65rem;
  letter-spacing: 0.05em;
}
.ss-status-dot {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.72rem;
  color: var(--text-muted);
  letter-spacing: 0.04em;
}
.ss-status-dot::before {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 6px var(--success);
  animation: pulse 2.5s ease-in-out infinite;
  flex-shrink: 0;
}

/* ── Rule ── */
.ss-rule {
  height: 1px;
  background: var(--border-faint);
  margin: 0 0 2.5rem;
  animation: fadeUp 0.5s 0.05s ease both;
}

/* ── Hero ── */
.ss-hero {
  padding: 0.5rem 0 2.8rem;
  animation: fadeUp 0.55s 0.08s ease both;
}
.ss-hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--accent);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  background: var(--accent-dim);
  border: 1px solid var(--border-accent);
  border-radius: 99px;
  padding: 0.25rem 0.75rem;
  margin-bottom: 1.1rem;
}
.ss-hero-eyebrow::before {
  content: '◆';
  font-size: 0.5rem;
  opacity: 0.7;
}
.ss-hero-heading {
  font-family: var(--font-display);
  font-size: clamp(2rem, 4.5vw, 3rem);
  font-weight: 800;
  line-height: 1.12;
  letter-spacing: -0.04em;
  color: var(--text-primary);
  margin-bottom: 1rem;
}
.ss-hero-heading em {
  font-style: normal;
  color: var(--accent);
}
.ss-hero-sub {
  color: var(--text-secondary);
  font-size: 1rem;
  line-height: 1.7;
  max-width: 540px;
  font-weight: 400;
}
.ss-hero-caps {
  display: flex;
  align-items: center;
  gap: 1.8rem;
  margin-top: 1.8rem;
  flex-wrap: wrap;
}
.ss-cap {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8rem;
  color: var(--text-muted);
  letter-spacing: 0.02em;
}
.ss-cap-icon {
  width: 22px;
  height: 22px;
  background: var(--bg-overlay);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
}

/* ── Section header ── */
.ss-section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.7rem;
}
.ss-section-label {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.ss-section-label::before {
  content: '';
  width: 3px;
  height: 12px;
  background: var(--accent);
  border-radius: 2px;
  display: inline-block;
}

/* ── Counter ── */
.ss-counter {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-muted);
}
.ss-counter .num { color: var(--text-secondary); }
.ss-counter .sep { opacity: 0.3; }
.ss-counter .status-ok  { color: var(--success); }
.ss-counter .status-warn { color: var(--warning); }

/* ── Output card ── */
.ss-output-card {
  background: var(--bg-raised);
  border: 1px solid var(--border-subtle);
  border-top: 2px solid var(--accent);
  border-radius: var(--r-xl);
  padding: 1.8rem;
  margin-top: 0.6rem;
  box-shadow: var(--shadow-lg), var(--shadow-accent);
  animation: fadeUp 0.45s ease both;
  position: relative;
  overflow: hidden;
}
.ss-output-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 80px;
  background: linear-gradient(to bottom, rgba(56,189,248,0.04), transparent);
  pointer-events: none;
  border-radius: inherit;
}
.ss-output-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.3rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-faint);
}
.ss-output-label {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-family: var(--font-display);
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--text-secondary);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.ss-output-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 8px var(--success);
}
.ss-copy-btn {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.3rem 0.8rem;
  border-radius: var(--r-sm);
  border: 1px solid var(--border-subtle);
  background: var(--bg-overlay);
  color: var(--text-muted);
  cursor: pointer;
  transition: border-color 0.2s, color 0.2s, background 0.2s;
}
.ss-copy-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-dim);
}
.ss-summary-text {
  color: var(--text-primary) !important;
  font-size: 1.05rem;
  line-height: 1.85;
  font-weight: 400;
  letter-spacing: 0.01em;
  position: relative;
  z-index: 1;
}

/* ── Stats row ── */
.ss-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin-top: 1.5rem;
}
@media (max-width: 600px) {
  .ss-stats { grid-template-columns: repeat(2, 1fr); }
}
.ss-stat {
  background: var(--bg-overlay);
  border: 1px solid var(--border-faint);
  border-radius: var(--r-lg);
  padding: 1rem 1.1rem;
  text-align: left;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.ss-stat:hover {
  border-color: var(--border-subtle);
}
.ss-stat::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--accent);
  opacity: 0;
  transition: opacity 0.2s;
}
.ss-stat:hover::before { opacity: 1; }
.ss-stat-value {
  font-family: var(--font-display);
  font-size: 1.7rem;
  font-weight: 800;
  color: var(--text-primary);
  line-height: 1.1;
  letter-spacing: -0.03em;
}
.ss-stat-label {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--text-muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 0.35rem;
}
.ss-stat.accent .ss-stat-value { color: var(--accent); }

/* ── Warning / Error ── */
.ss-warning {
  background: var(--error-dim);
  border: 1px solid rgba(248,113,113,0.25);
  border-left: 3px solid var(--error);
  border-radius: var(--r-md);
  padding: 0.9rem 1.1rem;
  color: #fca5a5;
  font-size: 0.9rem;
  display: flex;
  align-items: flex-start;
  gap: 0.7rem;
  animation: fadeUp 0.3s ease;
  line-height: 1.6;
}

/* ── Loading card ── */
.ss-loading {
  background: var(--bg-raised);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-xl);
  padding: 2.5rem;
  text-align: center;
  animation: fadeUp 0.35s ease;
}
.ss-loading-spinner {
  width: 36px; height: 36px;
  border: 3px solid var(--border-subtle);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 1.1rem;
}
.ss-loading-title {
  font-family: var(--font-display);
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 0.35rem;
}
.ss-loading-sub {
  color: var(--text-muted);
  font-size: 0.82rem;
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
}

/* ── Footer ── */
.ss-footer {
  text-align: center;
  margin-top: 5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border-faint);
  color: var(--text-disabled);
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  animation: fadeUp 0.5s 0.3s ease both;
}
.ss-footer span { color: var(--text-muted); }
.ss-footer a {
  color: var(--text-muted);
  text-decoration: none;
  border-bottom: 1px solid var(--border-faint);
  transition: color 0.2s, border-color 0.2s;
}
.ss-footer a:hover { color: var(--accent); border-color: var(--accent); }

/* ── Sidebar components ── */
.ss-sb-logo {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.4rem 0 1.2rem;
}
.ss-sb-mark {
  width: 30px; height: 30px;
  background: var(--accent);
  border-radius: 7px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 0.75rem;
  color: #050a14;
  flex-shrink: 0;
}
.ss-sb-name {
  font-family: var(--font-display);
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-primary);
}
.ss-sb-ver {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  color: var(--text-muted);
  letter-spacing: 0.08em;
  margin-top: 0.1rem;
}
.ss-sb-card {
  background: var(--bg-raised);
  border: 1px solid var(--border-faint);
  border-radius: var(--r-md);
  padding: 0.85rem 1rem;
  margin-bottom: 0.65rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.65;
}
.ss-sb-card strong { color: var(--text-primary); font-weight: 600; }
.ss-sb-card code {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  background: var(--bg-overlay);
  border: 1px solid var(--border-faint);
  border-radius: 4px;
  padding: 0.1rem 0.4rem;
  color: var(--accent);
}
.ss-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.2rem 0.55rem;
  border-radius: 99px;
  border: 1px solid var(--border-faint);
  background: var(--bg-overlay);
  color: var(--text-muted);
  margin: 0.15rem 0.12rem 0 0;
}
.ss-gh-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--accent);
  border-radius: var(--r-sm);
  padding: 0.5rem 0.9rem;
  color: #050a14 !important;
  font-family: var(--font-body);
  font-size: 0.8rem;
  font-weight: 600;
  text-decoration: none !important;
  transition: filter 0.18s, transform 0.18s;
  margin-top: 0.5rem;
  box-shadow: 0 2px 10px var(--accent-glow);
}
.ss-gh-btn:hover { filter: brightness(1.1); transform: translateY(-1px); }

/* Hide Streamlit default alerts */
.stAlert { display: none !important; }

/* Hide keyboard shortcut hints */
kbd, [aria-label*="keyboard"] { display: none !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 3. SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="ss-sb-logo">
      <div class="ss-sb-mark">SS</div>
      <div>
        <div class="ss-sb-name">S.S_AI</div>
        <div class="ss-sb-ver">Summarizer · v2.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### About")
    st.markdown("""
    <div class="ss-sb-card">
        <strong>S.S_AI Summarizer</strong> condenses long-form text into precise,
        AI-generated summaries using fine-tuned Transformer models.
        Runs entirely on your local machine.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Developer")
    st.markdown("""
    <div class="ss-sb-card">
        <strong>S.S</strong><br>
        AI/ML Engineer · NLP Specialist<br>
        <span style="color:#4c5668;font-size:0.78rem;">
            Building intelligent language systems.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model")
    st.markdown("""
    <div class="ss-sb-card">
        Architecture: <code>T5 / BART</code><br>
        Decoding: <code>Beam Search (n=4)</code><br>
        Max Input: <code>512 tokens</code><br>
        Max Output: <code>120 tokens</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Tech Stack")
    st.markdown("""
    <div class="ss-sb-card">
        <span class="ss-tag">🤗 HuggingFace</span>
        <span class="ss-tag">⚡ PyTorch</span>
        <span class="ss-tag">🧠 T5 / BART</span>
        <span class="ss-tag">🎈 Streamlit</span>
        <span class="ss-tag">🐍 Python 3.10+</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Links")
    st.markdown("""
    <div class="ss-sb-card">
        <a class="ss-gh-btn" href="https://github.com" target="_blank">
            ★ View on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.66rem;
                color:#2e3647;text-align:center;line-height:2;letter-spacing:0.05em;">
        MODEL RUNS LOCALLY<br>
        CUDA ACCELERATED WHEN AVAILABLE
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 4. MODEL LOADER
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    """Load tokenizer + model once and cache them across sessions."""
    model_path = "facebook/bart-large-cnn"   # ← Update this path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


# ══════════════════════════════════════════════════════
# 5. TOPBAR
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="ss-topbar">
  <div class="ss-logo">
    <div class="ss-logo-mark">SS</div>
    <div class="ss-logo-text">S.S<span>_AI</span></div>
  </div>
  <div class="ss-topbar-right">
    <div class="ss-status-dot">System Ready</div>
    <div class="ss-version-tag">v2.0</div>
  </div>
</div>
<div class="ss-rule"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 6. HERO SECTION
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="ss-hero">
  <div class="ss-hero-eyebrow">Transformer Intelligence</div>
  <div class="ss-hero-heading">
    Transform Long Text<br>into <em>Smart Summaries</em>
  </div>
  <div class="ss-hero-sub">
    Fine-tuned Transformer models instantly condense articles,
    research papers, and documents into clear, accurate summaries —
    running entirely on your machine.
  </div>
  <div class="ss-hero-caps">
    <div class="ss-cap">
      <div class="ss-cap-icon">⚡</div>
      GPU Accelerated
    </div>
    <div class="ss-cap">
      <div class="ss-cap-icon">🧠</div>
      Beam Search Decoding
    </div>
    <div class="ss-cap">
      <div class="ss-cap-icon">🔒</div>
      Fully Local · No API
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 7. INPUT SECTION
# ══════════════════════════════════════════════════════
# Character & word count (computed before render)
input_val   = st.session_state.get("input_text", "")
char_count  = len(input_val)
word_count_in = len(input_val.split()) if input_val.strip() else 0
limit       = 2000

status_cls  = "status-ok"  if char_count <= limit else "status-warn"
status_txt  = "Good length" if char_count <= limit else "Will be truncated"

st.markdown(f"""
<div class="ss-section-head">
  <div class="ss-section-label">Input Text</div>
  <div class="ss-counter">
    <span><span class="num">{word_count_in:,}</span> words</span>
    <span class="sep">·</span>
    <span><span class="num">{char_count:,}</span> chars</span>
    <span class="sep">·</span>
    <span class="{status_cls}">{status_txt}</span>
  </div>
</div>
""", unsafe_allow_html=True)

text = st.text_area(
    label="Input Text",
    height=230,
    placeholder="Paste your article, report, or research paper here…",
    label_visibility="collapsed",
    key="input_text",
)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ── Generate button ──
col_btn, _ = st.columns([1, 0.001])
with col_btn:
    generate = st.button("Generate Summary →", use_container_width=True)


# ══════════════════════════════════════════════════════
# 8. GENERATION LOGIC + OUTPUT
# ══════════════════════════════════════════════════════
if generate:

    # ── Validation ──
    if not text.strip():
        st.markdown("""
        <div class="ss-warning">
          ⚠ <span><strong>Empty input.</strong>
          Please paste some text before generating a summary.</span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Load model ──
    try:
        tokenizer, model, device = load_model()
    except Exception as e:
        st.markdown(f"""
        <div class="ss-warning">
          ⚠ <span><strong>Model failed to load.</strong>
          Check your <code>model_path</code> in the source code.<br>
          <small style="opacity:0.55">{e}</small></span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Loading UI ──
    loading_ph  = st.empty()
    progress_ph = st.empty()

    loading_ph.markdown("""
    <div class="ss-loading">
      <div class="ss-loading-spinner"></div>
      <div class="ss-loading-title">Processing your text</div>
      <div class="ss-loading-sub">Running beam search decoding…</div>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = progress_ph.progress(0)
    for pct in range(0, 72, 6):
        time.sleep(0.04)
        progress_bar.progress(pct)

    # ── Inference ──
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=120,
                num_beams=4,
                early_stopping=True,
            )

        summary = tokenizer.decode(output[0], skip_special_tokens=True)

    except Exception as e:
        loading_ph.empty()
        progress_ph.empty()
        st.markdown(f"""
        <div class="ss-warning">
          ⚠ <span><strong>Inference error:</strong> {e}</span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Finish progress
    for pct in range(72, 101, 4):
        time.sleep(0.025)
        progress_bar.progress(pct)
    time.sleep(0.12)
    loading_ph.empty()
    progress_ph.empty()

    # ── Stats ──
    word_count_out = len(summary.split())
    compression    = round((1 - word_count_out / max(word_count_in, 1)) * 100)
    device_lbl     = "GPU" if device == "cuda" else "CPU"

    # ── Output card ──
    st.markdown(f"""
    <div class="ss-output-card">
      <div class="ss-output-head">
        <div class="ss-output-label">
          <div class="ss-output-dot"></div>
          Summary Output
        </div>
        <button class="ss-copy-btn"
          onclick="navigator.clipboard.writeText('{summary.replace("'","&#39;").replace(chr(10)," ")}');
                   this.textContent='✓ Copied';
                   setTimeout(()=>this.textContent='Copy',1600);">
          Copy
        </button>
      </div>
      <div class="ss-summary-text">{summary}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat cards ──
    st.markdown(f"""
    <div class="ss-stats">
      <div class="ss-stat">
        <div class="ss-stat-value">{word_count_in:,}</div>
        <div class="ss-stat-label">Original Words</div>
      </div>
      <div class="ss-stat">
        <div class="ss-stat-value">{word_count_out:,}</div>
        <div class="ss-stat-label">Summary Words</div>
      </div>
      <div class="ss-stat accent">
        <div class="ss-stat-value">{compression}%</div>
        <div class="ss-stat-label">Compression</div>
      </div>
      <div class="ss-stat">
        <div class="ss-stat-value">{device_lbl}</div>
        <div class="ss-stat-label">Device Used</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.download_button(
        label="↓  Download Summary (.txt)",
        data=summary,
        file_name="ss_ai_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════
# 9. FOOTER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="ss-footer">
  <span>S.S_AI Summarizer</span> &nbsp;·&nbsp;
  Built with Transformer Intelligence &nbsp;·&nbsp;
  <a href="https://github.com" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
