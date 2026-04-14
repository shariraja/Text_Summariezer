"""
╔══════════════════════════════════════════════════════════════════╗
║              S.S_AI Summarizer — Premium SaaS UI                ║
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
# 2. GLOBAL CSS — Dark Glass SaaS Design System
# ══════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* ── CSS Variables ── */
:root {
  --bg-base:        #080c14;
  --bg-surface:     #0d1321;
  --bg-glass:       rgba(255,255,255,0.035);
  --bg-glass-hover: rgba(255,255,255,0.065);
  --border-subtle:  rgba(255,255,255,0.07);
  --border-glow:    rgba(99,179,237,0.35);

  --neon-blue:      #63b3ed;
  --neon-purple:    #b794f4;
  --neon-cyan:      #4fd1c5;
  --glow-blue:      rgba(99,179,237,0.25);
  --glow-purple:    rgba(183,148,244,0.25);

  --grad-primary:   linear-gradient(135deg, #63b3ed 0%, #b794f4 100%);
  --grad-subtle:    linear-gradient(135deg, rgba(99,179,237,0.15) 0%, rgba(183,148,244,0.15) 100%);

  --text-primary:   #e8edf5;
  --text-secondary: #8a95a8;
  --text-muted:     #4a5568;

  --font-display:   'Syne', sans-serif;
  --font-body:      'Space Grotesk', sans-serif;

  --radius-sm:      8px;
  --radius-md:      14px;
  --radius-lg:      22px;
  --radius-xl:      32px;

  --shadow-glow-blue:   0 0 40px rgba(99,179,237,0.18);
  --shadow-glow-purple: 0 0 40px rgba(183,148,244,0.18);
  --shadow-card:        0 8px 40px rgba(0,0,0,0.55);
}

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-base) !important;
  font-family: var(--font-body) !important;
  color: var(--text-primary) !important;
}

/* Remove default Streamlit chrome */
#MainMenu, footer, header { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--neon-blue); border-radius: 99px; }

/* ── Main container width ── */
[data-testid="stAppViewContainer"] > .main > .block-container {
  max-width: 920px !important;
  padding: 2rem 2rem 4rem !important;
  margin: 0 auto !important;
}

/* ════════════════════════════
   SIDEBAR
════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
  color: var(--neon-blue) !important;
  font-family: var(--font-display) !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  margin-bottom: 0.6rem !important;
}

/* ════════════════════════════
   TEXTAREA
════════════════════════════ */
textarea {
  background: var(--bg-glass) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.97rem !important;
  line-height: 1.7 !important;
  padding: 1.1rem 1.3rem !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
  resize: vertical !important;
}
textarea:focus {
  border-color: var(--neon-blue) !important;
  box-shadow: 0 0 0 3px rgba(99,179,237,0.12), var(--shadow-glow-blue) !important;
  outline: none !important;
}
textarea::placeholder { color: var(--text-muted) !important; }

/* Label above textarea */
[data-testid="stTextArea"] label {
  color: var(--text-secondary) !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

/* ════════════════════════════
   BUTTONS
════════════════════════════ */
.stButton > button {
  width: 100% !important;
  background: var(--grad-primary) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  color: #fff !important;
  font-family: var(--font-body) !important;
  font-size: 1rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  padding: 0.85rem 1.5rem !important;
  cursor: pointer !important;
  transition: transform 0.2s, box-shadow 0.2s, filter 0.2s !important;
  box-shadow: 0 4px 24px rgba(99,179,237,0.3), 0 2px 8px rgba(183,148,244,0.2) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) scale(1.012) !important;
  filter: brightness(1.12) !important;
  box-shadow: 0 8px 36px rgba(99,179,237,0.45), 0 4px 16px rgba(183,148,244,0.3) !important;
}
.stButton > button:active {
  transform: translateY(0) scale(0.99) !important;
}

/* ════════════════════════════
   PROGRESS BAR
════════════════════════════ */
.stProgress > div > div {
  background: var(--grad-primary) !important;
  border-radius: 99px !important;
}
.stProgress > div {
  background: var(--bg-glass) !important;
  border-radius: 99px !important;
  height: 5px !important;
}

/* ════════════════════════════
   SPINNER
════════════════════════════ */
.stSpinner > div {
  border-top-color: var(--neon-blue) !important;
}

/* ════════════════════════════
   SELECT / SLIDER
════════════════════════════ */
.stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--neon-blue) !important;
}

/* ════════════════════════════
   ANIMATIONS
════════════════════════════ */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes pulse-ring {
  0%   { box-shadow: 0 0 0 0 rgba(99,179,237,0.4); }
  70%  { box-shadow: 0 0 0 14px rgba(99,179,237,0); }
  100% { box-shadow: 0 0 0 0 rgba(99,179,237,0); }
}
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}

/* ════════════════════════════
   COMPONENT CLASSES
════════════════════════════ */

/* ── Header ── */
.ss-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.4rem 0 1rem;
  animation: fadeInUp 0.55s ease both;
}
.ss-logo {
  font-family: var(--font-display);
  font-size: 1.35rem;
  font-weight: 800;
  background: var(--grad-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.02em;
  cursor: default;
}
.ss-logo span {
  display: inline-block;
  animation: pulse-ring 2.8s cubic-bezier(0.455,0.03,0.515,0.955) infinite;
  border-radius: 6px;
  padding: 2px 8px;
  border: 1px solid rgba(99,179,237,0.3);
}
.ss-header-title {
  font-family: var(--font-body);
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--text-secondary);
  letter-spacing: 0.01em;
}
.ss-tagline {
  font-size: 0.72rem;
  color: var(--text-muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  text-align: right;
}
.ss-tagline span {
  color: var(--neon-blue);
}

/* ── Divider ── */
.ss-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--neon-blue), var(--neon-purple), transparent);
  opacity: 0.5;
  margin: 0.2rem 0 1.8rem;
  animation: fadeInUp 0.6s 0.1s ease both;
}

/* ── Hero ── */
.ss-hero {
  text-align: center;
  padding: 1.2rem 0 2.2rem;
  animation: fadeInUp 0.6s 0.15s ease both;
}
.ss-hero-heading {
  font-family: var(--font-display);
  font-size: clamp(1.9rem, 4.5vw, 2.9rem);
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #e8edf5 0%, var(--neon-blue) 40%, var(--neon-purple) 80%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: shimmer 4s linear infinite;
  margin-bottom: 0.9rem;
}
.ss-hero-sub {
  color: var(--text-secondary);
  font-size: 1.02rem;
  line-height: 1.6;
  max-width: 560px;
  margin: 0 auto;
}
.ss-hero-badges {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  margin-top: 1.2rem;
  flex-wrap: wrap;
}
.ss-badge {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  padding: 0.3rem 0.8rem;
  border-radius: 99px;
  border: 1px solid var(--border-subtle);
  background: var(--bg-glass);
  color: var(--text-secondary);
}
.ss-badge.blue  { border-color: rgba(99,179,237,0.35); color: var(--neon-blue); }
.ss-badge.purple{ border-color: rgba(183,148,244,0.35); color: var(--neon-purple); }
.ss-badge.cyan  { border-color: rgba(79,209,197,0.35);  color: var(--neon-cyan); }

/* ── Section label ── */
.ss-label {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 0.55rem;
}

/* ── Counter row ── */
.ss-counter-row {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 1rem;
  margin-top: 0.45rem;
  font-size: 0.77rem;
  color: var(--text-muted);
}
.ss-counter-row .count { color: var(--neon-blue); font-weight: 600; }
.ss-counter-row .divider-dot { opacity: 0.3; }

/* ── Output card ── */
.ss-output-card {
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.8rem;
  margin-top: 0.6rem;
  box-shadow: var(--shadow-card);
  animation: fadeInUp 0.5s ease both;
  position: relative;
  overflow: hidden;
}
.ss-output-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: var(--grad-subtle);
  border-radius: inherit;
  opacity: 0.6;
  pointer-events: none;
}
.ss-output-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}
.ss-output-title {
  font-family: var(--font-display);
  font-size: 1rem;
  font-weight: 700;
  background: var(--grad-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.ss-output-actions {
  display: flex;
  gap: 0.5rem;
}
.ss-action-btn {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.3rem 0.75rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
  background: var(--bg-glass);
  color: var(--text-secondary);
  cursor: pointer;
  transition: border-color 0.2s, color 0.2s, background 0.2s;
}
.ss-action-btn:hover {
  border-color: var(--neon-blue);
  color: var(--neon-blue);
  background: rgba(99,179,237,0.08);
}
.ss-summary-text {
  color: var(--text-primary);
  font-size: 1.03rem;
  line-height: 1.8;
  position: relative;
  z-index: 1;
}

/* ── Stats row ── */
.ss-stats {
  display: flex;
  gap: 1rem;
  margin-top: 1.3rem;
  flex-wrap: wrap;
}
.ss-stat {
  flex: 1;
  min-width: 120px;
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 0.9rem 1rem;
  text-align: center;
}
.ss-stat-value {
  font-family: var(--font-display);
  font-size: 1.6rem;
  font-weight: 800;
  background: var(--grad-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
}
.ss-stat-label {
  font-size: 0.7rem;
  color: var(--text-muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-top: 0.3rem;
}

/* ── Warning / Error ── */
.ss-warning {
  background: rgba(245,101,101,0.08);
  border: 1px solid rgba(245,101,101,0.3);
  border-radius: var(--radius-md);
  padding: 0.85rem 1.1rem;
  color: #fc8181;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  animation: fadeInUp 0.3s ease;
}

/* ── Loading card ── */
.ss-loading {
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 2rem;
  text-align: center;
  animation: fadeInUp 0.4s ease;
}
.ss-loading-title {
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-weight: 700;
  background: var(--grad-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.4rem;
}
.ss-loading-sub { color: var(--text-muted); font-size: 0.85rem; }

/* ── Footer ── */
.ss-footer {
  text-align: center;
  margin-top: 4rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border-subtle);
  color: var(--text-muted);
  font-size: 0.8rem;
  letter-spacing: 0.04em;
  animation: fadeInUp 0.6s 0.3s ease both;
}
.ss-footer span { color: var(--neon-blue); }

/* ── Sidebar styles ── */
.ss-sidebar-logo {
  font-family: var(--font-display);
  font-size: 1.4rem;
  font-weight: 800;
  background: var(--grad-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.3rem;
}
.ss-sidebar-section {
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 0.9rem 1rem;
  margin-bottom: 0.8rem;
  font-size: 0.87rem;
  color: var(--text-secondary);
  line-height: 1.6;
}
.ss-sidebar-section strong { color: var(--text-primary); }
.ss-chip {
  display: inline-block;
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  padding: 0.2rem 0.6rem;
  border-radius: 99px;
  border: 1px solid var(--border-subtle);
  background: var(--bg-glass);
  color: var(--neon-purple);
  margin: 0.15rem 0.15rem 0.15rem 0;
}
.ss-gh-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--grad-primary);
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.9rem;
  color: #fff !important;
  font-size: 0.82rem;
  font-weight: 600;
  text-decoration: none !important;
  width: fit-content;
  margin-top: 0.5rem;
  transition: filter 0.2s, transform 0.2s;
}
.ss-gh-link:hover { filter: brightness(1.15); transform: translateY(-1px); }

/* Override Streamlit success/info/warning boxes */
.stAlert { display: none !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 3. SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="ss-sidebar-logo">S.S_AI</div>
    <div style="font-size:0.72rem;color:#4a5568;letter-spacing:0.1em;
                text-transform:uppercase;margin-bottom:1.2rem;">
        Summarizer · v2.0
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 About")
    st.markdown("""
    <div class="ss-sidebar-section">
        <strong>S.S_AI Summarizer</strong> condenses long-form text into precise,
        AI-generated summaries using fine-tuned Transformer models (T5 / BART).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👨‍💻 Developer")
    st.markdown("""
    <div class="ss-sidebar-section">
        <strong>S.S</strong><br>
        AI/ML Engineer · NLP Specialist<br>
        <span style="color:#4a5568;font-size:0.78rem;">Building intelligent language systems.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔧 Tech Stack")
    st.markdown("""
    <div class="ss-sidebar-section">
        <span class="ss-chip">🤗 HuggingFace</span>
        <span class="ss-chip">⚡ PyTorch</span>
        <span class="ss-chip">🧠 T5 / BART</span>
        <span class="ss-chip">🎈 Streamlit</span>
        <span class="ss-chip">🐍 Python 3.10+</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔗 Links")
    st.markdown("""
    <div class="ss-sidebar-section">
        <a class="ss-gh-link" href="https://github.com" target="_blank">
            ⭐ View on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem;color:#2d3748;text-align:center;line-height:1.8;">
        Model runs locally on your machine.<br>
        GPU accelerated when CUDA is available.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 4. MODEL LOADER
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    """Load tokenizer + model once and cache them across sessions."""
    model_path = "G:/My Drive/my_summarizer"   # ← Update this path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


# ══════════════════════════════════════════════════════
# 5. HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="ss-header">
  <div class="ss-logo"><span>S.S_AI</span></div>
  <div class="ss-header-title">AI Text Summarizer</div>
  <div class="ss-tagline">Powered by <span>Transformer Intelligence</span></div>
</div>
<div class="ss-divider"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 6. HERO SECTION
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="ss-hero">
  <div class="ss-hero-heading">Transform Long Text<br>into Smart Summaries</div>
  <div class="ss-hero-sub">
    Leverage fine-tuned Transformer intelligence to instantly condense articles,
    research papers, and documents into clear, accurate summaries.
  </div>
  <div class="ss-hero-badges">
    <span class="ss-badge blue">⚡ GPU Accelerated</span>
    <span class="ss-badge purple">🧠 Transformer Model</span>
    <span class="ss-badge cyan">✨ Beam Search Decoding</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 7. INPUT SECTION
# ══════════════════════════════════════════════════════
st.markdown('<div class="ss-label">📝 Input Text</div>', unsafe_allow_html=True)

text = st.text_area(
    label="Input Text",
    height=220,
    placeholder="Paste your long text here — articles, reports, research papers…",
    label_visibility="collapsed",
    key="input_text",
)

# Character & word counter
char_count = len(text)
word_count_in = len(text.split()) if text.strip() else 0
limit        = 2000  # soft display limit (model truncates at 512 tokens)

st.markdown(f"""
<div class="ss-counter-row">
  <span><span class="count">{word_count_in:,}</span> words</span>
  <span class="divider-dot">•</span>
  <span><span class="count">{char_count:,}</span> characters</span>
  <span class="divider-dot">•</span>
  <span style="color:{'#68d391' if char_count <= limit else '#fc8181'}">
    {'✓ Good length' if char_count <= limit else '⚠ Very long — will be truncated to 512 tokens'}
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Generate button ──
col_btn, _ = st.columns([1, 0.001])
with col_btn:
    generate = st.button("✨ Generate Summary", use_container_width=True)


# ══════════════════════════════════════════════════════
# 8. GENERATION LOGIC + OUTPUT
# ══════════════════════════════════════════════════════
if generate:

    # ── Validation ──
    if not text.strip():
        st.markdown("""
        <div class="ss-warning">
          ⚠️ <strong>Empty input.</strong> Please paste some text before generating.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Load model (with graceful error) ──
    try:
        tokenizer, model, device = load_model()
    except Exception as e:
        st.markdown(f"""
        <div class="ss-warning">
          ⚠️ <strong>Model failed to load.</strong>
          Check your <code>model_path</code> in the code.<br>
          <small style="opacity:0.6">{e}</small>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Loading UI ──
    loading_placeholder = st.empty()
    progress_placeholder = st.empty()

    loading_placeholder.markdown("""
    <div class="ss-loading">
      <div class="ss-loading-title">🧠 AI is thinking…</div>
      <div class="ss-loading-sub">Running beam search decoding on your text</div>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = progress_placeholder.progress(0)

    # Animate progress bar while model runs
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
        loading_placeholder.empty()
        progress_placeholder.empty()
        st.markdown(f"""
        <div class="ss-warning">
          ⚠️ <strong>Inference error:</strong> {e}
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Finish progress bar
    for pct in range(72, 101, 4):
        time.sleep(0.025)
        progress_bar.progress(pct)

    time.sleep(0.15)
    loading_placeholder.empty()
    progress_placeholder.empty()

    # ── Output card ──
    word_count_out = len(summary.split())
    compression    = round((1 - word_count_out / max(word_count_in, 1)) * 100)

    st.markdown(f"""
    <div class="ss-output-card">
      <div class="ss-output-top">
        <div class="ss-output-title">📌 AI Summary</div>
        <div class="ss-output-actions">
          <button class="ss-action-btn"
            onclick="navigator.clipboard.writeText('{summary.replace("'", "&#39;")}');
                     this.textContent='✓ Copied!';
                     setTimeout(()=>this.textContent='📋 Copy',1500);">
            📋 Copy
          </button>
        </div>
      </div>
      <div class="ss-summary-text">{summary}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats ──
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
      <div class="ss-stat">
        <div class="ss-stat-value">{compression}%</div>
        <div class="ss-stat-label">Compression</div>
      </div>
      <div class="ss-stat">
        <div class="ss-stat-value">{"GPU" if device == "cuda" else "CPU"}</div>
        <div class="ss-stat-label">Device Used</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download button ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️  Download Summary (.txt)",
        data=summary,
        file_name="ss_ai_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # ── Expandable full summary ──
    with st.expander("🔍 View Full Summary in Expanded Mode"):
        st.markdown(f"""
        <div style="font-size:1.05rem;line-height:1.9;color:var(--text-primary,#e8edf5);
                    padding:0.5rem 0;">
          {summary}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 9. FOOTER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="ss-footer">
  Built with <span>❤️</span> using AI &nbsp;·&nbsp;
  <span>S.S_AI Summarizer</span> &nbsp;·&nbsp;
  Transformer Intelligence
</div>
""", unsafe_allow_html=True)