/*
 * AgriScan — full-width layout.
 * Left panel: image upload.
 * Center panel: crop selector + readiness.
 * Right panel: model info + scan button.
 * Pipeline: full-width visual walkthrough of every processing stage.
 */
import { ChangeEvent, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  Camera,
  Check,
  ChevronDown,
  CircleHelp,
  Cpu,
  Eye,
  FileImage,
  FlaskConical,
  Layers,
  Leaf,
  Microscope,
  RefreshCw,
  ScanLine,
  ShieldCheck,
  Sparkles,
  SunMedium,
  Upload,
  X,
  Zap,
} from "lucide-react";

const crops = ["Auto-detect", "Wheat", "Rice", "Potato", "Cotton", "Sugarcane", "Tomato"];

// ─── Types ────────────────────────────────────────────────────────────────────

interface PipelineStep {
  id: string;
  label: string;
  description: string;
  image: string;
  meta?: { green_ratio?: number; is_leaf?: boolean };
}

interface PipelineResult {
  steps: PipelineStep[];
  summary: {
    original_size: string;
    original_mode: string;
    model_input: string;
    green_ratio: number;
    is_leaf: boolean;
  };
}

interface PredictionResult {
  disease_class: string;
  label: string;
  confidence: number;
  is_healthy: boolean;
  is_leaf: boolean;
  green_ratio: number;
  description: string;
  treatment: string;
  selected_crop: string;
  top_crops: Record<string, { label: string; confidence: number }>;
}

interface GradCamResult {
  overlay: string;     // base64 PNG — heatmap blended onto original
  heatmap: string;     // base64 PNG — raw false-colour heatmap
  layer_used: string;
  class_index: number;
  class_name: string;
  alpha: number;
}

// ─── Step colour palette ──────────────────────────────────────────────────────
const STEP_COLORS: Record<string, { bg: string; border: string; accent: string; tag: string }> = {
  original:   { bg: "#f6faf5", border: "#c9dece", accent: "#16724d", tag: "SOURCE"       },
  rgb:        { bg: "#f0f8ff", border: "#b8d6f0", accent: "#1e6fa8", tag: "CONVERT"      },
  resized:    { bg: "#faf5ff", border: "#d4b8f0", accent: "#7c3aed", tag: "RESIZE"        },
  green_mask: { bg: "#f0fdf4", border: "#86efac", accent: "#15803d", tag: "LEAF DETECT"  },
  normalized: { bg: "#fffbeb", border: "#fde68a", accent: "#b45309", tag: "NORMALIZE"    },
  channel_r:  { bg: "#fff1f2", border: "#fda4af", accent: "#be123c", tag: "RED CH"       },
  channel_g:  { bg: "#f0fdf4", border: "#86efac", accent: "#15803d", tag: "GREEN CH"     },
  channel_b:  { bg: "#eff6ff", border: "#93c5fd", accent: "#1d4ed8", tag: "BLUE CH"      },
};

function getStepColor(id: string) {
  return STEP_COLORS[id] ?? { bg: "#f6faf5", border: "#c9dece", accent: "#16724d", tag: "STEP" };
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Home() {
  const [selectedCrop, setSelectedCrop] = useState("Auto-detect");
  const [fileName, setFileName]         = useState("");
  const [previewUrl, setPreviewUrl]     = useState("");
  const [imageSize, setImageSize]       = useState<{ w: number; h: number } | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Pipeline
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [pipeline, setPipeline]               = useState<PipelineResult | null>(null);
  const [activeStep, setActiveStep]           = useState<string | null>(null);

  // Scan
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult]         = useState<PredictionResult | null>(null);
  const [error, setError]           = useState<string | null>(null);
  const [modelReady, setModelReady] = useState<boolean | null>(null);

  // Grad-CAM
  const [gradCam, setGradCam]             = useState<GradCamResult | null>(null);
  const [gradCamLoading, setGradCamLoading] = useState(false);
  const [gradCamError, setGradCamError]   = useState<string | null>(null);
  const [gradCamAlpha, setGradCamAlpha]   = useState(0.4);

  const fileInput   = useRef<HTMLInputElement>(null);
  const pipelineRef = useRef<HTMLDivElement>(null);
  const gradCamRef  = useRef<HTMLDivElement>(null);

  // ── Model health check ──────────────────────────────────────────────────────
  useEffect(() => {
    fetch("/api/health")
      .then((r) => setModelReady(r.ok))
      .catch(() => setModelReady(false));
  }, []);

  // ── File selection ──────────────────────────────────────────────────────────
  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);

    const url = URL.createObjectURL(file);
    setFileName(file.name);
    setPreviewUrl(url);
    setSelectedFile(file);
    setResult(null);
    setError(null);
    setPipeline(null);
    setActiveStep(null);

    const tmp = new Image();
    tmp.onload = () => setImageSize({ w: tmp.naturalWidth, h: tmp.naturalHeight });
    tmp.src = url;

    await runPipeline(file);
  }

  async function runPipeline(file: File) {
    setPipelineLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/process", { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data: PipelineResult = await res.json();
      setPipeline(data);
      setActiveStep(data.steps[0]?.id ?? null);
      setTimeout(() => pipelineRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 200);
    } catch (e) {
      console.warn("Pipeline failed:", e);
    } finally {
      setPipelineLoading(false);
    }
  }

  // ── Scan ────────────────────────────────────────────────────────────────────
  async function handleScan() {
    if (!selectedFile) return;
    setIsScanning(true);
    setError(null);
    setResult(null);
    setGradCam(null);
    setGradCamError(null);
    try {
      const fd = new FormData();
      fd.append("file", selectedFile);
      fd.append("crop", selectedCrop);
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.error ?? `Server error ${res.status}`);
      }
      setResult(await res.json());
      // Auto-run Grad-CAM after a successful prediction
      runGradCam(selectedFile, gradCamAlpha);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsScanning(false);
    }
  }

  // ── Grad-CAM ─────────────────────────────────────────────────────────────────
  async function runGradCam(file: File, alpha = gradCamAlpha) {
    setGradCamLoading(true);
    setGradCamError(null);
    setGradCam(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("alpha", String(alpha));
      const res = await fetch("/api/gradcam", { method: "POST", body: fd });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.error ?? `Grad-CAM error ${res.status}`);
      }
      const data: GradCamResult = await res.json();
      setGradCam(data);
      setTimeout(() => gradCamRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
    } catch (e: unknown) {
      setGradCamError(e instanceof Error ? e.message : String(e));
    } finally {
      setGradCamLoading(false);
    }
  }

  // ── Clear ───────────────────────────────────────────────────────────────────
  function clearImage() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFileName(""); setPreviewUrl(""); setSelectedFile(null);
    setResult(null); setError(null); setImageSize(null);
    setPipeline(null); setActiveStep(null);
    setGradCam(null); setGradCamError(null);
    if (fileInput.current) fileInput.current.value = "";
  }

  // ── Derived ─────────────────────────────────────────────────────────────────
  const pillClass  = modelReady === null ? "model-status-pill checking" : modelReady ? "model-status-pill" : "model-status-pill offline";
  const pillLabel  = modelReady === null ? "Connecting…" : modelReady ? "Model ready" : "API offline";
  const orbColor   = modelReady === null ? "#f5a623" : modelReady ? "#42a66d" : "#e05c5c";
  const orbShadow  = modelReady === null ? "0 0 0 3px #fde9c0" : modelReady ? "0 0 0 3px #d2edda" : "0 0 0 3px #f8d7d7";

  const isNatural = (id: string) => id === "original" || id === "rgb";
  const expandedStep = pipeline?.steps.find((s) => s.id === activeStep) ?? null;

  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">

      {/* ── Sticky top bar ────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-mark" aria-hidden="true">
            <img className="mark-image" src="/assets/agriscan-mark.png" alt="" />
          </div>
          <div>
            <div className="brand-name">AgriScan</div>
            <div className="brand-caption">Plant health intelligence</div>
          </div>
        </div>

        <nav className="topbar-center" aria-label="Page navigation">
          <a className="topbar-nav-link" href="#scan"><ScanLine size={15} /> New scan</a>
          <a className="topbar-nav-link" href="#pipeline"><Layers size={15} /> Processing</a>
          <a className="topbar-nav-link" href="#how-it-works"><CircleHelp size={15} /> How it works</a>
        </nav>

        <div className="topbar-actions">
          <span className="privacy-note"><ShieldCheck size={14} /> Local only</span>
          <div className={pillClass}>
            <span className="status-orb" style={{ background: orbColor, boxShadow: orbShadow }} />
            {pillLabel}
          </div>
          <button className="help-button" aria-label="Help"><CircleHelp size={18} /></button>
        </div>
      </header>

      {/* ── Main canvas ───────────────────────────────────────────────────── */}
      <main className="main-canvas">

        {/* Hero */}
        <section className="hero-block" id="scan">
          <div className="hero-copy">
            <div className="section-kicker">01 / Observe</div>
            <h1>See what your<br /><em>plants are telling you.</em></h1>
            <p className="hero-intro">
              Upload a leaf photo to get a real-time visual walkthrough of every
              processing step — and an instant AI-powered disease diagnosis.
            </p>
          </div>
          <div className="hero-aside">
            <div className="hero-photo">
              <img src="/assets/agriscan-hero-leaf.jpg" alt="Close-up of a healthy leaf" />
            </div>
            <div className="hero-aside-line" />
            <p>"Natural light, steady hand, full leaf in frame — the best scan starts here."</p>
            <span>FIELD NOTE 014</span>
          </div>
        </section>

        {/* API offline banner */}
        {modelReady === false && (
          <div className="api-banner">
            <AlertTriangle size={16} style={{ flexShrink: 0 }} />
            <span>
              <strong>Flask API is not running.</strong> Start it:{" "}
              <code>cd AgriScan &amp;&amp; python api.py</code>
            </span>
          </div>
        )}

        {/* ── 3-column scan workspace ─────────────────────────────────────── */}
        <section className="scan-workspace" aria-label="Leaf scan workspace">

          {/* ── Col 1: Image upload ───────────────────────────────────────── */}
          <div className="upload-panel paper-card">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">02 / Upload image</div>
                <h2>Add a leaf image</h2>
              </div>
              <span className="format-note">JPG · PNG</span>
            </div>

            {!previewUrl ? (
              <button
                className="dropzone"
                onClick={() => fileInput.current?.click()}
                aria-label="Choose a leaf image"
              >
                <div className="dropzone-orbit">
                  <img className="dropzone-mark" src="/assets/agriscan-mark.png" alt="" />
                  <Upload size={16} />
                </div>
                <span className="dropzone-title">Drop a leaf image here</span>
                <span className="dropzone-copy">or choose a file from your device</span>
                <span className="dropzone-meta"><Camera size={13} /> Camera-ready</span>
              </button>
            ) : (
              <div className="preview-stage">
                <img src={previewUrl} alt="Uploaded leaf preview" />
                <button className="clear-preview" onClick={clearImage} aria-label="Remove image">
                  <X size={16} />
                </button>
                <div className="preview-file"><FileImage size={13} /> {fileName}</div>
              </div>
            )}

            <input
              ref={fileInput}
              className="sr-only"
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleFileChange}
            />

            {/* Processing metadata chips */}
            <div className="image-meta-strip">
              <div className="meta-chip">
                <span className="meta-chip-label">Original size</span>
                <span className="meta-chip-value amber">
                  {imageSize ? `${imageSize.w} × ${imageSize.h}` : "— × —"}
                </span>
              </div>
              <div className="meta-chip">
                <span className="meta-chip-label">Model input</span>
                <span className="meta-chip-value">224 × 224</span>
              </div>
              <div className="meta-chip">
                <span className="meta-chip-label">Leaf signal</span>
                <span className={`meta-chip-value ${pipeline ? (pipeline.summary.is_leaf ? "green" : "red") : ""}`}>
                  {pipeline ? `${pipeline.summary.green_ratio}%` : "—"}
                </span>
              </div>
            </div>

            <div className="upload-footnote">
              <SunMedium size={14} /> Natural light gives the model a clearer signal.
              {pipelineLoading && <span style={{ marginLeft: 8, color: "var(--agri-green)", fontWeight: 700 }}>Running pipeline…</span>}
            </div>
          </div>

          {/* ── Col 2: Crop + readiness ───────────────────────────────────── */}
          <div className="crop-panel paper-card">
            <div className="panel-kicker">03 / Crop focus</div>
            <h2>Select a crop</h2>

            <label className="crop-select-inline">
              <span className="sr-only">Crop focus</span>
              <select value={selectedCrop} onChange={(e) => setSelectedCrop(e.target.value)}>
                {crops.map((c) => <option key={c}>{c}</option>)}
              </select>
              <ChevronDown size={14} />
            </label>
            <p className="crop-hint">
              Lock to one crop for a focused result, or use Auto-detect across all 22 classes.
            </p>

            <hr className="readiness-divider" />
            <div className="readiness-heading">Scan readiness</div>

            <div className="readiness-list">
              <div className={`readiness-row ${previewUrl ? "complete" : ""}`}>
                <span className="readiness-icon">{previewUrl ? <Check size={12} /> : "01"}</span>
                <span>Image uploaded</span>
                <span className="readiness-state">{previewUrl ? "Ready" : "Waiting"}</span>
              </div>
              <div className={`readiness-row ${pipeline ? "complete" : ""}`}>
                <span className="readiness-icon">{pipeline ? <Check size={12} /> : "02"}</span>
                <span>Pipeline processed</span>
                <span className="readiness-state">{pipelineLoading ? "Running…" : pipeline ? "Done" : "Waiting"}</span>
              </div>
              <div className={`readiness-row ${modelReady ? "complete" : ""}`}>
                <span className="readiness-icon">{modelReady ? <Check size={12} /> : "03"}</span>
                <span>API connected</span>
                <span className="readiness-state">
                  {modelReady === null ? "Checking…" : modelReady ? "Online" : "Offline"}
                </span>
              </div>
              <div className="readiness-row">
                <span className="readiness-icon">04</span>
                <span>Crop scope</span>
                <span className="readiness-state">{selectedCrop}</span>
              </div>
            </div>
          </div>

          {/* ── Col 3: Model info + scan ──────────────────────────────────── */}
          <div className="controls-panel paper-card">
            <div className="panel-kicker">04 / Run scan</div>
            <h2>Start analysis</h2>

            <div className="model-info-block">
              <div className="mi-label">Model details</div>
              <div className="model-info-row"><span>Architecture</span><strong>MobileNetV2</strong></div>
              <div className="model-info-row"><span>Classes</span><strong>22 diseases</strong></div>
              <div className="model-info-row"><span>Input size</span><strong>224 × 224 px</strong></div>
              <div className="model-info-row"><span>Pipeline steps</span><strong>8 stages</strong></div>
              <div className="model-info-row">
                <span>Status</span>
                <span className="model-status-inline" style={{ color: orbColor }}>
                  <Cpu size={12} /> {pillLabel}
                </span>
              </div>
            </div>

            <button
              className="scan-button"
              disabled={!previewUrl || isScanning || !modelReady}
              onClick={handleScan}
            >
              {isScanning
                ? <><RefreshCw size={16} className="spin" /> Analyzing…</>
                : <><Sparkles size={16} /> Run health scan <ArrowUpRight size={16} /></>}
            </button>

            <div className="readiness-disclaimer">
              <ShieldCheck size={13} /> Image sent only to your local server
            </div>
          </div>
        </section>

        {/* ── Error ─────────────────────────────────────────────────────────── */}
        {error && (
          <div className="error-banner">
            <AlertTriangle size={16} style={{ marginTop: 1, flexShrink: 0 }} />
            <div><strong>Scan failed:</strong> {error}</div>
          </div>
        )}

        {/* ─────────────────────────────────────────────────────────────────── */}
        {/* ── IMAGE PROCESSING PIPELINE — full visual walkthrough ─────────── */}
        {/* ─────────────────────────────────────────────────────────────────── */}
        {(pipelineLoading || pipeline) && (
          <section className="pipeline-section" id="pipeline" ref={pipelineRef}>

            {/* Section header */}
            <div className="pipeline-header">
              <div>
                <div className="pipeline-section-kicker">05 / Processing pipeline</div>
                <h2 className="pipeline-section-title">
                  <Layers size={22} style={{ verticalAlign: "middle", marginRight: 8 }} />
                  Every stage, made visible
                </h2>
                <p className="pipeline-section-sub">
                  Watch your image transform step-by-step through the full 8-stage preprocessing pipeline before it reaches the AI model.
                </p>
              </div>
              {pipeline && (
                <div className="pipeline-summary-badges">
                  <span className="pipeline-chip"><Leaf size={11} /> {pipeline.summary.original_size} → 224×224</span>
                  <span className="pipeline-chip"><Eye size={11} /> {pipeline.summary.original_mode} → RGB</span>
                  <span className={`pipeline-chip ${pipeline.summary.is_leaf ? "" : "red"}`}>
                    <Zap size={11} /> {pipeline.summary.green_ratio}% leaf signal — {pipeline.summary.is_leaf ? "✓ PASS" : "✗ FAIL"}
                  </span>
                </div>
              )}
            </div>

            {/* Loading state */}
            {pipelineLoading && (
              <div className="pipeline-loading-full">
                <div className="pipeline-loading-inner">
                  <div className="pipeline-spinner-large" />
                  <div>
                    <div className="pipeline-loading-title">Running image processing pipeline…</div>
                    <div className="pipeline-loading-sub">Extracting all 8 processing stages from your image</div>
                  </div>
                </div>
                <div className="pipeline-skeleton-grid">
                  {Array.from({ length: 8 }).map((_, i) => (
                    <div key={i} className="pipeline-skeleton-card">
                      <div className="pipeline-skeleton-img" style={{ animationDelay: `${i * 0.08}s` }} />
                      <div className="pipeline-skeleton-bar" style={{ animationDelay: `${i * 0.08 + 0.05}s` }} />
                      <div className="pipeline-skeleton-bar short" style={{ animationDelay: `${i * 0.08 + 0.1}s` }} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Full pipeline grid — visible when loaded */}
            {pipeline && !pipelineLoading && (
              <>
                {/* Step progress bar */}
                <div className="pipeline-progress-bar">
                  {pipeline.steps.map((step, idx) => {
                    const col = getStepColor(step.id);
                    const isActive = activeStep === step.id;
                    return (
                      <button
                        key={step.id}
                        className={`pipeline-progress-step ${isActive ? "active" : ""}`}
                        onClick={() => setActiveStep(isActive ? null : step.id)}
                        style={{
                          "--step-accent": col.accent,
                          "--step-border": col.border,
                          "--step-bg": col.bg,
                        } as React.CSSProperties}
                      >
                        <span className="psb-num">{idx + 1}</span>
                        <span className="psb-label">{col.tag}</span>
                        {idx < pipeline.steps.length - 1 && (
                          <ArrowRight size={10} className="psb-arrow" />
                        )}
                      </button>
                    );
                  })}
                </div>

                {/* ── Big image grid — ALL stages shown at once ───────────── */}
                <div className="pipeline-big-grid">
                  {pipeline.steps.map((step, idx) => {
                    const col = getStepColor(step.id);
                    const isActive = activeStep === step.id;
                    return (
                      <div
                        key={step.id}
                        className={`pipeline-big-card ${isActive ? "active" : ""}`}
                        onClick={() => setActiveStep(isActive ? null : step.id)}
                        style={{
                          "--step-accent": col.accent,
                          "--step-border": col.border,
                          "--step-bg": col.bg,
                        } as React.CSSProperties}
                      >
                        {/* Step number badge */}
                        <div className="pbc-badge">
                          <span className="pbc-num">{idx + 1}</span>
                          <span className="pbc-tag">{col.tag}</span>
                        </div>

                        {/* Image */}
                        <div className="pbc-img-wrap">
                          <img
                            className={`pbc-img ${isNatural(step.id) ? "natural" : ""}`}
                            src={step.image}
                            alt={step.label}
                          />
                          {isActive && <div className="pbc-img-ring" />}
                        </div>

                        {/* Label + description */}
                        <div className="pbc-body">
                          <div className="pbc-label">{step.label}</div>
                          <p className="pbc-desc">{step.description}</p>
                          {step.meta?.green_ratio !== undefined && (
                            <div className={`pbc-meta-badge ${step.meta.is_leaf ? "green" : "red"}`}>
                              <Zap size={9} />
                              {step.meta.green_ratio}% leaf signal — {step.meta.is_leaf ? "PASS ✓" : "FAIL ✗"}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* ── Expanded full-screen detail view ───────────────────── */}
                {expandedStep && (() => {
                  const col = getStepColor(expandedStep.id);
                  const stepIdx = pipeline.steps.findIndex(s => s.id === expandedStep.id);
                  return (
                    <div
                      className="pipeline-detail-view"
                      style={{ "--step-accent": col.accent, "--step-bg": col.bg, "--step-border": col.border } as React.CSSProperties}
                    >
                      <div className="pdv-left">
                        <div className="pdv-badge-row">
                          <span className="pdv-step-num">Step {stepIdx + 1}</span>
                          <span className="pdv-tag" style={{ background: col.accent }}>{col.tag}</span>
                        </div>
                        <h3 className="pdv-title">{expandedStep.label}</h3>
                        <p className="pdv-desc">{expandedStep.description}</p>

                        {expandedStep.meta?.green_ratio !== undefined && (
                          <div className={`pdv-meta ${expandedStep.meta.is_leaf ? "pass" : "fail"}`}>
                            <Zap size={13} />
                            <div>
                              <strong>{expandedStep.meta.green_ratio}% leaf signal</strong>
                              <span>{expandedStep.meta.is_leaf ? " — Passes leaf threshold (≥10%)" : " — Below leaf threshold"}</span>
                            </div>
                          </div>
                        )}

                        {/* Step navigation */}
                        <div className="pdv-nav">
                          <button
                            className="pdv-nav-btn"
                            disabled={stepIdx === 0}
                            onClick={(e) => { e.stopPropagation(); setActiveStep(pipeline.steps[stepIdx - 1].id); }}
                          >← Prev</button>
                          <span className="pdv-nav-count">{stepIdx + 1} / {pipeline.steps.length}</span>
                          <button
                            className="pdv-nav-btn"
                            disabled={stepIdx === pipeline.steps.length - 1}
                            onClick={(e) => { e.stopPropagation(); setActiveStep(pipeline.steps[stepIdx + 1].id); }}
                          >Next →</button>
                        </div>
                      </div>

                      <div className="pdv-right">
                        <img
                          className={`pdv-img ${isNatural(expandedStep.id) ? "natural" : ""}`}
                          src={expandedStep.image}
                          alt={expandedStep.label}
                        />
                        <div className="pdv-img-caption">
                          Click any card above to inspect that stage
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </>
            )}
          </section>
        )}

        {/* ── Prediction Result ──────────────────────────────────────────── */}
        {result && (
          <section className="result-dossier paper-card" aria-live="polite">
            <div className="result-topline">
              <div className="panel-kicker">06 / Scan result</div>
              <div className="result-badge">
                <span style={{ background: result.is_healthy ? "#42a66d" : "#e05c5c" }} />
                {result.is_healthy ? "Healthy" : "Issue detected"}
              </div>
            </div>

            {!result.is_leaf && (
              <div style={{
                display: "flex", alignItems: "center", gap: "0.5rem",
                background: "#fffbea", border: "1px solid #f5c842", borderRadius: "10px",
                padding: "8px 13px", marginTop: "12px", color: "#92620a", fontSize: "12px",
              }}>
                <AlertTriangle size={14} />
                Low leaf signal ({result.green_ratio}%) — results may be less accurate.
              </div>
            )}

            <div className="result-heading-row">
              <div>
                <h2 style={{ color: result.is_healthy ? "#16724d" : "#b9634d" }}>
                  {result.is_healthy ? "Healthy leaf detected" : "Potential issue detected"}
                </h2>
                <div className="result-label">{result.label}</div>
              </div>
              <div className="result-confidence">
                <strong>{result.confidence}%</strong>
                <span>model confidence</span>
              </div>
            </div>

            <div className="confidence-track">
              <div
                className="confidence-fill"
                style={{
                  width: `${result.confidence}%`,
                  background: result.confidence >= 70
                    ? "linear-gradient(90deg,#e7b94a,#65ad67,#16724d)"
                    : "linear-gradient(90deg,#f5c842,#e07b3a)",
                }}
              />
            </div>

            {result.confidence < 70 && (
              <div style={{
                fontSize: "11px", color: "#765e22", background: "#fff9e9",
                border: "1px solid #f4e5ba", borderRadius: "10px",
                padding: "8px 12px", marginBottom: "12px",
              }}>
                Confidence below 70% — retake in natural light with the full leaf in frame.
              </div>
            )}

            <div className="result-columns">
              <div className="advice-block">
                <span className="advice-index">A</span>
                <div><h3>What we found</h3><p>{result.description}</p></div>
              </div>
              <div className="advice-block">
                <span className="advice-index green">B</span>
                <div><h3>Recommended next step</h3><p>{result.treatment}</p></div>
              </div>
            </div>

            {Object.keys(result.top_crops).length > 0 && (
              <details style={{ marginTop: "14px" }}>
                <summary style={{
                  cursor: "pointer", fontSize: "9px", fontWeight: 700,
                  textTransform: "uppercase", letterSpacing: ".12em", color: "#71827a",
                  display: "flex", alignItems: "center", gap: "5px",
                }}>
                  <Zap size={12} /> All-crop breakdown
                </summary>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginTop: "12px" }}>
                  {Object.entries(result.top_crops).map(([crop, info]) => (
                    <div key={crop} style={{
                      background: "#f6faf6", border: "1px solid #e5eee8",
                      borderRadius: "10px", padding: "8px 12px", fontSize: "11px",
                    }}>
                      <div style={{ fontWeight: 700, color: "#18332a" }}>{crop}</div>
                      <div style={{ color: "#71827a" }}>{info.label} · {info.confidence}%</div>
                    </div>
                  ))}
                </div>
              </details>
            )}

            <div className="result-footer">
              <span>Real model output · {result.disease_class}</span>
              <button onClick={clearImage}>Start another scan <ArrowUpRight size={14} /></button>
            </div>
          </section>
        )}

        {/* ── Grad-CAM explainability panel ──────────────────────────────── */}
        {(gradCamLoading || gradCam || gradCamError) && result && (
          <section className="gradcam-section" ref={gradCamRef} id="gradcam">

            {/* Header */}
            <div className="gradcam-header">
              <div>
                <div className="gradcam-kicker"><Microscope size={12} /> 07 / AI Visual Explanation</div>
                <h2 className="gradcam-title">Grad-CAM Heatmap</h2>
                <p className="gradcam-subtitle">
                  Which regions of the leaf drove this prediction?
                  Red/yellow = high influence · Blue = low influence.
                </p>
              </div>

              {/* Alpha slider */}
              <div className="gradcam-controls">
                <label className="gradcam-alpha-label">
                  <span>Heatmap intensity</span>
                  <input
                    type="range" min={0.1} max={0.8} step={0.05}
                    value={gradCamAlpha}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      setGradCamAlpha(v);
                      if (selectedFile) runGradCam(selectedFile, v);
                    }}
                  />
                  <span className="gradcam-alpha-val">{Math.round(gradCamAlpha * 100)}%</span>
                </label>
              </div>
            </div>

            {/* Loading */}
            {gradCamLoading && (
              <div className="gradcam-loading">
                <div className="gradcam-spinner" />
                <div>
                  <div className="gradcam-loading-title">Computing Grad-CAM…</div>
                  <div className="gradcam-loading-sub">Tracing gradients through the model backbone</div>
                </div>
              </div>
            )}

            {/* Error */}
            {gradCamError && !gradCamLoading && (
              <div className="error-banner">
                <AlertTriangle size={14} />
                <span><strong>Grad-CAM failed:</strong> {gradCamError}</span>
              </div>
            )}

            {/* Result */}
            {gradCam && !gradCamLoading && (
              <div className="gradcam-result">

                {/* Two image panels */}
                <div className="gradcam-images">
                  {/* Overlay */}
                  <div className="gradcam-img-card">
                    <div className="gradcam-img-badge overlay">OVERLAY</div>
                    <img
                      className="gradcam-img"
                      src={gradCam.overlay}
                      alt="Grad-CAM heatmap overlay on leaf image"
                    />
                    <div className="gradcam-img-caption">
                      Heatmap blended with original leaf
                    </div>
                  </div>

                  {/* Raw heatmap */}
                  <div className="gradcam-img-card">
                    <div className="gradcam-img-badge heatmap">HEATMAP</div>
                    <img
                      className="gradcam-img"
                      src={gradCam.heatmap}
                      alt="Raw Grad-CAM false-colour heatmap"
                    />
                    <div className="gradcam-img-caption">
                      Raw activation map (no original)
                    </div>
                  </div>
                </div>

                {/* Colour legend */}
                <div className="gradcam-legend">
                  <span className="gradcam-legend-label">Low activation</span>
                  <div className="gradcam-legend-bar" />
                  <span className="gradcam-legend-label">High activation</span>
                </div>

                {/* Technical metadata */}
                <div className="gradcam-meta">
                  <div className="gradcam-meta-chip">
                    <FlaskConical size={11} />
                    <span>Conv layer: <strong>{gradCam.layer_used}</strong></span>
                  </div>
                  <div className="gradcam-meta-chip">
                    <Zap size={11} />
                    <span>Class explained: <strong>{gradCam.class_name.replace(/___/g, " · ").replace(/_/g, " ")}</strong></span>
                  </div>
                  <div className="gradcam-meta-chip">
                    <Eye size={11} />
                    <span>Blend alpha: <strong>{Math.round(gradCam.alpha * 100)}%</strong></span>
                  </div>
                </div>

              </div>
            )}
          </section>
        )}

        {/* ── How it works ─────────────────────────────────────────────────── */}
        <section className="how-section" id="how-it-works">
          <div className="how-intro">
            <div className="section-kicker">The method</div>
            <h2>From leaf<br /><em>to field note.</em></h2>
            <img className="method-image" src="/assets/agriscan-crop-detail.jpg" alt="Macro detail of a crop leaf" />
          </div>
          <div className="method-steps">
            <div className="method-step">
              <span>01</span><h3>Frame it</h3>
              <p>Fill the view with one leaf and find soft, natural light.</p>
            </div>
            <div className="method-step">
              <span>02</span><h3>Process it</h3>
              <p>Watch the 8-stage pipeline — RGB conversion, resize, leaf detection, channel splits, and normalisation.</p>
            </div>
            <div className="method-step">
              <span>03</span><h3>Act on it</h3>
              <p>Use the diagnosis as a first read, then confirm with a local agronomist.</p>
            </div>
          </div>
        </section>

        <footer className="site-footer">
          <div className="footer-brand">
            <div className="brand-mark small"><img className="mark-image" src="/assets/agriscan-mark.png" alt="" /></div>
            <span>AgriScan</span>
          </div>
          <span>Transfer-learning plant disease detection</span>
          <span>Built for better field decisions.</span>
        </footer>
      </main>
    </div>
  );
}
