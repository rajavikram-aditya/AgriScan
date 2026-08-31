/*
 * AgriScan field-notes editorial direction:
 * The scan is staged as a calm field report: a persistent workspace rail,
 * tactile paper surfaces, numbered steps, and clear green/amber signals.
 */
import { ChangeEvent, useMemo, useRef, useState } from "react";
import {
  ArrowUpRight,
  Camera,
  Check,
  ChevronDown,
  CircleHelp,
  FileImage,
  Leaf,
  Menu,
  RefreshCw,
  ScanLine,
  ShieldCheck,
  Sparkles,
  SunMedium,
  Upload,
  X,
} from "lucide-react";

const crops = ["Auto-detect", "Wheat", "Rice", "Potato", "Cotton", "Sugarcane"];

const sampleResult = {
  title: "Potential issue detected",
  label: "Wheat · Leaf blight",
  confidence: 86,
  description:
    "Leaf blight causes yellowish-brown spots that can expand into irregular areas of damaged tissue. Early attention can help protect new growth.",
  nextStep:
    "Use a resistant variety where possible and inspect nearby leaves. If symptoms continue to spread, consult a local agronomist about an early foliar treatment.",
};

export default function Home() {
  const [selectedCrop, setSelectedCrop] = useState("Auto-detect");
  const [fileName, setFileName] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [isScanning, setIsScanning] = useState(false);
  const [hasResult, setHasResult] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const activeResult = useMemo(() => {
    if (selectedCrop === "Auto-detect") return sampleResult;
    return { ...sampleResult, label: `${selectedCrop} · Leaf health scan` };
  }, [selectedCrop]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFileName(file.name);
    setPreviewUrl(URL.createObjectURL(file));
    setHasResult(false);
  }

  function handleScan() {
    setIsScanning(true);
    window.setTimeout(() => {
      setIsScanning(false);
      setHasResult(true);
    }, 720);
  }

  function clearImage() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFileName("");
    setPreviewUrl("");
    setHasResult(false);
    if (fileInput.current) fileInput.current.value = "";
  }

  return (
    <div className="app-shell">
      <aside className={`workspace-rail ${mobileNavOpen ? "open" : ""}`}>
        <div className="rail-brand">
          <div className="brand-mark" aria-hidden="true">
            <img className="mark-image" src="/manus-storage/agriscan-mark_0ca6187a.png" alt="" />
          </div>
          <div>
            <div className="brand-name">AgriScan</div>
            <div className="brand-caption">Plant health intelligence</div>
          </div>
          <button className="rail-close" onClick={() => setMobileNavOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>

        <div className="rail-section">
          <div className="rail-eyebrow">Workspace</div>
          <nav className="rail-nav" aria-label="Primary navigation">
            <a className="rail-link active" href="#scan"><ScanLine size={17} /> New scan <span className="rail-dot" /></a>
            <a className="rail-link" href="#how-it-works"><CircleHelp size={17} /> How it works</a>
          </nav>
        </div>

        <div className="rail-section crop-section">
          <div className="rail-eyebrow">Crop focus</div>
          <label className="crop-select">
            <span className="sr-only">Crop focus</span>
            <select value={selectedCrop} onChange={(event) => setSelectedCrop(event.target.value)}>
              {crops.map((crop) => <option key={crop}>{crop}</option>)}
            </select>
            <ChevronDown size={15} />
          </label>
          <p className="rail-hint">Lock the scan to one crop when you already know what is growing in the field.</p>
        </div>

        <div className="rail-bottom">
          <div className="model-status"><span className="status-orb" /> Model ready</div>
          <div className="model-note">22 crop and disease classes<br />MobileNetV2 transfer learning</div>
        </div>
      </aside>

      <main className="main-canvas">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileNavOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
          <div className="topbar-note"><span className="topbar-leaf"><Leaf size={14} /></span> Field-ready crop intelligence</div>
          <div className="topbar-actions"><span className="privacy-note"><ShieldCheck size={15} /> Local preview only</span><button className="help-button" aria-label="Help"><CircleHelp size={19} /></button></div>
        </header>

        <section className="hero-block" id="scan">
          <div className="hero-copy">
            <div className="section-kicker">01 / Observe</div>
            <h1>See what your<br /><em>plants are telling you.</em></h1>
            <p className="hero-intro">A clear leaf photo is the beginning of a better field decision. AgriScan reads visual signals and turns them into a practical first look.</p>
          </div>
          <div className="hero-aside">
            <div className="hero-photo"><img src="/manus-storage/agriscan-hero-leaf_eb265ec3.jpg" alt="Close-up of a healthy leaf in morning light" /></div>
            <div className="hero-aside-line" />
            <p>“The best scan starts with natural light, a steady hand, and the whole leaf in frame.”</p>
            <span>FIELD NOTE 014</span>
          </div>
        </section>

        <section className="scan-grid" aria-label="Leaf scan workspace">
          <div className="upload-panel paper-card">
            <div className="panel-header">
              <div><div className="panel-kicker">02 / Add an image</div><h2>Start a new scan</h2></div>
              <span className="format-note">JPG · PNG</span>
            </div>

            {!previewUrl ? (
              <button className="dropzone" onClick={() => fileInput.current?.click()} aria-label="Choose a leaf image">
                <div className="dropzone-orbit"><img className="dropzone-mark" src="/manus-storage/agriscan-mark_0ca6187a.png" alt="" /><Upload size={16} /></div>
                <span className="dropzone-title">Drop a leaf image here</span>
                <span className="dropzone-copy">or choose a file from your device</span>
                <span className="dropzone-meta"><Camera size={14} /> Camera-ready · up to 200 MB</span>
              </button>
            ) : (
              <div className="preview-stage">
                <img src={previewUrl} alt="Uploaded leaf preview" />
                <button className="clear-preview" onClick={clearImage} aria-label="Remove uploaded image"><X size={17} /></button>
                <div className="preview-file"><FileImage size={14} /> {fileName}</div>
              </div>
            )}
            <input ref={fileInput} className="sr-only" type="file" accept="image/png,image/jpeg" onChange={handleFileChange} />
            <div className="upload-footnote"><SunMedium size={15} /> Natural light gives the model a clearer signal.</div>
            <div className="measurement-strip"><span>RGB</span><span>224 × 224</span><span>LEAF SIGNAL</span></div>
          </div>

          <div className="readiness-panel paper-card">
            <div className="panel-kicker">03 / Scan readiness</div>
            <h2>One image,<br />three checks.</h2>
            <div className="readiness-list">
              <div className={`readiness-row ${previewUrl ? "complete" : ""}`}><span className="readiness-icon">{previewUrl ? <Check size={14} /> : "01"}</span><span>Leaf fills the frame</span><span className="readiness-state">{previewUrl ? "Ready" : "Waiting"}</span></div>
              <div className="readiness-row"><span className="readiness-icon">02</span><span>Image is well lit</span><span className="readiness-state">Required</span></div>
              <div className="readiness-row"><span className="readiness-icon">03</span><span>Crop scope is set</span><span className="readiness-state">{selectedCrop}</span></div>
            </div>
            <button className="scan-button" disabled={!previewUrl || isScanning} onClick={handleScan}>
              {isScanning ? <><RefreshCw size={17} className="spin" /> Reading leaf signals…</> : <><Sparkles size={17} /> Run health scan <ArrowUpRight size={17} /></>}
            </button>
            <div className="readiness-disclaimer"><ShieldCheck size={14} /> Private by design · image stays in this browser demo</div>
          </div>
        </section>

        {hasResult && (
          <section className="result-dossier paper-card" aria-live="polite">
            <div className="result-topline"><div className="panel-kicker">04 / Scan result</div><div className="result-badge"><span /> Demo analysis</div></div>
            <div className="result-heading-row"><div><h2>{activeResult.title}</h2><div className="result-label">{activeResult.label}</div></div><div className="result-confidence"><strong>{activeResult.confidence}%</strong><span>model confidence</span></div></div>
            <div className="confidence-track"><div className="confidence-fill" style={{ width: `${activeResult.confidence}%` }} /></div>
            <div className="result-columns">
              <div className="advice-block"><span className="advice-index">A</span><div><h3>What we found</h3><p>{activeResult.description}</p></div></div>
              <div className="advice-block"><span className="advice-index green">B</span><div><h3>Recommended next step</h3><p>{activeResult.nextStep}</p></div></div>
            </div>
            <div className="result-footer"><span>Result shown is a static UI demonstration.</span><button onClick={clearImage}>Start another scan <ArrowUpRight size={15} /></button></div>
          </section>
        )}

        <section className="how-section" id="how-it-works">
          <div className="how-intro"><div className="section-kicker">The method</div><h2>From leaf<br /><em>to field note.</em></h2><img className="method-image" src="/manus-storage/agriscan-crop-detail_03d55572.jpg" alt="Macro detail of a crop leaf" /></div>
          <div className="method-steps">
            <div className="method-step"><span>01</span><h3>Frame it</h3><p>Fill the view with one leaf and find soft, natural light.</p></div>
            <div className="method-step"><span>02</span><h3>Read it</h3><p>The model looks for visual patterns across 22 crop classes.</p></div>
            <div className="method-step"><span>03</span><h3>Act on it</h3><p>Use the result as a first read, then confirm with an expert.</p></div>
          </div>
        </section>

        <footer className="site-footer"><div className="footer-brand"><div className="brand-mark small"><img className="mark-image" src="/manus-storage/agriscan-mark_0ca6187a.png" alt="" /></div><span>AgriScan</span></div><span>Transfer-learning plant disease detection</span><span>Built for better field decisions.</span></footer>
      </main>
    </div>
  );
}
