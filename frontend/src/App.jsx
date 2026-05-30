import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
const THEME_KEY = "deepfake-detector-theme";

const LABEL_META = {
  FAKE: { tone: "danger", badge: "Likely manipulated" },
  SUSPICIOUS: { tone: "warning", badge: "Suspicious" },
  REAL: { tone: "success", badge: "Likely authentic" },
  UNCERTAIN: { tone: "warning", badge: "Needs review" },
  ERROR: { tone: "neutral", badge: "Unavailable" },
};

const PROGRESS_STEPS = [
  "Uploading video",
  "Extracting frames",
  "Running detectors",
  "Calculating verdict",
  "Preparing results",
];

const pct = (value = 0) => `${(value * 100).toFixed(1)}%`;

const formatSeconds = (value) =>
  typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)}s` : "N/A";

const formatResolution = (meta) =>
  meta?.width && meta?.height ? `${meta.width} x ${meta.height}` : "N/A";

function apiUrl(path) {
  return API ? `${API}${path}` : path;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function readErrorMessage(response, fallbackMessage) {
  try {
    const data = await response.json();
    return data?.detail || data?.error || fallbackMessage;
  } catch {
    return fallbackMessage;
  }
}

function normalizeError(error) {
  if (error instanceof Error && error.name === "TypeError") {
    return "Cannot reach the backend API. Start FastAPI and check the frontend API URL.";
  }
  return error instanceof Error ? error.message : "Something went wrong.";
}

function getInitialTheme() {
  if (typeof window === "undefined") return "dark";
  const savedTheme = window.localStorage.getItem(THEME_KEY);
  if (savedTheme === "light" || savedTheme === "dark") return savedTheme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function toneForScore(score = 0) {
  if (score >= 0.6) return "danger";
  if (score >= 0.4) return "warning";
  return "success";
}

function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={onToggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      aria-pressed={theme === "dark"}
    >
      <span className="theme-toggle__track">
        <span className={`theme-toggle__thumb theme-toggle__thumb--${theme}`} />
        <span className={theme === "light" ? "is-active" : ""}>Light</span>
        <span className={theme === "dark" ? "is-active" : ""}>Dark</span>
      </span>
    </button>
  );
}

function StickyNav({ visible, theme, onToggle }) {
  return (
    <header className={`sticky-nav ${visible ? "sticky-nav--visible" : ""}`}>
      <a href="#top" className="sticky-nav__title">Multimodal Deepfake Detector</a>
      <ThemeToggle theme={theme} onToggle={onToggle} />
    </header>
  );
}

function UploadZone({ onFile }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const acceptFile = (file) => {
    if (file) onFile(file);
  };

  return (
    <button
      type="button"
      className={`upload-zone ${dragging ? "upload-zone--dragging" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        acceptFile(event.dataTransfer.files?.[0]);
      }}
    >
      <input
        ref={inputRef}
        className="sr-only"
        type="file"
        accept="video/*"
        onChange={(event) => acceptFile(event.target.files?.[0])}
      />
      <span className="upload-zone__mark">+</span>
      <strong>Upload video</strong>
      <span>Drag and drop or browse</span>
      <small>MP4, MOV, AVI, MKV, WebM</small>
    </button>
  );
}

function VideoPreview({ file, src }) {
  if (!file || !src) return null;

  return (
    <section className="video-preview card">
      <video src={src} controls playsInline preload="metadata" />
      <div className="video-preview__meta">
        <strong>{file.name}</strong>
        <span>{(file.size / (1024 ** 2)).toFixed(1)} MB</span>
      </div>
    </section>
  );
}

function ProgressPanel({ phase, step }) {
  if (phase !== "uploading" && phase !== "polling") return null;

  const index = Math.min(step, PROGRESS_STEPS.length - 1);
  const progress = Math.max(10, Math.round(((index + 1) / PROGRESS_STEPS.length) * 100));

  return (
    <section className="progress-panel card" aria-live="polite">
      <div>
        <strong>{PROGRESS_STEPS[index]}</strong>
        <span>{progress}%</span>
      </div>
      <div className="progress-track">
        <span style={{ width: `${progress}%` }} />
      </div>
    </section>
  );
}

function MetricCard({ label, value, helper }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {helper ? <small>{helper}</small> : null}
    </article>
  );
}

function Verdict({ result }) {
  const meta = LABEL_META[result.label] || LABEL_META.ERROR;
  const fakeRatio = result.frame_analysis?.fake_frame_ratio;
  const scoreHelper =
    typeof fakeRatio === "number" ? `Frame ratio ${pct(fakeRatio)}` : undefined;

  return (
    <section className={`card verdict verdict--${meta.tone}`}>
      <div>
        <span className={`status-pill status-pill--${meta.tone}`}>{meta.badge}</span>
        <h2>{result.label}</h2>
      </div>
      <div className="metric-grid">
        <MetricCard label="Decision score" value={pct(result.fake_probability)} helper={scoreHelper} />
        <MetricCard label="Confidence" value={result.confidence || "N/A"} />
        <MetricCard label="Processing time" value={formatSeconds(result.processing_time_s)} />
      </div>
    </section>
  );
}

function VideoMeta({ meta }) {
  if (!meta || !Object.keys(meta).length) return null;
  return (
    <section className="metric-grid">
      <MetricCard label="Duration" value={formatSeconds(meta.duration_s)} />
      <MetricCard label="Resolution" value={formatResolution(meta)} />
      <MetricCard label="Frame rate" value={meta.fps ? `${meta.fps.toFixed(0)} fps` : "N/A"} />
    </section>
  );
}

function SignalBar({ label, score }) {
  const tone = toneForScore(score);

  return (
    <article className="signal-row">
      <div>
        <span>{label}</span>
        <strong>{score == null ? "N/A" : pct(score)}</strong>
      </div>
      <div className="signal-bar">
        <span className={`signal-bar__fill signal-bar__fill--${tone}`} style={{ width: pct(score || 0) }} />
      </div>
    </article>
  );
}

function ModalityBreakdown({ result }) {
  return (
    <section className="card result-section">
      <div className="section-heading">
        <span>Modality Breakdown</span>
        <h2>Evidence</h2>
      </div>
      {result.breakdown_plot ? (
        <figure className="chart-frame">
          <img src={`data:image/png;base64,${result.breakdown_plot}`} alt="Modality breakdown" />
        </figure>
      ) : null}
    </section>
  );
}

function SuspiciousFrames({ frames, frameAnalysis, result }) {
  const [selectedFrame, setSelectedFrame] = useState(null);
  if (!frames?.length && !frameAnalysis) return null;

  const safeFrames = frames || [];
  const sortedFrames = [...safeFrames].sort((a, b) => b.score - a.score);
  const fakeRatio = frameAnalysis?.fake_frame_ratio ?? 0;
  const realRatio = frameAnalysis?.real_frame_ratio ?? 0;
  const formula = frameAnalysis?.score_formula;
  const audioAvailable = formula?.audio_available ?? result?.audio_score != null;
  const finalScore = formula?.final_fake_score ?? result?.fake_probability ?? 0;
  const ratioVerdict = frameAnalysis?.frame_ratio_verdict || "REAL";
  const ratioTone = ratioVerdict === "FAKE" ? "danger" : ratioVerdict === "SUSPICIOUS" ? "warning" : "success";

  return (
    <section className="card result-section">
      <div className="section-heading section-heading--row">
        <div>
          <span>Suspicious Frames</span>
          <h2>Frame Review</h2>
        </div>
        <span className={`status-pill status-pill--${ratioTone}`}>{ratioVerdict}</span>
      </div>

      <div className="metric-grid metric-grid--dense">
        <MetricCard label="Fake frames" value={frameAnalysis?.fake_frames ?? safeFrames.length} />
        <MetricCard label="Fake ratio" value={pct(fakeRatio)} />
        <MetricCard label="Real ratio" value={pct(realRatio)} />
        <MetricCard label="Frame component" value={pct(frameAnalysis?.frame_component ?? 0)} />
        <MetricCard label="Decision score" value={pct(finalScore)} />
        <MetricCard label="Audio included" value={audioAvailable ? "Yes" : "No"} />
      </div>

      {sortedFrames.length ? (
        <div className="frame-grid">
          {sortedFrames.map((frame, index) => {
            const tone = toneForScore(frame.score);
            return (
              <button
                type="button"
                className="frame-card"
                key={`${frame.frame_idx}-${index}`}
                onClick={() => setSelectedFrame(frame)}
              >
                <img src={`data:image/jpeg;base64,${frame.image}`} alt={`Frame ${frame.frame_idx}`} />
                <span>Frame {frame.frame_idx}</span>
                <strong className={`text-${tone}`}>{pct(frame.score)}</strong>
              </button>
            );
          })}
        </div>
      ) : null}

      {selectedFrame ? (
        <div className="frame-modal" onClick={() => setSelectedFrame(null)}>
          <div className="frame-modal__inner" onClick={(event) => event.stopPropagation()}>
            <button type="button" onClick={() => setSelectedFrame(null)} aria-label="Close frame preview">
              Close
            </button>
            <img
              src={`data:image/jpeg;base64,${selectedFrame.image}`}
              alt={`Frame ${selectedFrame.frame_idx} detail`}
            />
          </div>
        </div>
      ) : null}
    </section>
  );
}

function AudioVisual({ result }) {
  if (!result.audio_plot) return null;
  return (
    <section className="card result-section">
      <div className="section-heading">
        <span>Audio</span>
        <h2>Waveform Review</h2>
      </div>
      <figure className="chart-frame">
        <img src={`data:image/png;base64,${result.audio_plot}`} alt="Audio analysis" />
      </figure>
    </section>
  );
}

function ExplanationPanel({ explanation }) {
  if (!explanation?.length) return null;
  return (
    <section className="card result-section">
      <div className="section-heading">
        <span>Trace</span>
        <h2>Reasoning</h2>
      </div>
      <ol className="reasoning-list">
        {explanation.slice(0, 10).map((line, index) => (
          <li key={`${line}-${index}`}>{line}</li>
        ))}
      </ol>
    </section>
  );
}

function ErrorPanel({ error, onReset }) {
  return (
    <section className="card error-panel">
      <strong>Analysis failed</strong>
      <p>{error}</p>
      <button type="button" className="primary-button" onClick={onReset}>Try another video</button>
    </section>
  );
}

export default function App() {
  const [phase, setPhase] = useState("idle");
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [step, setStep] = useState(0);
  const [theme, setTheme] = useState(getInitialTheme);
  const [showStickyNav, setShowStickyNav] = useState(false);
  const headingRef = useRef(null);

  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    document.body.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (!headingRef.current) return undefined;
    const observer = new IntersectionObserver(
      ([entry]) => setShowStickyNav(!entry.isIntersecting),
      { threshold: 0.1 },
    );
    observer.observe(headingRef.current);
    return () => observer.disconnect();
  }, []);

  const toggleTheme = () => {
    setTheme((currentTheme) => (currentTheme === "dark" ? "light" : "dark"));
  };

  const reset = () => {
    setPhase("idle");
    setFile(null);
    setJobId(null);
    setResult(null);
    setError(null);
    setStep(0);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleFile = async (selectedFile) => {
    setFile(selectedFile);
    setPhase("uploading");
    setError(null);
    setResult(null);
    setStep(0);

    try {
      const form = new FormData();
      form.append("file", selectedFile);

      const uploadResponse = await fetch(apiUrl("/api/upload"), {
        method: "POST",
        body: form,
      });
      if (!uploadResponse.ok) {
        throw new Error(await readErrorMessage(uploadResponse, "Upload failed"));
      }

      const { job_id } = await uploadResponse.json();
      setJobId(job_id);
      setPhase("polling");
      setStep(1);

      const analyseResponse = await fetch(apiUrl(`/api/analyze/${job_id}`), {
        method: "POST",
      });
      if (!analyseResponse.ok) {
        throw new Error(await readErrorMessage(analyseResponse, "Unable to start analysis"));
      }

      let attempts = 0;
      while (attempts < 120) {
        setStep(Math.min(PROGRESS_STEPS.length - 1, 2 + Math.floor(attempts / 10)));
        const response = await fetch(apiUrl(`/api/result/${job_id}`));
        if (!response.ok) {
          throw new Error(await readErrorMessage(response, "Unable to fetch analysis result"));
        }

        const data = await response.json();
        if (data.status === "done") {
          setResult(data);
          setStep(PROGRESS_STEPS.length - 1);
          setPhase("done");
          return;
        }
        if (data.status === "error") {
          throw new Error(data.error || "Analysis failed");
        }

        attempts += 1;
        await sleep(2000);
      }

      throw new Error("Timed out waiting for result");
    } catch (err) {
      setError(normalizeError(err));
      setPhase("error");
    }
  };

  const hasVideo = Boolean(file && videoUrl);

  return (
    <div className="app-shell" id="top">
      <StickyNav visible={showStickyNav} theme={theme} onToggle={toggleTheme} />

      <main className="app">
        <div className="hero__toolbar">
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
        <section className="hero">
          <div className="hero__content">

            <p className="eyebrow">Review suspicious media with a cleaner, faster forensic workspace.</p>
            <h1 ref={headingRef}>Multimodal Deepfake Detector</h1>
            <p>Video, audio, and transcript signals for media verification.</p>
          </div>

          <div className="hero__media">
            {hasVideo ? <VideoPreview file={file} src={videoUrl} /> : <UploadZone onFile={handleFile} />}
          </div>
        </section>

        <ProgressPanel phase={phase} step={step} jobId={jobId} />

        {phase === "error" ? <ErrorPanel error={error} onReset={reset} /> : null}

        {phase === "done" && result ? (
          <section className="results" id="results">
            <Verdict result={result} />
            <VideoMeta meta={result.video_metadata} />
            <ModalityBreakdown result={result} />
            <SuspiciousFrames
              frames={result.suspicious_frames}
              frameAnalysis={result.frame_analysis}
              result={result}
            />
            <AudioVisual result={result} />
            <ExplanationPanel explanation={result.explanation} />
            <button type="button" className="primary-button primary-button--wide" onClick={reset}>
              Analyse another video
            </button>
          </section>
        ) : null}
      </main>

      <footer className="app-footer">
        <nav aria-label="Project links">
          <a href="https://github.com/c0derashish/multimodal-deepfake-detector" target="_blank" rel="noreferrer"><i class="fa-brands fa-github"></i>GitHub</a>
          <a href="https://www.linkedin.com/in/ashish-chandra-552528296/" target="_blank" rel="noreferrer"><i class="fa-brands fa-square-linkedin"></i>LinkedIn</a>
        </nav>
        <span>Made with ❤️ by Code Monks</span>
      </footer>
    </div>
  );
}
