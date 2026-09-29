import React, { useMemo, useState } from "react";
import {
  Activity, AlertTriangle, ArrowRight, BarChart3, Blocks, CheckCircle2,
  ChevronRight, Cpu, Database, FileSearch, Gauge, LayoutDashboard,
  Menu, Network, RefreshCw, Search, ShieldCheck, Sparkles, Wallet, X
} from "lucide-react";
import {
  AreaChart, Area, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";

const API_BASE = import.meta.env.VITE_FRAUD_API || "http://localhost:8001";

const demo = {
  borrower_wallet_address: "",
  verified_monthly_income: 12500,
  total_liabilities: 18000,
  requested_loan_amount: 15000,
  discrepancy_ratio: 1.2,
  dr_risk_penalty: 0
};

const nav = [
  ["overview", "Overview", LayoutDashboard],
  ["assessment", "Risk Assessment", ShieldCheck],
  ["explain", "AI Explanation", Sparkles],
  ["risk", "Risk Analysis", Gauge],
  ["technical", "Technical", Cpu]
];

function clamp(v) {
  return Math.min(1, Math.max(0, Number(v) || 0));
}

function riskMeta(score) {
  const s = clamp(score);
  if (s >= 0.70) return { label: "CRITICAL", decision: "BLOCK TRANSACTION", tone: "critical" };
  if (s >= 0.40) return { label: "MODERATE", decision: "REVIEW REQUIRED", tone: "warning" };
  return { label: "LOW", decision: "APPROVED", tone: "safe" };
}

function getAssessment(body) {
  const raw = body?.assessment || body?.result || body?.data || body || {};

  // The FastAPI/model-service uses the names local_shap and
  // boosting_progression. Normalize them once so the whole UI
  // can use the simple names shap and boosting.
  const shap = raw.shap?.length
    ? raw.shap
    : (raw.local_shap || []);

  const boosting = raw.boosting?.length
    ? raw.boosting
    : (raw.boosting_progression || []);

  const genai = raw.genai_explanation || null;
  const genaiPoints = raw.genai_points?.length
    ? raw.genai_points
    : (genai?.points || []);

  return {
    ...raw,
    shap,
    local_shap: shap,
    boosting,
    boosting_progression: boosting,
    genai_explanation: genai,
    genai_points: genaiPoints
  };
}

function getCheckpointStages(rows) {
  if (!Array.isArray(rows) || !rows.length) return [];

  const cleaned = rows
    .map((r) => ({
      iteration: Number(r.iteration ?? r.stage ?? r.trees ?? 0),
      probability: clamp(r.fraud_probability ?? r.probability)
    }))
    .filter((r) => Number.isFinite(r.iteration) && r.iteration > 0)
    .sort((a, b) => a.iteration - b.iteration);

  if (!cleaned.length) return [];

  const targets = [100, 200, 300, 400, 500];
  const selected = [];

  for (const target of targets) {
    const exact = cleaned.find((r) => r.iteration === target);
    if (exact) {
      selected.push(exact);
    } else {
      const nearest = cleaned.reduce((best, r) =>
        Math.abs(r.iteration - target) < Math.abs(best.iteration - target)
          ? r
          : best
      );
      if (
        nearest &&
        nearest.iteration <= target + 40 &&
        !selected.some((x) => x.iteration === nearest.iteration)
      ) {
        selected.push(nearest);
      }
    }
  }

  const final = cleaned[cleaned.length - 1];
  if (final && !selected.some((x) => x.iteration === final.iteration)) {
    selected.push(final);
  }

  return selected;
}

function Stat({ label, value, sub, tone = "" }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${tone}`}>{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  );
}

function Progress({ value, tone = "blue" }) {
  return (
    <div className="progress">
      <span className={`progress-fill ${tone}`} style={{ width: `${clamp(value) * 100}%` }} />
    </div>
  );
}

function App() {
  const [page, setPage] = useState("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [form, setForm] = useState(demo);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const finalRisk = clamp(result?.final_risk);
  const onChain = clamp(result?.on_chain_risk ?? result?.base_xgboost_probability ?? result?.fraud_probability);
  const offChain = clamp(result?.off_chain_risk);
  const meta = useMemo(() => riskMeta(finalRisk || onChain), [finalRisk, onChain]);

  const chartData = result?.boosting?.length
    ? result.boosting.map((x) => ({
        stage: `T${x.iteration ?? x.stage ?? x.trees ?? ""}`,
        probability: clamp(x.fraud_probability ?? x.probability) * 100
      }))
    : [];

  const checkpoints = getCheckpointStages(result?.boosting);

  const update = (key, value) => setForm((old) => ({ ...old, [key]: value }));


  async function assess(e) {
    e.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      const payload = {
        borrower_wallet_address: form.borrower_wallet_address,
        verified_monthly_income: Number(form.verified_monthly_income),
        total_liabilities: Number(form.total_liabilities),
        requested_loan_amount: Number(form.requested_loan_amount),
        discrepancy_ratio: Number(form.discrepancy_ratio),
        dr_risk_penalty: Number(form.dr_risk_penalty || 0),
        include_shap: true,
        include_boosting: true,
        forward_to_module3: true,
        forward_to_blockchain: true,
        forward_to_gateway: false
      };

      const response = await fetch(`${API_BASE}/api/v1/assess-risk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const body = await response.json();

      if (!response.ok) {
        throw new Error(body?.detail || body?.message || `HTTP ${response.status}`);
      }

      const assessment = getAssessment(body);
      setResult(assessment);
      setMessage("Risk assessment completed — SHAP, XGBoost progression and AI explanation loaded.");
      setPage("overview");
    } catch (e) {
      setMessage(`Assessment failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  function loadDemo() {
    const demoResult = getAssessment({
      ...demo,
      borrower_wallet_address: form.borrower_wallet_address || "Demo wallet",
      on_chain_risk: 0.23,
      off_chain_risk: 0.18,
      final_risk: 0.22,
      risk_level: "LOW",
      decision: "APPROVED",
      base_xgboost_probability: 0.23,
      genai_explanation: {
        title: "Why is this transaction considered safe?",
        points: [
          "The wallet activity is closer to normal transaction behaviour.",
          "The transaction pattern does not show a strong suspicious signal.",
          "The overall fraud probability remains relatively low."
        ],
        source: "Demo"
      },
      boosting_progression: [
        { iteration: 100, fraud_probability: 0.31 },
        { iteration: 200, fraud_probability: 0.26 },
        { iteration: 300, fraud_probability: 0.24 },
        { iteration: 400, fraud_probability: 0.23 },
        { iteration: 500, fraud_probability: 0.22 },
        { iteration: 552, fraud_probability: 0.22 }
      ],
      local_shap: [
        { feature: "Sent tnx", shap_value: -0.21, abs_shap: 0.21 },
        { feature: "total Ether sent", shap_value: -0.17, abs_shap: 0.17 },
        { feature: "Received Tnx", shap_value: -0.12, abs_shap: 0.12 },
        { feature: "ERC20 total Ether sent", shap_value: 0.08, abs_shap: 0.08 },
        { feature: "avg val received", shap_value: -0.06, abs_shap: 0.06 }
      ]
    });

    setResult(demoResult);
    setMessage("Demo assessment loaded.");
  }

  function go(id) {
    setPage(id);
    setMobileOpen(false);
  }

  const topShap = (result?.shap || []).slice(0, 10);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><ShieldCheck size={22} /></div>
          <div><strong>DeFiLens</strong><span>Fraud Intelligence</span></div>
        </div>

        <div className="workspace">
          <div className="eyebrow">WORKSPACE</div>
          <div className="workspace-row">
            <span className="online-dot" /> DeFi Lending <ChevronRight size={15} />
          </div>
        </div>

        <nav>
          {nav.map(([id, label, Icon]) => (
            <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => go(id)}>
              <Icon size={18} /> <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="security-mini">
            <ShieldCheck size={17} />
            <div><b>Protected pipeline</b><span>AI → Risk → Blockchain</span></div>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileOpen(!mobileOpen)}><Menu size={21} /></button>
          <div className="crumb">
            <span>Security Intelligence</span>
            <ChevronRight size={14} />
            <b>{nav.find((x) => x[0] === page)?.[1]}</b>
          </div>
          <div className="top-actions">
            <button className="icon-btn" onClick={() => window.location.reload()}><RefreshCw size={17} /></button>
          </div>
        </header>

        {message && (
          <div className="toast">
            <Activity size={15} /> {message}
            <button onClick={() => setMessage("")}><X size={15} /></button>
          </div>
        )}

        <div className="content">
          <section className="hero">
            <div>
              <div className="eyebrow">DECENTRALIZED FINANCE · AI SECURITY</div>
              <h1>Transaction Risk Center</h1>
              <p>Detect suspicious DeFi activity, understand the decision, inspect the model reasoning, and pass a clean risk signal to the lending pipeline.</p>
            </div>
            <div className="hero-status"><span className="pulse" /> MODEL ONLINE</div>
          </section>

          {page === "overview" && (
            <>
              <div className="stats-grid">
                <Stat label="Final risk" value={`${(finalRisk * 100).toFixed(1)}%`} sub="Hybrid risk" tone={meta.tone} />
                <Stat label="Fraud probability" value={`${(onChain * 100).toFixed(1)}%`} sub="XGBoost output" />
                <Stat label="Decision" value={result?.decision || "Awaiting"} sub="Policy output" />
                <Stat label="Wallet" value={(result?.borrower_address || result?.borrower_wallet_address || form.borrower_wallet_address) ? `${(result?.borrower_address || result?.borrower_wallet_address || form.borrower_wallet_address).slice(0, 8)}…` : "Not set"} sub="Borrower" />
              </div>

              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head">
                    <div><div className="eyebrow">RISK SIGNAL</div><h2>Current assessment</h2></div>
                    <span className={`badge ${meta.tone}`}>{result?.risk_level || "NO RESULT"}</span>
                  </div>
                  <div className="score-area">
                    <div className={`score-ring ${meta.tone}`} style={{ "--score": `${(finalRisk || onChain) * 100}%` }}>
                      <div><strong>{((finalRisk || onChain) * 100).toFixed(1)}%</strong><span>final risk</span></div>
                    </div>
                    <div>
                      <h3>{result ? (result.decision || meta.decision) : "Run an assessment"}</h3>
                      <p>The final signal combines the wallet's XGBoost fraud risk with the off-chain financial checks supplied by Module 1.</p>
                      <button className="primary-btn" onClick={() => go("assessment")}>Open assessment <ArrowRight size={16} /></button>
                    </div>
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <div><div className="eyebrow">GENERATIVE AI</div><h2>{result?.genai_explanation?.title || "Decision explanation"}</h2></div>
                    <Sparkles size={19} />
                  </div>
                  <div className="explain-list">
                    {(result?.genai_points?.length ? result.genai_points : [
                      "Run an assessment to generate a plain-language explanation.",
                      "Gemini explains the strongest signals identified by the fraud pipeline.",
                      "The explanation does not change the underlying fraud decision."
                    ]).slice(0, 3).map((p, i) => (
                      <div className="explain-item" key={i}><span>{String(i + 1).padStart(2, "0")}</span><p>{p}</p></div>
                    ))}
                  </div>
                  {result?.genai_explanation?.source && <div className="source-note">{result.genai_explanation.source}</div>}
                </div>
              </div>

              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head"><div><div className="eyebrow">BOOSTING TRACE</div><h2>XGBoost decision progression</h2></div><BarChart3 size={19} /></div>
                  {chartData.length ? (
                    <ResponsiveContainer width="100%" height={250}>
                      <AreaChart data={chartData}>
                        <defs>
                          <linearGradient id="riskFillOverview" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#7c6cff" stopOpacity=".28" />
                            <stop offset="100%" stopColor="#7c6cff" stopOpacity="0" />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="#1d2636" vertical={false} />
                        <XAxis dataKey="stage" stroke="#68758c" tickLine={false} axisLine={false} />
                        <YAxis stroke="#68758c" tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} />
                        <Tooltip contentStyle={{ background: "#0d1320", border: "1px solid #253149", borderRadius: 12 }} formatter={(v) => [`${Number(v).toFixed(2)}%`, "Fraud probability"]} />
                        <Area type="monotone" dataKey="probability" stroke="#8d80ff" fill="url(#riskFillOverview)" strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-state"><BarChart3 size={24} /><b>No boosting data loaded</b><span>Run an assessment with boosting enabled.</span></div>
                  )}
                </div>

                <div className="panel">
                  <div className="panel-head"><div><div className="eyebrow">EXPLAINABLE AI</div><h2>Top feature contributions</h2></div><Search size={19} /></div>
                  {topShap.length ? topShap.slice(0, 6).map((s, i) => {
                    const value = Number(s.shap_value ?? s.value ?? s.contribution ?? 0);
                    return (
                      <div className="shap-row" key={i}>
                        <div><b>{s.feature || `Feature ${i + 1}`}</b><span>{value >= 0 ? "pushes toward fraud" : "pushes toward legitimate"}</span></div>
                        <strong className={value >= 0 ? "positive" : "negative"}>{value >= 0 ? "+" : ""}{value.toFixed(3)}</strong>
                      </div>
                    );
                  }) : (
                    <div className="empty-state"><Search size={24} /><b>No local SHAP data loaded</b><span>Run an assessment with SHAP enabled.</span></div>
                  )}
                </div>
              </div>

              <div className="panel">
                <div className="panel-head"><div><div className="eyebrow">PIPELINE</div><h2>Security decision flow</h2></div></div>
                <div className="pipeline">
                  {[
                    ["01", "Document AI", "Financial facts", FileSearch],
                    ["02", "Fraud Engine", "Wallet → XGBoost + SHAP", ShieldCheck],
                    ["03", "Risk Engine", "Hybrid score", Gauge],
                    ["04", "Blockchain", "Approved lending action", Blocks]
                  ].map(([n, title, sub, Icon], i) => (
                    <React.Fragment key={title}>
                      <div className="pipeline-node">
                        <div className="node-icon"><Icon size={18} /></div>
                        <div><b>{title}</b><span>{sub}</span></div>
                        <small>{n}</small>
                      </div>
                      {i < 3 && <ArrowRight className="pipeline-arrow" size={18} />}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </>
          )}

          {page === "assessment" && (
            <>
              <div className="grid-2">
                <form className="panel" onSubmit={assess}>
                  <div className="panel-head">
                    <div><div className="eyebrow">LIVE INPUT</div><h2>Run risk assessment</h2></div>
                    <span className="badge blue">FASTAPI</span>
                  </div>
                  <div className="field-grid">
                    <label className="field wide"><span>Borrower wallet address</span><input value={form.borrower_wallet_address} onChange={(e) => update("borrower_wallet_address", e.target.value)} placeholder="0x..." required /></label>
                    <label className="field"><span>Verified monthly income</span><input type="number" value={form.verified_monthly_income} onChange={(e) => update("verified_monthly_income", e.target.value)} /></label>
                    <label className="field"><span>Total liabilities</span><input type="number" value={form.total_liabilities} onChange={(e) => update("total_liabilities", e.target.value)} /></label>
                    <label className="field"><span>Requested loan amount</span><input type="number" value={form.requested_loan_amount} onChange={(e) => update("requested_loan_amount", e.target.value)} /></label>
                    <label className="field"><span>Discrepancy ratio</span><input type="number" step="0.01" value={form.discrepancy_ratio} onChange={(e) => update("discrepancy_ratio", e.target.value)} /></label>
                    <label className="field"><span>DR risk penalty</span><input type="number" step="0.01" value={form.dr_risk_penalty} onChange={(e) => update("dr_risk_penalty", e.target.value)} /></label>
                  </div>
                  <div className="form-actions">
                    <button type="button" className="secondary-btn" onClick={loadDemo}>Load demo</button>
                    <button className="primary-btn" disabled={loading}>{loading ? "Analyzing…" : "Run assessment"} <ArrowRight size={16} /></button>
                  </div>
                </form>

                <div className="panel">
                  <div className="panel-head"><div><div className="eyebrow">PROCESS</div><h2>Live execution stages</h2></div></div>
                  {[
                    ["Wallet lookup", "Find the Module 1 wallet in transaction_dataset.csv.", Database, !!result],
                    ["XGBoost", "Apply imputation, selected features and the trained ensemble.", ShieldCheck, !!result?.base_xgboost_probability],
                    ["SHAP", "Calculate local feature contributions for this wallet.", Search, !!result?.shap?.length],
                    ["GenAI", "Translate the strongest signals into three simple points.", Sparkles, !!result?.genai_points?.length],
                    ["Risk handoff", "Create the clean Module 3 / blockchain payload.", Network, !!result?.module3_payload]
                  ].map(([title, text, Icon, done], i) => (
                    <div className="step" key={title}>
                      <div className={`step-number ${done ? "done" : ""}`}>{done ? "✓" : `0${i + 1}`}</div>
                      <Icon size={18} />
                      <div><b>{title}</b><p>{text}</p></div>
                    </div>
                  ))}
                </div>
              </div>

              {result && (
                <div className="panel">
                  <div className="panel-head"><div><div className="eyebrow">LIVE RESULT</div><h2>Assessment summary</h2></div><span className={`badge ${meta.tone}`}>{result.risk_level}</span></div>
                  <div className="stats-grid compact">
                    <Stat label="Wallet" value={(result.borrower_address || "—").slice(0, 12) + (result.borrower_address ? "…" : "")} sub={`${result.match_count ?? 0} dataset match(es)`} />
                    <Stat label="XGBoost" value={`${(onChain * 100).toFixed(2)}%`} sub="Fraud probability" />
                    <Stat label="Final risk" value={`${(finalRisk * 100).toFixed(2)}%`} sub="Hybrid score" tone={meta.tone} />
                    <Stat label="Features" value={`${result.feature_pipeline?.selected_features ?? result.features_used ?? 20}`} sub={`${result.feature_pipeline?.original_features ?? 38} original → selected`} />
                  </div>
                </div>
              )}
            </>
          )}

          {page === "explain" && (
            <>
              <div className="panel">
                <div className="panel-head"><div><div className="eyebrow">GENERATIVE AI</div><h2>{result?.genai_explanation?.title || "Plain-language explanation"}</h2></div><Sparkles size={19} /></div>
                <div className="explain-list">
                  {(result?.genai_points?.length ? result.genai_points : ["No assessment loaded."]).slice(0, 3).map((p, i) => (
                    <div className="explain-item" key={i}><span>{String(i + 1).padStart(2, "0")}</span><p>{p}</p></div>
                  ))}
                </div>
                <div className="source-note">{result?.genai_explanation?.source || "Gemini GenAI explanation is generated after the XGBoost decision."}</div>
              </div>

              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head"><div><div className="eyebrow">MODEL BOUNDARY</div><h2>Decision ownership</h2></div></div>
                  <div className="boundary"><CheckCircle2 size={18} /><div><b>XGBoost decides</b><p>The fraud model produces the transaction fraud probability.</p></div></div>
                  <div className="boundary"><Search size={18} /><div><b>SHAP explains</b><p>Local SHAP values show which wallet features increased or reduced the prediction.</p></div></div>
                  <div className="boundary"><Sparkles size={18} /><div><b>Gemini translates</b><p>GenAI converts the strongest signals into simple language without changing the decision.</p></div></div>
                </div>

                <div className="panel">
                  <div className="panel-head"><div><div className="eyebrow">TOP SIGNALS</div><h2>Why the model moved</h2></div></div>
                  {topShap.slice(0, 8).map((s, i) => {
                    const v = Number(s.shap_value ?? 0);
                    return <div className="shap-row" key={i}><div><b>{s.feature}</b><span>{v >= 0 ? "increases fraud score" : "reduces fraud score"}</span></div><strong className={v >= 0 ? "positive" : "negative"}>{v >= 0 ? "+" : ""}{v.toFixed(4)}</strong></div>;
                  })}
                </div>
              </div>
            </>
          )}

          {page === "risk" && (
            <>
              <div className="stats-grid">
                <Stat label="On-chain risk" value={`${(onChain * 100).toFixed(1)}%`} sub="XGBoost fraud probability" />
                <Stat label="Off-chain risk" value={`${(offChain * 100).toFixed(1)}%`} sub="Module 1 financial checks" />
                <Stat label="Final hybrid risk" value={`${(finalRisk * 100).toFixed(1)}%`} sub="Weighted signal" tone={meta.tone} />
                <Stat label="Decision" value={result?.decision || "—"} sub="Policy output" />
              </div>

              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head"><div><div className="eyebrow">RISK COMPOSITION</div><h2>Signal breakdown</h2></div></div>
                  <div className="risk-row"><div><span>On-chain fraud</span><b>{(onChain * 100).toFixed(1)}%</b></div><Progress value={onChain} tone="purple" /></div>
                  <div className="risk-row"><div><span>Off-chain financial</span><b>{(offChain * 100).toFixed(1)}%</b></div><Progress value={offChain} tone="blue" /></div>
                  <div className="risk-row"><div><span>Final hybrid</span><b>{(finalRisk * 100).toFixed(1)}%</b></div><Progress value={finalRisk} tone={meta.tone === "critical" ? "red" : meta.tone === "warning" ? "amber" : "green"} /></div>
                  <div className="formula-box"><span>Current formula</span><code>R = clip(0.70 × on_chain + 0.30 × off_chain + DR_penalty)</code></div>
                </div>

                <div className="panel chart-panel">
                  <div className="panel-head"><div><div className="eyebrow">BOOSTING TRACE</div><h2>How XGBoost builds the decision</h2></div></div>
                  {chartData.length ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <AreaChart data={chartData}>
                        <CartesianGrid stroke="#1d2636" vertical={false} />
                        <XAxis dataKey="stage" stroke="#68758c" tickLine={false} axisLine={false} />
                        <YAxis stroke="#68758c" tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} />
                        <Tooltip contentStyle={{ background: "#0d1320", border: "1px solid #253149", borderRadius: 12 }} formatter={(v) => [`${Number(v).toFixed(2)}%`, "Fraud probability"]} />
                        <Area type="monotone" dataKey="probability" stroke="#8d80ff" fill="#7c6cff" fillOpacity={0.12} strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-state"><BarChart3 size={24} /><b>No boosting data loaded</b><span>Run an assessment with boosting enabled.</span></div>
                  )}
                </div>
              </div>

              <div className="panel">
                <div className="panel-head"><div><div className="eyebrow">INTERMEDIATE TREES</div><h2>Cumulative ensemble checkpoints</h2></div></div>
                {checkpoints.length ? (
                  <>
                    <div className="checkpoint-grid">
                      {checkpoints.map((r, i) => {
                        const prev = checkpoints[i - 1];
                        const delta = prev ? r.probability - prev.probability : 0;
                        return (
                          <div className="checkpoint" key={r.iteration}>
                            <span>{i === checkpoints.length - 1 ? "FINAL" : `TREE ${r.iteration}`}</span>
                            <strong>{(r.probability * 100).toFixed(2)}%</strong>
                            {prev && <small>{delta >= 0 ? "+" : ""}{(delta * 100).toFixed(2)} pp vs previous</small>}
                            {!prev && <small>First cumulative checkpoint</small>}
                          </div>
                        );
                      })}
                    </div>
                    <div className="stage-note">These are cumulative ensemble outputs as additional boosting trees are included; they are not independent models.</div>
                  </>
                ) : (
                  <div className="empty-state"><b>No intermediate stages loaded</b><span>Run an assessment with boosting enabled.</span></div>
                )}
              </div>
            </>
          )}

          {page === "technical" && (
            <>
              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head"><div><div className="eyebrow">EXPLAINABLE AI</div><h2>Local SHAP signals</h2></div><Search size={19} /></div>
                  {topShap.length ? topShap.map((s, i) => {
                    const value = Number(s.shap_value ?? s.value ?? s.contribution ?? 0);
                    return (
                      <div className="shap-row" key={i}>
                        <div><b>{s.feature || `Feature ${i + 1}`}</b><span>{value >= 0 ? "increases fraud score" : "reduces fraud score"}</span></div>
                        <strong className={value >= 0 ? "positive" : "negative"}>{value >= 0 ? "+" : ""}{value.toFixed(4)}</strong>
                      </div>
                    );
                  }) : (
                    <div className="empty-state"><Search size={24} /><b>No local SHAP data loaded</b><span>Run an assessment with SHAP enabled.</span></div>
                  )}
                </div>

                <div className="panel">
                  <div className="panel-head"><div><div className="eyebrow">ARCHITECTURE</div><h2>Fraud model pipeline</h2></div></div>
                  {[
                    ["Input", `${result?.feature_pipeline?.original_features ?? 38} transaction features`],
                    ["Preprocessing", "SimpleImputer on original feature vector"],
                    ["Selection", `${result?.feature_pipeline?.selected_features ?? 20} SHAP-selected features`],
                    ["Classifier", `Optuna-tuned XGBoost · ${result?.boosting?.length ? result.boosting[result.boosting.length - 1].iteration : "—"} rounds observed`],
                    ["Explainability", `${topShap.length ? "Local SHAP loaded" : "SHAP pending"}`],
                    ["GenAI", `${result?.genai_points?.length ? "Gemini explanation loaded" : "Gemini pending"}`],
                    ["Output", "Fraud probability + hybrid risk + Module 3 payload"]
                  ].map(([a, b]) => (
                    <div className="tech-row" key={a}><Cpu size={16} /><div><b>{a}</b><span>{b}</span></div><ChevronRight size={14} /></div>
                  ))}
                </div>
              </div>

              <div className="panel">
                <div className="panel-head"><div><div className="eyebrow">BOOSTING DETAIL</div><h2>Tree-by-tree cumulative probability</h2></div></div>
                {checkpoints.length ? checkpoints.map((r, i) => {
                  const prev = checkpoints[i - 1];
                  const delta = prev ? r.probability - prev.probability : 0;
                  return (
                    <div className="tech-row" key={r.iteration}>
                      <BarChart3 size={16} />
                      <div><b>{i === checkpoints.length - 1 ? `Final · Tree ${r.iteration}` : `Tree ${r.iteration}`}</b><span>Cumulative fraud probability</span></div>
                      <strong className="tech-prob">{(r.probability * 100).toFixed(2)}% {prev ? `(${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} pp)` : ""}</strong>
                    </div>
                  );
                }) : <div className="empty-state"><b>No boosting progression available.</b></div>}
              </div>
            </>
          )}

          <footer>
            <span><span className="online-dot" /> Fraud API: {API_BASE}</span>
            <span>Module 1 → Wallet Lookup → XGBoost → SHAP → GenAI → Risk → Blockchain</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
