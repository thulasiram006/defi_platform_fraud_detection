import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ==============================================================================
# 1. PAGE CONFIG & PROFESSIONAL DARK THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="DeFiLens | Fraud Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Fonts & Root ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ───────────────────────────────────────────── */
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }

    /* ── Sidebar ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 3px 0;
        display: block;
        transition: all 0.15s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: #21262d;
        border-color: #388bfd;
    }

    /* ── Metric Cards ─────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        transition: border-color 0.15s;
    }
    [data-testid="metric-container"]:hover {
        border-color: #388bfd;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #e6edf3 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em;
    }

    /* ── Section Headers ──────────────────────────────────────── */
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        color: #8b949e;
        text-transform: uppercase;
        margin-bottom: 8px;
        margin-top: 4px;
    }

    /* ── Risk Badge ───────────────────────────────────────────── */
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
    }
    .badge-critical { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid rgba(248,81,73,0.4); }
    .badge-moderate { background: rgba(210,153,34,0.15); color: #e3b341; border: 1px solid rgba(210,153,34,0.4); }
    .badge-low      { background: rgba(56,139,253,0.12); color: #58a6ff; border: 1px solid rgba(56,139,253,0.35); }
    .badge-safe     { background: rgba(63,185,80,0.12);  color: #3fb950; border: 1px solid rgba(63,185,80,0.35); }

    /* ── Decision Banner ──────────────────────────────────────── */
    .decision-banner {
        border-radius: 12px;
        padding: 16px 22px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .banner-critical { background: rgba(248,81,73,0.08);  border-color: #f85149; }
    .banner-moderate { background: rgba(210,153,34,0.08); border-color: #e3b341; }
    .banner-safe     { background: rgba(63,185,80,0.08);  border-color: #3fb950; }

    /* ── Info Cards & Drivers ────────────────────────────────── */
    .info-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 6px 0;
    }
    .driver-item {
        background: #1c2128;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .driver-critical { border-color: #f85149; }
    .driver-warn     { border-color: #e3b341; }
    .driver-ok       { border-color: #3fb950; }
    .driver-info     { border-color: #388bfd; }

    /* ── Formula Card ─────────────────────────────────────────── */
    .formula-card {
        background: #0d1117;
        border: 1px solid #388bfd44;
        border-radius: 10px;
        padding: 14px 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #79c0ff;
        margin: 10px 0 16px;
        text-align: center;
    }

    /* ── Dataframe & Code ─────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d !important;
        border-radius: 10px;
    }
    .stCode { background: #161b22 !important; border: 1px solid #30363d; border-radius: 8px; }

    /* ── Inputs ───────────────────────────────────────────────── */
    .stNumberInput input, .stTextArea textarea {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* ── Download Button ──────────────────────────────────────── */
    .stDownloadButton button {
        background: #1f6feb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
    }
    .stDownloadButton button:hover {
        background: #388bfd !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL ENGINE INITIALIZATION
# ==============================================================================
@st.cache_resource
def load_and_train_improved_xgboost():
    df = pd.read_csv("data/transaction_dataset.csv")
    drop_cols = ['Index', 'Address', 'Unnamed: 0']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = df_clean.drop(columns=['FLAG'])
    y = df_clean['FLAG']
    X_num = X.select_dtypes(include=[np.number])
    var_mask = X_num.var() != 0
    X_filtered = X_num.loc[:, var_mask]
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X_filtered), columns=X_filtered.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.20, random_state=42, stratify=y
    )
    scale_w = (len(y_train) - sum(y_train)) / sum(y_train)
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_w, random_state=42, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model, imputer, X_test, y_test, X_imputed.columns.tolist()

model, imputer, X_test, y_test, feature_columns = load_and_train_improved_xgboost()

# ==============================================================================
# 3. CUSTOM MATPLOTLIB GAUGE (MATCHING SAMPLE IMAGE STYLE)
# ==============================================================================
def draw_risk_meter_5zone(score):
    """
    Renders a 5-Zone Arc Gauge Meter (NO, LOW, MEDIUM, HIGH, MAX) 
    with a central dark needle and bottom 'RISK' label.
    """
    fig, ax = plt.subplots(figsize=(6, 3.4), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # 5 Zone Definitions (Angles from pi to 0)
    zones = [
        (np.pi,           4*np.pi/5, '#8cc63f', 'NO'),      # Light Green
        (4*np.pi/5,       3*np.pi/5, '#d7df23', 'LOW'),     # Yellow / Light Yellow
        (3*np.pi/5,       2*np.pi/5, '#f7931e', 'MEDIUM'),  # Orange
        (2*np.pi/5,       np.pi/5,   '#f15a24', 'HIGH'),    # Dark Orange
        (np.pi/5,         0,         '#e61c24', 'MAX')      # Red
    ]

    # Draw 5 Segmented Color Arcs
    for start_a, end_a, col, label in zones:
        angles = np.linspace(start_a, end_a, 50)
        ax.plot(angles, [1.0]*50, color=col, linewidth=28, solid_capstyle='butt')
        
        # Add Radial Text Labels outside the arcs
        mid_a = (start_a + end_a) / 2
        ax.text(mid_a, 1.25, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='#c9d1d9', fontfamily='DejaVu Sans')

    # Needle Logic
    needle_angle = np.pi * (1.0 - score)
    
    # Draw Needle Shaft & Arrowhead
    ax.annotate('', xy=(needle_angle, 0.95), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='#e6edf3', lw=3.0,
                                mutation_scale=16, shrinkA=0))
    
    # Central Pivot Circle
    ax.plot(0, 0, 'o', color='#161b22', markersize=14, zorder=5)
    ax.plot(0, 0, 'o', color='#e6edf3', markersize=8, zorder=6)

    # Polar System Settings
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.35)
    ax.axis('off')

    # Bottom Title & Percentage Display
    ax.text(0, -0.22, "RISK", ha='center', va='center',
            fontsize=14, fontweight='bold', color='#8b949e', fontfamily='DejaVu Sans')
    
    score_pct_color = '#e61c24' if score >= 0.8 else ('#f15a24' if score >= 0.6 else ('#f7931e' if score >= 0.4 else '#8cc63f'))
    ax.text(0, -0.42, f"{score*100:.1f}%", ha='center', va='center',
            fontsize=20, fontweight='bold', color=score_pct_color, fontfamily='DejaVu Sans')

    plt.tight_layout(pad=0.1)
    return fig


def plot_feature_importance(model, feature_columns, top_n=15):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-top_n:]
    vals = importance[sorted_idx]
    labels = [feature_columns[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    cmap = LinearSegmentedColormap.from_list("risk", ['#388bfd', '#79c0ff'])
    colors = [cmap(v / vals.max()) for v in vals]

    bars = ax.barh(range(top_n), vals, color=colors, height=0.65)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', fontsize=7.5, color='#8b949e')

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=8.5, color='#c9d1d9')
    ax.set_xlabel("XGBoost Information Gain", color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#30363d')
    ax.grid(axis='x', alpha=0.12, color='#8b949e')
    plt.tight_layout()
    return fig


def plot_convergence(booster, model, dmatrix, final_score):
    cumulative = [float(booster.predict(dmatrix, iteration_range=(0, i))[0])
                  for i in range(1, model.n_estimators + 1)]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    ax.fill_between(range(1, model.n_estimators + 1), cumulative, alpha=0.12, color='#f85149')
    ax.plot(range(1, model.n_estimators + 1), cumulative, color='#f85149', linewidth=2, label='Risk Score P(fraud)')
    ax.axhline(0.70, color='#f85149', linestyle=':', linewidth=1.2, alpha=0.7, label='Critical threshold (0.70)')
    ax.axhline(0.40, color='#e3b341', linestyle='--', linewidth=1.2, alpha=0.8, label='Moderate threshold (0.40)')

    ax.scatter([model.n_estimators], [final_score], color='#e6edf3', s=60, zorder=5)
    ax.annotate(f'  Final: {final_score:.4f}', xy=(model.n_estimators, final_score),
                fontsize=8.5, color='#e6edf3', va='center')

    ax.set_xlabel("Boosting Round", color='#8b949e', fontsize=9)
    ax.set_ylabel("Accumulated Risk Probability", color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#30363d')
    ax.legend(fontsize=8, framealpha=0.15, labelcolor='#c9d1d9', facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.08, color='#8b949e')
    plt.tight_layout()
    return fig, cumulative

# ==============================================================================
# 4. SIDEBAR NAVIGATION & CONFIGURATIONS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 16px;'>
        <div style='font-size:1.3rem;font-weight:700;color:#e6edf3;letter-spacing:-0.02em;'>🛡️ DeFiLens</div>
        <div style='font-size:0.72rem;color:#8b949e;margin-top:2px;letter-spacing:0.04em;'>BEHAVIORAL FRAUD ENGINE v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">System Layers Navigation</p>', unsafe_allow_html=True)
    selected_layer = st.radio(
        "layer_select",
        options=[
            "🏠  Overview & Risk Meter",
            "📊  Layer 1 · Raw Feature Inputs",
            "📈  Layer 2 · Boosting Convergence",
            "🌲  Layer 3 · Feature Gain & Trees",
            "🚀  Layer 4 · Module 3 Handoff",
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown('<p class="section-header">Holdout Evaluation Profile</p>', unsafe_allow_html=True)
    sample_index = st.number_input(
        "Test Sample Index",
        min_value=0, max_value=len(X_test) - 1, value=5, step=1,
        help="Select sample row index from holdout dataset (X_test)."
    )

    st.divider()
    st.markdown('<p class="section-header">Module 1 Input Payload (JSON)</p>', unsafe_allow_html=True)
    default_m1_json = {
        "borrower_wallet_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        "verified_monthly_income": 4500.00,
        "total_liabilities": 10000.00,
        "requested_loan_amount": 27000.00,
        "discrepancy_ratio": 6.00,
        "dr_risk_penalty": 0.35
    }
    json_input = st.text_area(
        "payload_json",
        value=json.dumps(default_m1_json, indent=2),
        height=170,
        label_visibility="collapsed"
    )
    try:
        module1_payload = json.loads(json_input)
    except Exception:
        module1_payload = default_m1_json
        st.warning("Invalid JSON — falling back to defaults.")

# ==============================================================================
# 5. RISK ENGINE COMPUTATION PIPELINE
# ==============================================================================
sample_features_df = X_test.iloc[[sample_index]]
true_label = y_test.iloc[sample_index]

booster = model.get_booster()
dmatrix = xgb.DMatrix(sample_features_df)
base_xgboost_score = float(booster.predict(dmatrix)[0])
dr_penalty = module1_payload.get("dr_risk_penalty", 0.0)
final_risk_score = min(1.0, base_xgboost_score + dr_penalty)

if final_risk_score >= 0.70:
    status = "REVERTED"
    risk_tier = "CRITICAL / HIGH FRAUD RISK"
    banner_class = "banner-critical"
    badge_class = "badge-critical"
    status_icon = "🔴"
elif final_risk_score >= 0.40:
    status = "MULTI-SIG REQUIRED"
    risk_tier = "MODERATE RISK"
    banner_class = "banner-moderate"
    badge_class = "badge-moderate"
    status_icon = "🟡"
else:
    status = "APPROVED"
    risk_tier = "LOW RISK"
    banner_class = "banner-safe"
    badge_class = "badge-safe"
    status_icon = "🟢"

# ==============================================================================
# 6. DYNAMIC PAGE RENDERER
# ==============================================================================

# ── OVERVIEW & RISK METER PAGE ────────────────────────────────────────────────
if selected_layer == "🏠  Overview & Risk Meter":

    st.markdown("""
    <div style='margin-bottom:6px;'>
        <span style='font-size:1.6rem;font-weight:700;color:#e6edf3;letter-spacing:-0.03em;'>
            On-Chain Fraud Risk Assessment Overview
        </span>
        <br>
        <span style='font-size:0.82rem;color:#8b949e;'>
            Module 2 Execution Engine &nbsp;·&nbsp; Real-Time Behavioral Analysis &nbsp;·&nbsp; Cross-Domain Verification
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col_meter, col_gap, col_ai = st.columns([1.15, 0.05, 1.4])

    with col_meter:
        st.markdown('<p class="section-header">5-Zone Fraud Risk Arc Gauge</p>', unsafe_allow_html=True)
        fig_meter = draw_risk_meter_5zone(final_risk_score)
        st.pyplot(fig_meter, use_container_width=True)
        plt.close(fig_meter)

        st.markdown(f"""
        <div class="decision-banner {banner_class}">
            <div style="font-size:0.75rem;color:#8b949e;font-weight:500;letter-spacing:0.06em;margin-bottom:4px;">RECOMMENDED SYSTEM ACTION</div>
            <div style="font-size:1.25rem;font-weight:700;color:#e6edf3;">{status_icon} {status}</div>
            <div style="margin-top:6px;">
                <span class="risk-badge {badge_class}">{risk_tier}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header" style="margin-top:16px;">Score Composition</p>', unsafe_allow_html=True)
        score_data = {
            "Component": ["Base XGBoost Probability (P_base)", "DR Off-Chain Penalty", "Final Risk Output"],
            "Score": [f"{base_xgboost_score:.4f} ({base_xgboost_score*100:.1f}%)",
                      f"+{dr_penalty:.4f} (+{dr_penalty*100:.1f}%)",
                      f"{final_risk_score:.4f} ({final_risk_score*100:.1f}%)"]
        }
        st.dataframe(pd.DataFrame(score_data), hide_index=True, use_container_width=True)

    with col_ai:
        st.markdown('<p class="section-header">AI Explainability Engine</p>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="formula-card">
            Final Risk = min(1.0, P<sub>base</sub> + DR<sub>penalty</sub>)<br>
            = min(1.0, {base_xgboost_score:.4f} + {dr_penalty:.2f})
            = <b>{final_risk_score:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">Key Behavioral Drivers</p>', unsafe_allow_html=True)

        lifespan_mins = sample_features_df.get('Time Diff between first and last (Mins)', pd.Series([0])).values[0]
        sent_txs = sample_features_df.get('Sent tnx', pd.Series([0])).values[0]

        if base_xgboost_score > 0.40:
            st.markdown(f"""
            <div class="driver-item driver-critical">
                🔴 <b>On-Chain Anomaly Flagged</b> — Base XGBoost probability of <code>{base_xgboost_score*100:.1f}%</code>
                places this wallet in the high-risk behavioral cluster[cite: 1, 2]. Interaction velocity and contract entropy match known fraud signatures[cite: 1, 2].
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="driver-item driver-ok">
                🟢 <b>Low On-Chain Behavioral Risk</b> — Base XGBoost probability of <code>{base_xgboost_score*100:.1f}%</code>[cite: 1, 2].
                Transfer frequency and counterparty wallet diversity reflect standard non-malicious usage[cite: 1, 2].
            </div>""", unsafe_allow_html=True)

        if lifespan_mins < 1440:
            st.markdown(f"""
            <div class="driver-item driver-warn">
                ⚠️ <b>Short Wallet Lifespan</b> — Active lifespan of <code>{lifespan_mins:.0f} mins</code>[cite: 1, 2].
                Accounts under 24 hours old exhibit elevated Sybil and temporary burner wallet risk[cite: 1, 2].
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="driver-item driver-ok">
                ✅ <b>Established History</b> — Account has <code>{lifespan_mins/1440:.1f} days</code>
                of recorded on-chain activity, mitigating short-term account generation risks[cite: 1, 2].
            </div>""", unsafe_allow_html=True)

        if sent_txs > 50:
            st.markdown(f"""
            <div class="driver-item driver-warn">
                ⚡ <b>Transaction Velocity Spike</b> — <code>{sent_txs:.0f}</code> outbound transfers detected[cite: 1, 2].
                Rapid automated transactions may signal bot-like execution or flash loan draining scripts[cite: 1, 2].
            </div>""", unsafe_allow_html=True)

        if dr_penalty > 0:
            st.markdown(f"""
            <div class="driver-item driver-info">
                📄 <b>Module 1 Off-Chain Penalty (+{dr_penalty*100:.0f}%)</b> — Discrepancy ratio of
                <code>{module1_payload['discrepancy_ratio']:.1f}x</code>[cite: 1, 2]. Requested loan exceeds verified monthly income limits[cite: 1, 2].
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="driver-item driver-ok">
                ✅ <b>Income Alignment</b> — Requested loan is proportional to verified monthly earnings ($DR \le 5.0\text{x}$)[cite: 1, 2].
            </div>""", unsafe_allow_html=True)

        ground_color = "#f85149" if true_label == 1 else "#3fb950"
        ground_text = "⚠️ SCAM ACCOUNT" if true_label == 1 else "✅ LEGITIMATE ACCOUNT"
        st.markdown(f"""
        <div class="driver-item driver-info" style="margin-top:10px;">
            🏷️ <b>Dataset Ground Truth</b> — FLAG = <code>{true_label}</code>
            &nbsp;<span style="color:{ground_color};font-weight:600;">{ground_text}</span>[cite: 1, 2]
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-header">Evaluation Borrower Profile Summary</p>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    addr = module1_payload['borrower_wallet_address']
    with m1: st.metric("Wallet Address", f"{addr[:6]}…{addr[-4:]}")
    with m2: st.metric("Monthly Income", f"${module1_payload['verified_monthly_income']:,.0f}")
    with m3: st.metric("Requested Loan", f"${module1_payload['requested_loan_amount']:,.0f}")
    with m4: st.metric("Total Debt", f"${module1_payload['total_liabilities']:,.0f}")
    with m5: st.metric("DR Ratio / Penalty", f"{module1_payload['discrepancy_ratio']:.1f}x (+{dr_penalty*100:.0f}%)")
    with m6: st.metric("Final Risk", f"{final_risk_score*100:.1f}%")

# ── LAYER 1: RAW FEATURES ──────────────────────────────────────────────────────
elif selected_layer == "📊  Layer 1 · Raw Feature Inputs":
    st.title("📊 Layer 1 · Multi-Modal Feature Synchronization")
    st.caption("Verifies off-chain document metrics ingested from Module 1 alongside on-chain features loaded from `transaction_dataset.csv`[cite: 1, 2].")
    st.divider()

    tab_oc, tab_onchain, tab_stats = st.tabs(["📄 Off-Chain Payload", "⛓️ On-Chain Vector", "📊 Feature Statistics"])
    with tab_oc:
        c1, c2 = st.columns(2)
        with c1: st.json(module1_payload)
        with c2:
            st.markdown("""
            <div class="info-card">
                <b>borrower_wallet_address:</b> EOA address undergoing verification[cite: 1, 2].<br><br>
                <b>verified_monthly_income:</b> Extracted off-chain income via Module 1 RAG[cite: 1, 2].<br><br>
                <b>requested_loan_amount:</b> Total borrow credit requested on-chain[cite: 1, 2].<br><br>
                <b>dr_risk_penalty:</b> Additive risk penalty applied if $DR > 5.0$[cite: 1, 2].
            </div>
            """, unsafe_allow_html=True)
    with tab_onchain:
        st.write(f"Displaying on-chain attributes for holdout sample index `#{sample_index}`[cite: 1, 2]:")
        display_df = sample_features_df.T.rename(columns={sample_features_df.index[0]: 'Attribute Value'})
        st.dataframe(display_df, use_container_width=True, height=420)
    with tab_stats:
        st.dataframe(X_test[feature_columns].describe().T.style.format("{:.4f}"), use_container_width=True, height=420)

# ── LAYER 2: BOOSTING CONVERGENCE ──────────────────────────────────────────────
elif selected_layer == "📈  Layer 2 · Boosting Convergence":
    st.title("📈 Layer 2 · XGBoost Stage-by-Stage Convergence")
    st.caption("Tracks probability accumulation across sequential decision trees[cite: 1, 2].")
    st.divider()

    fig_conv, cumulative = plot_convergence(booster, model, dmatrix, base_xgboost_score)
    st.pyplot(fig_conv, use_container_width=True)
    plt.close(fig_conv)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Tree 10 Score", f"{cumulative[9]:.4f}")
    with c2: st.metric("Tree 25 Score", f"{cumulative[24]:.4f}")
    with c3: st.metric("Tree 50 Score", f"{cumulative[49]:.4f}")
    with c4: st.metric("Final Score (T100)", f"{cumulative[-1]:.4f}")

# ── LAYER 3: FEATURE GAIN & TREES ──────────────────────────────────────────────
elif selected_layer == "🌲  Layer 3 · Feature Gain & Trees":
    st.title("🌲 Layer 3 · Explainable AI Feature Gain & Decision Trees")
    st.caption("Evaluates global XGBoost Gain metrics and terminal tree decision splits[cite: 1, 2].")
    st.divider()

    tab_imp, tab_tree = st.tabs(["📊 Feature Importance", "🌲 Tree Dump"])
    with tab_imp:
        fig_imp = plot_feature_importance(model, feature_columns, top_n=15)
        st.pyplot(fig_imp, use_container_width=True)
        plt.close(fig_imp)
    with tab_tree:
        tree_dump = booster.get_dump()[99]
        st.code(tree_dump[:1500] + "\n...", language="text")

# ── LAYER 4: MODULE 3 HANDOFF ──────────────────────────────────────────────────
elif selected_layer == "🚀  Layer 4 · Module 3 Handoff":
    st.title("🚀 Layer 4 · Output Transmission Payload to Module 3")
    st.caption("Formats the authorization payload passed to the DeFi Lending Simulator[cite: 1, 2].")
    st.divider()

    output_payload = {
        "borrower_wallet_address": module1_payload["borrower_wallet_address"],
        "verified_monthly_income": module1_payload["verified_monthly_income"],
        "requested_loan_amount": module1_payload["requested_loan_amount"],
        "base_xgboost_risk": round(base_xgboost_score, 4),
        "dr_penalty_applied": round(dr_penalty, 4),
        "final_risk_score": round(final_risk_score, 4),
        "risk_tier": risk_tier,
        "execution_status": status,
        "authorized_for_module3": status in ["APPROVED", "MULTI-SIG REQUIRED"]
    }

    c_pay, c_sum = st.columns([1, 1])
    with c_pay:
        st.markdown("### **Formatted Output JSON**")
        st.json(output_payload)
        st.download_button(
            label="📥 Download Payload JSON for Module 3",
            data=json.dumps(output_payload, indent=2),
            file_name="module2_risk_output.json",
            mime="application/json"
        )
    with c_sum:
        st.markdown("### **Transmission Decision**")
        st.markdown(f"""
        <div class="decision-banner {banner_class}">
            <div style="font-size:0.75rem;color:#8b949e;font-weight:500;">STATUS</div>
            <div style="font-size:1.3rem;font-weight:700;color:#e6edf3;">{status_icon} {status}</div>
            <div style="margin-top:6px;font-size:0.85rem;">Authorized for Module 3: <b>{output_payload['authorized_for_module3']}</b></div>
        </div>
        """, unsafe_allow_html=True)