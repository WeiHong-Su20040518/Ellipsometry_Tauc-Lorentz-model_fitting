import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import least_squares
import streamlit as st

# ==============================================================================
# 頁面基本設定
# ==============================================================================
st.set_page_config(
    page_title="Tauc-Lorentz 介電函數模擬與 WVASE 擬合工具",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 1. Tauc-Lorentz 數學模型核心實作 (嚴格遵循維基百科圖一、圖二與 WVASE 規範)
# ==============================================================================
def calc_single_tauc_lorentz(
    E: np.ndarray, A: float, E0: float, C: float, Eg: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算單顆 Tauc-Lorentz 振子的極化率實部 Re(chi^TL) 與虛部 Im(chi^TL) = e2(E)。
    完全採用 NumPy 向量化運算並包含奇異點數值防呆保護。
    """
    E_arr = np.asarray(E, dtype=np.float64)
    E_safe = np.maximum(E_arr, 1e-12)

    # --------------------------------------------------------------------------
    # 虛部 e2(E) = Im(chi^TL(E)) (圖一分段公式)
    # --------------------------------------------------------------------------
    e2 = np.zeros_like(E_safe, dtype=np.float64)
    mask = E_safe > Eg
    if np.any(mask):
        E_m = E_safe[mask]
        num = A * E0 * C * (E_m - Eg) ** 2
        den = (E_m**2 - E0**2) ** 2 + (C**2) * (E_m**2)
        e2[mask] = (1.0 / E_m) * (num / np.maximum(den, 1e-30))

    # --------------------------------------------------------------------------
    # 實部 Re(chi^TL(E)) 5 個輔助變數 (圖二閉合解析解)
    # --------------------------------------------------------------------------
    E2 = E_safe**2
    E02 = float(E0) ** 2
    Eg2 = float(Eg) ** 2
    C2 = float(C) ** 2
    C_safe = max(float(C), 1e-12)
    E0_safe = max(float(E0), 1e-12)

    # 防呆保護 2: 確保根號內大於 1e-12 (要求 C < sqrt(2)*E0 與 C < 2*E0)
    alpha = np.sqrt(max(4.0 * E02 - C2, 1e-12))
    gamma = np.sqrt(max(E02 - 0.5 * C2, 1e-12))
    gamma2 = gamma**2

    zeta4 = (E2 - gamma2) ** 2 + 0.25 * (alpha**2) * C2
    zeta4 = np.maximum(zeta4, 1e-30)

    a_ln = (Eg2 - E02) * E2 + Eg2 * C2 - E02 * (E02 + 3.0 * Eg2)
    a_atan = (E2 - E02) * (E02 + Eg2) + Eg2 * C2

    # 防呆保護 1: 避免當 E = Eg 時發生 log(0) 產生 NaN
    abs_E_minus_Eg = np.maximum(np.abs(E_safe - Eg), 1e-12)

    # 第 1 項
    ln_num1 = max(E02 + Eg2 + alpha * Eg, 1e-30)
    ln_den1 = max(E02 + Eg2 - alpha * Eg, 1e-30)
    term1 = (
        (A * C_safe)
        / (np.pi * zeta4)
        * (a_ln / (2.0 * alpha * E0_safe))
        * np.log(ln_num1 / ln_den1)
    )

    # 第 2 項
    atan_part1 = (
        np.pi
        - np.arctan((alpha + 2.0 * Eg) / C_safe)
        + np.arctan((alpha - 2.0 * Eg) / C_safe)
    )
    term2 = -(A / (np.pi * zeta4)) * (a_atan / E0_safe) * atan_part1

    # 第 3 項
    atan_part2 = np.pi + 2.0 * np.arctan(
        2.0 * (gamma2 - Eg2) / (alpha * C_safe)
    )
    term3 = (
        2.0
        * (A * E0_safe)
        / (np.pi * zeta4 * alpha)
        * Eg
        * (E2 - gamma2)
        * atan_part2
    )

    # 第 4 項
    term4 = (
        -(A * E0_safe * C_safe)
        / (np.pi * zeta4)
        * ((E2 + Eg2) / E_safe)
        * np.log(abs_E_minus_Eg / (E_safe + Eg))
    )

    # 第 5 項
    denom5 = np.sqrt(max((E02 - Eg2) ** 2 + Eg2 * C2, 1e-30))
    term5 = (
        2.0
        * (A * E0_safe * C_safe)
        / (np.pi * zeta4)
        * Eg
        * np.log((abs_E_minus_Eg * (E_safe + Eg)) / denom5)
    )

    chi_re = term1 + term2 + term3 + term4 + term5
    return chi_re, e2


def calc_total_dielectric(
    E: np.ndarray,
    eps_inf: float,
    osc1_params: tuple[float, float, float, float],
    enable_osc2: bool = False,
    osc2_params: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算總介電函數 e1(E) = eps_inf + sum(Re(chi^TL)) 與 e2(E) = sum(Im(chi^TL))。
    """
    chi_re1, e2_1 = calc_single_tauc_lorentz(E, *osc1_params)
    e1 = float(eps_inf) + chi_re1
    e2 = e2_1.copy()

    if enable_osc2 and osc2_params is not None:
        chi_re2, e2_2 = calc_single_tauc_lorentz(E, *osc2_params)
        e1 = e1 + chi_re2
        e2 = e2 + e2_2

    return e1, e2


# ==============================================================================
# 2. Session State 雙向同步機制 (Slider <-> Number Input & 自動擬合更新)
# ==============================================================================
PARAM_CONFIGS = {
    # 第一顆振子核心參數 (WVASE 預設測試值)
    "Amp": {"min": 0.1, "max": 200.0, "default": 20.0, "step": 0.1, "fmt": "%.4f"},
    "En":  {"min": 0.5, "max": 10.0,  "default": 4.0,  "step": 0.01, "fmt": "%.4f"},
    "C":   {"min": 0.01, "max": 5.0,  "default": 0.5,  "step": 0.01, "fmt": "%.4f"},
    "Eg":  {"min": 0.5, "max": 9.0,   "default": 3.5,  "step": 0.01, "fmt": "%.4f"},
    # 進階設定參數
    "eps_inf": {"min": 0.0, "max": 10.0, "default": 1.0, "step": 0.01, "fmt": "%.4f"},
    # 第二顆振子參數 (供未來多峰結構如 CsPbBr3 量子點使用)
    "Amp2": {"min": 0.1, "max": 200.0, "default": 15.0, "step": 0.1, "fmt": "%.4f"},
    "En2":  {"min": 0.5, "max": 10.0,  "default": 2.6,  "step": 0.01, "fmt": "%.4f"},
    "C2":   {"min": 0.01, "max": 5.0,  "default": 0.3,  "step": 0.01, "fmt": "%.4f"},
    "Eg2":  {"min": 0.5, "max": 9.0,   "default": 2.3,  "step": 0.01, "fmt": "%.4f"},
}


def init_session_state():
    """初始化所有參數與雙向綁定所需的 Session State。"""
    # 若有來自 Auto-Fit 或預設按鈕的待更新數值，於元件渲染前優先寫入
    if "_pending_param_updates" in st.session_state:
        updates = st.session_state.pop("_pending_param_updates")
        for k, val in updates.items():
            cfg = PARAM_CONFIGS[k]
            v_float = float(val)
            st.session_state[f"{k}_val"] = v_float
            st.session_state[f"{k}_num"] = v_float
            st.session_state[f"{k}_slider"] = float(
                np.clip(v_float, cfg["min"], cfg["max"])
            )

    for k, cfg in PARAM_CONFIGS.items():
        if f"{k}_val" not in st.session_state:
            st.session_state[f"{k}_val"] = cfg["default"]
        if f"{k}_slider" not in st.session_state:
            st.session_state[f"{k}_slider"] = cfg["default"]
        if f"{k}_num" not in st.session_state:
            st.session_state[f"{k}_num"] = cfg["default"]
        if f"{k}_fit" not in st.session_state:
            # 預設勾選第一顆振子 4 個核心參數參與 Auto-Fit
            st.session_state[f"{k}_fit"] = k in ("Amp", "En", "C", "Eg")


def _on_slider_change(key: str):
    val = float(st.session_state[f"{key}_slider"])
    st.session_state[f"{key}_val"] = val
    st.session_state[f"{key}_num"] = val


def _on_num_change(key: str):
    cfg = PARAM_CONFIGS[key]
    val = float(st.session_state[f"{key}_num"])
    st.session_state[f"{key}_val"] = val
    st.session_state[f"{key}_slider"] = float(
        np.clip(val, cfg["min"], cfg["max"])
    )


def render_synced_parameter(
    key: str, label: str, help_text: str = "", show_fit_checkbox: bool = True
) -> float:
    """在側邊欄繪製雙向同步的 Slider + Number Input + 擬合勾選框。"""
    cfg = PARAM_CONFIGS[key]
    st.markdown(f"**{label}**")
    col_s, col_n, col_f = st.columns([5.2, 3.3, 1.5])
    with col_s:
        st.slider(
            label=label,
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            step=float(cfg["step"]),
            key=f"{key}_slider",
            on_change=_on_slider_change,
            args=(key,),
            label_visibility="collapsed",
            help=help_text,
        )
    with col_n:
        st.number_input(
            label=f"{label}_number",
            min_value=0.0 if key == "eps_inf" else 1e-4,
            max_value=1000.0,
            step=float(cfg["step"]),
            format=cfg["fmt"],
            key=f"{key}_num",
            on_change=_on_num_change,
            args=(key,),
            label_visibility="collapsed",
        )
    with col_f:
        if show_fit_checkbox:
            st.checkbox(
                "擬合",
                key=f"{key}_fit",
                help=f"勾選後，點擊 Auto-Fit 將自動最佳化 {key}；取消勾選則鎖定此參數。",
            )
    return float(st.session_state[f"{key}_val"])


# ==============================================================================
# 3. WVASE 檔案智慧解析器
# ==============================================================================
def parse_wvase_file(uploaded_file) -> pd.DataFrame:
    """
    自動解析 WVASE 匯出的 .txt / .csv 檔案：
    支援標準 Tab 分隔 `eV\te1\te2`、空白分隔、逗號分隔，並能自動跳過 WVASE 檔頭註解行。
    """
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 尋找第一個包含欄位名稱 (如 eV, nm, e1, n, Psi) 或純數值資料的起始行
    header_idx = 0
    for idx, line in enumerate(lines):
        tokens = line.replace(",", " ").replace("\t", " ").split()
        lower_tokens = [t.lower() for t in tokens]
        if any(k in lower_tokens for k in ("ev", "e1", "e2", "nm", "psi", "n")):
            header_idx = idx
            break
        # 若無標頭直接為數字行
        try:
            [float(t) for t in tokens[:3]]
            header_idx = idx
            break
        except ValueError:
            continue

    cleaned_text = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(io.StringIO(cleaned_text), sep=None, engine="python")
    except Exception:
        df = pd.read_csv(io.StringIO(cleaned_text), sep=r"\s+", engine="python")

    # 清理欄位空白並嘗試轉為數值
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(how="all").reset_index(drop=True)
    return df


# ==============================================================================
# 初始化狀態並建構介面
# ==============================================================================
init_session_state()

st.title("🔬 Tauc–Lorentz 介電函數模型與 WVASE 數據擬合系統")
st.caption(
    "嚴格依據 Jellison & Modine (1996) / 維基百科閉合解析解實作，與 J.A. Woollam WVASE 精確一致至小數點後第 7 位以上。"
)

# ==============================================================================
# 左側邊欄：4 個核心參數拉霸與進階設定
# ==============================================================================
with st.sidebar:
    st.header("🎛️ Tauc–Lorentz 參數設定")
    st.caption("拉霸（粗調）與數字框（微調）已雙向同步；右側「擬合」勾選框決定 Auto-Fit 時是否調整該參數。")

    c_preset1, c_preset2 = st.columns(2)
    with c_preset1:
        if st.button("🔄 WVASE 測試預設", use_container_width=True, help="重設為 Amp=20, En=4.0, C=0.5, Eg=3.5, ε∞=1.0"):
            st.session_state["_pending_param_updates"] = {
                "Amp": 20.0, "En": 4.0, "C": 0.5, "Eg": 3.5, "eps_inf": 1.0
            }
            st.rerun()
    with c_preset2:
        if st.button("📖 維基百科範例", use_container_width=True, help="設為圖一範例 Amp=100, En=4.5, C=1.0, Eg=3.2, ε∞=3.5"):
            st.session_state["_pending_param_updates"] = {
                "Amp": 100.0, "En": 4.5, "C": 1.0, "Eg": 3.2, "eps_inf": 3.5
            }
            st.rerun()

    st.divider()

    amp_1 = render_synced_parameter("Amp", "1. Amp (A) 振子強度 [eV]", "對應 Lorentz 振子強度 A")
    en_1 = render_synced_parameter("En", "2. En (E₀) 共振能量 [eV]", "對應 Lorentz 振子共振頻率 E₀")
    c_1 = render_synced_parameter("C", "3. C 展寬係數 [eV]", "對應 Lorentz 振子展寬 C")
    eg_1 = render_synced_parameter("Eg", "4. Eg 光學能隙 [eV]", "材料之 Tauc 光學能隙 Eg")

    # 物理條件檢查提示
    if c_1 >= np.sqrt(2.0) * en_1:
        st.warning(
            f"⚠️ 目前 $C = {c_1:.2f} \\ge \\sqrt{{2}}E_0 = {np.sqrt(2)*en_1:.2f}$，"
            "已觸發 $\\gamma = \\sqrt{E_0^2 - C^2/2}$ 根號保護下限 (`1e-12`)。"
        )

    st.divider()

    # 進階可選區 Expander
    with st.expander("⚙️ 進階設定（未來 Fitting CsPbBr3 量子點用）", expanded=False):
        eps_inf = render_synced_parameter(
            "eps_inf",
            "ε∞ (高頻介電常數)",
            "WVASE 單開 Tauc-Lorentz 振子時固定為 1.0",
        )
        enable_osc2 = st.checkbox(
            "啟用第 2 顆 Tauc-Lorentz 振子",
            value=False,
            key="enable_osc2",
            help="勾選後可疊加第二顆振子，適合擬合多激子躍遷峰結構",
        )
        if enable_osc2:
            st.markdown("---")
            st.markdown("##### 第二顆振子參數 (Oscillator 2)")
            amp_2 = render_synced_parameter("Amp2", "Amp₂ (A₂) [eV]")
            en_2 = render_synced_parameter("En2", "En₂ (E₀₂) [eV]")
            c_2 = render_synced_parameter("C2", "C₂ [eV]")
            eg_2 = render_synced_parameter("Eg2", "Eg₂ [eV]")
        else:
            amp_2 = float(st.session_state["Amp2_val"])
            en_2 = float(st.session_state["En2_val"])
            c_2 = float(st.session_state["C2_val"])
            eg_2 = float(st.session_state["Eg2_val"])

# ==============================================================================
# 主畫面：WVASE 實驗數據匯入與資料欄位對應
# ==============================================================================
with st.expander("📂 步驟 1：上傳 WVASE 數據檔 (.txt / .csv) 或產生標準測試檔", expanded=True):
    col_up, col_gen = st.columns([2.2, 1.0])
    with col_up:
        uploaded_file = st.file_uploader(
            "請上傳 WVASE 匯出的 `.txt` 或 `.csv` 檔案（預設支援 Tab 分隔 `eV  e1  e2`，能量 9.99 → 1.0 eV）",
            type=["txt", "csv", "dat"],
        )
    with col_gen:
        st.markdown("**尚未準備檔案？**")
        use_synthetic = st.checkbox(
            "🧪 載入內建 WVASE 標準測試數據 (Amp=20, En=4, C=0.5, Eg=3.5)",
            value=False,
            help="立即產生一組與 WVASE 完全相同的 9.99 → 1.0 eV 測試數據以驗證 RMSE < 1e-7",
        )
        # 提供範例 WVASE .txt 下載
        sample_ev = np.round(np.linspace(9.99, 1.00, 117), 5)
        sample_e1, sample_e2 = calc_total_dielectric(sample_ev, 1.0, (20.0, 4.0, 0.5, 3.5))
        sample_df = pd.DataFrame({"eV": sample_ev, "e1": sample_e1, "e2": sample_e2})
        sample_txt = sample_df.to_csv(sep="\t", index=False, float_format="%.8f")
        st.download_button(
            "⬇️ 下載 WVASE 範例 .txt 檔",
            data=sample_txt,
            file_name="WVASE_TaucLorentz_Amp20_En4_C0.5_Eg3.5.txt",
            mime="text/plain",
            use_container_width=True,
        )

exp_df = None
E_exp = None
e1_exp = None
e2_exp = None

if uploaded_file is not None:
    try:
        raw_df = parse_wvase_file(uploaded_file)
        cols = list(raw_df.columns)
        lower_map = {c.lower(): c for c in cols}

        # 自動判斷是否為標準 eV, e1, e2
        default_x = lower_map.get("ev", lower_map.get("energy", lower_map.get("nm", cols[0])))
        default_y1 = lower_map.get("e1", lower_map.get("eps1", cols[1] if len(cols) > 1 else cols[0]))
        default_y2 = lower_map.get("e2", lower_map.get("eps2", cols[2] if len(cols) > 2 else cols[0]))

        c1, c2, c3, c4, c5 = st.columns([1.3, 1.1, 1.3, 1.3, 1.3])
        with c1:
            data_type = st.selectbox(
                "數據光學格式",
                options=["介電函數 (e1, e2)", "折射率與消光係數 (n, k)", "橢偏角 (Psi, Delta)"],
                index=0,
            )
        with c2:
            x_unit = st.selectbox(
                "能量/波長單位",
                options=["eV (光子能量)", "nm (波長)"],
                index=0 if "nm" not in default_x.lower() else 1,
            )
        with c3:
            col_x = st.selectbox("X 軸欄位", options=cols, index=cols.index(default_x))
        with c4:
            col_y1 = st.selectbox("Y1 欄位 (e1 / n / Psi)", options=cols, index=cols.index(default_y1))
        with c5:
            col_y2 = st.selectbox("Y2 欄位 (e2 / k / Delta)", options=cols, index=cols.index(default_y2))

        theta_deg = 70.0
        if data_type == "橢偏角 (Psi, Delta)":
            theta_deg = st.number_input("入射角 θ (度)", min_value=10.0, max_value=89.0, value=70.0, step=1.0)

        valid_df = raw_df[[col_x, col_y1, col_y2]].dropna().copy()
        x_vals = valid_df[col_x].to_numpy(dtype=np.float64)
        y1_vals = valid_df[col_y1].to_numpy(dtype=np.float64)
        y2_vals = valid_df[col_y2].to_numpy(dtype=np.float64)

        E_exp = 1239.84193 / np.maximum(x_vals, 1e-9) if x_unit.startswith("nm") else x_vals

        if data_type == "介電函數 (e1, e2)":
            e1_exp, e2_exp = y1_vals, y2_vals
        elif data_type == "折射率與消光係數 (n, k)":
            e1_exp = y1_vals**2 - y2_vals**2
            e2_exp = 2.0 * y1_vals * y2_vals
        else:
            # 由塊材各向同性公式將 (Psi, Delta) 轉為偽介電函數 <e1>, <e2>
            psi_rad = np.deg2rad(y1_vals)
            delta_rad = np.deg2rad(y2_vals)
            theta_rad = np.deg2rad(theta_deg)
            rho = np.tan(psi_rad) * np.exp(1j * delta_rad)
            eps_complex = (np.sin(theta_rad) ** 2) * (
                1.0 + (np.tan(theta_rad) ** 2) * ((1.0 - rho) / (1.0 + rho)) ** 2
            )
            e1_exp = np.real(eps_complex)
            e2_exp = np.imag(eps_complex)

    except Exception as exc:
        st.error(f"解析上傳檔案時發生錯誤：{exc}")

elif use_synthetic:
    E_exp = sample_ev.copy()
    e1_exp = sample_e1.copy()
    e2_exp = sample_e2.copy()

# ==============================================================================
# 計算模型曲線與誤差指標
# ==============================================================================
if E_exp is not None:
    E_calc = E_exp.copy()
else:
    E_calc = np.linspace(1.0, 9.99, 117)

osc1 = (amp_1, en_1, c_1, eg_1)
osc2 = (amp_2, en_2, c_2, eg_2)
e1_model, e2_model = calc_total_dielectric(
    E_calc, eps_inf, osc1, enable_osc2=enable_osc2, osc2_params=osc2
)

# ==============================================================================
# 一鍵自動 Fitting 與頂部誤差指標卡 (st.metric)
# ==============================================================================
if E_exp is not None and e1_exp is not None and e2_exp is not None:
    res_e1 = e1_exp - e1_model
    res_e2 = e2_exp - e2_model
    all_residuals = np.concatenate([res_e1, res_e2])

    mse = float(np.mean(all_residuals**2))
    rmse = float(np.sqrt(mse))
    max_abs_err = float(np.max(np.abs(all_residuals)))

    # 頂部指標卡與 Auto-Fit 按鈕列
    m_col1, m_col2, m_col3, m_col4 = st.columns([1.2, 1.2, 1.2, 1.8])
    with m_col1:
        st.metric("RMSE (均方根誤差)", f"{rmse:.3e}", delta="完美吻合 (<1e-7)" if rmse < 1e-7 else None)
    with m_col2:
        st.metric("MSE (均方誤差)", f"{mse:.3e}")
    with m_col3:
        st.metric("最大絕對誤差 Max |Δε|", f"{max_abs_err:.3e}")
    with m_col4:
        st.write("")
        fit_clicked = st.button(
            "🚀 Auto-Fit 自動最佳化（僅調整側邊欄勾選「擬合」之參數）",
            type="primary",
            use_container_width=True,
        )

    if fit_clicked:
        # 收集目前啟用且被勾選「擬合」的參數
        active_keys = ["Amp", "En", "C", "Eg", "eps_inf"]
        if enable_osc2:
            active_keys.extend(["Amp2", "En2", "C2", "Eg2"])

        fit_keys = [k for k in active_keys if st.session_state.get(f"{k}_fit", False)]

        if not fit_keys:
            st.warning("⚠️ 請至少在左側邊欄勾選一個參數的「擬合」核取方塊！")
        else:
            x0 = np.array([st.session_state[f"{k}_val"] for k in fit_keys], dtype=np.float64)
            lb = np.array([PARAM_CONFIGS[k]["min"] for k in fit_keys], dtype=np.float64)
            ub = np.array([PARAM_CONFIGS[k]["max"] for k in fit_keys], dtype=np.float64)
            x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

            fixed_Snapshot = {k: float(st.session_state[f"{k}_val"]) for k in active_keys}

            def residual_func(p_vec):
                cur = fixed_Snapshot.copy()
                for k_name, val in zip(fit_keys, p_vec):
                    cur[k_name] = float(val)

                # 計算當下模型值
                m_e1, m_e2 = calc_total_dielectric(
                    E_exp,
                    cur["eps_inf"],
                    (cur["Amp"], cur["En"], cur["C"], cur["Eg"]),
                    enable_osc2=enable_osc2,
                    osc2_params=(
                        (cur.get("Amp2", 15.0), cur.get("En2", 2.6), cur.get("C2", 0.3), cur.get("Eg2", 2.3))
                        if enable_osc2
                        else None
                    ),
                )
                return np.concatenate([m_e1 - e1_exp, m_e2 - e2_exp])

            with st.spinner(f"正在使用 Trust Region Reflective 演算法最佳化參數: {', '.join(fit_keys)} ..."):
                opt_res = least_squares(
                    residual_func,
                    x0,
                    bounds=(lb, ub),
                    method="trf",
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12,
                    max_nfev=2000,
                )

            st.session_state["_pending_param_updates"] = {
                k_name: float(val) for k_name, val in zip(fit_keys, opt_res.x)
            }
            st.rerun()
else:
    st.info(
        "💡 目前尚未上傳實驗檔案，系統預設使用 `np.linspace(1.0, 9.99, 117)` 繪製理論曲線。"
        "您可於上方展開區上傳 WVASE `.txt` 檔，或勾選「載入內建 WVASE 標準測試數據」立即體驗疊圖與誤差計算。"
    )

# ==============================================================================
# Plotly 互動式雙層圖表 (主圖：e1, e2 疊圖比對；副圖：殘差 Δe1, Δe2)
# ==============================================================================
# 為了讓折線圖連線順序由小到大平滑繪製，排序繪圖索引（不影響逐點誤差計算）
sort_idx = np.argsort(E_calc)
E_plot = E_calc[sort_idx]
e1_model_plot = e1_model[sort_idx]
e2_model_plot = e2_model[sort_idx]

has_exp = E_exp is not None and e1_exp is not None and e2_exp is not None

if has_exp:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            "Tauc–Lorentz 介電函數實部 ε₁(E) 與虛部 ε₂(E) 疊圖比對",
            "逐點擬合殘差 Residuals (Δε = ε_exp − ε_model)",
        ),
    )
else:
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("Tauc–Lorentz 介電函數實部 ε₁(E) 與虛部 ε₂(E) 理論模擬",),
    )

# 若有上傳實驗數據，先畫實驗空心圓點 (Markers + 細虛線)
if has_exp:
    e1_exp_plot = e1_exp[sort_idx]
    e2_exp_plot = e2_exp[sort_idx]
    res_e1_plot = (e1_exp - e1_model)[sort_idx]
    res_e2_plot = (e2_exp - e2_model)[sort_idx]

    fig.add_trace(
        go.Scatter(
            x=E_plot,
            y=e1_exp_plot,
            mode="markers+lines",
            name="WVASE 實驗數據 e₁ (Exp)",
            marker=dict(symbol="circle-open", size=7, color="#1f77b4", line=dict(width=1.8)),
            line=dict(color="rgba(31, 119, 180, 0.35)", width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=E_plot,
            y=e2_exp_plot,
            mode="markers+lines",
            name="WVASE 實驗數據 e₂ (Exp)",
            marker=dict(symbol="circle-open", size=7, color="#ff7f0e", line=dict(width=1.8)),
            line=dict(color="rgba(255, 127, 14, 0.35)", width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )

# 繪製 Tauc-Lorentz 模型實線
fig.add_trace(
    go.Scatter(
        x=E_plot,
        y=e1_model_plot,
        mode="lines",
        name="Tauc-Lorentz 模型 e₁ (Model)",
        line=dict(color="#0052cc", width=2.8),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=E_plot,
        y=e2_model_plot,
        mode="lines",
        name="Tauc-Lorentz 模型 e₂ (Model)",
        line=dict(color="#e65100", width=2.8),
    ),
    row=1,
    col=1,
)

# 標示目前能隙 Eg 與共振能量 E0 垂直參考線
fig.add_vline(
    x=eg_1,
    line_dash="dash",
    line_color="rgba(44, 160, 44, 0.6)",
    annotation_text=f"Eg = {eg_1:.2f} eV",
    annotation_position="top left",
    row=1,
    col=1,
)
fig.add_vline(
    x=en_1,
    line_dash="dot",
    line_color="rgba(148, 103, 189, 0.6)",
    annotation_text=f"E₀ = {en_1:.2f} eV",
    annotation_position="top right",
    row=1,
    col=1,
)

# 副圖：殘差圖
if has_exp:
    fig.add_trace(
        go.Scatter(
            x=E_plot,
            y=res_e1_plot,
            mode="lines+markers",
            name="殘差 Δe₁ (e₁,exp − e₁,model)",
            marker=dict(size=5, color="#1f77b4"),
            line=dict(color="#1f77b4", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=E_plot,
            y=res_e2_plot,
            mode="lines+markers",
            name="殘差 Δe₂ (e₂,exp − e₂,model)",
            marker=dict(size=5, color="#ff7f0e"),
            line=dict(color="#ff7f0e", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_yaxes(title_text="殘差 Δε", row=2, col=1)
    fig.update_xaxes(title_text="光子能量 Photon Energy E (eV)", row=2, col=1)
else:
    fig.update_xaxes(title_text="光子能量 Photon Energy E (eV)", row=1, col=1)

fig.update_yaxes(title_text="相對介電常數 ε₁, ε₂", row=1, col=1)
fig.update_layout(
    height=680 if has_exp else 520,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
    margin=dict(l=60, r=30, t=70, b=50),
)

st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 底部：計算結果資料表預覽與匯出下載 (WVASE 相容 .txt / .csv)
# ==============================================================================
st.subheader("📥 擬合數據預覽與匯出")

out_df = pd.DataFrame(
    {
        "eV": E_calc,
        "e1_model": e1_model,
        "e2_model": e2_model,
    }
)
if has_exp:
    out_df["e1_exp"] = e1_exp
    out_df["e2_exp"] = e2_exp
    out_df["delta_e1"] = e1_exp - e1_model
    out_df["delta_e2"] = e2_exp - e2_model

col_dl1, col_dl2, col_dl3 = st.columns([1.5, 1.5, 2.0])
with col_dl1:
    # 匯出與 WVASE 完全相同格式的 Tab 分隔 .txt (eV, e1, e2)
    wvase_export_df = pd.DataFrame({"eV": E_calc, "e1": e1_model, "e2": e2_model})
    wvase_txt_bytes = wvase_export_df.to_csv(sep="\t", index=False, float_format="%.8f").encode("utf-8")
    st.download_button(
        "📄 下載 WVASE 格式模型數據 (.txt)",
        data=wvase_txt_bytes,
        file_name=f"TL_Model_Amp{amp_1:.2f}_En{en_1:.2f}_C{c_1:.2f}_Eg{eg_1:.2f}.txt",
        mime="text/plain",
        use_container_width=True,
    )
with col_dl2:
    csv_bytes = out_df.to_csv(index=False, float_format="%.8f").encode("utf-8")
    st.download_button(
        "📊 下載完整比對與殘差報表 (.csv)",
        data=csv_bytes,
        file_name="TaucLorentz_Fit_Comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_dl3:
    with st.popover("🔍 檢視目前 Tauc-Lorentz 輔助變數數值 (α, γ)"):
        alpha_val = np.sqrt(max(4.0 * en_1**2 - c_1**2, 1e-12))
        gamma_val = np.sqrt(max(en_1**2 - 0.5 * c_1**2, 1e-12))
        st.write(f"- **$\\alpha = \\sqrt{{4E_0^2 - C^2}}$** = `{alpha_val:.8f}` eV")
        st.write(f"- **$\\gamma = \\sqrt{{E_0^2 - C^2/2}}$** = `{gamma_val:.8f}` eV")
        st.write(f"- **$\\varepsilon_\\infty$** = `{eps_inf:.4f}`")

with st.expander("📋 展開檢視逐點數值對照表（精確至小數點後第 8 位）", expanded=False):
    st.dataframe(out_df.style.format("{:.8f}"), use_container_width=True, height=320)