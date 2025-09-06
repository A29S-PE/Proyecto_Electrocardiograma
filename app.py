# app_ecg.py
# Dash de ECG multi-paciente con FC (RR) + alertas
import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import streamlit as st
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import tempfile
from drive import *

st.set_page_config(page_title="ECG (FC & RR)", layout="wide")

# ----------------- Parámetros iniciales -----------------
DEFAULT_ROOT = r"D:\Documentos\Cursos\Diplomatura IA\Modulo3\Redes Neuronales para el Análisis de Series Temporales\Proyecto\ECG\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords"
PREFERRED_LEADS = ["II", "MLII", "V2", "V5", "I", "AVF", "V1", "III"]
REMOTE_MODE = True

# ----------------- Utilidades -----------------
@st.cache_data(show_spinner=True)
def index_records(root_dir: str, remote: bool = False) -> pd.DataFrame:
    """
    Escanea todas las .hea bajo root_dir y extrae metadatos livianos vía rdheader.
    Retorna DataFrame con: id, base (path sin extensión), carpeta, fs, sig_len, n_sig, sig_names
    """
    if remote:
        try:
            df = pd.DataFrame(records_index)
            if not df.empty:
                df = df.sort_values(["carpeta", "id"]).reset_index(drop=True)
            return df
        except Exception as e:
            st.error(f"No se pudo obtener el índice remoto: {e}")
            return pd.DataFrame()
    else:
        root = Path(root_dir)
        if not root.exists():
            return pd.DataFrame()

        rows = []
        for hea in root.rglob("*.hea"):
            base = hea.with_suffix("")  # path sin extensión
            try:
                h = wfdb.rdheader(str(base))
                sig_names = h.sig_name if getattr(h, "sig_name", None) else [f"ch{i+1}" for i in range(h.nsig)]
                try:
                    # carpeta relativa si es posible
                    folder_rel = str(hea.parent.relative_to(root))
                except Exception:
                    folder_rel = str(hea.parent)
                rows.append({
                    "id": hea.stem,
                    "base": str(base),
                    "carpeta": folder_rel,
                    "fs": float(getattr(h, "fs", np.nan)),
                    "sig_len": int(getattr(h, "sig_len", 0) or 0),
                    "n_sig": int(getattr(h, "nsig", 0) or 0),
                    "sig_names": ",".join(sig_names),
                })
            except Exception:
                # Si algún header está corrupto, lo omitimos
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["carpeta", "id"]).reset_index(drop=True)
        return df

@st.cache_resource(show_spinner=False)
def load_record(base_path: str):
    """Carga completa del registro para graficar/procesar."""
    signals, fields = wfdb.rdsamp(base_path)
    return signals, fields

@st.cache_resource(show_spinner=False)
def load_record_drive(record_id: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        for ext in ["hea", "mat"]:
            file_id = find_file_id(record_id, ext)
            if not file_id:
                raise FileNotFoundError(f"No se encontró el archivo {record_id}.{ext} en Drive")

            request = drive_service.files().get_media(fileId=file_id)
            output_path = os.path.join(tmpdir, f"{record_id}.{ext}")

            with open(output_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"[{ext}] Progreso: {int(status.progress() * 100)}%")

        signals, fields = wfdb.rdsamp(os.path.join(tmpdir, record_id))
        return signals, fields
    
def choose_best_lead(sig_names: list[str]) -> int:
    """Intenta elegir una derivación 'ideal' (II/MLII ...). Devuelve índice."""
    upper = [str(s).strip().upper() for s in sig_names]
    for name in PREFERRED_LEADS:
        if name in upper:
            return upper.index(name)
    return 0

def fmt_seconds(s):
    try:
        s = float(s)
        m, ss = divmod(s, 60)
        h, mm = divmod(m, 60)
        if h >= 1:
            return f"{int(h)}h {int(mm)}m {ss:0.1f}s"
        if m >= 1:
            return f"{int(m)}m {ss:0.1f}s"
        return f"{ss:0.1f}s"
    except Exception:
        return "-"

def slice_window(y, fs, t0, win_s):
    i0 = max(0, int(t0 * fs))
    i1 = min(len(y), int((t0 + win_s) * fs))
    return i0, i1, y[i0:i1]

def downsample_for_plot(t, y, max_points=25000):
    """Muestreo para trazar rápido sin perder forma general."""
    n = len(y)
    if n <= max_points:
        return t, y
    step = int(np.ceil(n / max_points))
    return t[::step], y[::step]

def compute_hr_from_peaks(rpeaks_idx: np.ndarray, fs: float):
    """RR en segundos y FC en lpm (FC = 60 / RR) + tiempos interlatido."""
    if rpeaks_idx is None or len(rpeaks_idx) < 2:
        return np.array([]), np.array([]), np.array([])
    rr_s = np.diff(rpeaks_idx) / fs
    hr = 60.0 / rr_s
    # tiempo representativo de cada FC: punto medio entre picos
    t_between = (rpeaks_idx[1:] + rpeaks_idx[:-1]) / 2.0 / fs
    return rr_s, hr, t_between

def apply_ecg_layout(fig: Figure, t_max, y_data, t_start=0):
    fig.layout.shapes = ()

    # proteger cuando la ventana inicial está vacía
    if y_data is None or len(y_data) == 0:
        y_min, y_max = -1.0, 1.0
    else:
        ymin = float(np.min(y_data))
        ymax = float(np.max(y_data))
        # ajustar a múltiplos de 0.5 mV (papel real)
        y_min = np.floor(ymin / 0.5) * 0.5
        y_max = np.ceil(ymax / 0.5) * 0.5
    
    # === grid vertical (tiempo) ===
    for x in np.arange(t_start, t_start + t_max + 0.04, 0.04):  # 1 mm = 0.04s
        fig.add_shape(type="line", x0=x, x1=x, y0=y_min, y1=y_max,
                      line=dict(color="lightpink", width=0.5), layer="below")
    for x in np.arange(0, t_max + 0.20, 0.20):  # 5 mm = 0.20s
        fig.add_shape(type="line", x0=x, x1=x, y0=y_min, y1=y_max,
                      line=dict(color="red", width=1.2), layer="below")

    # === grid horizontal (amplitud) ===
    for y in np.arange(y_min, y_max + 0.1, 0.1):  # 1 mm = 0.1 mV
        fig.add_shape(type="line", x0=0, x1=t_max, y0=y, y1=y,
                      line=dict(color="lightpink", width=0.5), layer="below")
    for y in np.arange(y_min, y_max + 0.5, 0.5):  # 5 mm = 0.5 mV
        fig.add_shape(type="line", x0=0, x1=t_max, y0=y, y1=y,
                      line=dict(color="red", width=1.2), layer="below")

    # Layout final
    fig.update_layout(
        xaxis=dict(
            range=[t_start, t_max],
            title="Tiempo (s)",
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            range=[y_min, y_max],
            title="mV",
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

# ----------------- Sidebar -----------------
st.sidebar.header("⚙️ Configuración")

root_dir = st.sidebar.text_input(
    "Ubicación de los WFDBRecords",
    value=DEFAULT_ROOT,
    help="Carpeta que contiene todas las subcarpetas con .hea/.mat"
)

with st.sidebar.expander("📂 Lectura de registros", expanded=True):
    st.caption("Se indexan archivos *.hea en todas las subcarpetas.")
    df_index = index_records(root_dir, remote=REMOTE_MODE)
    if df_index.empty:
        st.warning("No se encontraron headers (.hea). Verifica la ruta.")
    else:
        st.success(f"Registros encontrados: {len(df_index):,}")

search = st.sidebar.text_input("🔎 Buscar (ID o carpeta contiene)", value="")
if not df_index.empty:
    mask = (
        df_index["id"].str.contains(search, case=False, na=False) |
        df_index["carpeta"].str.contains(search, case=False, na=False)
    ) if search else np.ones(len(df_index), dtype=bool)
    df_view = df_index.loc[mask].reset_index(drop=True)

    # Selección con índice para no cargar miles de strings en el widget
    options = df_view.index.tolist()
    sel_idx = st.sidebar.selectbox(
        "🧑‍⚕️ Selecciona registro/paciente",
        options,
        format_func=lambda i: f"{df_view.loc[i,'id']}  —  {df_view.loc[i,'carpeta']}"
    )
else:
    df_view = pd.DataFrame()
    sel_idx = None

lead_mode = st.sidebar.radio("Derivación", ["Automática (recomendada)", "Elegir manualmente"], horizontal=False)

desired_win = st.sidebar.slider("Ventana a visualizar (s)", min_value=1, max_value=15, value=1, step=1)
show_hr_curve = st.sidebar.checkbox("Mostrar curva de FC (lpm)", value=True)
download_csv = st.sidebar.checkbox("Preparar CSV RR/FC para descarga", value=True)

# ----------------- Main -----------------
st.title("Análisis de ECG – Frecuencia Cardíaca (RR) y Picos R")
st.caption("Método: FC = 60 / RR (segundos). La detección de picos R se realiza con neurokit2.")

if sel_idx is not None and not df_view.empty:
    row = df_view.loc[sel_idx]
    # Carga de registro
    try:
        if REMOTE_MODE:
            signals, fields = load_record_drive(row["id"])
        else:
            signals, fields = load_record(row["base"])
    except Exception as e:
        st.error(f"No se pudo cargar el registro: {e}")
        st.stop()

    fs = float(fields.get("fs", row["fs"]))
    sig_names = fields.get("sig_name", [f"ch{i+1}" for i in range(signals.shape[1])])
    duration_s = signals.shape[0] / fs

    # Elegir derivación
    if lead_mode == "Elegir manualmente":
        lead_name = st.selectbox("Elige derivación disponible", sig_names)
        lead_idx = sig_names.index(lead_name)
    else:
        lead_idx = choose_best_lead(sig_names)
        lead_name = sig_names[lead_idx]

    max_win_allowed = max(1, int(np.floor(duration_s))) if duration_s > 0 else 1
    win_seconds = min(desired_win, max_win_allowed)

    if win_seconds < desired_win:
        st.info(f"Ventana ajustada a {win_seconds}s porque la duración del registro es {duration_s:.2f}s.")

    # Selector de inicio de ventana
    max_start = max(0.0, duration_s - win_seconds)
    t0 = st.slider("Inicio de la ventana (s)", min_value=0.0, max_value=float(max(0.5, max_start)), value=0.0, step=0.5)


    # Señal a mostrar (ventana)
    y_full = signals[:, lead_idx].astype("float64")
    i0, i1, y = slice_window(y_full, fs, t0, win_seconds)
    t = np.arange(i0, i1) / fs
    # Limpieza y picos
    # (manejo NaNs por si acaso)
    if np.isnan(y).any():
        y = pd.Series(y).interpolate(limit_direction="both").to_numpy()

    y_clean = nk.ecg_clean(y, sampling_rate=fs, method="neurokit")
    peaks, info = nk.ecg_peaks(y_clean, sampling_rate=fs)
    r_idx_local = info.get("ECG_R_Peaks", np.array([], dtype=int))
    # Ajustar índices a tiempo absoluto
    r_idx_abs = r_idx_local + i0

    # RR y FC
    rr_s, hr, t_hr = compute_hr_from_peaks(r_idx_local, fs)

    # ----------------- Layout de visualización -----------------
    colA, colB = st.columns([3.1, 1.0], vertical_alignment="top")

    with colA:
        st.subheader(f"Señal ECG – {row['id']} | Derivación: {lead_name} | fs={fs:g} Hz")
        # t_segment = np.linspace(t0, t0 + win_seconds, len(y), endpoint=False)
        t_plot, y_plot = downsample_for_plot(t, y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_plot, y=y_plot, mode="lines", name=f"ECG {lead_name}", line=dict(width=1)))
        # Picos
        if len(r_idx_local) > 0:
            fig.add_trace(go.Scatter(
                x=t[r_idx_local], y=y[r_idx_local],
                mode="markers", name="Picos R", marker=dict(size=7, symbol="x", color="red")
            ))
        fig = apply_ecg_layout(fig, t_max=(t0 + win_seconds), y_data=y_plot, t_start=t0)
        st.plotly_chart(fig, use_container_width=True)

        if show_hr_curve and hr.size > 0:
            fig_hr = go.Figure()
            fig_hr.add_trace(go.Scatter(x=t_hr + t0, y=hr, mode="lines+markers", name="FC (lpm)"))
            fig_hr.update_layout(
                xaxis_title="Tiempo (s)",
                yaxis_title="lpm",
                height=260,
                margin=dict(l=40, r=20, t=30, b=40),
                showlegend=False
            )
            st.plotly_chart(fig_hr, use_container_width=True)

    with colB:
        st.subheader("Resumen")
        st.metric("Duración registro", fmt_seconds(duration_s))
        st.metric("Muestras", f"{signals.shape[0]:,}")
        st.metric("Canales", f"{signals.shape[1]}")

        if hr.size > 0:
            hr_mean = float(np.nanmean(hr))
            hr_min = float(np.nanmin(hr))
            hr_max = float(np.nanmax(hr))

            # Alerta de rango
            if (hr_mean < 60) or (hr_mean > 100):
                st.error(f"⚠️ FC media fuera de rango: {hr_mean:.1f} lpm (min={hr_min:.1f}, max={hr_max:.1f})")
            else:
                st.success(f"✅ FC media: {hr_mean:.1f} lpm (min={hr_min:.1f}, max={hr_max:.1f})")

            st.caption("Regla aplicada: FC = 60 / RR (seg), con RR entre picos R detectados.")

            # Tabla breve
            rr_df = pd.DataFrame({
                "t_medio_s": np.round(t_hr + t0, 3),
                "RR_s": np.round(rr_s, 4),
                "FC_lpm": np.round(hr, 1)
            })
            st.dataframe(rr_df.head(30), use_container_width=True, height=300)

            if download_csv:
                csv_bytes = rr_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Descargar RR/FC (CSV)",
                    data=csv_bytes,
                    file_name=f"{row['id']}_{lead_name}_RR_FC.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No se detectaron suficientes picos R en la ventana seleccionada. Prueba otra derivación o mueve la ventana.")

    # Notas técnicas
    with st.expander("📌 Detalles técnicos"):
        st.markdown(
            """
            - **Derivación**: selección automática prioriza II/MLII por mayor prominencia de ondas R; si no existe, intenta V2/V5/I/AVF/V1/III; finalmente el primer canal disponible.
            - **Limpieza**: `nk.ecg_clean(..., method='biosppy')` sobre la ventana de interés.
            - **Picos R**: `nk.ecg_peaks` → índices de picos; **RR** en segundos por diferencia de índices / fs; **FC (lpm)** = 60 / RR.
            - **Rango normal**: 60–100 lpm. Se muestra **alerta** si la **FC media** sale de ese rango.
            - **Rendimiento**: el gráfico se submuestrea a ~25k puntos máximos para evitar bloqueos en señales largas.
            """
        )
else:
    st.info("Selecciona una ruta válida y un registro para comenzar.")

