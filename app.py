import streamlit as st
import pandas as pd
import io
import os
import signal
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Passing Zone Tool", layout="wide")

# -----------------------
# Config & Initialization
# -----------------------
DEFAULT_FORMATIONS = ["Spread", "Bunch", "Trips", "Pre-BFA"]
DEFAULT_RECEIVERS  = ["C","W1","S1","S2"]

DEFAULT_LAYOUT = {
    "sections": [
        {
            "title": "Redzone",
            "grid": [
                ["E7","E8","E9"],
                ["E4","E5","E6"],
                ["E1","E2","E3"]
            ]
        },
        {
            "title": "Open field Opp",
            "grid": [
                ["Opp-7","Opp-8","Opp-9"],
                ["Opp-4","Opp-5","Opp-6"],
                ["Opp-1","Opp-2","Opp-3"],
                ["BFA-Opp-1","BFA-Opp-2","BFA-Opp-3"]
            ]
        },
        {
            "title": "Passing Zone Own",
            "grid": [
                ["Own-7","Own-8","Own-9"],
                ["Own-4","Own-5","Own-6"],
                ["Own-1","Own-2","Own-3"],
                ["BFA-Own-1","BFA-Own-2","BFA-Own-3"]
            ]
        }
    ]
}

# Session State
if "records" not in st.session_state:
    st.session_state.records = []  # list of dicts: {timestamp, field_id, formation, receiver}
if "last_clicked_field" not in st.session_state:
    st.session_state.last_clicked_field = None
if "formations" not in st.session_state:
    st.session_state.formations = DEFAULT_FORMATIONS.copy()
if "selected_formation" not in st.session_state:
    st.session_state.selected_formation = None
if "receivers" not in st.session_state:
    st.session_state.receivers = DEFAULT_RECEIVERS.copy()
if "selected_receiver" not in st.session_state:
    st.session_state.selected_receiver = None
if "layout" not in st.session_state:
    st.session_state.layout = DEFAULT_LAYOUT
if "_excel_bytes" not in st.session_state:
    st.session_state._excel_bytes = None
if "_pdf_bytes" not in st.session_state:
    st.session_state._pdf_bytes = None

# -----------------------
# Helpers & Caching
# -----------------------
def add_record(field_id, formation, receiver):
    st.session_state.records.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "field_id": field_id,
        "formation": formation,
        "receiver": receiver
    })
    # Exporte invalidieren, damit bewusst neu gebaut werden
    st.session_state._excel_bytes = None
    st.session_state._pdf_bytes = None

@st.cache_data(show_spinner=False)
def compute_pivot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    piv = pd.pivot_table(
        df, index="receiver", columns="field_id",
        values="timestamp", aggfunc="count", fill_value=0
    )
    return piv.astype(int)

@st.cache_data(show_spinner=False)
def counts_by_field(df_filtered: pd.DataFrame) -> dict:
    """Schnelles Zählen pro field_id (nach Filtern)."""
    if df_filtered.empty:
        return {}
    vc = df_filtered["field_id"].value_counts()
    return {k: int(v) for k, v in vc.items()}

# Erstellt eine Heatmap-Figur mit Beschriftungen
def draw_heatmap(counts_matrix: np.ndarray, grid, title: str):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    fig, ax = plt.subplots(figsize=(4.5, 4.5))  # quadratisch, gut für 3 nebeneinander
    im = ax.imshow(counts_matrix, aspect="equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([str(c+1) for c in range(cols)])
    ax.set_yticklabels([str(r+1) for r in range(rows)])
    ax.set_xlabel("Spalte")
    ax.set_ylabel("Zeile")
    ax.set_title(title)
    # Zellen beschriften
    for r in range(rows):
        for c in range(cols):
            fid = grid[r][c]
            if fid in (None, "", "-"):
                continue
            ax.text(c, r, f"{fid}\n{int(counts_matrix[r,c])}", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    return fig, im

# -----------------------
# Sidebar (ganz Links)
# -----------------------
st.sidebar.header("⚙️ Aktionen")
if st.sidebar.button("🗑️ Alle Einträge löschen"):
    st.session_state.records = []
    st.session_state.last_clicked_field = None
    st.session_state.selected_formation = None
    st.session_state.selected_receiver = None
    st.session_state._excel_bytes = None
    st.session_state._pdf_bytes = None
    st.sidebar.success("Zurückgesetzt.")

if st.sidebar.button("🛑 Server beenden"):
    os.kill(os.getpid(), signal.SIGTERM) 


st.sidebar.markdown("---")
st.sidebar.subheader("📤 Export")

# State für Download-Bytes
if "_excel_bytes" not in st.session_state:
    st.session_state._excel_bytes = None
if "_pdf_bytes" not in st.session_state:
    st.session_state._pdf_bytes = None

# Buttons zum Öffnen der Konfiguration
col_pdf, col_xlsx = st.sidebar.columns(2)

# --- PDF-Konfiguration & Export ---
with st.sidebar.expander("PDF-Export", expanded=False):
    with st.form("pdf_export_form", clear_on_submit=False):
        st.markdown("**Was soll in die PDF?**")
        include_pivot = st.checkbox("Statistik / Pivot", value=True)
        include_bar   = st.checkbox("Balkendiagramm gesamt", value=True)
        include_hm    = st.checkbox("Heatmaps (3 Sektionen)", value=True)
        use_filters_hm = st.checkbox("Für Heatmaps: aktuelle Filter anwenden", value=True,
                                        help="Verwendet die oben gesetzten Empfänger-/Formationsfilter.")
        # Sektionen auswählen
        section_list = st.session_state.layout.get("sections", [])
        section_titles = [s.get("title","Sektion") for s in section_list]
        default_sel = section_titles[:3] if len(section_titles) >= 3 else section_titles
        selected_titles = st.multiselect("Sektionen auswählen", options=section_titles, default=default_sel)

        make_pdf = st.form_submit_button("Export starten", use_container_width=True)

    if make_pdf:
        # Basistabellen
        df_all_raw = pd.DataFrame(st.session_state.records) if st.session_state.records else \
                        pd.DataFrame(columns=["timestamp","field_id","formation","receiver"])
        piv_all = compute_pivot(df_all_raw)

        # PDF bauen
        import matplotlib.backends.backend_pdf as pdf_backend
        pdf_buf = io.BytesIO()
        pdf = pdf_backend.PdfPages(pdf_buf)

        # --- Seite 1: Statistik (optional)
        if include_pivot or include_bar:
            # Flexible Seitenaufteilung
            fig1 = plt.figure(figsize=(11.69, 8.27))  # A4 quer
            fig1.suptitle("Passing Matrix – Übersicht", fontsize=16, y=0.98)

            # Gridspec dynamisch, je nachdem was gewählt ist
            rows = 2 if (include_pivot and include_bar) else 1
            cols = 1 if (include_pivot and not include_bar) or (include_bar and not include_pivot) else 2
            if include_pivot and include_bar:
                import matplotlib.gridspec as gridspec
                gs = gridspec.GridSpec(2, 2, height_ratios=[2, 3], width_ratios=[3, 2], hspace=0.25, wspace=0.25)
                # Pivot oben über beide Spalten
                ax_table = fig1.add_subplot(gs[0, :])
                ax_bar   = fig1.add_subplot(gs[1, 1])
                ax_info  = fig1.add_subplot(gs[1, 0])
                ax_info.axis('off')
            else:
                ax = fig1.add_subplot(111)

            if include_pivot:
                if include_bar:
                    ax_t = ax_table
                    ax_t.axis('off')
                else:
                    ax.axis('off')
                    ax_t = ax
                if not piv_all.empty:
                    piv_disp = piv_all.copy()
                    piv_disp["Summe"] = piv_disp.sum(axis=1)
                    header = ["Empfänger"] + list(piv_all.columns) + ["Summe"]
                    body = [[idx] + [int(v) for v in row] + [int(sum(row))] for idx, row in piv_all.iterrows()]
                    table = ax_t.table(cellText=[header] + body, loc="center")
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 1.2)
                else:
                    ax_t.text(0.5, 0.5, "Keine Daten", ha="center", va="center")

            if include_bar:
                if include_pivot:
                    ax_b = ax_bar
                else:
                    ax_b = ax
                if not df_all_raw.empty:
                    totals = df_all_raw.groupby("receiver").size().reset_index(name="count")
                    ax_b.bar(totals["receiver"], totals["count"])
                    ax_b.set_title("Catches pro Empfänger (gesamt)")
                    ax_b.set_xlabel("Empfänger")
                    ax_b.set_ylabel("Anzahl")
                else:
                    ax_b.text(0.5, 0.5, "Keine Daten", ha="center", va="center")

            # Infofeld nur, wenn beide aktiv sind
            if include_pivot and include_bar:
                info_lines = [
                    f"Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Anzahl Einträge: {len(df_all_raw)}",
                ]

            fig1.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

        # --- Seite 2: Heatmaps (optional)
        if include_hm and selected_titles:
            # Datenbasis für Heatmaps (gefiltert oder komplett)
            df_hm = df_all_raw.copy()
            if use_filters_hm:
                if 'receiver_filter' in locals() and receiver_filter:
                    df_hm = df_hm[df_hm["receiver"].isin(receiver_filter)]
                if 'formation_filter' in locals() and formation_filter:
                    df_hm = df_hm[df_hm["formation"].isin(formation_filter)]
            fc = counts_by_field(df_hm)

            # Hole die ausgewählten Sektionen in Reihenfolge
            title_to_section = {s.get("title","Sektion"): s for s in section_list}
            chosen_sections = [title_to_section[t] for t in selected_titles if t in title_to_section]

            n = min(3, len(chosen_sections))
            fig2, axes = plt.subplots(1, n, figsize=(11.69, 4.5))  # A4 quer
            if n == 1:
                axes = [axes]
            vmax = 0
            ims = []

            for ax, section in zip(axes, chosen_sections[:3]):
                title = section.get("title","")
                grid  = section.get("grid", [])
                rows = len(grid)
                cols = len(grid[0]) if rows else 0
                counts = np.zeros((rows, cols), dtype=int)
                for r in range(rows):
                    for c in range(cols):
                        fid = grid[r][c]
                        if fid not in (None,"","-"):
                            counts[r, c] = fc.get(fid, 0)
                vmax = max(vmax, int(counts.max()))
                im = ax.imshow(counts, aspect="equal", vmin=0)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(title, fontsize=12)
                for r in range(rows):
                    for c in range(cols):
                        fid = grid[r][c]
                        if fid in (None,"","-"): 
                            continue
                        ax.text(c, r, f"{fid}\n{int(counts[r,c])}", ha="center", va="center", fontsize=7)
                ims.append(im)

            for im in ims:
                im.set_clim(0, max(1, vmax))
            if ims:
                cbar = fig2.colorbar(ims[-1], ax=axes, fraction=0.03, pad=0.02)
                cbar.ax.set_ylabel("Anzahl", rotation=90)

            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

        pdf.close()
        st.session_state._pdf_bytes = pdf_buf.getvalue()
        st.sidebar.success("PDF erstellt. Jetzt herunterladen 👇")

# Download PDF (falls vorhanden)
if st.session_state._pdf_bytes:
    st.sidebar.download_button(
        "⬇️ PDF herunterladen",
        data=st.session_state._pdf_bytes,
        file_name="passing_matrix_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# --- Excel-Konfiguration & Export ---

with st.sidebar.expander("Excel-Export"):
    with st.form("xlsx_export_form", clear_on_submit=False):
        st.markdown("**Welche Blätter sollen rein?**")
        inc_entries = st.checkbox("Rohdaten (Eintraege)", value=True)
        inc_pivot   = st.checkbox("Übersicht (Pivot)", value=True)

        make_xlsx = st.form_submit_button("Export starten", use_container_width=True)

    if make_xlsx:
        df_all_raw = pd.DataFrame(st.session_state.records) if st.session_state.records else \
                        pd.DataFrame(columns=["timestamp","field_id","formation","receiver"])
        df_out = df_all_raw.copy()
        piv_out = compute_pivot(df_out) if inc_pivot else None

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            if inc_entries:
                df_out.to_excel(writer, index=False, sheet_name="Eintraege")
            if inc_pivot and piv_out is not None and not piv_out.empty:
                piv_out.to_excel(writer, sheet_name="Uebersicht")
        st.session_state._excel_bytes = excel_buffer.getvalue()
        st.sidebar.success("Excel erstellt.")

# Download Excel (falls vorhanden)
if st.session_state._excel_bytes:
    st.sidebar.download_button(
        "⬇️ Excel herunterladen",
        data=st.session_state._excel_bytes,
        file_name="passing_matrix_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )


# -----------------------
# Hauptlayout: 3 Spalten
# -----------------------
left, mid, right = st.columns([3,2,3])

# ---- Left: Feld wählen (aus Layout)
def _select_field(fid: str):
    st.session_state.last_clicked_field = fid

with left:
    st.subheader("1) Feld wählen")
    st.caption("Klicke zuerst ein Feld. Danach wähle Empfänger & Formation und speichere.")

    for section in st.session_state.layout.get("sections", []):
        st.markdown(f"**{section.get('title','')}**")
        for row in section.get("grid", []):
            cols = st.columns(len(row))
            for c_idx, fid in enumerate(row):
                if fid in (None, "", "-"):
                    cols[c_idx].markdown("&nbsp;")
                    continue
                is_selected = (st.session_state.last_clicked_field == fid)
                btn_type = "primary" if is_selected else "secondary"
                cols[c_idx].button(
                    fid,
                    key=f"btn_{fid}",
                    type=btn_type,
                    use_container_width=True,
                    on_click=_select_field,
                    args=(fid,)
                )


# ---- Mid: Formation, Empfänger, Speichern
with mid:
    st.subheader("2) Formation & Receiver wählen")
    st.caption("Wähle Formation & Receiver und speichere den Eintrag.")
    with st.form("entry_form", clear_on_submit=False):
        # Formation
        formations = st.session_state.formations
        cur_form = st.session_state.selected_formation
        idx_form = formations.index(cur_form) if (cur_form in formations) else 0
        selected_form = st.radio("Formation", formations, index=idx_form, horizontal=True)

        # Empfänger
        receivers = st.session_state.receivers
        cur_rcv = st.session_state.selected_receiver
        idx_rcv = receivers.index(cur_rcv) if (cur_rcv in receivers) else 0
        selected_rcv = st.radio("Receiver", receivers, index=idx_rcv, horizontal=True)

        submitted = st.form_submit_button(
            "✅ Eintrag speichern",
            use_container_width=True,
            disabled=(st.session_state.last_clicked_field is None)
        )

    if submitted:
        st.session_state.selected_formation = selected_form
        st.session_state.selected_receiver  = selected_rcv
        add_record(st.session_state.last_clicked_field, selected_form, selected_rcv)
        st.success(f"Gespeichert: {selected_rcv} in {st.session_state.last_clicked_field}, ({selected_form})")

    st.markdown("---")
    if st.session_state.records:
        df_all_raw = pd.DataFrame(st.session_state.records)
        st.markdown("**Letzten 3 Einträge**")
        st.dataframe(df_all_raw.tail(3), use_container_width=True, hide_index=True)
    else:
        st.caption("Noch keine Einträge.")

# ---- Right: Übersicht (live) – Statistik über ALLE Einträge (ungefiltert)
with right:
    st.subheader("Übersicht (live)")
    if st.session_state.records:
        df_all_raw = pd.DataFrame(st.session_state.records)
        pivot_all = compute_pivot(df_all_raw)
        totals = df_all_raw.groupby("receiver").size().reset_index(name="count")

        # Pivot + Summenspalte
        if not pivot_all.empty:
            totals_receiver = pivot_all.sum(axis=1).rename("Summe")
            pivot_with_total = pivot_all.copy()
            pivot_with_total["Summe"] = totals_receiver
            st.dataframe(pivot_with_total.astype(int), use_container_width=True)
        else:
            st.caption("Noch keine Daten für Pivot.")

        # Balken gesamt pro Empfänger
        st.markdown("**Gesamt nach Empfänger**")
        fig, ax = plt.subplots()
        ax.bar(totals["receiver"], totals["count"])
        ax.set_xlabel("Empfänger")
        ax.set_ylabel("Anzahl Catches")
        ax.set_title("Catches pro Empfänger (gesamt)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.caption("Die Übersicht erscheint, sobald Einträge vorhanden sind.")

# -----------------------
# Heatmaps (nebeneinander, mit optionalen Filtern)
# -----------------------
st.markdown("---")
st.subheader("Heatmaps")

receiver_filter = None
formation_filter = None

if st.session_state.records:
    with st.expander("Filter (optional)", expanded=False):
        selected_receivers = st.multiselect(
            "Nach Empfängern filtern",
            options=st.session_state.receivers,
            default=st.session_state.receivers
        )
        receiver_filter = (
            selected_receivers
            if selected_receivers and len(selected_receivers) < len(st.session_state.receivers)
            else None
        )

        selected_formations = st.multiselect(
            "Nach Formationen filtern",
            options=st.session_state.formations,
            default=st.session_state.formations
        )
        formation_filter = (
            selected_formations
            if selected_formations and len(selected_formations) < len(st.session_state.formations)
            else None
        )

# gefiltertes DF für Heatmaps
if st.session_state.records:
    df_all = pd.DataFrame(st.session_state.records)
    if receiver_filter:
        df_all = df_all[df_all["receiver"].isin(receiver_filter)]
    if formation_filter:
        df_all = df_all[df_all["formation"].isin(formation_filter)]
else:
    df_all = pd.DataFrame(columns=["timestamp","field_id","receiver","formation"])

# schnelle Zählung pro Feld (nach Filtern)
field_counts = counts_by_field(df_all)

# drei Heatmaps nebeneinander
sections = st.session_state.layout.get("sections", [])
cols3 = st.columns(3)
for i, section in enumerate(sections[:3]):  # wir gehen von 3 Sektionen aus
    with cols3[i]:
        title = section.get("title","Sektion")
        grid  = section.get("grid", [])
        if not grid:
            st.caption("Kein Grid definiert.")
            continue
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        counts = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                fid = grid[r][c]
                if fid in (None,"","-"):
                    counts[r, c] = 0
                else:
                    counts[r, c] = field_counts.get(fid, 0)

        fig_hm, im_hm = draw_heatmap(counts, grid, title)
        st.pyplot(fig_hm, use_container_width=True)
        plt.close(fig_hm)

# Footer
st.markdown("---")
st.caption("Passing Zone Tool \n Ver. 0.1 ")
