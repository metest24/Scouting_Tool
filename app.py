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
DEFAULT_RECEIVERS  = ["C","W1","S1","S2","QB"]
DEFAULT_CAT = ["C","W1","S1","S2"]
DEFAULT_DOWN = ["1st", "2nd", "3rd", "4th", "1xp", "2xp"]

LAYOUT_FIELD = {
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
            ]
        },
        {
            "title": "Passing Zone Own",
            "grid": [
                ["Own-7","Own-8","Own-9"],
                ["Own-4","Own-5","Own-6"],
                ["Own-1","Own-2","Own-3"],
            ]
        }
    ]
}

LAYOUT_BFA = {
    "title": "Back Field Action",
    "grid":[
            ["BFA-1","BFA-2","BFA-3"],
    ]
}

# -----------------------
# Session State
# -----------------------
if "records" not in st.session_state:
    st.session_state.records = []  # list of dicts

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

if "downs" not in st.session_state:
    st.session_state.downs = DEFAULT_DOWN.copy()
if "selected_down" not in st.session_state:
    st.session_state.selected_down = None

if "layout" not in st.session_state:
    st.session_state.layout = LAYOUT_FIELD
if "backfield" not in st.session_state:
    st.session_state.backfield = LAYOUT_BFA
if "cat" not in st.session_state:
    st.session_state.cat = DEFAULT_CAT.copy()

if "bfa_enabled" not in st.session_state:
    st.session_state.bfa_enabled = False
if "selected_cat" not in st.session_state:
    st.session_state.selected_cat = None
if "selected_bfa" not in st.session_state:
    st.session_state.selected_bfa = None

if "_excel_bytes" not in st.session_state:
    st.session_state._excel_bytes = None
if "_pdf_bytes" not in st.session_state:
    st.session_state._pdf_bytes = None

# Team-Namen für PDF
if "team1_name" not in st.session_state:
    st.session_state.team1_name = ""
if "team2_name" not in st.session_state:
    st.session_state.team2_name = ""

# ---- Defaults sauber setzen
def _ensure_defaults():
    if st.session_state.selected_down is None and st.session_state.downs:
        st.session_state.selected_down = st.session_state.downs[0]
    if st.session_state.selected_formation is None and st.session_state.formations:
        st.session_state.selected_formation = st.session_state.formations[0]
    if st.session_state.selected_receiver is None and st.session_state.receivers:
        st.session_state.selected_receiver = st.session_state.receivers[0]
    if st.session_state.bfa_enabled:
        if st.session_state.selected_cat is None and st.session_state.cat:
            st.session_state.selected_cat = st.session_state.cat[0]
        bfa_flat = [it for row in st.session_state.backfield.get("grid", []) for it in row]
        if st.session_state.selected_bfa is None and bfa_flat:
            st.session_state.selected_bfa = bfa_flat[0]

_ensure_defaults()

# -----------------------
# Helpers & Caching
# -----------------------
def add_record(field_id, formation, receiver, backfield=None, cat=None, down=None):
    st.session_state.records.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "field_id": field_id,
        "formation": formation,
        "receiver": receiver,
        "backfield": backfield,
        "cat": cat,
        "down": down
    })
    # Exporte invalidieren
    st.session_state._excel_bytes = None
    st.session_state._pdf_bytes = None

@st.cache_data(show_spinner=False)
def compute_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Formations-Tabelle: Gesamt + davon mit BFA (wirkt nach vorgelagertem Filter)."""
    if df.empty:
        return pd.DataFrame(columns=["Formation", "Gesamt Anzahl", "Davon mit BFA"])
    total_formation = df.groupby("formation").size()
    counts_bfa = df[df["backfield"].notna() & (df["backfield"] != "")].groupby("formation").size()
    result = pd.DataFrame({
        "Gesamt Anzahl": total_formation,
        "Davon mit BFA": counts_bfa
    }).fillna(0).astype(int).reset_index().rename(columns={"formation": "Formation"})
    return result

@st.cache_data(show_spinner=False)
def counts_by_field(df_filtered: pd.DataFrame) -> dict:
    """Schnelles Zählen pro field_id (nach Filtern)."""
    if df_filtered.empty:
        return {}
    vc = df_filtered["field_id"].value_counts()
    return {k: int(v) for k, v in vc.items()}

# Heatmap zeichnen
def draw_heatmap(counts_matrix: np.ndarray, grid, title: str):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(counts_matrix, aspect="equal")
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xticklabels([str(c+1) for c in range(cols)])
    ax.set_yticklabels([str(r+1) for r in range(rows)])
    ax.set_xlabel("Spalte"); ax.set_ylabel("Zeile")
    ax.set_title(title)
    for r in range(rows):
        for c in range(cols):
            fid = grid[r][c]
            if fid in (None, "", "-"):
                continue
            ax.text(c, r, f"{fid}\n{int(counts_matrix[r,c])}", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    return fig, im

# ===== PDF HELPERS =====
def _filter_df_for_pdf(df, rx_sel, fx_sel, dx_sel):
    """Filter nach Receivern/Formationen/Downs; None/leer => kein Filter."""
    if df.empty:
        return df
    if rx_sel:
        df = df[df["receiver"].isin(rx_sel)]
    if fx_sel:
        df = df[df["formation"].isin(fx_sel)]
    if dx_sel:
        df = df[df["down"].isin(dx_sel)]
    return df

def _render_pivot_table(ax, df_pivot):
    ax.axis("off")
    if df_pivot.empty:
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
        return
    table = ax.table(cellText=df_pivot.values, colLabels=df_pivot.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

def _render_receiver_bar(ax, df_filtered):
    if df_filtered.empty:
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
        return
    totals = df_filtered.groupby("receiver").size().reset_index(name="count")
    if "receivers" in st.session_state:
        totals = totals.set_index("receiver").reindex(st.session_state.receivers, fill_value=0).reset_index()
    ax.bar(totals["receiver"], totals["count"])
    ax.set_title("Catches pro Receiver (gefiltert)")
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Anzahl")

def _build_heatmaps_figure(df_filtered, section_list, selected_titles, bfa_row):
    import matplotlib.pyplot as plt
    # Zählen pro Feld
    def _counts_by_field_loc(df_):
        if df_.empty: return {}
        vc = df_["field_id"].value_counts()
        return {k: int(v) for k, v in vc.items()}
    fc = _counts_by_field_loc(df_filtered)

    title_to_section = {s.get("title","Sektion"): s for s in section_list}
    chosen = [title_to_section[t] for t in selected_titles if t in title_to_section]
    n = min(3, len(chosen)) or 0
    if n == 0:
        fig = plt.figure(figsize=(11.69, 4.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "Keine Sektion gewählt", ha="center", va="center")
        return fig

    prepared = []
    vmax_global = 0
    for section in chosen[:3]:
        grid = section.get("grid", [])
        rows = len(grid); cols = len(grid[0]) if rows else 0
        counts = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                fid = grid[r][c]
                counts[r, c] = 0 if fid in (None,"","-") else fc.get(fid, 0)

        if bfa_row:
            section_fids = [cell for row in grid for cell in row if cell not in (None,"","-")]
            df_sec = df_filtered[df_filtered["field_id"].isin(section_fids)]
            bfa_labels = ["BFA-1","BFA-2","BFA-3"]
            bfa_counts = [
                int((df_sec["backfield"] == lab).sum()) if not df_sec.empty else 0
                for lab in bfa_labels
            ]
            counts_ext = np.vstack([counts, np.array(bfa_counts).reshape(1, 3)])
            grid_ext = grid + [bfa_labels]
            vmax_global = max(vmax_global, int(counts_ext.max()))
            prepared.append((section.get("title",""), grid_ext, counts_ext))
        else:
            vmax_global = max(vmax_global, int(counts.max()))
            prepared.append((section.get("title",""), grid, counts))

    fig, axes = plt.subplots(1, n, figsize=(11.69, 4.5))
    if n == 1:
        axes = [axes]
    ims = []
    for ax, (title, grid, counts) in zip(axes, prepared[:3]):
        im = ax.imshow(counts, aspect="equal", vmin=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12)
        rows = len(grid); cols = len(grid[0]) if rows else 0
        for r in range(rows):
            for c in range(cols):
                fid = grid[r][c]
                if fid in (None,"","-"):
                    continue
                ax.text(c, r, f"{fid}\n{int(counts[r,c])}", ha="center", va="center", fontsize=7)
        ims.append(im)
    for im in ims:
        im.set_clim(0, max(1, vmax_global))
    if ims:
        cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.03, pad=0.02)
        cbar.ax.set_ylabel("Anzahl", rotation=90)
    fig.tight_layout()
    return fig

def make_pdf_bytes(df_all_raw, *, title, include_pivot, include_bar, include_hm,
                   rx_sel, fx_sel, dx_sel, hm_sections, hm_bfa_row, section_list):
    import matplotlib.backends.backend_pdf as pdf_backend
    df_filtered = _filter_df_for_pdf(df_all_raw.copy(), rx_sel, fx_sel, dx_sel)

    pdf_buf = io.BytesIO()
    pdf = pdf_backend.PdfPages(pdf_buf)

    # Seite 1
    if include_pivot or include_bar:
        fig1 = plt.figure(figsize=(11.69, 8.27))  # A4 quer
        fig1.suptitle(title or "Passing Matrix – Report", fontsize=16, y=0.98)

        if include_pivot and include_bar:
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(2, 2, height_ratios=[2, 3], width_ratios=[3, 2], hspace=0.25, wspace=0.25)
            ax_table = fig1.add_subplot(gs[0, :])
            ax_bar   = fig1.add_subplot(gs[1, 1])
            ax_info  = fig1.add_subplot(gs[1, 0])
            ax_info.axis("off")
            piv = compute_pivot(df_filtered)
            _render_pivot_table(ax_table, piv)
            _render_receiver_bar(ax_bar, df_filtered)
            lines = [
                f"Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Einträge (gefiltert): {len(df_filtered)}",
            ]
            ax_info.text(0, 1, "\n".join(lines), va="top", fontsize=10)
        else:
            ax = fig1.add_subplot(111)
            if include_pivot:
                piv = compute_pivot(df_filtered)
                _render_pivot_table(ax, piv)
            else:
                _render_receiver_bar(ax, df_filtered)

        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

    # Seite 2
    if include_hm:
        fig2 = _build_heatmaps_figure(
            df_filtered,
            section_list=section_list,
            selected_titles=hm_sections,
            bfa_row=hm_bfa_row
        )
        pdf.savefig(fig2)
        plt.close(fig2)

    pdf.close()
    return pdf_buf.getvalue()
# ===== END PDF HELPERS =====


# -----------------------
# Sidebar (ganz Links)
# -----------------------
st.sidebar.header("⚙️ Aktionen")
if st.sidebar.button("🗑️ Alle Einträge löschen", type="secondary", key="btn_clear_all", help="Alle gespeicherten Einträge verwerfen."):
    st.session_state.records = []
    st.session_state.last_clicked_field = None
    st.session_state.selected_formation = None
    st.session_state.selected_receiver = None
    st.session_state.selected_down = None
    st.session_state._excel_bytes = None
    st.session_state._pdf_bytes = None
    st.sidebar.success("Zurückgesetzt.")

if st.sidebar.button("↩️ Letzten Eintrag löschen", type="secondary", key="btn_clear_last", help="Nur den letzten Eintrag entfernen."):
    if st.session_state.records:
        removed = st.session_state.records.pop(-1)
        st.sidebar.success("Letzter Eintrag gelöscht!")
        st.session_state._excel_bytes = None
        st.session_state._pdf_bytes = None
    else:
        st.sidebar.warning("Keine Einträge zum Löschen vorhanden.")

st.sidebar.markdown("---")
st.sidebar.subheader("📤 Export")

# PDF: Dialog (Popup)
@st.dialog("PDF-Export konfigurieren")
def pdf_export_dialog():
    with st.form("pdf_dialog_form", clear_on_submit=False):
        st.markdown("**Teams**")
        team1 = st.text_input("Team 1:", value=st.session_state.team1_name, key="pdf_team1_input")
        team2 = st.text_input("Team 2:", value=st.session_state.team2_name, key="pdf_team2_input")

        auto_title = (team1.strip() != "" and team2.strip() != "")
        suggested_title = f'{team1.strip()} gg. {team2.strip()} Livescouting' if auto_title else "Passing Matrix – Report"

        st.markdown("**Titel**")
        report_title = st.text_input(
            "PDF-Titel:",
            value=suggested_title,
            key="pdf_title_input",
            help="Wenn beide Teamnamen gesetzt sind, wird automatisch 'Team1 gg. Team2 Livescouting' vorgeschlagen."
        )

        st.markdown("**Inhalte auswählen**")
        include_pivot = st.checkbox("Tabelle: Formationen & BFA", value=True)
        include_bar   = st.checkbox("Balkendiagramm: Catches pro Receiver", value=True)
        include_hm    = st.checkbox("Heatmaps (3 Sektionen)", value=True)

        st.markdown("**Filter**")
        rx_sel = st.multiselect("Receiver filtern", options=st.session_state.receivers, default=st.session_state.receivers)
        fx_sel = st.multiselect("Formationen filtern", options=st.session_state.formations, default=st.session_state.formations)
        dx_sel = st.multiselect("Downs filtern", options=st.session_state.downs, default=st.session_state.downs)

        st.markdown("**Heatmap-Optionen**")
        section_list = st.session_state.layout.get("sections", [])
        section_titles = [s.get("title","Sektion") for s in section_list]
        default_sel = section_titles[:3] if len(section_titles) >= 3 else section_titles
        hm_sections = st.multiselect("Sektionen (max. 3)", options=section_titles, default=default_sel)
        hm_bfa_row  = st.checkbox("BFA-Zeile in Heatmaps anzeigen (BFA-1/2/3)", value=False)

        create = st.form_submit_button("PDF erstellen", type="primary", use_container_width=False)  # Streamlit <=1.39
        # Wenn du >=1.40 bist: replace use_container_width=False mit width="content"

    if create:
        # Teams persistieren
        st.session_state.team1_name = team1
        st.session_state.team2_name = team2

        df_all_raw = pd.DataFrame(st.session_state.records) if st.session_state.records else \
            pd.DataFrame(columns=["timestamp","field_id","formation","receiver","backfield","cat","down"])

        bytes_pdf = make_pdf_bytes(
            df_all_raw,
            title=report_title,
            include_pivot=include_pivot,
            include_bar=include_bar,
            include_hm=include_hm,
            rx_sel=rx_sel,
            fx_sel=fx_sel,
            dx_sel=dx_sel,
            hm_sections=hm_sections,
            hm_bfa_row=hm_bfa_row,
            section_list=section_list
        )
        st.session_state._pdf_bytes = bytes_pdf
        st.success("PDF erstellt. Du kannst es jetzt herunterladen.")
        st.download_button(
            "⬇️ PDF jetzt herunterladen",
            data=bytes_pdf,
            file_name="passing_matrix_report.pdf",
            mime="application/pdf",
            width="stretch"
        )
        # Sidebar-Download sichtbar machen:
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

# Button, um den Dialog zu öffnen
if st.sidebar.button("🧾 PDF exportieren…", type="primary"):
    pdf_export_dialog()

# Download PDF (falls vorhanden)
if st.session_state._pdf_bytes:
    st.sidebar.download_button(
        "⬇️ PDF herunterladen",
        data=st.session_state._pdf_bytes,
        file_name="passing_matrix_report.pdf",
        mime="application/pdf",
        width="stretch"
    )

# --- Excel-Konfiguration & Export ---
with st.sidebar.expander("Excel-Export", expanded=False):
    with st.form("xlsx_export_form", clear_on_submit=False):
        st.markdown("**Welche Blätter sollen rein?**")
        inc_entries = st.checkbox("Rohdaten (Eintraege)", value=True)
        make_xlsx = st.form_submit_button("Export starten", type="primary")

    if make_xlsx:
        df_all_raw = pd.DataFrame(st.session_state.records) if st.session_state.records else \
            pd.DataFrame(columns=["timestamp","field_id","formation","receiver","backfield","cat","down"])
        df_out = df_all_raw.copy()

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            if inc_entries:
                df_out.to_excel(writer, index=False, sheet_name="Eintraege")
        st.session_state._excel_bytes = excel_buffer.getvalue()
        st.sidebar.success("Excel erstellt.")

# Download Excel (falls vorhanden)
if st.session_state._excel_bytes:
    st.sidebar.download_button(
        "⬇️ Excel herunterladen",
        data=st.session_state._excel_bytes,
        file_name="passing_matrix_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch"
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
    st.caption("Klicke zuerst ein Feld. Danach wähle Down, Formation & Receiver und speichere.")

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
                    width="stretch",
                    on_click=_select_field,
                    args=(fid,)
                )

# ---- Mid: Down, Formation, Empfänger, Speichern
with mid:
    st.subheader("2) Down, Formation & Receiver")
    st.caption("Wähle Down, Formation & Receiver und speichere den Eintrag. Falls es eine Back-Field Action gibt, aktiviere BFA.")
    st.checkbox("BFA aktivieren", key="bfa_enabled")
    _ensure_defaults()

    with st.form("entry_form", clear_on_submit=False):
        downs      = st.session_state.downs
        formations = st.session_state.formations
        receivers  = st.session_state.receivers

        # Down
        st.radio(
            "Down",
            downs,
            index=downs.index(st.session_state.selected_down) if st.session_state.selected_down in downs else 0,
            key="down_radio",
            horizontal=True
        )

        # Formation
        st.radio(
            "Formation",
            formations,
            index=formations.index(st.session_state.selected_formation) if st.session_state.selected_formation in formations else 0,
            key="formation_radio",
            horizontal=True
        )

        # Receiver
        st.radio(
            "Receiver",
            receivers,
            index=receivers.index(st.session_state.selected_receiver) if st.session_state.selected_receiver in receivers else 0,
            key="receiver_radio",
            horizontal=True
        )

        st.markdown("---")

        if st.session_state.bfa_enabled:
            st.markdown("**BFA Details**")
            cats = st.session_state.cat
            st.radio(
                "Cat",
                cats,
                index=cats.index(st.session_state.selected_cat) if st.session_state.selected_cat in cats else 0,
                key="cat_radio",
                horizontal=True
            )
            bfa_grid = [item for row in st.session_state.backfield.get("grid", []) for item in row]
            st.radio(
                "Back Field Action",
                bfa_grid,
                index=bfa_grid.index(st.session_state.selected_bfa) if st.session_state.selected_bfa in bfa_grid else 0,
                key="bfa_radio",
                horizontal=True
            )

        submitted = st.form_submit_button(
            "✅ Eintrag speichern",
            type="primary",
            disabled=(st.session_state.last_clicked_field is None)
        )

    if submitted:
        st.session_state.selected_down      = st.session_state.get("down_radio")
        st.session_state.selected_formation = st.session_state.get("formation_radio")
        st.session_state.selected_receiver  = st.session_state.get("receiver_radio")

        if st.session_state.bfa_enabled:
            st.session_state.selected_cat = st.session_state.get("cat_radio")
            st.session_state.selected_bfa = st.session_state.get("bfa_radio")
            add_record(
                st.session_state.last_clicked_field,
                st.session_state.selected_formation,
                st.session_state.selected_receiver,
                backfield=st.session_state.selected_bfa,
                cat=st.session_state.selected_cat,
                down=st.session_state.selected_down
            )
        else:
            add_record(
                st.session_state.last_clicked_field,
                st.session_state.selected_formation,
                st.session_state.selected_receiver,
                backfield=None,
                cat=None,
                down=st.session_state.selected_down
            )

        st.success(
            f"Gespeichert: {st.session_state.selected_receiver} in {st.session_state.last_clicked_field} "
            f"({st.session_state.selected_formation}, {st.session_state.selected_down})"
            + (f" | {st.session_state.selected_bfa} mit {st.session_state.selected_cat} als Cat"
               if st.session_state.bfa_enabled else "")
        )

    if st.session_state.records:
        df_all_raw = pd.DataFrame(st.session_state.records)
        st.markdown("---")
        st.markdown("**Letzten 3 Einträge**")
        st.dataframe(df_all_raw.tail(3), use_container_width=True, hide_index=True)
    else:
        st.caption("Noch keine Einträge.")

# ---- Right: Übersicht (live)
with right:
    st.subheader("Übersicht (live)")
    if st.session_state.records:
        df_all_raw = pd.DataFrame(st.session_state.records)
        pivot_all = compute_pivot(df_all_raw)
        if not pivot_all.empty:
            st.markdown("**Anzahl der Formationen (davon BFA)**")
            st.dataframe(pivot_all, use_container_width=True, hide_index=True)
        else:
            st.caption("Noch keine Daten für Pivot.")
        
        st.markdown("---")

        # Balken: Catches pro Receiver (WR)
        wr_order = st.session_state.receivers if "receivers" in st.session_state else None
        wr_counts = df_all_raw.groupby("receiver").size()
        if wr_order:
            wr_counts = wr_counts.reindex(wr_order, fill_value=0)
        wr_counts = wr_counts.reset_index()
        wr_counts.columns = ["receiver", "count"]

        st.markdown("**Catches pro Receiver**")
        if len(wr_counts) > 0:
            fig_wr, ax_wr = plt.subplots()
            ax_wr.bar(wr_counts["receiver"], wr_counts["count"])
            ax_wr.set_xlabel("Receiver")
            ax_wr.set_ylabel("Anzahl Catches")
            ax_wr.set_title("Catches pro Receiver (gesamt)")
            st.pyplot(fig_wr, use_container_width=True)
            plt.close(fig_wr)
        else:
            st.caption("Noch keine Catches erfasst.")
    else:
        st.caption("Die Übersicht erscheint, sobald Einträge vorhanden sind.")

# -----------------------
# Heatmaps (nebeneinander, mit optionalen Filtern)
# -----------------------
st.markdown("---")
st.subheader("Heatmaps")

receiver_filter = None
formation_filter = None
down_filter = None
bfa_row_enabled = False

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

        selected_downs = st.multiselect(
            "Nach Downs filtern",
            options=st.session_state.downs,
            default=st.session_state.downs
        )
        down_filter = (
            selected_downs
            if selected_downs and len(selected_downs) < len(st.session_state.downs)
            else None
        )

        bfa_row_enabled = st.checkbox("BFA-Zeile anzeigen (BFA-1/2/3)", value=False, key="hm_bfa_enabled")

# gefiltertes DF für Heatmaps
if st.session_state.records:
    df_all = pd.DataFrame(st.session_state.records)
    if receiver_filter:
        df_all = df_all[df_all["receiver"].isin(receiver_filter)]
    if formation_filter:
        df_all = df_all[df_all["formation"].isin(formation_filter)]
    if down_filter:
        df_all = df_all[df_all["down"].isin(down_filter)]
else:
    df_all = pd.DataFrame(columns=["timestamp","field_id","receiver","formation","backfield","cat","down"])

# schnelle Zählung pro Feld (nach Filtern)
field_counts = counts_by_field(df_all)

# drei Heatmaps nebeneinander – vorbereiten, globales vmax bestimmen
sections = st.session_state.layout.get("sections", [])
prepared = []
vmax_global = 0

for section in sections[:3]:
    title = section.get("title", "Sektion")
    grid  = section.get("grid", [])
    if not grid:
        prepared.append({"title": title, "grid": [], "counts": None})
        continue

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    counts = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            fid = grid[r][c]
            if fid in (None, "", "-"):
                counts[r, c] = 0
            else:
                counts[r, c] = field_counts.get(fid, 0)

    if bfa_row_enabled:
        section_fids = [cell for row in grid for cell in row if cell not in (None, "", "-")]
        df_sec = df_all[df_all["field_id"].isin(section_fids)]
        bfa_labels = ["BFA-1", "BFA-2", "BFA-3"]
        bfa_counts = [
            int((df_sec["backfield"] == lab).sum()) if not df_sec.empty else 0
            for lab in bfa_labels
        ]
        counts_ext = np.vstack([counts, np.array(bfa_counts).reshape(1, 3)])
        grid_ext = grid + [bfa_labels]
        vmax_global = max(vmax_global, int(counts_ext.max()))
        prepared.append({"title": title, "grid": grid_ext, "counts": counts_ext})
    else:
        vmax_global = max(vmax_global, int(counts.max()))
        prepared.append({"title": title, "grid": grid, "counts": counts})

# Rendern mit globalem vmax
cols3 = st.columns(3)
for i, sec in enumerate(prepared[:3]):
    with cols3[i]:
        title = sec["title"]
        grid  = sec["grid"]
        counts = sec["counts"]

        if counts is None or not grid:
            st.caption("Kein Grid definiert.")
            continue

        fig_hm, im_hm = draw_heatmap(counts, grid, title)
        im_hm.set_clim(0, max(1, vmax_global))
        st.pyplot(fig_hm, use_container_width=True)
        plt.close(fig_hm)

# Footer
st.markdown("---")
if st.session_state.team1_name and st.session_state.team2_name:
    st.caption(f"Passing Zone Tool – {st.session_state.team1_name} gg. {st.session_state.team2_name} – Ver. 0.3")
else:
    st.caption("Passing Zone Tool – Ver. 0.3")
