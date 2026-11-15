import os
import warnings
import logging

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from tracklab.pipeline.videolevel_module import VideoLevelModule

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def _save_cluster_plot(X_pca, labels, centers_pca, title, path):
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cluster in enumerate(np.unique(labels)):
        idxs = np.where(labels == cluster)[0]
        ax.scatter(X_pca[idxs, 0], X_pca[idxs, 1], c=colors[i % len(colors)],
                   label=f"Cluster {cluster}", s=80)
        if centers_pca is not None and cluster < centers_pca.shape[0]:
            ax.scatter(centers_pca[cluster, 0], centers_pca[cluster, 1],
                       c="black", marker="*", s=300)
        for j in idxs:
            ax.text(X_pca[j, 0], X_pca[j, 1], str(j), fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    log.info(f"[team_cluster] debug plot saved: {path}")


_PITCH_X_MAX         = 52.5   # pitch half-length — filter calibration artefacts
_ONE_SIDED_THRESH    = 0.85   # ≥85% of tracklets on same half → one-sided (legacy rule)
_REID_WEAK_L2        = 2.04   # inter-centroid L2 below this → weak ReID (legacy rule)
_REID_SIL_WEAK       = 0.28   # silhouette of raw-ReID clusters below this → ReID unreliable
_COLOR_MIN_CLUSTER_FRAC = 0.20  # color clustering must not be degenerate (min cluster ≥20%)
_COLOR_TAIL_DIM      = 27     # trailing color-histogram dims of embeddings_with_colors

# --- auditoria de agrupamento degenerado (2026-09-22) ---------------------------------------
# Um agrupamento que deixe quase todas as deteções de jogadores num único grupo está a afirmar
# que todos os jogadores em campo pertencem à mesma equipa, o que não pode acontecer num jogo.
# Quando isso sucede, o resultado é descartado e os tracklets são reagrupados pela cor do
# equipamento, medida diretamente nos frames originais (histograma HSV da região do tronco, com
# os píxeis de relva excluídos). O resgate só é aceite se produzir dois grupos equilibrados.
_DEGENERATE_DET_FRAC   = 0.08   # grupo menor com < 8% das deteções de jogadores → degenerado
_RESCUE_MIN_FRAC       = 0.20   # o agrupamento de resgate tem de ter ≥20% dos tracklets E das deteções
_MAX_REFEREE_RATIO     = 0.25   # árbitros/jogadores acima disto → falta uma equipa, não reagrupar
_RESCUE_SAMPLES        = 24     # recortes por tracklet usados no histograma de cor
_RESCUE_MIN_TRACKLETS  = 6      # abaixo disto não há amostra para reagrupar
_TORSO_BOX             = (0.20, 0.12, 0.60, 0.40)  # x, y, largura, altura, em fração da caixa
_GRASS_HSV             = ((30, 90), 50, 40)        # tom, saturação mínima, valor mínimo da relva


def _torso_color_hist(crop_bgr):
    """Histograma HSV (8 bins por canal, 24 dims) da região do tronco, sem píxeis de relva."""
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
        return np.zeros(24, np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    (h_lo, h_hi), s_min, v_min = _GRASS_HSV
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (h >= h_lo) & (h <= h_hi) & (s >= s_min) & (v >= v_min)
    mask = (~grass).astype(np.uint8)
    if mask.sum() < 8:
        mask = np.ones(hsv.shape[:2], np.uint8)
    parts = []
    for ch, hi in ((0, 180), (1, 256), (2, 256)):
        hist = cv2.calcHist([hsv], [ch], mask, [8], [0, hi]).flatten().astype(np.float32)
        total = hist.sum()
        parts.append(hist / total if total > 0 else hist)
    return np.concatenate(parts)


def _tracklet_color_features(player_det, metadatas, track_ids):
    """Média, por tracklet, do histograma de cor do tronco, sobre _RESCUE_SAMPLES recortes
    distribuídos uniformemente no tempo. Devolve (N_tracklets, 24) na ordem de track_ids."""
    paths = dict(zip(metadatas.index.astype(str), metadatas["file_path"])) if "file_path" in metadatas else {}
    if "id" in metadatas.columns:
        paths.update(dict(zip(metadatas["id"].astype(str), metadatas["file_path"])))
    wanted = []
    for tid in track_ids:
        g = player_det[player_det.track_id == tid].sort_values("image_id")
        if len(g) == 0:
            continue
        idx = np.unique(np.linspace(0, len(g) - 1, min(_RESCUE_SAMPLES, len(g))).round().astype(int))
        wanted.append(g.iloc[idx])
    if not wanted:
        return None
    samp = pd.concat(wanted)
    hists, owner = [], []
    for image_id, group in samp.groupby(samp["image_id"].astype(str), sort=True):
        path = paths.get(image_id)
        image = cv2.imread(path) if path else None
        if image is None:
            continue
        H, W = image.shape[:2]
        fx, fy, fw, fh = _TORSO_BOX
        for row in group.itertuples():
            bbox = row.bbox_ltwh
            if bbox is None or len(bbox) < 4:
                continue
            x, y, w, h = (float(t) for t in bbox)
            x0 = max(0, int(x + fx * w)); y0 = max(0, int(y + fy * h))
            x1 = min(W, x0 + max(2, int(fw * w))); y1 = min(H, y0 + max(2, int(fh * h)))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            hists.append(_torso_color_hist(image[y0:y1, x0:x1])); owner.append(row.track_id)
    if not hists:
        return None
    hists = np.vstack(hists); owner = np.array(owner)
    out = []
    for tid in track_ids:
        sel = owner == tid
        out.append(hists[sel].mean(0) if sel.any() else np.zeros(24, np.float32))
    return np.vstack(out)


class TrackletTeamClustering(VideoLevelModule):
    """
    KMeans clustering on per-tracklet mean ReID embeddings → team_cluster 0/1.

    When the raw-ReID clustering is weak (silhouette < _REID_SIL_WEAK — the
    two clusters are not well separated in ReID space), falls back to
    clustering on the jersey-color tail of embeddings_with_colors alone —
    mixing it with the full 256-dim ReID block drowns out the color signal
    under StandardScaler+KMeans. The color clustering is only trusted when it
    is not degenerate (its smaller cluster holds ≥ _COLOR_MIN_CLUSTER_FRAC of
    tracklets).

    This rule was calibrated jointly on test(39)+valid(54) with a hard
    zero-regression constraint on test (2026-07-06). It replaces the earlier
    one_sided+L2<2.04 heuristic, which never fired on balanced valid videos
    where color was the better signal.
    """

    input_columns = ["track_id", "embeddings", "role"]
    output_columns = ["team_cluster"]

    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame,
                tracklet_images: dict = None):
        player_det = detections[detections.role == "player"]

        has_colors = (
            "embeddings_with_colors" in player_det.columns
            and player_det["embeddings_with_colors"].notna().any()
            and not np.allclose(
                np.vstack(player_det["embeddings_with_colors"].dropna().values)[:, -27:], 0
            )
        )

        tracklet_rows = []
        for track_id, group in player_det.groupby("track_id"):
            if np.isnan(track_id):
                continue
            # Collect both embedding types and pitch position
            emb_reid   = np.mean(np.vstack(group["embeddings"].values), axis=0)
            emb_colors = (np.mean(np.vstack(group["embeddings_with_colors"].values), axis=0)
                          if has_colors else emb_reid)
            px_vals = group["bbox_pitch"].apply(
                lambda b: b["x_bottom_middle"]
                if isinstance(b, dict) and abs(b.get("x_bottom_middle", 999)) <= _PITCH_X_MAX
                else np.nan
            ).dropna()
            mean_px = float(px_vals.mean()) if len(px_vals) >= 3 else np.nan
            tracklet_rows.append({
                "track_id":    track_id,
                "emb_reid":    emb_reid,
                "emb_colors":  emb_colors,
                "mean_px":     mean_px,
            })

        if not tracklet_rows:
            detections["team_cluster"] = np.nan
            return detections

        embedding_tracklet = pd.DataFrame(tracklet_rows)
        px_raw = embedding_tracklet["mean_px"].values.astype(float)

        # Detect if video is one-sided (≥85% of tracklets on same half of pitch)
        px_valid = px_raw[~np.isnan(px_raw)]
        one_sided = (
            len(px_valid) >= 6 and (
                (px_valid < 0).mean() >= _ONE_SIDED_THRESH or
                (px_valid > 0).mean() >= _ONE_SIDED_THRESH
            )
        )

        # Always compute raw ReID clusters first (appearance-first principle)
        X_reid = np.vstack(embedding_tracklet["emb_reid"].values)
        km_reid = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_reid)
        c0_reid = X_reid[km_reid.labels_ == 0].mean(0)
        c1_reid = X_reid[km_reid.labels_ == 1].mean(0)
        reid_l2 = float(np.linalg.norm(c0_reid - c1_reid))
        sil_reid = (silhouette_score(X_reid, km_reid.labels_)
                    if len(set(km_reid.labels_)) == 2 else -1.0)

        # Two triggers for the color branch (union — either suffices):
        #   legacy: one-sided clip + centroids close in ReID space
        #   new:    ReID clusters poorly separated (silhouette), color not degenerate
        legacy_trigger = one_sided and reid_l2 < _REID_WEAK_L2
        sil_trigger = sil_reid < _REID_SIL_WEAK

        use_color, km_color, X_col_scaled = False, None, None
        if (legacy_trigger or sil_trigger) and has_colors:
            X = np.vstack(embedding_tracklet["emb_colors"].values)[:, -_COLOR_TAIL_DIM:]
            X_col_scaled = StandardScaler().fit_transform(X)
            km_color = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_col_scaled)
            counts = np.bincount(km_color.labels_, minlength=2)
            min_frac = counts.min() / counts.sum()
            # degeneracy guard only gates the new (silhouette) trigger; the legacy
            # trigger keeps its original behavior (validated at 97.9% on test)
            if legacy_trigger or min_frac >= _COLOR_MIN_CLUSTER_FRAC:
                use_color = True
            else:
                log.info(f"[team_cluster] color clustering degenerate "
                         f"(min_frac={min_frac:.2f} < {_COLOR_MIN_CLUSTER_FRAC}) — keeping ReID")

        if use_color:
            X_scaled = X_col_scaled
            kmeans = km_color
            log.info(f"[team_cluster] color_tail({_COLOR_TAIL_DIM}d)+scaled "
                     f"(legacy={legacy_trigger} sil={sil_reid:.3f} L2={reid_l2:.3f})")
        else:
            X_scaled = X_reid
            kmeans = km_reid
            log.info(f"[team_cluster] raw ReID (sil={sil_reid:.3f} L2={reid_l2:.3f})")


        labels = np.asarray(kmeans.labels_)

        # --- auditoria: o agrupamento diz que estão todos na mesma equipa? -------------------
        det_cluster = player_det["track_id"].map(
            dict(zip(embedding_tracklet.track_id, labels))
        )
        det_counts = np.array([(det_cluster == c).sum() for c in (0, 1)], dtype=float)
        det_frac = det_counts.min() / det_counts.sum() if det_counts.sum() else 0.0
        n_ref = (detections.role == "referee").groupby(detections.track_id).any().sum() \
            if "track_id" in detections.columns else 0
        n_pla = len(embedding_tracklet)
        ref_ratio = n_ref / n_pla if n_pla else 0.0
        if det_frac < _DEGENERATE_DET_FRAC and ref_ratio > _MAX_REFEREE_RATIO:
            log.warning(f"[team_cluster] agrupamento degenerado, mas {n_ref} tracklets dados como "
                        f"arbitragem para {n_pla} de jogador ({ref_ratio:.0%} > "
                        f"{_MAX_REFEREE_RATIO:.0%}) — falta uma equipa ao agrupamento, "
                        f"reagrupar o resto não faria sentido; mantém-se a partição")
        elif det_frac < _DEGENERATE_DET_FRAC and len(embedding_tracklet) >= _RESCUE_MIN_TRACKLETS:
            log.warning(f"[team_cluster] agrupamento degenerado: grupo menor com "
                        f"{det_frac:.1%} das deteções (<{_DEGENERATE_DET_FRAC:.0%}) — "
                        f"a reagrupar pela cor do equipamento")
            X_rescue = _tracklet_color_features(player_det, metadatas,
                                                list(embedding_tracklet.track_id))
            if X_rescue is None:
                log.warning("[team_cluster] resgate por cor impossível (sem frames) — mantém")
            else:
                km_rescue = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_rescue)
                cnt = np.bincount(km_rescue.labels_, minlength=2)
                frac = cnt.min() / cnt.sum()
                # o resgate tem de passar o mesmo teste que reprovou a partição anterior,
                # medido nas deteções e não apenas nos tracklets
                nd = np.array([(player_det.track_id == t).sum()
                               for t in embedding_tracklet.track_id], dtype=float)
                d_cnt = np.array([nd[km_rescue.labels_ == c].sum() for c in (0, 1)])
                d_frac = d_cnt.min() / d_cnt.sum() if d_cnt.sum() else 0.0
                if frac < _RESCUE_MIN_FRAC or d_frac < _RESCUE_MIN_FRAC:
                    log.warning(f"[team_cluster] resgate por cor também degenerado "
                                f"(tracklets {frac:.2f}, deteções {d_frac:.2f} < "
                                f"{_RESCUE_MIN_FRAC}) — mantém o agrupamento anterior")
                else:
                    labels = km_rescue.labels_
                    X_scaled = X_rescue
                    log.info(f"[team_cluster] resgate por cor aceite: {cnt[0]}/{cnt[1]} tracklets")

        embedding_tracklet["team_cluster"] = labels

        counts = np.bincount(labels)
        log.info(f"[team_cluster] cluster sizes: {counts[0]}/{counts[1]}")
        log.info(f"[team_cluster] KMeans labels: {dict(zip(embedding_tracklet.track_id, labels))}")

        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            centers_pca = pca.transform(
                np.array([X_scaled[labels == c].mean(0) for c in [0, 1]])
            )
            plot_path = os.path.join(os.getcwd(), "debug_kmeans_clusters_pca.png")
            _save_cluster_plot(X_pca, labels, centers_pca, "KMeans Clustering", plot_path)
        except Exception as e:
            log.warning(f"[team_cluster] debug plot failed: {e}")

        detections = detections.merge(
            embedding_tracklet[["track_id", "team_cluster"]],
            on="track_id", how="left", sort=False,
        )
        return detections
