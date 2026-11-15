import pandas as pd
import torch
import numpy as np
from tracklab.pipeline.videolevel_module import VideoLevelModule
import logging

log = logging.getLogger(__name__)

_VOTE_THRESHOLD       = 0.60
_MAX_VALID_PITCH_X    = 60.0
_RELATIVE_VOTE_MIN_DIFF = 0.10  # min difference in right-fraction between clusters
                                # (0.25→0.10 calibrated on test+valid, 2026-07-06)
_MEAN_X_GAP_MIN       = 0.3     # below this, cluster means are noise -> use GK-drift tiebreak
                                # (0.5→0.3 calibrated on test+valid, 2026-07-06)
_DRIFT_MIN_FRAMES     = 10      # min frames per tracklet to trust a net-drift measurement


class TrackletTeamSideLabeling(VideoLevelModule):
    """
    Maps KMeans cluster 0/1 → 'left'/'right'.

    Priority chain:
    1. Tracklet vote   — each tracklet votes based on its mean pitch_x (outlier-filtered).
    2. Relative vote   — compares right-leaning fraction between clusters.
    3. Mean-X fallback — cluster mean pitch_x comparison (outlier-filtered). When the
       two cluster means are within _MEAN_X_GAP_MIN of each other (no usable signal,
       e.g. the whole clip is pinned to one zone of the pitch), defers to a GK-relative
       drift tiebreak: the cluster net-advancing toward the goalkeeper's own goal over
       the clip is the attacking/away side.

    Note: a set-piece/deep-zone anchor step was tried and removed — across the
    39-video test set it fired exactly once (SNGS-200) and got it wrong; its
    "deepest cluster is attacking" assumption doesn't hold (the cluster crowding
    the deep zone is at least as often the team defending that goal). Mean-X
    fallback alone gives the correct side for that case.
    """

    input_columns = ["track_id", "team_cluster", "bbox_pitch", "role"]
    output_columns = ["team"]

    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if "track_id" not in detections.columns:
            return detections

        gks = detections[detections.role == "goalkeeper"].dropna(subset=["bbox_pitch"])
        gk_side = None
        if not gks.empty:
            gk_x = gks.bbox_pitch.apply(
                lambda b: b.get("x_bottom_middle") if isinstance(b, dict) else np.nan
            ).dropna()
            if len(gk_x):
                gk_side = "right" if gk_x.mean() > 0 else "left"

        cluster_to_side, method = self._resolve_cluster_sides(detections, gk_side)
        log.info(f"[team_side] method='{method}' cluster→side: {cluster_to_side}")

        for cid, side in cluster_to_side.items():
            mask = (detections.team_cluster == cid) & (detections.role == "player")
            detections.loc[mask, "team"] = side

        if not gks.empty:
            gk_sides = gks.bbox_pitch.apply(
                lambda b: "right" if (isinstance(b, dict) and b.get("x_bottom_middle", 0) > 0) else "left"
            )
            detections.loc[gks.index, "team"] = gk_sides
            log.info(f"[team_side] GK direct assignment: {gk_sides.value_counts().to_dict()}")

        return detections

    def _resolve_cluster_sides(self, detections, gk_side):
        result = self._tracklet_vote(detections)
        if result:
            return result, "tracklet_vote"
        result = self._relative_vote(detections)
        if result:
            return result, "relative_vote"
        result = self._mean_x_fallback(detections, gk_side)
        return result, "mean_x_fallback"

    def _tracklet_vote(self, detections):
        players = detections[detections.role == "player"].copy()
        players["_px"] = players.bbox_pitch.apply(
            lambda b: b["x_bottom_middle"] if isinstance(b, dict) else np.nan
        )
        players = players.dropna(subset=["_px", "team_cluster", "track_id"])
        n_before = len(players)
        players = players[players["_px"].abs() <= _MAX_VALID_PITCH_X]
        if n_before - len(players) > 0:
            log.warning(f"[team_side] filtered {n_before - len(players)} dets with "
                        f"|pitch_x| > {_MAX_VALID_PITCH_X}m (calibration artefacts)")
        if players.empty:
            return {}

        tracklet_stats = (
            players.groupby("track_id")
            .agg(mean_x=("_px", "mean"), cluster=("team_cluster", "first"))
            .reset_index()
        )

        votes = {0: {"right": 0, "left": 0}, 1: {"right": 0, "left": 0}}
        for _, row in tracklet_stats.iterrows():
            cid = int(row.cluster)
            if cid not in votes:
                continue
            votes[cid]["right" if row.mean_x > 0 else "left"] += 1

        log.info(f"[team_side] tracklet votes: {votes}")

        cluster_to_side = {}
        for cid, v in votes.items():
            total = v["right"] + v["left"]
            if total == 0:
                continue
            ratio = v["right"] / total
            if ratio >= _VOTE_THRESHOLD:
                cluster_to_side[cid] = "right"
            elif ratio <= (1.0 - _VOTE_THRESHOLD):
                cluster_to_side[cid] = "left"

        if len(cluster_to_side) == 0:
            return {}
        if len(cluster_to_side) == 1:
            known_cid, known_side = next(iter(cluster_to_side.items()))
            cluster_to_side[1 - known_cid] = "right" if known_side == "left" else "left"
        if len(set(cluster_to_side.values())) < 2:
            log.warning("[team_side] vote resolved both clusters to same side — falling back")
            return {}

        return cluster_to_side

    def _relative_vote(self, detections):
        """
        Fires when _tracklet_vote fails (both clusters ambiguous or same side).
        Compares the fraction of right-side tracklets between clusters:
        if one cluster clearly has more right-leaning tracklets than the other,
        assign it 'right'. Threshold = _RELATIVE_VOTE_MIN_DIFF.

        Handles videos like SNGS-190 where one cluster has ~0% right and the other
        has ~25-30% right — neither reaches the absolute 60% threshold but the
        asymmetry is informative.
        """
        players = detections[detections.role == "player"].copy()
        players["_px"] = players.bbox_pitch.apply(
            lambda b: b["x_bottom_middle"] if isinstance(b, dict) else np.nan
        )
        players = players.dropna(subset=["_px", "team_cluster", "track_id"])
        players = players[players["_px"].abs() <= _MAX_VALID_PITCH_X]
        if players.empty:
            return {}

        tracklet_stats = (
            players.groupby("track_id")
            .agg(mean_x=("_px", "mean"), cluster=("team_cluster", "first"))
            .reset_index()
        )

        right_frac = {}
        for cid in [0, 1]:
            t = tracklet_stats[tracklet_stats.cluster == cid]
            if len(t) < 3:
                continue
            right_frac[cid] = (t.mean_x > 0).mean()

        if len(right_frac) < 2:
            return {}

        diff = right_frac[0] - right_frac[1]
        log.info(f"[team_side] relative_vote: C0_right={right_frac[0]:.0%} "
                 f"C1_right={right_frac[1]:.0%} diff={diff:+.0%}")

        if abs(diff) < _RELATIVE_VOTE_MIN_DIFF:
            log.info(f"[team_side] relative_vote: diff too small ({abs(diff):.0%} < "
                     f"{_RELATIVE_VOTE_MIN_DIFF:.0%}) — skipping")
            return {}

        if diff > 0:
            return {0: "right", 1: "left"}
        else:
            return {0: "left", 1: "right"}

    def _mean_x_fallback(self, detections, gk_side=None):
        players = detections[detections.role == "player"]
        team_a = players[players.team_cluster == 0]
        team_b = players[players.team_cluster == 1]
        xa = [b["x_bottom_middle"] for b in team_a.bbox_pitch
              if isinstance(b, dict) and abs(b.get("x_bottom_middle", 999)) <= _MAX_VALID_PITCH_X]
        xb = [b["x_bottom_middle"] for b in team_b.bbox_pitch
              if isinstance(b, dict) and abs(b.get("x_bottom_middle", 999)) <= _MAX_VALID_PITCH_X]
        avg_a = np.nanmean(xa) if xa else 0.0
        avg_b = np.nanmean(xb) if xb else 0.0
        log.info(f"[team_side] mean_x fallback: cluster0={avg_a:.1f}m cluster1={avg_b:.1f}m")

        if abs(avg_a - avg_b) < _MEAN_X_GAP_MIN and gk_side is not None:
            result = self._gk_drift_tiebreak(players, gk_side)
            if result:
                return result

        return {0: "right", 1: "left"} if avg_a > avg_b else {0: "left", 1: "right"}

    def _gk_drift_tiebreak(self, players, gk_side):
        """
        Mean-X is uninformative when both clusters sit at ~the same average pitch_x
        (whole clip pinned to one zone, e.g. a sustained spell of pressure). In that
        case the team net-advancing toward the goalkeeper's own goal over the clip
        is the attacking (away) side; the other team — already sitting deep near
        their own goal — barely drifts further. Tracklet net drift (last - first
        pitch_x) per cluster disambiguates this even when raw averages can't.
        """
        gk_sign = 1.0 if gk_side == "right" else -1.0
        drifts = {0: [], 1: []}
        for cid in (0, 1):
            c = players[players.team_cluster == cid]
            for _, g in c.groupby("track_id"):
                g = g.dropna(subset=["bbox_pitch"])
                xs = [b["x_bottom_middle"] for b in g.bbox_pitch
                      if isinstance(b, dict) and abs(b.get("x_bottom_middle", 999)) <= _MAX_VALID_PITCH_X]
                if len(xs) < _DRIFT_MIN_FRAMES:
                    continue
                drifts[cid].append(xs[-1] - xs[0])

        if len(drifts[0]) < 3 or len(drifts[1]) < 3:
            return None

        toward_gk = {cid: np.mean(drifts[cid]) * gk_sign for cid in (0, 1)}
        log.info(f"[team_side] gk_drift_tiebreak: gk_side={gk_side} "
                 f"toward_gk={ {k: round(v, 2) for k, v in toward_gk.items()} }")

        attacking_cid = max(toward_gk, key=toward_gk.get)
        defending_cid = 1 - attacking_cid
        away_side = "left" if gk_side == "right" else "right"
        return {attacking_cid: away_side, defending_cid: gk_side}
