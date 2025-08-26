import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def plot_clusters(X_pca, labels, centers_pca=None, title="Clustering", path=None):
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    plt.figure(figsize=(10,8))
    unique_labels = np.unique(labels)
    for i, cluster in enumerate(unique_labels):
        idxs = np.where(labels == cluster)[0]
        plt.scatter(X_pca[idxs,0], X_pca[idxs,1], c=colors[i%len(colors)], label=f'Cluster {cluster}', s=80)
        if centers_pca is not None and cluster < centers_pca.shape[0]:
            plt.scatter(centers_pca[cluster,0], centers_pca[cluster,1], c='black', marker='*', s=300, label=f'Centro {cluster}')
            for j in idxs:
                plt.plot([X_pca[j,0], centers_pca[cluster,0]], [X_pca[j,1], centers_pca[cluster,1]], c=colors[i%len(colors)], alpha=0.3, linewidth=1)
        for j in idxs:
            plt.text(X_pca[j,0], X_pca[j,1], str(j), fontsize=9)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path)
        print(f"[DEBUG] {title} plot saved at: {path}")
    plt.show()
    plt.close()


log = logging.getLogger(__name__)


class TrackletTeamClustering(VideoLevelModule):
    """
    This module performs KMeans clustering on the embeddings of the tracklets to cluster the detections with role "player" into two teams.
    Teams are labeled as 0 and 1, and transformer into 'left' and 'right' in a separate module.
    """
    input_columns = ["track_id", "embeddings", "role"]
    output_columns = ["team_cluster"]
    
    def __init__(self, **kwargs):
        super().__init__()
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame, tracklet_images: dict = None):
        """
        Adiciona suporte para clustering por cor usando imagens dos tracklets.
        tracklet_images: dict[track_id] -> list de imagens (np.ndarray)
        """
        player_detections = detections[detections.role == "player"]
        embeddings_list = []
        color_features = []
        for track_id, group in player_detections.groupby("track_id"):
            if np.isnan(track_id):
                continue
            embeddings = np.mean(np.vstack(group.embeddings.values), axis=0)
            embeddings_list.append({'track_id': track_id, 'embeddings': embeddings})
            # Se imagens disponíveis, extrai cor média
            if tracklet_images and track_id in tracklet_images:
                imgs = tracklet_images[track_id]
                # Extrai cor média de todas as imagens do tracklet
                mean_colors = [np.mean(img.reshape(-1, 3), axis=0) for img in imgs if img is not None and img.size > 0]
                if mean_colors:
                    color_features.append({'track_id': track_id, 'color': np.mean(mean_colors, axis=0)})
        if not embeddings_list:
            detections['team_cluster'] = np.nan
            return detections
        embedding_tracklet = pd.DataFrame(embeddings_list)
        X = np.vstack(embedding_tracklet.embeddings.values)
        # --- PCA global ---
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        # --- KMeans ---
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        centers_pca = pca.transform(kmeans.cluster_centers_)
        print(f"[DEBUG][KMeans] labels: {kmeans.labels_}")
        print(f"[DEBUG][KMeans] cluster centers (PCA): {centers_pca}")
        plot_clusters(X_pca, kmeans.labels_, centers_pca, title="KMeans Clustering", path=os.path.join(os.getcwd(), 'debug_kmeans_clusters_pca.png'))
        embedding_tracklet['team_cluster'] = kmeans.labels_
        # --- Agglomerative ---
        agglo = AgglomerativeClustering(n_clusters=2).fit(X)
        print(f"[DEBUG][Agglomerative] labels: {agglo.labels_}")
        plot_clusters(X_pca, agglo.labels_, None, title="Agglomerative Clustering", path=os.path.join(os.getcwd(), 'debug_agglo_clusters_pca.png'))
        # --- Gaussian Mixture ---
        gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
        gmm_labels = gmm.predict(X)
        gmm_centers = gmm.means_
        gmm_centers_pca = pca.transform(gmm_centers)
        print(f"[DEBUG][GMM] labels: {gmm_labels}")
        print(f"[DEBUG][GMM] cluster centers (PCA): {gmm_centers_pca}")
        plot_clusters(X_pca, gmm_labels, gmm_centers_pca, title="Gaussian Mixture Clustering", path=os.path.join(os.getcwd(), 'debug_gmm_clusters_pca.png'))
        detections = detections.merge(embedding_tracklet[['track_id', 'team_cluster']], on='track_id', how='left', sort=False)
        return detections
