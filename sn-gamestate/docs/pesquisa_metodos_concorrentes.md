# Pesquisa profunda — métodos dos concorrentes do SoccerNet

Extraído dos PDF oficiais em 2026-09-08. Fontes primárias:
- arXiv:2309.06006  SoccerNet 2023 Challenges Results (tem desafio DEDICADO a JNR)
- arXiv:2409.10587  SoccerNet 2024 Challenges Results (1a edicao GSR)
- arXiv:2508.19182  SoccerNet 2025 Challenges Results (2a edicao GSR)
- arXiv:2504.06357  Golovkin et al., artigo completo da equipa vencedora de 2024
- arXiv:2607.07320  SoccerNet 2026 Challenges Results (NAO tem tarefa GSR)

## 1. Desafio dedicado de JNR, 2023 — a peca que faltava
```
8 Jersey Number Recognition                                   Table 7: Jersey Number Recognition leaderboard.
                                                              The main metric for the leaderboard and best
8.1 Task description                                          performance are in bold. Team names with a
The task consists in identifying the jersey num-              superscript have provided a summary that may
ber of a player from a short video tracklet showing           be found in Appendix A, or in Section 8.4 for the
the soccer players. This jersey number recognition            winning team.
task is challenging because of the low quality of the          Participants            acc     Participants         acc
thumbnails (i.e. low resolution and high motion                ZZPMJ1                  92.85   Kalisteo             58.35
blur) and because the jersey numbers might be vis-             UniBw Munich - VISJ2    90.95   FindNum              54.91
                                                               zzzzzJ3                 88.08   jn                   47.55
ible on a minimal subset of the whole tracklet. The            Mike Azatov             82.05   Surya                28.40
SoccerNet Jersey Number dataset comprises 2,853                MT-IOTJ5                81.70   tony506672558        20.06
                                                               justplayJ6              77.77   lfriend               5.68
tracklets of players extracted from the SoccerNet              AIBrain Global TeamJ7   75.18   zhq                   4.07
tracking videos. The challenge set comprises 1,211             SARG UWaterloo          73.77   Baseline (Random)*    3.93
separate players’ tracklets with segregated annota-
tions. The target classes are therefore all the jersey
                                                              motion blur and low quality. Our solution was to
numbers from 1 to 99, and one extra ”−1” class
                                                              first perform text detection on the training set, fil-
when no jersey number is visible in the tracklet.
                                                              tering out a portion of the data that clearly did
                                                              not have numbers, using a modified pre-trained
8.2 Metrics                                                   DBNet++ model for the training set, and remov-
Since the jersey number recognition challenge was             ing images from the training set that did not
formulated as a classification task, with one class           contain a detection box with a confidence level of
for each possible jersey number in the [0, 99]                90 or higher. Data augmentation was then per-
range, we use the overall classification accuracy             formed on the dataset, including image rotation
as a target metric to rank participants’ solutions.           and flipping, image scaling and cropping, color
Therefore, we compute the jersey number predic-               scrambling, noise addition, and multi-frame image
tion accuracy as the number of correctly predicted            overlay methods. We found that the data augmen-
jersey numbers (including −1 for non-visible num-             tation method of multi-frame fusion can provide
bers) over the total number of tracklets in the               more information and increase the robustness to
challenge set.                                                features such as number shape, color and texture,
                                                              thus improving the accuracy of recognition. The
8.3 Leaderboard                                               enhanced data were then trained for text recogni-
                                                              tion using multiple models, including SVTR-tiny,
This year, 15 teams participated in the jer-                  SVTR-small, SATRN, NRTR, and ASTER. The
sey number recognition challenge, for a total of              final result consists of a fusion of votes from
157 submissions, with the best method reaching                multiple models.
92.85% accuracy. The leaderboard may be found
in Table 7.                                                   8.5 Results
8.4 Winner                                                    Most teams adopted a standard three-stage
                                                              approach. A first text detection step is performed
The winners for this task are Rui Peng et al.. A              with various state-of-the-art methods (DBNet++,
summary of their method is given hereafter.                   MMOCR, Deepsolo, YOLO) to predict the jer-
J1 - ZZPM                                                     sey number location in the image. A second text
Rui Peng, Kexin Zhang, Junpei Zhang, Yanbiao                  recognition step is performed to recognize the
Ma, and Licheng Jiao                                          corresponding number using fine-tuned state-of-
{22171214876,22171214672,22171214671,                         the-art OCR methods (PP-OCRv3, PaddleOCR).
ybmamail}@stu.xidian.edu.cn,                                  Finally, majority voting is employed to aggregate
lchjiao@mail.xidian.edu.cn                                    image-level results within a tracklet and output
    The task requires the identification of play-             the final video-level prediction.
ers’ numbers, which become difficult due to high


                                                         10
9 Conclusion                                                [2] Deliège, A., Cioppa, A., Giancola, S., Seika-
                                                                vandi, M.J., Dueholm, J.V., Nasrollahi, K.,
This paper summarizes the outcome of the Soc-                   Ghanem, B., Moeslund, T.B., Van Droogen-
cerNet 2023 challenges. In total, we present the                broeck, M.: SoccerNet-v2: A dataset and
results on seven tasks: action spotting, ball action            benchmarks for holistic understanding of
spotting, dense video captioning, camera cali-                  broadcast soccer videos. In: IEEE Int.
bration, player re-identification, player tracking,             Conf. Comput. Vis. Pattern Recognit. Work.
and jersey number recognition. These challenges                 (CVPRW), CVsports, Nashville, TN, USA,
```

## 2. Golovkin — trabalho relacionado (taxonomia de JNR e de equipa)
```
Since game state reconstruction was first introduced in           clustering. In the feature extraction stage, player appear-
2024, few papers explicitly address the task. However, it         ance information is collected, often in the form of color his-
can be decomposed into four critical subtasks: player de-         tograms or Re-ID embeddings trained via contrastive learn-
tection and tracking , pitch localization , team recogni-         ing [33]. Extracted features are then clustered using unsu-
tion , and jersey number detection.                               pervised methods such as DBSCAN [11] or k-means [31],
    Players detection and tracking in sports analyt-              grouping players into up to 2–5 clusters corresponding to
ics is dominated by the tracking-by-detection paradigm,           referees, field players, and goalkeepers [46]. An alterna-
which typically combines one-stage object detectors like          tive approach, Associative Embedding (AE), eliminates
YOLOv8 [19] or other [3–5, 44, 61][45, 50, 55, 56] with           the need for clustering by training a CNN to generate team-
DeepSORT-like algorithms [52, 59], augmented by pre-              discriminative embeddings directly [20]. Unlike clustering-
trained re-identification (ReID) feature extractors. While        based pipelines, AE does not require explicit position-based
these methods enable real-time performance due to their           assignment, making it more adaptable to different sports
computational efficiency, they struggle with tracklet frag-       and camera views.
mentation in dynamic scenes and identity swaps in crowded
scenarios. These issues stem from Kalman filter limita-               Jersey Number Detection methods are broadly cate-
tions in predicting non-linear motion [22] and the chal-          gorized into two approaches: (1) adaptations of general-
lenges of ReID in football, where uniform similarity, occlu-      purpose OCR techniques [26, 27, 46] and (2) custom solu-
sions, and drastic pose variations degrade feature matching       tions tailored for sports scenarios [2, 14, 23, 25, 29]. The
[8, 9, 17, 37, 40, 49, 51]. An alternative approach, tracking-    first approach typically follows a two-step pipeline: text
by-attention [57, 59], integrates detection and tracking into     detection on localized player regions using methods like
a unified transformer-based framework, achieving state-of-        Differentiable Binarization [27], followed by recognition
the-art results on benchmarks like DanceTrack [47] and            via models such as Show, Attend and Read [26]. While
MOTChallenge [10]. However, these methods prioritize              these methods offer development simplicity, their accu-
spatial accuracy over temporal consistency, often neglect-        racy suffers in scenarios with distant players, occlusions, or
ing smooth tracklet association, which is critical for sports     sideways orientations. Custom solutions address domain-
analytics. Additionally, the computational complexity of          specific challenges by integrating sports-aware designs,
transformer-based tracking makes it impractical for real-         such as one-stage number classification [14, 25], pose-
time applications.                                                guided detectors with post-processing [29], or keyframe
    Pitch localization is crucial for game state reconstruc-      identification modules leveraging spatiotemporal networks
tion, enabling the conversion of player positions from image      to extract critical frames with visible numbers [2]. These
space to real-world coordinates by estimating homography          methods boost accuracy using contextual features but need
parameters. The most common method relies on detecting            dataset-specific training, adding implementation complex-
correspondences between predicted keypoints [7, 21, 34]           ity.
or extracting field lines and circles using semantic seg-
mentation [18, 58], aligning them with a predefined field
model using RANSAC [12]. While keypoint-based meth-               3. Methodology
ods are computationally efficient, they suffer from instabil-
ity in broadcast footage, where only a small subset of field      Our game state reconstruction framework is a robust and
markings is visible at any moment.                                modular pipeline designed to track players and reconstruct
    Alternative approaches estimate homography directly           their positions on a 2D top-view (minimap) of the football
via CNN-based regression [48] or use search-based opti-           pitch using single-camera video footage (Figure 1). The
mization, matching image features with a database of pre-         tracking pipeline consists of three key stages: raw track-
computed homographies [6, 41]. While these methods offer          ing, team detection, and post-processing.
greater stability than keypoint-based approaches, they re-           In the raw tracking stage, athletes are detected, pitch
quire large-scale databases for robust feature matching and       localization is performed to determine their coordinates on
still tend to be less precise and often require post-prediction   the field, initial tracking is established, and feature vectors
```

## 3. Golovkin — modelo de numero de camisola (2 cabecas de classificacao)
```
                                                                       ±2 meters for positions. To enable its use, predictions are
                                                                       delayed by 15 frames (0.5 sec), allowing the filter to access
   L1parameters represents the difference between predicted            both past and future values.
and real camera parameters (camera position, pitch, yaw,                   The Jersey Number Recognition model is based on a
roll, field of view):                                                  modified ResNet architecture designed for 32×32 input im-
                                X                                      ages. Instead of detecting individual digits using bounding
              L1parameters =          ∥(ypred − ygt )∥1          (4)   boxes, the model employs two classification heads: the first
                                  y                                    determines whether a jersey number contains a leading digit
                                                                       (1–9 or none), while the second predicts the second digit
where y is a specific camera parameter. Term of the loss
                                                                       (0–9). This structure eliminates the need for precise bound-
function Eq. (1)
                                                                       ing box annotations, simplifying data labeling and improv-
                      X                                                ing robustness.
```

## 4. Golovkin — extraccao de caracteristicas, deteccao de equipa e pos-processamento
```
points Model, based on ResNet18 [16], detects 74 keypoints             dicts two digits per image, allowing for recognition even
representing the intersections of pitch lines with grass lines.        when part of the number is occluded.
These keypoints refine the initial camera parameter predic-                At Feature Extraction stage secondary networks infer
tions by minimizing the reprojection error when mapped to              additional attributes for each athlete: ReID Embeddings
a canonical (two-dimensional aerial) field view.                       are extracted using a ResNet50-based [16] model trained
    This refinement process employs a brute-force optimiza-            on athlete crops, enabling DeepSORT [52] to maintain con-
tion strategy, evaluating predefined combinations of camera            sistent player identities across frames. TeamID Embed-
parameter corrections and selecting the best one. A prede-             dings are generated by an OSNet [60] model trained on
fined set of delta values, such as [−0.15, −0.10, . . . , 0.15],       111 uniform classes, providing uniform-specific features
is used for each camera parameter. These deltas are added              for team detection in subsequent stages. Each class repre-
to the SegFormer [54] model’s outputs, generating a delta              sents a unique uniform of the football players (for example
matrix. Using these adjusted camera parameters, a series of            red t-shirt and white shorts). OSNet[60] is designed for per-
homography matrices are computed to project the detected               son re-identification (ReID), utilizing multi-scale residual
keypoints from the Field Keypoints Model onto planar co-               blocks and a unified aggregation gate (AG) to dynamically
ordinates (in the world space).                                        fuse spatial features, ensuring both efficiency and strong
    To ensure alignment, keypoints are converted into lines,           discriminative power. Player Orientation is predicted us-
where applicable, based on the expectation that certain key-           ing a ResNet18 [16] model trained to classify athletes’ di-
points lie along the same line. Outliers — keypoints deviat-           rections (left/up/right/down). Player Anomaly is predicted
ing significantly from the expected line are discarded. Mul-           by a lightweight custom model identifies and discards crops
tiple sets of lines are then generated, and the L2 distance            containing multiple athletes or irrelevant content to improve
between these planar lines and the ideal pitch model is cal-           tracking quality.
culated. The optimal camera parameters are determined by                   DeepSORT-based algorithm [52] is used for prelimi-
selecting the set with the smallest L2 difference, ensuring            nary tracking and operates in real-world pitch coordinates
the most accurate alignment and projection.                            derived from the camera parameters instead of pixel co-
    The final camera parameters derived from this process              ordinates. To ensure accurate and consistent tracking, re-
enable the calculation of real-world 3D coordinates for all            strictions based on player orientation (e.g., preventing 180-
detected athletes. These coordinates are used by the Deep-             degree direction changes in one frame) and TeamID Em-
SORT algorithm [52] for tracking player movements on the               beddings are applied.
pitch, providing a bird’s-eye view of the game.                            Raw Tracking stage lays the foundation for subsequent
    To reduce frame-to-frame fluctuations and ensure tempo-            steps, enabling efficient, real-time tracking while providing
ral smoothness in camera parameter predictions, a Savitzky-            essential features and initial tracking results for further re-
Golay filter [13] is applied. This method efficiently smooths          finement.
                                                                           mined based on the majority vote count.
                                                                       This voting mechanism is more robust than simply calcu-
                                                                       lating the mean X-coordinate across all frames, which can
                                                                       fail in very short clips or scenarios where only part of the
                                                                       football field is visible.
                                                                       3.3. Post-Processing
                                                                       The post-processing stage plays a critical role in refining
                                                                       and consolidating the raw tracking results generated dur-
                                                                       ing the preliminary tracking and team detection stages.
Figure 3. Team Detection Process. (a) Frames are clustered into
                                                                       By leveraging information such as player positions on the
three main groups: the two largest clusters (left and right teams)
                                                                       pitch, detected jersey numbers, re-identification (ReID) em-
and the referee cluster. (b) Goalkeeper detection is performed sep-
arately by identifying athletes inside the penalty area and cluster-   beddings, and team labels, this stage addresses common
ing them based on embeddings.                                          tracking challenges, including tracklet fragmentation, iden-
                                                                       tity swaps, and temporal inconsistencies. The primary ob-
3.2. Team Detection                                                    jective of post-processing is to concatenate short tracklets
                                                                       into longer, coherent trajectories while ensuring consistency
In the previous step (raw-tracking), we gathered TeamID                across all attributes.
embeddings and pitch coordinates for every athlete across                  The post-processing pipeline consists of four key stages:
all frames. The next step is team detection , where we iden-           splitting tracklets by jersey numbers and team IDs, merg-
tify five embedding clusters: left/right team athletes, ref-           ing tracklets by jersey numbers, merging tracklets by ReID
erees, and left/right goalkeepers . During post-processing,            similarity, tracklet interpolation.
each athlete is assigned to a team based on the minimum                    In the first stage, tracklets are split based on the assump-
cosine distance to the estimated clusters.                             tion that each track should correspond to a single recognized
    Estimating the Two Biggest Clusters (Left/Right Teams)             jersey number and team label. This ensures that no track-
and Referee Cluster (Figure 3a):                                       let contains conflicting or inconsistent attributes. After this
1. We query all athletes whose X-axis distance is less than            step, all tracklets are guaranteed to contain only detections
    30 meters from the pitch center and cluster them into              associated with a unique jersey number and team ID. This
    three groups: the two largest clusters representing the            process effectively eliminates ambiguities arising from in-
    left and right teams , and a smaller cluster for the refer-        correct associations during earlier stages.
    ees.                                                                   The second stage focuses on merging tracklets that share
2. If the input video contains only the left or right penalty          the same jersey number. This is performed under the fol-
    area (determined by the camera pan), or the clip is very           lowing conditions:
    short we instead select all athletes outside the penalty           • Temporal Non-Overlap : The tracklets must not overlap
    area . This strategy is not used by default, as goalkeepers           in time.
    occasionally leave their designated areas.                         • Physical Feasibility : The distance between the end posi-
Estimating Embeddings Clusters for Left and Right Goal-                   tion of one tracklet and the start position of another must
keepers (Figure 3b):                                                      be physically traversable by a player within the time gap
1. We query athletes positioned inside the penalty area (on               between the two tracklets.
    the left or right side).                                           • Consistent Team IDs : The team IDs of the tracklets being
2. Athletes near previously detected clusters are filtered out            merged must match.
    based on cosine distance to avoid overlap.                         This stage significantly reduces fragmentation caused by
3. The remaining athletes are used to calculate an embed-              temporary occlusions or misdetections.
    ding cluster for the respective goalkeeper.                            For tracklets that remain unmerged after the jersey
Defining Left vs. Right Teams. To distinguish the left                 number-based merging stage, often due to missing or unde-
team from the right team , we employ a voting mechanism                tected jersey numbers caused by occlusions or unfavorable
across frames (or at a defined stride for computational effi-          player orientations, we employ ReID feature similarity as
ciency):                                                               an alternative criterion. The merging process adheres to the
1. For each processed frame, we calculate the mean X-                  same foundational constraints outlined in Section 3.3, en-
    coordinate of athletes belonging to one of the two largest         suring temporal non-overlap, physical feasibility of move-
    clusters and collect votes indicating whether the cluster          ment, and consistency in team IDs. To determine whether
    is positioned more to the left or right.                           two tracklets belong to the same player, we leverage mul-
2. After collecting all votes, the team affiliation is deter-          tiple ReID vectors along each tracklet. For example, con-
sider Tracklet 1 with N ReID vectors and Tracklet 2 with           detailed explanation, which will be provided in a section
M ReID vectors. We apply two complementary methods                 below.
for comparing their ReID features:
• Mean Vector Comparison : For each tracklet, we compute
   the mean ReID vector by averaging all embeddings along          4.1. Camera parameters dataset
   the tracklet. The cosine distance between the mean vec-
   tors of Tracklet 1 and Tracklet 2 is then calculated. If this   Our dataset mainly consists of two parts: real-world im-
   distance falls below a predefined threshold, the tracklets      ages (sourced from the Field Keypoints Dataset) and syn-
   are merged. This approach provides a computationally            thetic images (generated using a modified version of the
   efficient way to assess overall similarity.                     Google Research Football Simulator [24]). The combina-
• Cross-Multiplication Matrix (Pairwise Comparison) : To           tion of these datasets ensures a balance between real-world
   achieve more robust matching, we construct an N × M             variability and controlled synthetic diversity, enhancing the
   matrix where each entry represents the cosine similar-          robustness of the training process. Visualized distributions
   ity between a pair of ReID vectors—one from Tracklet            of camera parameters and example images can be found in
   1 and one from Tracklet 2. By identifying the maximum           Supplementary Material.
   value in this matrix (or equivalently, the minimum cosine       Real-World Dataset. To construct our dataset, we modi-
   distance), we ensure that even noisy or inconsistent em-        fied the TVCalib framework [49] to work with labeled key-
   beddings do not hinder accurate matching. This method           points instead of segmentation masks. The dataset consists
   effectively captures the most similar pair of embeddings        of images annotated with keypoints corresponding to the
   across the two tracklets.                                       pitch lines in the image space. Originally TVCalib calcu-
By combining these two strategies, our pipeline achieves           lates the loss in the image space. In our implementation,
higher accuracy in merging tracklets, particularly in chal-        the optimization process operates in the real-world space
lenging scenarios where individual ReID embeddings may             (canonical view). Specifically, labeled keypoints from the
be unreliable. This dual-strategy approach leverages both          image space are projected onto the canonical view, and the
global (mean vector) and local (pairwise comparison) in-           loss is defined as the sum of L2 distances between the pro-
formation, ensuring precise identity association across frag-      jected keypoints and the corresponding pitch lines in the
mented tracklets.                                                  real-world space. This adjustment to the framework proved
```

## 5. SoccerNet 2026 — cinco tarefas, nenhuma delas GSR
```
        dicting the timing and class of ball-related actions within a short future
        window from a preceding observation window; (2) Player-Centric Ball
        Action Spotting, temporally localizing and classifying ball-related actions
        while assigning each action to the acting player through team affiliation
        and jersey number; (3) Novel View Synthesis, rendering images unob-
        served from camera poses in multi-view football scenes; (4) Spiideo Soc-
        cerNet Synloc, localizing athletes in real-world pitch coordinates from a
        single calibrated static-camera image; and (5) Visual Question Answer-
        ing, answering multiple-choice questions about football broadcasts across
        text, image, and video inputs. For each task, participants were provided
        with annotated data, a unified evaluation protocol, and a public base-
        line. This edition saw broad participation, with 427 teams submitting
        1,129 entries across the five tasks and 28 teams contributing reviewed
        technical reports. This paper describes each task and its evaluation pro-
        tocol, presents the challenge leaderboards, and summarizes the leading
```

---

## 6. Oo et al., Broadcast2Pitch (WACV 2026) — artigo completo dos vencedores de 2025
Fonte: openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_..._paper.pdf

### 6.1 Tabela 5 — comparacao de abordagens de identificacao (mesmo pipeline)
```
PRTReID + EasyOCR + ResNet-18 (modelos especializados) : GS-HOTA 18.11  IDF1 10.62
CLIP                                                    : GS-HOTA 60.13  IDF1 62.20
LLaMA-3.2-Vision (proposto)                             : GS-HOTA 61.48  IDF1 64.20
```

### 6.2 Tabela 7 — decomposicao do GS-HOTA por atributo (CHAVE)
```
Pitch                       : 79.52   <- limite superior, so localizacao
Pitch + Role                : 79.20   (queda marginal)
Pitch + Team                : 73.58   (77.39 excluindo SNGS-126/131/197)
Pitch + Jersey              : 64.70   <- QUEDA GRANDE
Pitch + Role + Team + Jersey: 61.48
```
Conclusao textual dos autores: "the inclusion of jersey numbers leads to a
substantial decline in performance, identifying jersey recognition as the
primary bottleneck".

### 6.3 ANOTACAO ERRADA no conjunto de teste (verificar!)
Nota da Tabela 7: "Samples SNGS-126, 131, 197 are incorrectly annotated for
the team attribute in the SoccerNet-GSR test set."
-> SNGS-126 e exactamente o nosso unico caso 'sem solucao segura' (0% nos 3 estados).

### 6.4 Tabela 3 — registo do campo (corrige uma afirmacao minha)
```
TVCalib     JaC5 19.88 | NBJW 37.14 | BroadTrack 56.88 | Oo et al. 69.38
```
O NBJW (que usamos) NAO e estado da arte em 2025, esta atras do BroadTrack.

### 6.5 Metodo
```
tracklet appears, and let ∆f = fi+1 − fi denote the tem-                remain meaningfully separated over time, as offensive
poral gap between consecutive frames. We define the                     advances are typically met with coordinated defensive
split points as G = {g | ∆f > τgap }, and segment                       retreats that preserve this spatial distinction. We further
the tracklet into [S1 , S2 , ..., S|G|+1 ]. We then evaluate            discuss the validity of our assumption in Section 4.6.
whether adjacent segments Si and Si+1 should remain                     A final post-processing stage improves identity consis-
split by checking consistency in jersey number, role, and               tency and team-level coherence. This includes majority
color, using an identity inconsistency indicator, I:                    voting within each tracklet for jersey number, role, and
                                                                       team affiliations, and filtering unreliable tracklets (e.g.,
                    
                           if Maj(RX ) ̸= Maj(RY )                     tracklets with length less than 20 and detections outside
                    
                    1 ∨ Maj(J ) ̸= Maj(J )
                                       X           Y                    of the soccer field in pitch coordinate) to ensure robust
       I(X, Y ) =                                        (8)            reconstruction.
                    
                    
                            ∨ Maj(CX ) ̸= Maj(CY )
                        0 otherwise,
                    
                                                                        4. Experiments
where Maj(RX ), Maj(JX ), and Maj(CX ) denote the
majority-voted role, jersey number, and color for X, re-                4.1. Dataset
spectively. If I(Si , Si+1 ) is equal to 1, then split occurs.          We evaluate our GSR framework on the SoccerNet-GSR
Merge. After splitting, we perform a merging step to                    dataset [39], which contains 200 broadcast video clips,
produce coherent tracklets. Two tracklets Ta and Tb are                 each 30 seconds long. It extends SoccerNet-Tracking
merged if their ReID features ra and rb are sufficiently                [9] and provides dense geometric annotations over 9.37
similar, as measured by cosine distance:                                million line points for pitch localization and camera cal-
                                   ra · rb                              ibration, as well as over 2.36 million athlete positions
          dcos (ra , rb ) = 1 −              < τcos .     (9)           on the pitch, role, team, and jersey number. The dataset
                                  ∥ra ∥∥rb ∥
```
