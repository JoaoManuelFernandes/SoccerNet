# Sistemas do desafio SoccerNet-GSR (2024 e 2025) — notas verificadas

Fontes: Cioppa et al. 2024 (arXiv:2409.10587) e Giancola et al. 2025 (arXiv:2508.19182),
seccoes 5 e apendices 7.3 / 7.4. Texto extraido dos PDF oficiais em 2026-09-08.

## Edicao 2024 (16 equipas, 34 submissoes, baseline 23.36 -> 63.81)
```
there was no contact, while the other could be classified as       Table 4. Game state reconstruction leaderboard. The main met-
“Offense + Yellow card” if the player made contact with            ric for the leaderboard and best performances are in bold. Team
the opponent. Small differences, such as the speed of the          names with a superscript have provided a summary that may be
                                                                   found in Appendix 7.3 or in Section 5.4 for the winning team.
foul, the point of contact or the intention to play the ball,
can result in different classifications. A second challenge is      Participant team            GS-HOTA (↑)   GS-DetA (↑)       GS-AssA (↑)
the subjective nature of the task. The interpretation of what       Constructor techGSR−1          63.81         49.52             82.23
constitutes a foul can vary between referees, leading to in-        UPCxMobiusGSR−2                43.15         30.46             61.16
consistencies. This subjectivity introduces variability and         JAM                            34.40         19.38             61.08
                                                                    XJTU MM (JNR)                  33.57         20.44             55.16
complicates the implementation of an objective classifica-          VIPLab                         29.82         16.16             55.03
tion model.                                                         JustTesting                    29.08         13.78             61.37
    The participants explored multiple directions for this          Robo SpaceGSR−7                26.92         11.04             65.66
                                                                    sjc                            24.20         10.16             57.68
first edition of the challenge. First, Hong et al. used the         ABL                            23.73          9.94             56.67
VideoChat2-based video foundation model [19] as a fea-              testingGS (test baseline)      23.73          9.94             56.67
ture extractor to more effectively capture temporal infor-          Baseline [25]                  23.36          9.80             55.69
                                                                    SAIVA Ball                     23.08         10.13             52.62
mation. Rüfenacht adopted a distinct approach by relying           Football AI Lab                22.96          9.92             53.17
only on a single view (the live feed). Given that the por-          playbox x NUSGGSR−14           21.26          7.55             59.90
tion of the frame covering the foul is small, he employed           UWIPL                          20.33          7.85             52.64
                                                                    eidos.aiGSR−16                 8.92           1.51             52.69
a lightweight feature extractor to process full-resolution          Eidos                          8.92           1.51             52.69
720 × 1280 grayscale images instead of the commonly used
224 × 224 resolution.
                                                                   Intersection-over-Union (IoU) as a similarity score to match
5. Game State Reconstruction                                       predictions (P) and ground truth (G) bounding boxes in the
                                                                   image space. The key distinction setting GS-HOTA apart
5.1. Task description
                                                                   from HOTA is the use of a new similarity score, that ac-
   SoccerNet Game State Reconstruction (GSR) is a novel            counts for the specificities of the GSR task, i.e. the addi-
computer vision task involving the tracking and identifi-          tional target attributes (jersey number, role, team) and the
cation of players from a single moving camera to con-              detections provided as 2D points in the real-world instead
struct a video game-like minimap without any specific hard-        of bounding boxes in the image plane. This new similar-
ware worn by the players. While this task holds immense            ity score, denoted SimGS−HOT A (P, G), is formulated as
```

## Descricoes 2024
```
are obtained through a 3-layered MLP. Our code is available at          7.3. Game State Reconstruction
https://github.com/csjihwanh/soccernet-MLV.
                                                                        GSR-2 - No bells, More Whistles
MVF-3 - Single-View Foul Recognition
                                                                        Marc Gutiérrez-Pérez and Dominic Rüfenacht
Dominic Rüfenacht
                                                                        (dominic.ruefenacht@gmail.com)
(dominic.ruefenacht@gmail.com)
                                                                            Our submission to the SoccerNet gamestate reconstruction
    Perhaps the most interesting aspect of the proposed method for
                                                                        challenge uses the Tracklab library [14] with three significant
foul recognition is that it solely relies upon view 0 of the multi-
                                                                        modifications to improve performance. First, we replaced the
view dataset (i.e., the live feed). As the portion of the frame in
                                                                        camera calibration method with the one from Gutierrez et al. [9],
view 0 that covers the foul is small, we opted for a lightweight fea-
                                                                        which involves a hierarchical keypoint generation process leverag-
ture extraction backbone that can consume full-resolution 720 ×
                                                                        ing the soccer field’s geometric properties, and uses an HRNetv2-
1280 grayscale images as input. These are stacked into “fake”
                                                                        based encoder-decoder network for keypoint and line detection.
RGB images, before being passed through a sequence of 2D and
                                                                        Second, we retrained the person detection model on around 2000
3D convnets that captures spatio-temporal relationships [1]. The
                                                                        custom-annotated frames from SoccerNet videos and directly clas-
output of the backbone is a 1280 dimensional vector, which is
                                                                        sified the detections as players, goalkeepers, and referees. Lastly,
passed through a number of MLPs. First, there is a shared MLP
                                                                        we trained a dedicated jersey digit detection model on publicly
for both offense severity and action, the output of which is then
                                                                        available data and around 1,000 custom annotated frames from
passed through individual MLPs for the two classification tasks.
                                                                        SoccerNet videos. The recognized digits are subsequently merged
Horizontal flipping is used as test-time augmentation, which im-
                                                                        to form the jersey numbers. This submission significantly im-
proved results by around 1.5% points. The full code is available
                                                                        proves over the baseline, addressing core issues effectively and
on https://github.com/druefena/MVFoul [23].
                                                                        enhancing the overall methodology.
MVF4 - TAdaFormer for foul recognition                                  GSR-7 - Robo Space
    Konrad Habel,          Fabian Deuser,        Norbert Oswald         Zehua Cheng, Yuan Li, Wei Dai, Menglong Li, and Yongqiang
({konrad.habel, fabian.deuser, norbert.oswald}@unibw.de)                Zhu (zehua.cheng@cs.ox.ac.uk, lyn@intellicloud.ai, loveis-
    Our approach leverages a TAdaFormer-L/14 [13] pre-trained           pdvd@gmail.com, mlli8803@163.com, alexzhu.vip@gmail.com)
on the Kinetics 400 and 710 datasets [15]. We use an ensemble               The methodology outlined commences with utilizing TrackLab
of three models using frame 75 as center frame with 3 different         as the core framework due to its proven effectiveness in analyz-
step sizes (1, 2, 3), what leads to a context window of (16, 32, 48)    ing behaviors and tracking, establishing a solid foundation. To
frames. The model is kept frozen until Transformer stage 21. We         improve object detection accuracy specific to the research needs,
only train the last three Transformer stages and the used classifica-   YOLOv8 is fine-tuned using a custom dataset, focusing on precise
tion heads on the foul recognition data. Similar to the baseline so-    localization of humans and their actions in videos. In conjunc-
lution [11], we use max-pooling on the output of the TAdaFormer         tion, the Body Pose-Based Re-Identification (BPBReID) model is
to aggregate multiple views before the classification heads. During     refined to strengthen individual recognition even in visually com-
training, we randomly select two views, whereas during inference        plex situations, contributing to consistent tracking. To enhance
all views per action are used. An ensemble of three models trained      the system’s understanding of dynamic scenes over time, a Global
with different step size between consecutive frames achieves on         Temporal Attention mechanism is introduced, which selectively
the test set a combined metric score of 42.37 and on the challenge      highlights important spatiotemporal elements, improving the in-
set a combined metric score of 42.05.                                   terpretation of behaviors. Finally, time smoothing techniques are
MVF9 - PW ZZSN                                                          applied to refine detections, eliminating noise and ensuring coher-
    Milosz        Lopatto        and        Adam           Gorski       ence throughout the video, facilitating seamless analysis of contin-
({milosz.lopatto,adamsebastiangorski}@gmail.com)                        uous actions and paths. This comprehensive strategy combines es-
    We based our solution on the existing VARS model with the           tablished and innovative methods, from the solid base of TrackLab
MViTv2 encoder [21]. First, we refactored and wrapped the code          to the specialized tuning of YOLOv8 and BPBReID, integration of
in Pytorch Lightning, which allowed us to easily track experi-          temporal attention, and concluding with time smoothing.
ments using a custom Weights and Biases logger. We experi-              GSR-14 - playbox x NUSG
mented with multiple parameters, but FPS, start frame, and end          Atom Scott and Calvin Yeung
frame are the ones that have been changed compared to the base-         (atom.james.scott@gmail.com,
line model configuration. We kept the middle of the extracted clip      yeung.chikwong@g.sp.m.is.nagoya-u.ac.jp)
    We fine-tuned YOLOv8X for detection while retaining the
original tracking model implemented in tracklab. Camera cali-
bration was improved by correcting a tvcalib implementation er-
ror which excluded the center circle when calculating segment re-
projection loss and applying temporal filtering, using roll degrees
as an error proxy since broadcast cameras should not exhibit roll.
For jersey number recognition, we used a two-stage CLIP-based
method: one model detected numbers in bounding boxes, another
identified specific numbers. This approach, fine-tuned on a custom
dataset (we manually annotated 3,000 bounding boxes with jersey
numbers) achieved 0.78 tracklet accuracy. GPT-4O was used to re-
fine predicted roles, team affiliation and jersey numbers. Despite
qualitative improvements, especially in camera calibration, and a
4% increase in test set scores over the baseline, our methods didn’t
surpass the baseline on the final challenge set. We expect camera
calibration and jersey number recognition to have significant room
for improvement.
GSR-16 - Further than aspect: solving roles with expanded
information context
Francisco Cachón, Enzo Pacilio, Cristian Gonzalez, and Marcelo
Ortega
({francisco.cachon, enzo, cristian, ortegatron}@eidos.ai)
    Our approach is focused on improving role and team classi-
fication, solving it globally for the video instead of classifying
each track individually. We trained a Siamese Network to produce
bounding box embeddings, and cluster these embeddings together
with different KMeans values. To find the most plausible cluster-
ing and assign role and team to its clusters, we trained a Random
Forest classifier, on features incorporating additional game con-
text information derived from detected tracks and its fieldmap po-
sition. This method achieved a lower GS-HOTA, likely because
the model was trained with ground truth positional information,
but finally evaluated on inaccurate detection data.
```

## Edicao 2025 (14 equipas, 66 submissoes, baseline 29.01 -> 63.90)
```
Table 4. Game state reconstruction leaderboard. The main met-                          to further enhance performance. For the camera calibra-
ric for the leaderboard and best performances are in bold. Team                        tion task, popular approaches involved landmark detection
names with a superscript have provided a summary that may be                           followed by posterior calibration or homography estima-
found in Appendix 7.4 or in Section 5.4 for the winning team.
                                                                                       tion, with pipelines such as NBJW [11], PnLCalib [10],
Participant team                             GS-HOTA (↑)   GS-DetA (↑)   GS-AssA (↑)   and Broadtrack [23] commonly used. A novel development
KIST-GSRGSR-1                                   63.90         51.36         79.55      in this edition was the incorporation of optical flow tech-
Constructor.Tech                                63.81         49.52         82.23
lianyouGSR-3                                    62.76         46.17         85.33
                                                                                       niques to improve temporal smoothing. Significant innova-
Playbox & MIXIGSR-4                             61.64         45.04         84.36      tions were also observed in player feature extraction for re-
tyler durden                                    57.08         41.52         78.48
SJTU Multi-Modal Perception Group               50.06         32.98         75.99      identification, role classification, team affiliation, and jersey
eidos.aiGSR-7                                   46.24         31.06         68.85      number recognition. Notably, the use of CLIP- and OSNet-
UTokyo Football LabGSR-8                        40.13         21.33         75.51
hjkimGSR-9                                      36.90         17.56         77.58      based features for these tasks, including object tracking,
junkwang                                        33.62         18.09         62.51
lsmuqi                                          32.62         17.03         62.50
                                                                                       emerged as a strong approach. Furthermore, Vision Lan-
hayden97                                        32.59         16.96         62.61      guage models such as LLaMA-Vision [9] and Qwen2 VL
Baseline [26]                                   29.01         13.71         61.41
Laboratory for Biosignal Processing GSR-13      28.45         13.50         59.99      Instruct [29] were applied successfully for jersey number
johnstreet                                      14.26          9.67         21.10      recognition.

                                                                                       6. Conclusion
tracking and identity reconstruction in single-camera foot-
```

## Descricoes 2025
```
7.4. Game State Reconstruction
                                                                 {yeung.chikwong,       suzuki.tomohiro}@g.sp.m.is.nagoya-
GSR-1                                                            u.ac.jp, nkym20020422@gmail.com, {masato.tonouchi,
Yin May Oo, Yewon Hwang, Vanyi Chao, Amrulloh Robbani            rio.watanabe, takaya.hashiguchi}@mixi.co.jp, yamag-
Muhammad, Jinwook Kim (jinwook.kim21@gmail.com)                  ishi.shunzo@image.iit.tsukuba.ac.jp, ytkmkw@gmail.com)
Our game state reconstruction framework addresses player         Our approach builds on the TrackLab framework, updating
tracking and identity reconstruction in single-camera foot-      several modules with state-of-the-art techniques. For object
ball broadcasts by integrating modular components for            detection, we use RF-DETR, outperforming the YOLO
localization and identity recognition under challenging          series. Feature extraction used CLIP-ReID embeddings,
conditions. Athletes are detected via YOLO-X, tracked            feeding our tracking pipeline consisting of BoT-SORT and
with Deep-EIoU and OSNet-based ReID embeddings, and              Global Tracklet Association (GTA). Jersey number recog-
mapped to pitch coordinates using a multi-frame key-             nition closely follows previous work; we filter embeddings
point detection model. Athlete identity is formulated as a       via Gaussian outlier rejection, geometrically extract torso
multimodal autoregressive generation task with LLAMA-            regions using ViTPose keypoints, and recognize numbers
3.2-Vision, estimating role, jersey number, and color via        using PARSeq OCR. Camera calibration extends Broad-
instruction-based prompts for flexible, open-set recogni-        Track by propagating calibration parameters temporally
tion. Initial tracklets are refined using a split-and-merge      via optical flow, initialized from multiple stable frames.
strategy: splits are determined based on identity-prediction     To address visual ambiguity or occlusion, we enhance
consistency; merges are guided by ReID similarity un-            role classification heuristically, assigning goalkeeper roles
der spatial-temporal and identity constraints. Team affili-      based on spatial positioning (furthest player in penalty
ation (left/right) is inferred by clustering jersey colors and   areas), followed by greedy hill-climbing optimization
comparing cluster-wise mean x-positions, assuming players        penalizing invalid configurations, such as multiple goal-
tend to remain on their side. Final post-processing improves     keepers. Finally, linear interpolation and a Savitzky–Golay
identity consistency and team-level coherence using major-       filter are applied to ensure smoothness and handle short
ity voting within tracklets and filtering unreliable trajecto-   tracking gaps. This pipeline achieved 4th place (61.64
ries, resulting in a robust and structured game state recon-     GS-HOTA). Future work will leverage domain-specific
struction.                                                       knowledge to enhance overall robustness.
GSR-3                                                            GSR-7
Youxing         Lian         and       Tiancheng         Wang    Marcelo Ortega and Federico Méndez ({ortegatron, fed-
(jy02624985@gmail.com, tiancheng.wang91@gmail.com)               erico}@eidos.ai)
Our submission for the SoccerNet Game State Recon-               We boost player-ID accuracy by combining a large foun-
struction challenge implements a four-stage process. We          dational vision–language model (Qwen2-VL-Instruct) with
detect persons using YOLOv12 and 27 field line categories        function calling. Instead of classifying each detection in
(16 points each) via a custom RT-DETR model. For field           isolation, we feed the model a long sequence of detections
calibration, radial distortion correction (Wang et al.) is       arranged on a grid. Through chain-of-thought reasoning,
combined with PnLCalib (Gutiérrez-Pérez & Agudo); fixed        the model can invoke a “zoom” function to inspect specific
camera parameters are simplified to [pan,tilt,roll,focal]        frame ranges before choosing a jersey number, iterating un-
and temporally smoothed.           Our tracking, enhancing       til confident. We used this method also for the referee role
IOF-Tracker (Liu et al.), utilizes Farneback optical flow        recognition, and modified the clustering algorithm to search
to improve Kalman filter-based motion prediction, espe-          for a better balance between the number of tracks on the
cially during abrupt movements, applied to foreground            players clusters.
objects identified via IoU and score thresholds. Features        GSR-8
Sadao Hirose, Kazuto Iwai, Ayaha Motomochi and                     flow. These tracked correspondences enable homography
Ririko Inoue ({hirose.akka, kazutoiwai.jp, ayaha.mochi5,           updates in subsequent frames via RANSAC. This iterative
lyrico1202}@gmail.com)                                             process improves temporal consistency and robustness.
We present an enhanced pipeline built upon the SoccerNet           Evaluation using GS-HOTA and LocSim metrics shows
2025 GSR baseline. Images in the dataset were deblurred            improvements of up to 4% over the NBJW baseline,
using DeblurGAN-v2. We fine-tuned a YOLOv8x detec-                 particularly in frames lacking prominent pitch features.
tor on the SoccerNet dataset for improved athlete detection.       Future work will focus on analyzing failure cases and
Detected regions were then segmented with DeepLabV3                further fine-tuning the system to enhance performance.
and MobileNetV3-Large to isolate player silhouettes. Seg-
mented images were used to extract a 10-dimensional vec-
tor encoding uniform colors and their corresponding areas.
These vectors augmented inputs to downstream modules
for re-identification, role classification, and team identifi-
cation. We also introduced GTA-Link to refine tracklets
post-baseline tracking. A Gaussian blur filter was applied
prior to jersey number recognition. After passing through
the pipeline, we applied linear interpolation to fill in gaps in
detections and stabilize trajectories. Similarly to the orig-
inal baseline, our method had difficulties mostly with jer-
sey number recognition. Future work includes incorporat-
ing multimodal large language models for more accurate
jersey number recognition.
GSR-9
Hyungjung Kim, Joohyung Oh and Jae-Young Sim
({hjkim1113, joohyung0809, jysim}@unist.ac.kr)
We developed four modules to improve the performance of
tracking and re-identification. First, we applied motion de-
blurring to reduce the blurring artifacts in images and im-
prove the detection performance of the players and jersey
numbers. Second, we used optical flow estimation to com-
pensate for the camera movements and enhance the stabil-
ity of tracking. Third, we fine-tuned the detector model on
SoccerNet-GSR dataset to further improve the performance.
Lastly, we used uniqueness property that no more than 24
individuals (22 players and 2 referees) have different iden-
tities for reliable identity matching. The proposed method
achieves a GS-HOTA score of 36.90% in the challenge.
GSR-13
Julian Ziegler, Patrick Frenzel, Daniel Matthes
and Mirco Fuchs ({julian.ziegler, patrick.frenzel,
daniel.matthes,mirco.fuchs}@htwk-leipzig.de)
This report presents our submission to the SoccerNet
2025 Game State Reconstruction Challenge, specifically
addressing the camera calibration subtask. Our method
refines homography estimation across video frames using
a tracking-based approach. Reliable initialization frames
are identified by comparing a U-Net pitch segmentation
prediction with one derived from the NBJW-predicted
homography, using their Intersection over Union (IoU) as
a confidence measure. Keypoints from the NBJW model,
along with auxiliary points sampled via geometric priors
around randomly selected pitch locations, are projected into
pitch coordinates and tracked using Lucas-Kanade optical
```
