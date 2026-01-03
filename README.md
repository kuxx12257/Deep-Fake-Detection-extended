# Deep-Fake-Detection
 Dual-Model Deepfake Detection with Intelligent Gating
This project introduces a novel architecture for deepfake detection that leverages the complementary strengths of two powerful models — a Convolutional Neural Network (CNN) and a Vision Transformer (ViT) — running in parallel, with an intelligent gating system that dynamically prescribes how their outputs should be interpreted and fused.

- Parallel Model Execution: Both CNN and ViT models process the same video input simultaneously, ensuring no loss of temporal or spatial information.
- Gating System: A learned gating mechanism (CNN-based, RNN-based, or rule-based) analyzes the input and prescribes the optimal weighting between the two models.
- Adaptive Fusion: Instead of choosing one model over the other, the system intelligently blends their predictions based on content characteristics (e.g., blur, motion, brightness).
- Modular Design: Each component — feature extractor, gating model, CNN, ViT — is independently modular and swappable.

-
-## ARCHITECTURE 
-             +------------------+
                |  Input Video     |
                +--------+---------+
                         |
                         v
              +----------------------+
              | Feature Extraction   |  →  [blur, brightness, motion, etc.]
              +----------------------+
                         |
                         v
              +----------------------+
              |   Gating System      |  →  [w_CNN, w_ViT]
              +----------------------+
                 |              |
                 v              v
         +-----------+    +-----------+
         |   CNN     |    |   ViT     |
         |  Model    |    |  Model    |
         +-----------+    +-----------+
                 \              /
                  \            /
                   \          /
                    v        v
               +------------------+
               | Weighted Fusion  |
               +------------------+
                         |
                         v
               +------------------+
               | Final Prediction |
               +------------------+


## SAMPLE OUTPUT
 
 Gating Weights:
   - CNN: 0.68
   - ViT: 0.32

 CNN Prediction: 0.72
 ViT Prediction: 0.61

 Final Score: 0.675
 Final Verdict: Deepfake


  
