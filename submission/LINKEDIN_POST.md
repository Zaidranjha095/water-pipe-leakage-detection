# LinkedIn post

Copy the text below. Replace the bracketed URLs if they are still placeholders, then publish the post and paste its URL into `project.config.json` (`linkedinUrl`).

---

I built an acoustic water-pipe leak detector for my Deep Learning semester project.

A compact CNN classifies one-second clips as leak, no leak, or environmental noise. Training used 1,000 public field recordings (Zenodo 18631450). On a held-out test split the model reached 94.00% accuracy and 93.13% macro F1. The page runs the same network in the browser from a WAV file.

- Live demo and report: https://water-pipe-leakage-detection.vercel.app
- Code and weights: https://github.com/Zaidranjha095/water-pipe-leakage-detection
- Walkthrough: https://youtu.be/wDVgfvbutMc

Weights are at `weights/model.pth`. Training audio is from field sensors, so a phone clip is a domain shift — this is a course project, not a plumbing tool.

#DeepLearning #AudioClassification #ComputerVision
