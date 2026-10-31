# Submission checklist

Measured test accuracy: 94.00%. Macro F1: 93.13%.

- [x] Put your name and roll number in `project.config.json`.
- [x] Re-run `python build_docs.py` and `node prepare_site.js` so the report and slides use that name.
- [ ] Create a public GitHub repository and push this project. Do not enable Git LFS for the weights.
- [ ] Confirm these files open from the repository: `weights/model.pth`, `weights/model.onnx`, `weights/config.json`, `model.py`, `predict.py`, `model_manifest.json`.
- [ ] Put the repository URL in `project.config.json` as `githubUrl`.
- [ ] Deploy the site to Vercel from this folder (`npx vercel`). The build command is `node prepare_site.js`.
- [ ] Open the Vercel URL and check the demo, `report.html`, `downloads/model.pth`, and `downloads/presentation.pptx`.
- [ ] Put the Vercel URL in `project.config.json` as `vercelUrl`.
- [ ] Record the voiceover from `presentation/VOICEOVER_SCRIPT.md` while showing `presentation/presentation.pptx`.
- [ ] Upload the video to YouTube as Public or Unlisted. Store the URL as `youtubeUrl`.
- [ ] Publish `submission/LINKEDIN_POST.md` and store the post URL as `linkedinUrl`.
- [ ] Run `node prepare_site.js` again and redeploy so the live page contains the GitHub, YouTube, and LinkedIn links.
- [ ] Submit the instructor form with the Vercel URL, GitHub repository, YouTube video, and LinkedIn post.

Current placeholders:

- GitHub: [GitHub URL — add this before submission]
- Vercel: https://water-pipe-leakage-detection.vercel.app
- YouTube: [YouTube URL — add this before submission]
- LinkedIn: [LinkedIn URL — add this before submission]
