This work introduces an interactive system that couples text‑to‑image generation with region‑aware editing and semantic analysis. Users generate scenes, run object detection, and apply natural‑language edits (e.g., “make the person old”) in a closed loop. Attribute classifiers and probability plots then quantify changes such as young/old or safe/dangerous, turning each edit into a measurable intervention on the model’s latent behavior.

-----------------------------
<img width="3298" height="1933" alt="cv1" src="https://github.com/user-attachments/assets/a2e4bebb-733a-4207-87fd-216be45963f1" />
**Figure 1: Proposed Design of the Interface enabling controlled text-to-image generation and evaluation of prompt
modifier effects. Quantifying the Semantic Accuracy of AI-Generated Imagery. Designed to bridge the gap between
abstract text prompts and visual representation, this tool uses state-of-the-art multi-modal models to verify if an
image successfully captures the conceptual essence of its input prompt. In this image we have used "young man" so
we we have detected and cropped the man to evaluate separately. The evaluator is predicting with a very high score
that the man in the image is young**

<img width="3298" height="1946" alt="cv2" src="https://github.com/user-attachments/assets/b18b616a-e9d4-49e1-bca4-0ef01f60e1f7" />
**Figure 2: Our Model can edit or modify a generated image with prompt also. In the Editing prompt box "Box 1
highlighted in figure" and the modified image can be visually compared side-by-side B, C in the box 2 we can detect
objects by open dictionary. This model is so robust and user friendly that it can just modify the requested object
without changing the other portion and details of the image.**

-----------------------------
