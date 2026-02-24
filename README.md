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
To enable contextual understanding in CLIP based models, concept pairs (e.g., safe/dangerous, young/old, cute/ugly.....) are defined as contrastive
text prompts. Each pair describes opposite meanings through rich, situational sentences that go beyond single words — covering context, object
behavior, and environmental cues. This allows the model to reason contextually — distinguishing between similar visuals, such as a man holding
a kitchen knife to cook (safe) versus threatening someone (dangerous), or a lion cub playing (safe) versus an adult lion attacking (dangerous). Such
structured prompt design improves the model’s interpretive robustness across scenes, perspectives and contexts.

<p float="left">
  <img src="https://github.com/user-attachments/assets/95dcae0a-d79c-4af2-88e1-c2bf1477ec1e" alt="Screenshot_2025_11_05-3" width="49%" />
  <img src="https://github.com/user-attachments/assets/4c3f0291-76b8-4cb3-b8ed-b09a6fd42f84" alt="Screenshot_2025_11_05-4" width="49%" />
</p>

**Figure 3: A tiger caged in zoo is detected as safe (left) vs a Tiger attacking in jungle is detected as Dangerous (right)**

<p float="left">
<img width="1566" height="1397" alt="cv7" src="https://github.com/user-attachments/assets/b0a95714-36fa-437c-bddd-0d92a0e03a99" width="49%" />
<img width="1555" height="1366" alt="cv8" src="https://github.com/user-attachments/assets/056633ac-35fc-483c-947c-fc018ea723cc" width="49%" />
</p>

-------------------------
