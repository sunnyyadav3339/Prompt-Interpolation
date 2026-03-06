# Prompt Interpolation: Smooth Visual Transitions 

## 📖 Overview
**Prompt Interpolation** is an AI-powered pipeline that transforms two distinct text prompts into a seamless GIF, showcasing a smooth, object-to-object visual transition. By leveraging advanced language models to enrich prompt details and diffusion models for high-fidelity generation, this project creates semantically gradual and visually stunning frame-by-frame animations.

## ✨ Key Features
- **Smooth Visual Transitions:** Generates an animated GIF representing a gradual transformation between two completely different text concepts.
- **LLM Prompt Enrichment:** Integrates **LLaMA 3.3-70B** to automatically expand and enrich base prompts with detailed visual attributes for better image generation.
- **High-Fidelity Image Generation:** Utilizes **Stable Diffusion v1.5** to render high-quality, frame-by-frame outputs.
- **Advanced Embedding Interpolation:** Implements both **SLERP** (Spherical Linear Interpolation) and **Linear** interpolation techniques on the text embeddings to ensure semantically accurate and gradual frame transitions.
- **Interactive UI:** Features a user-friendly frontend built with **Gradio**, allowing users to easily input prompts and generate transition GIFs directly from their web browser.

## 🧠 How It Works
1. **Input:** The user provides a starting prompt (e.g., "A modern sports car") and an ending prompt (e.g., "A futuristic spaceship").
2. **Enrichment:** LLaMA 3.3-70B expands both prompts to include rich descriptive details (lighting, style, environment).
3. **Interpolation:** The pipeline calculates the text embeddings for both prompts and generates intermediate steps using SLERP or Linear interpolation.
4. **Generation:** Stable Diffusion v1.5 processes these interpolated embeddings to generate a sequence of images.
5. **Output:** The individual frames are compiled into a smooth GIF and displayed on the Gradio interface.

## 🛠️ Tech Stack
- **Frontend:** Gradio
- **Text Models:** LLaMA 3.3-70B
- **Image Models:** Stable Diffusion v1.5 (via Hugging Face `diffusers`)
- **Core Logic:** PyTorch (for SLERP/Linear embedding manipulation)


   
