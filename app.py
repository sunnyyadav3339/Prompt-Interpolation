import gradio as gr
from main2 import run_generation

def interface_fn(obj1, obj2):
    gif_path = run_generation(obj1, obj2)
    return gif_path

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Text-to-Image Interpolation")
    gr.Markdown("Enter two objects below and generate an interpolated image GIF using Stable Diffusion.")

    with gr.Row():
        with gr.Column(scale=1):
            obj1 = gr.Textbox(label="Object 1", placeholder="e.g., dog")
        with gr.Column(scale=1):
            obj2 = gr.Textbox(label="Object 2", placeholder="e.g., cat")

    gen_btn = gr.Button("🚀 Generate")

    output_img = gr.Image(
        type="filepath", 
        format="gif", 
        label="🔄 Interpolated GIF", 
        show_label=True
    )

    gen_btn.click(fn=interface_fn, inputs=[obj1, obj2], outputs=output_img)

    gr.Markdown("---")
    gr.Markdown("## 📦 Demo Outputs")

    with gr.Row():
        with gr.Column():
            gr.Image(value="output/22-02-51.gif", label="🐎 Horse → 🦓 Zebra", show_label=True)
        with gr.Column():
            gr.Image(value="output/17-07-58.gif", label="🦁 Lion → 🫏 Donkey", show_label=True)
        with gr.Column():
            gr.Image(value="output/22-02-51.gif", label="🐎 Horse → 🦓 Zebra", show_label=True)
        with gr.Column():
            gr.Image(value="output/17-07-58.gif", label="🦁 Lion → 🫏 Donkey", show_label=True)
    

demo.launch(allowed_paths=["./output"])
