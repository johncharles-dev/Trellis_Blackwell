import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
import shutil
import zipfile
from typing import *
import torch
import numpy as np
import imageio
import open3d as o3d
from easydict import EasyDict as edict
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh


def state_to_o3d_mesh(state: dict) -> o3d.geometry.TriangleMesh:
    """Convert packed state mesh data to open3d TriangleMesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(state['mesh']['vertices'])
    mesh.triangles = o3d.utility.Vector3iVector(state['mesh']['faces'])
    return mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def text_to_3d(
    prompt: str,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    req: gr.Request,
) -> Tuple[dict, str]:
    """
    Convert a text prompt to a 3D model.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    outputs = pipeline.run(
        prompt,
        seed=seed,
        formats=["gaussian", "mesh"],
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path


def text_variant(
    variant_prompt: str,
    seed: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    output_buf: dict,
    variant_mesh_file: str,
    req: gr.Request,
) -> Tuple[dict, str]:
    """
    Generate a variant of a 3D model using a text prompt.
    Uses either the previously generated mesh or an uploaded PLY file as base.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))

    # Load base mesh: prefer uploaded file, fall back to previous generation
    if variant_mesh_file is not None:
        base_mesh = o3d.io.read_triangle_mesh(variant_mesh_file)
    elif output_buf is not None:
        base_mesh = state_to_o3d_mesh(output_buf)
    else:
        raise gr.Error("No base mesh available. Generate a model first or upload a PLY file.")

    outputs = pipeline.run_variant(
        base_mesh,
        variant_prompt,
        seed=seed,
        formats=["gaussian", "mesh"],
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'variant.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path


def extract_mesh(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    export_format: str,
    req: gr.Request,
) -> Tuple[str, str]:
    """
    Extract a mesh file from the 3D model.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)

    if export_format == "STL":
        gr.Info("STL is geometry-only — textures are skipped for faster extraction.")
        mesh_obj = postprocessing_utils.to_mesh_geometry(mesh, simplify=mesh_simplify, verbose=False)
        stl_path = os.path.join(user_dir, 'sample.stl')
        mesh_obj.export(stl_path)
        torch.cuda.empty_cache()
        return stl_path, stl_path

    # GLB and OBJ both need the textured mesh from to_glb()
    glb_mesh = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)

    if export_format == "OBJ":
        obj_dir = os.path.join(user_dir, 'obj_export')
        os.makedirs(obj_dir, exist_ok=True)
        obj_path = os.path.join(obj_dir, 'sample.obj')
        glb_mesh.export(obj_path)
        zip_path = os.path.join(user_dir, 'sample.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(obj_dir):
                zf.write(os.path.join(obj_dir, fname), fname)
        glb_path = os.path.join(user_dir, 'sample.glb')
        glb_mesh.export(glb_path)
        torch.cuda.empty_cache()
        return glb_path, zip_path

    # Default: GLB
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb_mesh.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a Gaussian file from the 3D model.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Text to 3D Asset with [TRELLIS](https://trellis3d.github.io/)
    * Type a text prompt and click "Generate" to create a 3D asset.
    * Use the "Variant Editor" tab to modify an existing 3D model with a text description.
    * If you find the generated 3D asset satisfactory, choose a format (GLB, OBJ, or STL) and click "Extract Mesh" to extract and download it.
    """)

    with gr.Row():
        with gr.Column():
            with gr.Tabs() as input_tabs:
                with gr.Tab(label="Text to 3D", id=0) as text_tab:
                    text_prompt = gr.Textbox(label="Text Prompt", lines=3, placeholder="A chair that looks like an avocado")

                    with gr.Accordion(label="Generation Settings", open=False):
                        seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                        randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                        gr.Markdown("Stage 1: Sparse Structure Generation")
                        with gr.Row():
                            ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=25, step=1)
                        gr.Markdown("Stage 2: Structured Latent Generation")
                        with gr.Row():
                            slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=25, step=1)

                    generate_btn = gr.Button("Generate")

                with gr.Tab(label="Variant Editor", id=1) as variant_tab:
                    gr.Markdown("""
                    Generate a variant of an existing 3D model using a text description.
                    You can either use the previously generated model or upload your own PLY mesh file.
                    """)
                    variant_prompt = gr.Textbox(label="Variant Prompt", lines=3, placeholder="Rugged, metallic texture with orange paint finish")
                    variant_mesh_file = gr.File(label="Upload Base Mesh (PLY/OBJ)", file_types=[".ply", ".obj", ".glb"], type="filepath")
                    gr.Markdown("*Leave empty to use the previously generated model as base.*")

                    with gr.Accordion(label="Variant Settings", open=False):
                        variant_seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                        variant_randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                        gr.Markdown("Structured Latent Generation")
                        with gr.Row():
                            variant_slat_guidance = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            variant_slat_steps = gr.Slider(1, 50, label="Sampling Steps", value=25, step=1)

                    variant_btn = gr.Button("Generate Variant")

            with gr.Accordion(label="Mesh Extraction Settings", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 4096, label="Texture Size", value=1024, step=512)
                export_format = gr.Radio(
                    ["GLB", "OBJ", "STL"],
                    label="Export Format",
                    value="GLB",
                    info="GLB: textured PBR mesh | OBJ: textured mesh (zipped) | STL: geometry only (fast, no textures)",
                )

            with gr.Row():
                extract_mesh_btn = gr.Button("Extract Mesh", interactive=False)
                extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)
            gr.Markdown("*NOTE: Gaussian file can be very large (~50MB), it will take a while to display and download.*")

        with gr.Column():
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
            model_output = LitModel3D(label="Extracted Mesh/Gaussian", exposure=10.0, height=300)

            with gr.Row():
                download_mesh = gr.DownloadButton(label="Download Mesh", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)

    output_buf = gr.State()

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    # Text to 3D generation
    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        text_to_3d,
        inputs=[text_prompt, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps],
        outputs=[output_buf, video_output],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_mesh_btn, extract_gs_btn],
    )

    # Variant generation
    variant_btn.click(
        get_seed,
        inputs=[variant_randomize_seed, variant_seed],
        outputs=[variant_seed],
    ).then(
        text_variant,
        inputs=[variant_prompt, variant_seed, variant_slat_guidance, variant_slat_steps, output_buf, variant_mesh_file],
        outputs=[output_buf, video_output],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_mesh_btn, extract_gs_btn],
    )

    video_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[extract_mesh_btn, extract_gs_btn],
    )

    extract_mesh_btn.click(
        extract_mesh,
        inputs=[output_buf, mesh_simplify, texture_size, export_format],
        outputs=[model_output, download_mesh],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_mesh],
    )

    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_gs],
    )

    model_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[download_mesh],
    )


# Launch the Gradio app
if __name__ == "__main__":
    import argparse
    from trellis.utils.vram_manager import init_vram_manager

    parser = argparse.ArgumentParser(description="TRELLIS Text-to-3D (Blackwell)")
    parser.add_argument('--model', default='microsoft/TRELLIS-text-large',
                        help='Model to use (default: microsoft/TRELLIS-text-large). '
                             'Options: microsoft/TRELLIS-text-base, microsoft/TRELLIS-text-large, '
                             'microsoft/TRELLIS-text-xlarge')
    parser.add_argument('--precision', choices=['auto', 'full', 'half'], default='auto',
                        help='Precision mode: auto (detect VRAM), full (float32), half (float16)')
    parser.add_argument('--vram-tier', choices=['auto', 'high', 'medium', 'low'], default='auto',
                        help='VRAM tier: auto (detect), high (>=24GB), medium (12-23GB), low (8-11GB)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=7861, help='Port to bind to')
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    args = parser.parse_args()

    vm = init_vram_manager(precision=args.precision, vram_tier=args.vram_tier)

    print(f"Loading text-to-3D model: {args.model}")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    if vm.dtype == torch.float16:
        pipeline.to_dtype(torch.float16)

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
