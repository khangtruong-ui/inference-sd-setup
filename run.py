from inference import generate_from_prompts
from saving import save_images, upload_to_gcs
import os
import argparse

parser = argparse.ArgumentParser(description="Inferencer")
parser.add_argument("gcs_dir", type=str, help="GCS directory")
parser.add_argument("save_dir", type=str, help="save_directory")
args = parser.parse_args()

gcs_dir = args.gcs_dir
save_dir = args.save_dir
assert gcs_dir.startswith('gs://'), f"The GCS directory '{gcs_dir}' is not valid"
assert save_dir, f"The SAVE directory '{save_dir}' is not valid"

from diffusers import FlaxStableDiffusionPipeline
import jax
from PIL import Image

key = jax.random.key(0)

def load_model():
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        './sd-full-finetuned'
    )

    mesh = jax.sharding.Mesh(jax.devices(), ('data',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data',))
    no_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    params = jax.tree.map(lambda x: jax.device_put(x, no_sharding), params)
    return pipeline, params

def generate_from_prompts(pipeline, params, prompts: list[str]):
    global key
    key, subkey = jax.random.split(key)
    prompt_ids = pipeline.prepare_inputs(prompt)
    prompt_ids = jax.device_put(prompt_ids, sharding)
    images = pipeline(prompt_ids, params, jax.random.key(2), num_inference_steps=1000, guidance_scale=7.5, jit=False)


pipeline, params = load_model()
prompts = load_prompts()
images = generate_from_prompts()
if save_images(save_dir) and upload_to_gcs(save_dir, gcs_dir):
    print("===== Done =====")

