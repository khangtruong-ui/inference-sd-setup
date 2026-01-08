import os
import argparse

parser = argparse.ArgumentParser(description="Inferencer")
parser.add_argument("-gcs_dir", type=str, help="GCS directory")
parser.add_argument("-save_dir", type=str, help="save_directory")
args = parser.parse_args()

gcs_dir = args.gcs_dir
save_dir = args.save_dir
assert gcs_dir.startswith('gs://'), f"The GCS directory '{gcs_dir}' is not valid"
assert save_dir, f"The SAVE directory '{save_dir}' is not valid"

from diffusers import FlaxStableDiffusionPipeline
import jax
from PIL import Image
import numpy as np
import random
import string
import csv
import subprocess
from datasets import load_dataset

key = jax.random.key(0)
mesh = jax.sharding.Mesh(jax.devices(), ('data',))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data',))
no_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

def load_model():
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        'KhangTruong/SD_NWPU'
    )
    params = jax.tree.map(lambda x: jax.device_put(x, no_sharding), params)
    return pipeline, params

def generate_from_prompts(pipeline, params, prompts: list[str]):
    global key
    key, subkey = jax.random.split(key)
    prompt_ids = pipeline.prepare_inputs(prompts)
    prompt_ids = jax.device_put(prompt_ids, sharding)
    images = pipeline(prompt_ids, params, subkey, num_inference_steps=20, guidance_scale=7.5, jit=False)
    images = images.images.reshape((images.images.shape[0],) + images.images.shape[-3:])
    out = [Image.fromarray((image * 255.).astype(np.uint8)) for image in images]
    return out

def random_string(length=64):
    candidates = string.ascii_letters + string.digits
    name = ''.join(random.choices(candidates, k=length))
    return name

def save_images(images, prompts, save_dir):
    with open(save_dir + f'/{random_string()}.csv', 'w') as f:
        writer = csv.DictWriter(csvfile, fieldnames=['image', 'caption'])
        writer.writeheader() 
        for image, prompt in zip(images, prompts):
            name = random_string() + '.png'
            image.save(save_dir + '/' + name, format='PNG')
            writer.writerow(dict(image=name, caption=prompt))

def load_prompts():
    with open('checklist.txt', 'a+') as f:
        f.seek(0)
        check_list = f.read().split()
        print(f'Found {len(check_list)} processed files')

    def mapper(example):
        out = example['raw'] + example['raw_1'] + example['raw_2'] + example['raw_3'] + example['raw_4']
        with open('checklist.txt', 'a') as f:
            f.write(' ' + ' '.join(example['raw_filename']))
        return out

    def filter_out(example):
        return all(existing_file not in check_list for existing_file in example['raw_filename'])
        
    ds = load_dataset('KhangTruong/NWPU_Split')['train']
    iterables = ds.iter(batch_size=64, drop_last_batch=True)
    return map(mapper, filter(filter_out, iterables))

pipeline, params = load_model()
promptss = load_prompts()

for prompts in promptss:
    directory = save_dir + '/' + random_string()
    images = generate_from_prompts(pipeline, params, prompts)
    save_images(images, prompts, directory)
    subprocess.run(f'gsutil -m cp -r {directory} {gcs_dir}/generated/ &', shell=True)
    subprocess.run(f'gsutil cp checklist.txt {gcs_dir}/checklist.txt &', shell=True)
