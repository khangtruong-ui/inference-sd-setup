import os
import io
import threading
from flask import Flask, request, send_file, jsonify

from diffusers import FlaxStableDiffusionPipeline
import jax
import numpy as np
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_DIR = os.environ.get("STABLE_DIFFUSION_DIRECTORY")
if not MODEL_DIR:
    raise ValueError("Set STABLE_DIFFUSION_DIRECTORY")

NUM_DEVICES = len(jax.devices())
BATCH_SIZE = 32 * NUM_DEVICES

print(f"Using {NUM_DEVICES} devices")
print(f"Global batch size = {BATCH_SIZE}")

# =========================
# JAX SETUP
# =========================
key = jax.random.key(0)

mesh = jax.sharding.Mesh(jax.devices(), ('data',))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data',))
no_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

key_lock = threading.Lock()

# =========================
# LOAD MODEL
# =========================
print("Loading model...")
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(MODEL_DIR)
params = jax.tree.map(lambda x: jax.device_put(x, no_sharding), params)
print("Model loaded.")

# =========================
# CORE BATCH GENERATION
# =========================
def run_batch(prompts):
    global key

    with key_lock:
        key, subkey = jax.random.split(key)

    prompt_ids = pipeline.prepare_inputs(prompts)
    prompt_ids = jax.device_put(prompt_ids, sharding)

    images = pipeline(
        prompt_ids,
        params,
        subkey,
        num_inference_steps=400,
        guidance_scale=np.array([7.5]),
        height=256,
        width=256
    )

    images = images.images.reshape((len(prompts),) + images.images.shape[-3:])

    out = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
    return out

# =========================
# SMART BATCH HANDLER
# =========================
def generate_images(prompts):
    results = []

    for i in range(0, len(prompts), BATCH_SIZE):
        old_chunk = chunk = prompts[i:i + BATCH_SIZE]

        # pad if needed
        if len(chunk) < BATCH_SIZE:
            pad_count = BATCH_SIZE - len(chunk)
            chunk = chunk + chunk[:1] * pad_count

        images = run_batch(chunk)

        # remove padded outputs
        results.extend(images[:len(old_chunk)])

    return results

# =========================
# FLASK
# =========================
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Missing JSON"}), 400

    # support single or batch
    if "prompt" in data:
        prompts = [data["prompt"]]
    elif "prompts" in data:
        prompts = data["prompts"]
    else:
        return jsonify({"error": "Provide 'prompt' or 'prompts'"}), 400

    try:
        images = generate_images(prompts)

        # single image → return PNG
        if len(images) == 1:
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            buf.seek(0)
            return send_file(buf, mimetype="image/png")

        # multiple → zip
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for i, img in enumerate(images):
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                z.writestr(f"{i}.png", img_bytes.getvalue())

        buf.seek(0)
        return send_file(buf, mimetype="application/zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return {"status": "ok", "batch_size": BATCH_SIZE}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)