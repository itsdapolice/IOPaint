#!/usr/bin/env python3
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import hashlib
import traceback
from dataclasses import dataclass


import imghdr
import io
import logging
import multiprocessing
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from lama_cleaner.const import *
from lama_cleaner.file_manager import FileManager
from lama_cleaner.model.utils import torch_gc
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.plugins import (
    InteractiveSeg,
    RemoveBG,
    AnimeSeg,
    build_plugins,
)
from lama_cleaner.schema import Config


try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

from flask import (
    Flask,
    request,
    send_file,
    cli,
    make_response,
    send_from_directory,
    jsonify,
)
from flask_socketio import SocketIO

# Disable ability for Flask to display warning about using a development server in a production environment.
# https://gist.github.com/jerblack/735b9953ba1ab6234abb43174210d356
cli.show_server_banner = lambda *_: None
from flask_cors import CORS

from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
    pil_to_bytes,
    is_mac,
)

NUM_THREADS = str(multiprocessing.cpu_count())

# fix libomp problem on windows https://github.com/Sanster/lama-cleaner/issues/56
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "app/build")


class NoFlaskwebgui(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Running on http:" in msg:
            print(msg[msg.index("Running on http:") :])

        return (
            "flaskwebgui-keep-server-alive" not in msg
            and "socket.io" not in msg
            and "This is a development server." not in msg
        )


logging.getLogger("werkzeug").addFilter(NoFlaskwebgui())

app = Flask(__name__, static_folder=os.path.join(BUILD_DIR, "static"))
app.config["JSON_AS_ASCII"] = False
CORS(app, expose_headers=["Content-Disposition", "X-seed", "X-Height", "X-Width"])

sio_logger = logging.getLogger("sio-logger")
sio_logger.setLevel(logging.ERROR)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


@dataclass
class GlobalConfig:
    model_manager: ModelManager = None
    file_manager: FileManager = None
    output_dir: Path = None
    input_image_path: Path = None
    disable_model_switch: bool = False
    is_desktop: bool = False
    image_quality: int = 95
    plugins = {}

    @property
    def enable_auto_saving(self) -> bool:
        return self.output_dir is not None

    @property
    def enable_file_manager(self) -> bool:
        return self.file_manager is not None


global_config = GlobalConfig()


def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w


def diffuser_callback(i, t, latents):
    socketio.emit("diffusion_progress", {"step": i})


@app.route("/save_image", methods=["POST"])
def save_image():
    if global_config.output_dir is None:
        return "--output-dir is None", 500

    input = request.files
    filename = request.form["filename"]
    origin_image_bytes = input["image"].read()  # RGB
    # ext = get_image_ext(origin_image_bytes)
    ext = "png"
    image, alpha_channel, infos = load_img(origin_image_bytes, return_info=True)
    save_path = (global_config.output_dir / filename).with_suffix(f".{ext}")

    if alpha_channel is not None:
        if alpha_channel.shape[:2] != image.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(image.shape[1], image.shape[0])
            )
        image = np.concatenate((image, alpha_channel[:, :, np.newaxis]), axis=-1)

    pil_image = Image.fromarray(image).convert("RGBA")

    img_bytes = pil_to_bytes(
        pil_image,
        ext,
        quality=global_config.image_quality,
        infos=infos,
    )
    try:
        with open(save_path, "wb") as fw:
            fw.write(img_bytes)
    except:
        return f"Save image failed: {traceback.format_exc()}", 500

    return "ok", 200


@app.route("/medias/<tab>")
def medias(tab):
    if tab == "image":
        response = make_response(jsonify(global_config.file_manager.media_names), 200)
    else:
        response = make_response(
            jsonify(global_config.file_manager.output_media_names), 200
        )
    # response.last_modified = thumb.modified_time[tab]
    # response.cache_control.no_cache = True
    # response.cache_control.max_age = 0
    # response.make_conditional(request)
    return response


@app.route("/media/<tab>/<filename>")
def media_file(tab, filename):
    if tab == "image":
        return send_from_directory(global_config.file_manager.root_directory, filename)
    return send_from_directory(global_config.file_manager.output_dir, filename)


@app.route("/media_thumbnail/<tab>/<filename>")
def media_thumbnail_file(tab, filename):
    args = request.args
    width = args.get("width")
    height = args.get("height")
    if width is None and height is None:
        width = 256
    if width:
        width = int(float(width))
    if height:
        height = int(float(height))

    directory = global_config.file_manager.root_directory
    if tab == "output":
        directory = global_config.file_manager.output_dir
    thumb_filename, (width, height) = global_config.file_manager.get_thumbnail(
        directory, filename, width, height
    )
    thumb_filepath = f"{app.config['THUMBNAIL_MEDIA_THUMBNAIL_ROOT']}{thumb_filename}"

    response = make_response(send_file(thumb_filepath))
    response.headers["X-Width"] = str(width)
    response.headers["X-Height"] = str(height)
    return response


@app.route("/inpaint", methods=["POST"])
def process():
    input = request.files
    # RGB
    origin_image_bytes = input["image"].read()
    image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_info=True)

    mask, _ = load_img(input["mask"].read(), gray=True)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    if image.shape[:2] != mask.shape[:2]:
        return (
            f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]}",
            400,
        )

    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    form = request.form
    size_limit = max(image.shape)

    if "paintByExampleImage" in input:
        paint_by_example_example_image, _ = load_img(
            input["paintByExampleImage"].read()
        )
        paint_by_example_example_image = Image.fromarray(paint_by_example_example_image)
    else:
        paint_by_example_example_image = None

    config = Config(
        ldm_steps=form["ldmSteps"],
        ldm_sampler=form["ldmSampler"],
        hd_strategy=form["hdStrategy"],
        zits_wireframe=form["zitsWireframe"],
        hd_strategy_crop_margin=form["hdStrategyCropMargin"],
        hd_strategy_crop_trigger_size=form["hdStrategyCropTrigerSize"],
        hd_strategy_resize_limit=form["hdStrategyResizeLimit"],
        prompt=form["prompt"],
        negative_prompt=form["negativePrompt"],
        use_croper=form["useCroper"],
        croper_x=form["croperX"],
        croper_y=form["croperY"],
        croper_height=form["croperHeight"],
        croper_width=form["croperWidth"],
        use_extender=form["useExtender"],
        extender_x=form["extenderX"],
        extender_y=form["extenderY"],
        extender_height=form["extenderHeight"],
        extender_width=form["extenderWidth"],
        sd_scale=form["sdScale"],
        sd_mask_blur=form["sdMaskBlur"],
        sd_strength=form["sdStrength"],
        sd_steps=form["sdSteps"],
        sd_guidance_scale=form["sdGuidanceScale"],
        sd_sampler=form["sdSampler"],
        sd_seed=form["sdSeed"],
        sd_freeu=form["enableFreeu"],
        sd_freeu_config=json.loads(form["freeuConfig"]),
        sd_lcm_lora=form["enableLCMLora"],
        sd_match_histograms=form["sdMatchHistograms"],
        cv2_flag=form["cv2Flag"],
        cv2_radius=form["cv2Radius"],
        paint_by_example_example_image=paint_by_example_example_image,
        p2p_image_guidance_scale=form["p2pImageGuidanceScale"],
        enable_controlnet=form["enable_controlnet"],
        controlnet_conditioning_scale=form["controlnet_conditioning_scale"],
        controlnet_method=form["controlnet_method"],
        powerpaint_task=form["powerpaintTask"],
    )

    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 99999999)

    logger.info(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)

    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

    start = time.time()
    try:
        res_np_img = global_config.model_manager(image, mask, config)
    except RuntimeError as e:
        if "CUDA out of memory. " in str(e):
            # NOTE: the string may change?
            return "CUDA out of memory", 500
        elif "Invalid buffer size" in str(e) and is_mac():
            return "Out of memory", 500
        else:
            logger.exception(e)
            return f"{str(e)}", 500
    finally:
        logger.info(f"process time: {(time.time() - start) * 1000}ms")
        torch_gc()

    res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    ext = get_image_ext(origin_image_bytes)

    bytes_io = io.BytesIO(
        pil_to_bytes(
            Image.fromarray(res_np_img),
            ext,
            quality=global_config.image_quality,
            infos=exif_infos,
        )
    )

    response = make_response(
        send_file(
            # io.BytesIO(numpy_to_bytes(res_np_img, ext)),
            bytes_io,
            mimetype=f"image/{ext}",
        )
    )
    response.headers["X-Seed"] = str(config.sd_seed)

    socketio.emit("diffusion_finish")
    return response


@app.route("/run_plugin", methods=["POST"])
def run_plugin():
    form = request.form
    files = request.files
    name = form["name"]
    if name not in global_config.plugins:
        return "Plugin not found", 500

    origin_image_bytes = files["image"].read()  # RGB
    rgb_np_img, alpha_channel, infos = load_img(
        origin_image_bytes, return_info=True
    )

    start = time.time()
    try:
        form = dict(form)
        if name == InteractiveSeg.name:
            img_md5 = hashlib.md5(origin_image_bytes).hexdigest()
            form["img_md5"] = img_md5
        bgr_res = global_config.plugins[name](rgb_np_img, files, form)
    except RuntimeError as e:
        torch.cuda.empty_cache()
        if "CUDA out of memory. " in str(e):
            # NOTE: the string may change?
            return "CUDA out of memory", 500
        else:
            logger.exception(e)
            return "Internal Server Error", 500

    logger.info(f"{name} process time: {(time.time() - start) * 1000}ms")
    torch_gc()

    if name == InteractiveSeg.name:
        return make_response(
            send_file(
                io.BytesIO(numpy_to_bytes(bgr_res, "png")),
                mimetype="image/png",
            )
        )

    if name in [RemoveBG.name, AnimeSeg.name]:
        rgb_res = bgr_res
        ext = "png"
    else:
        rgb_res = cv2.cvtColor(bgr_res, cv2.COLOR_BGR2RGB)
        ext = get_image_ext(origin_image_bytes)
        if alpha_channel is not None:
            if alpha_channel.shape[:2] != rgb_res.shape[:2]:
                alpha_channel = cv2.resize(
                    alpha_channel, dsize=(rgb_res.shape[1], rgb_res.shape[0])
                )
            rgb_res = np.concatenate(
                (rgb_res, alpha_channel[:, :, np.newaxis]), axis=-1
            )

    response = make_response(
        send_file(
            io.BytesIO(
                pil_to_bytes(
                    Image.fromarray(rgb_res),
                    ext,
                    quality=global_config.image_quality,
                    infos=infos,
                )
            ),
            mimetype=f"image/{ext}",
        )
    )
    return response


@app.route("/server_config", methods=["GET"])
def get_server_config():
    return {
        "plugins": list(global_config.plugins.keys()),
        "enableFileManager": global_config.enable_file_manager,
        "enableAutoSaving": global_config.enable_auto_saving,
        "enableControlnet": global_config.model_manager.enable_controlnet,
        "controlnetMethod": global_config.model_manager.controlnet_method,
        "disableModelSwitch": global_config.disable_model_switch,
        "isDesktop": global_config.is_desktop,
    }, 200


@app.route("/models", methods=["GET"])
def get_models():
    return [it.model_dump() for it in global_config.model_manager.scan_models()]


@app.route("/model")
def current_model():
    return (
        global_config.model_manager.current_model,
        200,
    )


@app.route("/model", methods=["POST"])
def switch_model():
    if global_config.disable_model_switch:
        return "Switch model is disabled", 400

    new_name = request.form.get("name")
    if new_name == global_config.model_manager.name:
        return "Same model", 200

    try:
        global_config.model_manager.switch(new_name)
    except Exception as e:
        traceback.print_exc()
        error_message = f"{type(e).__name__} - {str(e)}"
        logger.error(error_message)
        return f"Switch model failed: {error_message}", 500
    return f"ok, switch to {new_name}", 200


@app.route("/")
def index():
    return send_file(os.path.join(BUILD_DIR, "index.html"))


@app.route("/inputimage")
def get_cli_input_image():
    if global_config.input_image_path:
        with open(global_config.input_image_path, "rb") as f:
            image_in_bytes = f.read()
        return send_file(
            global_config.input_image_path,
            as_attachment=True,
            download_name=Path(global_config.input_image_path).name,
            mimetype=f"image/{get_image_ext(image_in_bytes)}",
        )
    else:
        return "No Input Image"


def start(
    host: str,
    port: int,
    model: str,
    no_half: bool,
    cpu_offload: bool,
    disable_nsfw_checker,
    cpu_textencoder: bool,
    device: Device,
    gui: bool,
    disable_model_switch: bool,
    input: Path,
    output_dir: Path,
    quality: int,
    enable_interactive_seg: bool,
    interactive_seg_model: InteractiveSegModel,
    interactive_seg_device: Device,
    enable_remove_bg: bool,
    enable_anime_seg: bool,
    enable_realesrgan: bool,
    realesrgan_device: Device,
    realesrgan_model: RealESRGANModel,
    enable_gfpgan: bool,
    gfpgan_device: Device,
    enable_restoreformer: bool,
    restoreformer_device: Device,
):
    if input:
        if not input.exists():
            logger.error(f"invalid --input: {input} not exists")
            exit()
        if input.is_dir():
            logger.info(f"Initialize file manager")
            file_manager = FileManager(app)
            app.config["THUMBNAIL_MEDIA_ROOT"] = input
            app.config["THUMBNAIL_MEDIA_THUMBNAIL_ROOT"] = os.path.join(
                output_dir, "lama_cleaner_thumbnails"
            )
            file_manager.output_dir = output_dir
            global_config.file_manager = file_manager
        else:
            global_config.input_image_path = input

    global_config.image_quality = quality
    global_config.disable_model_switch = disable_model_switch
    global_config.is_desktop = gui
    build_plugins(
        global_config,
        enable_interactive_seg,
        interactive_seg_model,
        interactive_seg_device,
        enable_remove_bg,
        enable_anime_seg,
        enable_realesrgan,
        realesrgan_device,
        realesrgan_model,
        enable_gfpgan,
        gfpgan_device,
        enable_restoreformer,
        restoreformer_device,
        no_half,
    )
    if output_dir:
        output_dir = output_dir.expanduser().absolute()
        logger.info(f"Image will auto save to output dir: {output_dir}")
        if not output_dir.exists():
            logger.info(f"Create output dir: {output_dir}")
            output_dir.mkdir(parents=True)
        global_config.output_dir = output_dir

    global_config.model_manager = ModelManager(
        name=model,
        device=torch.device(device),
        no_half=no_half,
        disable_nsfw=disable_nsfw_checker,
        sd_cpu_textencoder=cpu_textencoder,
        cpu_offload=cpu_offload,
        callback=diffuser_callback,
    )

    if gui:
        from flaskwebgui import FlaskUI

        ui = FlaskUI(
            app,
            socketio=socketio,
            width=1200,
            height=800,
            host=host,
            port=port,
            close_server_on_exit=True,
            idle_interval=60,
        )
        ui.run()
    else:
        socketio.run(
            app,
            host=host,
            port=port,
            allow_unsafe_werkzeug=True,
        )
