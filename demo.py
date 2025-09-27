# Matplotlib with appropriate backend
import matplotlib
matplotlib.use('Agg')

# Native
import os, sys
from time import sleep
import torch.nn.functional
import types
import yaml
from argparse import ArgumentParser
from PIL import Image

# 3rd party
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torch.nn.functional as F
from torch.utils import model_zoo
from scipy.spatial import ConvexHull

# Debug
import gc
# from memory_profiler import profile

# Rooot
from device import device
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from super_resolution import SuperResolutionNet
from image import feather_torch, alpha_blending, center_crop, save_tensor_frame_to_png

sys.path.append(f"{os.getenv('HOME')}/IMU/synergize-motion-appearance")
from basicsr.utils.options import ordered_yaml
from basicsr.archs import build_network
from basicsr.demo import load_network

class DefaultOptions():
    def __init__(self):
        self.debug = True
        self.config = f"{os.getenv('HOME')}/IMU/synergize-motion-appearance/options/test.yml"
        # self.config = "./config/voxceleb1-hdam.yaml"
        self.checkpoint = "./checkpoints/voxceleb-hdam.pth.tar"
        # self.upscaler_path = "./upscaler/model.pt"
        self.upscaler_path = None
        self.upscaler_input_frames = 5
        self.source_image = "./upload/source.png" if not self.debug else "./upload/debug.png"
        self.driving_videos = [
            "./data/food0.mp4",
            "./data/food1.mp4",
            "./data/food2.mp4",
            "./data/food3.mp4",
            "./data/food4.mp4",
            "./data/food5.mp4",
            "./data/food6.mp4",
            "./data/food7.mp4",
            "./data/food8.mp4",
        ]
        self.relative = True
        self.adapt_scale = True
        self.find_best_frame = False
        self.best_frame = None
        self.cpu = device == 'cpu' # Checks using torch.cuda.is_available

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    if "synergize-motion-appearance" in config_path:
        return load_synergize_motion(config_path=config_path, cpu=cpu)

    with open(config_path) as f:
        config = yaml.load(f,  Loader=yaml.SafeLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = generator.cuda()
        kp_detector = kp_detector.cuda()

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

def load_synergize_motion(config_path="/home/luuk/IMU/synergize-motion-appearance/options/test.yml", cpu=False, **__):

    def path_root(path, root, oldroot="./"):
        assert path[:len(oldroot)] == oldroot, f"path root mismatch, expected: {oldroot}, found: {path[:len(oldroot)]}"
        return root + path[len(oldroot):]

    with open(config_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        config = yaml.load(f, Loader=Loader)

    # load model
    generator = build_network(config['network_g'])
    load_path = config['path'].get('pretrain_network_g', None)
    if load_path is not None:
        load_path = path_root(load_path, "/home/luuk/IMU/synergize-motion-appearance/")
        param_key = config['path'].get('param_key_g', 'params')
        generator = load_network(generator, load_path, config['path'].get('strict_load_g', True), param_key)
    generator.eval()

    kp_detector = build_network(config['network_motion_estimator'])
    load_path = config['path'].get('pretrain_network_motion_estimator', None)
    if load_path is not None:
        load_path = path_root(load_path, "/home/luuk/IMU/synergize-motion-appearance/")
        param_key = config['path'].get('param_key_m', 'params')
        kp_detector = load_network(kp_detector, load_path, config['path'].get('strict_load_motion_estimator', True), param_key)
    kp_detector.eval()

    if not cpu:
        generator.cuda()
        kp_detector.cuda()
    return generator, kp_detector

def load_upscaler(upscaler_path, cpu=False):
    if 'superres_epoch100-44c6958e' in upscaler_path:
        upscaler = SuperResolutionNet(upscale_factor=3)
        upscaler.load_state_dict(model_zoo.load_url(upscaler_path, torch.device('cpu' if cpu else 'cuda')))
    else:
        upscaler = torch.load(upscaler_path)

    if not cpu:
        upscaler.cuda()
    upscaler.eval()

    return upscaler

def print_globs():
    for var_name, var_val in globals().items():
        size = sys.getsizeof(var_val)
        if not isinstance(var_val, types.FunctionType):
            var_type = str(type(var_val)).split("'", maxsplit=1)[-1]
            var = f"{var_name}: {var_type}"
            print(f"{var:<75}SIZE: {size/1024/1024:.3f}mb")

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False, upscaler=False, upscaler_frames=None, video_writer=None):
    try:
        driving_length = driving_video.count_frames()  # If available, more reliable
    except AttributeError:
        driving_length = driving_video.get_length()    # Fallback method

    frames = 0
    predictions = []
    def append(p):
        p = p.data.cpu().numpy().squeeze(0)
        p = np.transpose(p, [1, 2, 0])   # transpose to (H, W, C)
        p = np.clip(p, a_min=-1, a_max=1)
        if video_writer is not None:
            video_writer.append_data(img_as_ubyte(p)) # convert to uint8
        else:
            predictions.append(img_as_ubyte(p)) # convert to uint8
        nonlocal frames
        frames += 1

    def feather(t):
        return feather_torch(t, each=10)

    def interpolate(x, **kwargs):
        return F.interpolate(x, **kwargs, mode='bilinear', align_corners=False)

    def load(f, size=(256, 256)):
        f = center_crop(f, *size)
        f = resize(f, size)[..., :3] # Ignore alpha channel
        f = torch.tensor(f[None, ...].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            f = f.cuda()
        return f

    with torch.no_grad():

        source = load(alpha_blending(source_image))
        driving0 = load(driving_video.get_data(0))

        kp_source = kp_detector.estimate_kp(source)
        kp_driving_initial = kp_detector.estimate_kp(driving0)

        # Upscaler buffer
        buffer = []
        mod = upscaler_frames // 2

        # Generator loop
        for driving_frame in tqdm(driving_video, total=driving_length):
            driving_frame = load(driving_frame)

            kp_driving = kp_detector.estimate_kp(driving_frame)

            kp_norm = normalize_kp(
                kp_source=kp_source,
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial,
                use_relative_movement=relative,
                use_relative_jacobian=relative,
                adapt_movement_scale=adapt_movement_scale
            )

            dense_motion = kp_detector.estimate_motion_w_kp(kp_source=kp_source, kp_driving=kp_norm, source_image=source)
            out_dict = generator(source, dense_motion, w=1, inference=True)
            p = out_dict['out']

            # Upscale center frame using video SR model with multiple input frames for temporal coherence
            if callable(upscaler):

                buffer.append(feather(p))

                print("Prediction shape: ", p.shape)

                if len(buffer) == upscaler_frames:

                    upscaled = upscaler(torch.stack(buffer[-upscaler_frames:], dim=1))

                    print("Upscaled shape: ", upscaled.shape)

                    if frames==100: # TEST
                        save_tensor_frame_to_png(p, 'test_raw.png')
                        save_tensor_frame_to_png(upscaled, 'test_upscaled.png')

                    if frames < mod:
                        for f in buffer[:mod]:
                            append(feather(interpolate(f, size=upscaled.shape)))

                    append(feather(upscaled))
                    buffer.pop(0)
                    assert len(buffer) == upscaler_frames-1, f"Buffer length exceeded limit: {len(buffer)} / {upscaler_frames}"

                    print(f"Frame {frames} / {driving_length}")

            else:

                append(feather(p))

    if len(buffer) > 0:
        for f in buffer[-mod:]:
            append(feather(interpolate(f, size=upscaled.shape)))

        print(f"Finished: {frames} / {driving_length}")

    assert frames == driving_length, f"Oh boy, we've lost frames! Deepfake is length {frames} and Food is length {driving_length}"

    if video_writer is None:
        return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

def generate(generator, kp_detector, upscaler=False, opt=DefaultOptions(), driver_index=0):
    driver = opt.driving_videos[driver_index]
    source_image = imageio.imread(opt.source_image)

    # reader = imageio.get_reader(driver)
    # fps = reader.get_meta_data()['fps']
    # video = []
    # try:
    #     for im in reader:
    #         video.append(im)
    # except RuntimeError:
    #     pass
    # reader.close()

    # source_image = resize(source_image, (256, 256))[..., :3]
    # video = [resize(frame, (256, 256))[..., :3] for frame in video]

    result_path = f"./results/result-{driver_index}.mp4"

    # TODO: Find best frame method currently NOT working. Add clause in make_animation to distinguish between video_reader and video_writer or numpy array
    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        forward = video[i:]
        backward = video[:(i+1)][::-1]
        forward = make_animation(
            source_image,
            forward,
            generator,
            kp_detector,
            relative=opt.relative,
            adapt_movement_scale=opt.adapt_scale,
            cpu=opt.cpu,
            upscaler=upscaler,
            upscaler_frames=opt.upscaler_input_frames
        )
        backward = make_animation(
            source_image,
            backward, generator,
            kp_detector,
            relative=opt.relative,
            adapt_movement_scale=opt.adapt_scale,
            cpu=opt.cpu,
            upscaler=upscaler,
            upscaler_frames=opt.upscaler_input_frames
        )
        video = backward[::-1] + forward[1:]
        # imageio.mimsave(result_path, [img_as_ubyte(frame) for frame in video], fps=fps)
    else:
        with imageio.get_reader(driver) as reader:
            fps = reader.get_meta_data()['fps']
            with imageio.get_writer(result_path, fps=fps) as video_writer:
                make_animation(
                    source_image,
                    reader,
                    generator,
                    kp_detector,
                    relative=opt.relative,
                    adapt_movement_scale=opt.adapt_scale,
                    cpu=opt.cpu,
                    upscaler=upscaler,
                    upscaler_frames=opt.upscaler_input_frames,
                    video_writer=video_writer,
                )
    print(f"{result_path} saved")

if __name__ == "__main__":
    opt=DefaultOptions()

    parser = ArgumentParser()
    parser.add_argument("--config", default=opt.config, help="path to config")
    parser.add_argument("--checkpoint", default=opt.checkpoint, help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='upload/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='data/food0.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='test-0.mp4', help="path to output")

    parser.add_argument("--relative", dest="relative", action="store_true" if opt.relative else "store_false",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true" if opt.adapt_scale else "store_false",
                        help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true" if opt.find_best_frame else "store_false",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=opt.best_frame,
                        help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true" if device == "cpu" else "store_false", help="cpu mode.")

    parser.add_argument("--upscaler", dest="upscaler_path", action="store_false", help="Path to upscaler model.pt")
    parser.set_defaults(upscaler_path=False)
    parser.add_argument("--upscaler_frames", dest="upscaler_frames", default=1, help="Amount of frames to pass to upscaler")


    opt = parser.parse_args()
    print("options")
    print(opt)

    generator, kp_detector = load_checkpoints(
        config_path=opt.config,
        checkpoint_path=opt.checkpoint,
        cpu=opt.cpu
    )
    if opt.upscaler_path:
        upscaler = load_upscaler(opt.upscaler_path)
    else:
        upscaler = False

    generate(generator, kp_detector, opt)
