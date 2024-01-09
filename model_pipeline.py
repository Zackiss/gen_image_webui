import os
import random
from time import sleep


import numpy
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionControlNetInpaintPipeline
from diffusers import ControlNetModel, UniPCMultistepScheduler
from matplotlib import pyplot as plt

from segment_anything import build_sam, SamPredictor
import cv2
import accelerate
from scipy.ndimage import gaussian_filter
import torch
from toolbox import load_model, load_image, get_grounding_output, show_mask, show_box, make_inpaint_condition, \
    make_canny_condition, request_to_sd_server, log_gpu, headline


class Pipeline:
    def __init__(self, config: dict, web_ui=False):
        self.cfg = config
        self.use_sdxl = False

        # load GroundDINO model
        self.grounding_dino_model = load_model(
            self.cfg["config_file"],
            self.cfg["grounded_checkpoint"],
            device=self.cfg["device"]
        )

        if self.use_sdxl:
            # load SDXL base model
            self.base_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                cache_dir=self.cfg["cache_dir"]
            )
            self.base_pipe.scheduler = UniPCMultistepScheduler.from_config(self.base_pipe.scheduler.config)
            self.base_pipe.enable_model_cpu_offload()

            # load SDXL refiner
            self.refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                cache_dir=self.cfg["cache_dir"]
            )
            self.refiner.scheduler = UniPCMultistepScheduler.from_config(self.refiner.scheduler.config)
            self.refiner.enable_model_cpu_offload()

        if not web_ui:
            # load Controlnet inpaint model
            self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint",
                torch_dtype=torch.float16,
                cache_dir=self.cfg["cache_dir"]
            )

            # load Controlnet canny model
            self.controlnet_canny = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16,
                cache_dir=self.cfg["cache_dir"]
            )

            # load SD 1.5 + Controlnet without canny model
            self.controlnet_pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                self.cfg["sd_checkpoint"],
                use_safetensors=True,
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
                cache_dir=self.cfg["cache_dir"],
            )
            self.controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.controlnet_pipe.scheduler.config
            )
            print(self.cfg["lora_path"])
            self.controlnet_pipe.load_lora_weights(self.cfg["lora_path"])
            self.controlnet_pipe.enable_model_cpu_offload()

            # load SD 1.5 + Controlnet with canny model
            self.controlnet_pipe_with_canny = StableDiffusionControlNetInpaintPipeline.from_single_file(
                self.cfg["sd_checkpoint"],
                use_safetensors=True,
                controlnet=self.controlnet_canny,
                torch_dtype=torch.float16,
                cache_dir=self.cfg["cache_dir"],
                controlnet_conditioning_scale=[1.0, 0.8],
                guidance_scale=3.0,
            )
            self.controlnet_pipe_with_canny.scheduler = UniPCMultistepScheduler.from_config(
                self.controlnet_pipe_with_canny.scheduler.config
            )
            self.controlnet_pipe_with_canny.enable_model_cpu_offload()

    def image_processing_pipeline(
            self, process_index,
            # various of prompts
            det_prompt, inpaint_prompt, negative_prompt,
            detailer_hand_prompt_pos, detailer_hand_prompt_neg,
            detailer_face_prompt_pos, detailer_face_prompt_neg,
            # parameters of controlnet
            box_threshold, text_threshold, num_inference_steps,
            with_canny=False, with_pose=False,
            # parameters of inpaint
            image_pil=None, image_path=None, seed=None, guess_mode=True,
            inpaint_mode="first", invert_mask=False, high_noise_frac=0,
            # sdxl usage, make sure self.use_sdxl is open
            enhance_with_sdxl=False, enhance_with_sdxl_refiner=False,
            # config whether to use ADetailer
            with_detailer=False,
            # config whether to use WebUI or local
            web_ui=False
    ):
        # load image
        image_pil, image = load_image(image_path, image_pil)

        # visualize raw image
        image_pil.save(os.path.join(self.cfg["test_dir"], "raw.jpg"))
        if not image_path:
            image_path = os.path.join(self.cfg["test_dir"], "raw.jpg")

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            self.grounding_dino_model, image, det_prompt, box_threshold, text_threshold, device=self.cfg["device"]
        )

        # initialize SAM
        predictor = SamPredictor(
            build_sam(checkpoint=self.cfg["sam_checkpoint"]).to(self.cfg["device"])
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.cfg["device"])

        # masks: [1, 1, 512, 512]
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.cfg["device"]),
            multimask_output=False,
        )

        # draw target detection output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        plt.savefig(os.path.join(self.cfg["test_dir"], "gsm.jpg"), bbox_inches="tight")

        # inpainting pipeline
        if inpaint_mode == 'merge':
            masks = torch.sum(masks, dim=0).unsqueeze(0)
            masks = torch.where(masks > 0, True, False)
        mask = masks[0][0].cpu().numpy()

        if invert_mask:
            mask = ~mask

        mask_pil = Image.fromarray(mask)
        mask_pil.save(os.path.join(self.cfg["test_dir"], "mask.jpg"))

        image_pil = Image.fromarray(image)

        image = Image.open(image_path)
        image_size_x, image_size_y = image.size
        image_size = (image_size_x, image_size_y)
        mask_pil = mask_pil.resize(image_size)
        image_pil = image_pil.resize(image_size)

        # load Generator
        if not seed and seed != -1:
            seed = random.randint(310721, 67280421)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        if not web_ui:
            if with_canny:
                control_image = make_canny_condition(image_pil)
                control_image.save(os.path.join(self.cfg["test_dir"], "canny.jpg"))
                image = self.controlnet_pipe_with_canny(
                    prompt=inpaint_prompt,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    image=image_pil,
                    mask_image=mask_pil,
                    control_image=control_image,
                    num_inference_steps=num_inference_steps,
                    controlnet_conditioning_scale=0.35,
                    control_guidance_end=0.8,
                    guess_mode=guess_mode
                ).images[0]
            else:
                control_image = make_inpaint_condition(image_pil, mask_pil)
                image = self.controlnet_pipe(
                    prompt=inpaint_prompt,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    image=image_pil,
                    mask_image=mask_pil,
                    control_image=control_image,
                    num_inference_steps=num_inference_steps,
                    controlnet_conditioning_scale=0.35,
                    control_guidance_end=0.8,
                    guess_mode=guess_mode
                ).images[0]
        else:
            if with_canny or with_pose:
                headline("image_processing_pipeline before request_to_sd_server")
                log_gpu()
                image = request_to_sd_server(
                    config=self.cfg,
                    image=image_pil,
                    mask=mask_pil,
                    prompt=inpaint_prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    base_model=self.cfg["WebUI"]["base_model"],
                    with_control=True,
                    control_model=self.cfg["WebUI"][
                        "controlnet_model_{0}".format("canny" if with_canny else "openpose")],
                    control_type="canny" if with_canny else "openpose",
                    inpaint_ctr_model=self.cfg["WebUI"]["inpaint_ctr_model"],
                    with_detailer=with_detailer,
                    detailer_model_face=self.cfg["WebUI"]["detailer_model_face"],
                    detailer_model_hand=self.cfg["WebUI"]["detailer_model_hand"],
                    detailer_hand_prompt_pos=detailer_hand_prompt_pos,
                    detailer_hand_prompt_neg=detailer_hand_prompt_neg,
                    detailer_face_prompt_pos=detailer_face_prompt_pos,
                    detailer_face_prompt_neg=detailer_face_prompt_neg
                )
                # sleep(10)
            else:
                headline("clean pipeline before request_to_sd_server")
                image = request_to_sd_server(
                    config=self.cfg,
                    image=image_pil,
                    mask=mask_pil,
                    prompt=inpaint_prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    base_model=self.cfg["WebUI"]["base_model"],
                    with_control=False,
                    inpaint_ctr_model=self.cfg["WebUI"]["inpaint_ctr_model"],
                    with_detailer=with_detailer,
                    detailer_model_face=self.cfg["WebUI"]["detailer_model_face"],
                    detailer_model_hand=self.cfg["WebUI"]["detailer_model_hand"],
                    detailer_hand_prompt_pos=detailer_hand_prompt_pos,
                    detailer_hand_prompt_neg=detailer_hand_prompt_neg,
                    detailer_face_prompt_pos=detailer_face_prompt_pos,
                    detailer_face_prompt_neg=detailer_face_prompt_neg
                )

        if enhance_with_sdxl and self.use_sdxl:
            image = self.base_pipe(
                prompt=inpaint_prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps + 10,
                denoising_end=high_noise_frac + 0.1,
                strength=0.5
            ).images[0]

        if enhance_with_sdxl_refiner and self.use_sdxl:
            image = self.refiner(
                prompt=inpaint_prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps + 5,
                denoising_start=high_noise_frac - 0.1,
                strength=0.8
            ).images[0]

        image = image.resize(size)
        image.save(os.path.join(self.cfg["test_dir"], "temp_image_{0}.jpg".format(process_index)))
        return image, seed
