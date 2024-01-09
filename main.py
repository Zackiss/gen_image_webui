import os
import toml as toml
import torch
from PIL import Image

# from model_pipeline import Pipeline
# from toolbox import process_images_from_zip
from torch.multiprocessing import Pool, set_start_method

from toolbox import get_iamge_paths
from workflow import workflow
from time import sleep
import shutil
import datetime

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def resize_image(input_image_directory='./image_templates_origin', output_image_directory='./image_templates',
                 size=1024):
    # size为512的n倍
    # 清除输出文件夹下的文件
    for file in os.listdir(output_image_directory):
        file_path = os.path.join(output_image_directory, file)
        try:
            os.remove(file_path)
            print(f"Successfully delete: {file_path}")
        except OSError as e:
            print(f"Error occurs when deleting: {file_path}, with message: {e.strerror}")

    for filename in os.listdir(input_image_directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            input_image_path = os.path.join(input_image_directory, filename)
            output_image_path = os.path.join(output_image_directory, filename)
            shutil.copy(input_image_path, output_image_path)

            # original_image = Image.open(input_image_path)  
            # width, height = original_image.size  
            # print(f'Original image size: {width}x{height}')
            # if width > size or height > size:
            #     ratio = min(size/width, size/height)
            #     new_width = int(width * ratio)  
            #     new_height = int(height * ratio)  
            #     resized_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)

            #     width, height = resized_image.size  
            #     print(f'Resized image size: {width}x{height}')  
            #     resized_image.save(output_image_path)  
            # else:
            #     print("没有大于%d阈值，直接搬运" % size)
            #     shutil.copy(input_image_path, output_image_path) 


def process_images_from_zip(config, directory, num_images, web_ui=True):
    # 调整图片大小
    resize_image()
    extracted_image_paths = get_iamge_paths(directory)
    params_list = []
    for image_path in extracted_image_paths:
        for _ in range(num_images):
            # image_path = random.choice(extracted_image_paths)
            if web_ui:
                # workflow(image_path=image_path, config=config)
                params_list.append((image_path, config))
            else:
                workflow(image_path=image_path, config=config)
                pass

    # 改为进程，解决GPU不足的问题
    with Pool(processes=1) as p:
        # log_gpu()
        p.starmap(workflow, params_list)


if __name__ == "__main__":
    begin = datetime.datetime.now()
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    # cfg
    cfg = toml.load("config.toml")
    # make dir
    os.makedirs(cfg["output_dir"], exist_ok=True)
    # start processing
    process_images_from_zip(config=cfg, directory="./image_templates", web_ui=True,
                            num_images=1)
    end = datetime.datetime.now()
    print(end - begin)
