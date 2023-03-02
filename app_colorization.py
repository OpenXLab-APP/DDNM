#!/usr/bin/env python

from __future__ import annotations

import json
import shlex
import subprocess

import gradio as gr


def run(image_path: str, class_index: int, sigma_y: float) -> str:
    out_name = image_path.split('/')[-1].split('.')[0]
    subprocess.run(shlex.split(
        f'python main.py --config confs/inet256.yml --deg colorization --scale 1 --class {class_index} --path_y {image_path} --save_path {out_name} --sigma_y {sigma_y}'
    ),
                   cwd='DDNM/hq_demo')
    return f'DDNM/hq_demo/results/{out_name}/final/00000.png'


def create_demo():
    examples = [
        [
            'sample_images/monarch_gray.png',
            'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
            0,
        ],
        [
            'sample_images/tiger_gray.png',
            'tiger, Panthera tigris',
            0,
        ],
    ]

    with open('imagenet_classes.json') as f:
        imagenet_class_names = json.load(f)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image = gr.Image(label='Input image', type='filepath')
                class_index = gr.Dropdown(label='Class name',
                                          choices=imagenet_class_names,
                                          type='index',
                                          value=950)
                sigma_y = gr.Number(label='sigma_y', value=0, precision=2)
                run_button = gr.Button('Run')
            with gr.Column():
                result = gr.Image(label='Result', type='filepath')

        gr.Examples(
            examples=examples,
            inputs=[
                image,
                class_index,
                sigma_y,
            ],
        )

        run_button.click(
            fn=run,
            inputs=[
                image,
                class_index,
                sigma_y,
            ],
            outputs=result,
        )
    return demo
