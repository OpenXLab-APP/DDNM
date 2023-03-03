#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

import gradio as gr

from app_colorization import create_demo as create_demo_colorization
from app_superresolution import create_demo as create_demo_superresolution

DESCRIPTION = '''# DDNM-HQ

This is an unofficial demo for [https://github.com/wyhuai/DDNM/tree/main/hq_demo](https://github.com/wyhuai/DDNM/tree/main/hq_demo).
'''
if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'''<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.<br/>
<a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
<p/>
'''

MODEL_DIR = pathlib.Path('DDNM/hq_demo/data/pretrained')
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir()
    subprocess.run(shlex.split(
        'wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt'
    ),
                   cwd=MODEL_DIR.as_posix())
    subprocess.run(shlex.split(
        'wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt'
    ),
                   cwd=MODEL_DIR.as_posix())

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem(label='Super-resolution'):
            create_demo_superresolution()
        with gr.TabItem(label='Colorization'):
            create_demo_colorization()
demo.queue(max_size=5, api_open=False).launch()
