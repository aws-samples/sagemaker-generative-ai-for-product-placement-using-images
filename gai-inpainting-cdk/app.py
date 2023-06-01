#!/usr/bin/env python3
import os
import aws_cdk as cdk
from cdk.gai_inpainting_sagemaker import GenAiInpaintStack

app = cdk.App()
GenAiInpaintStack(app, "GenAiInpaintStack",)

app.synth()