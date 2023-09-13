# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)

import random
from typing import Optional

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from pydantic import BaseModel

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.util.misc import SEED_MAX, get_random_seed


@invocation("film_grain", title="FilmGrain", tags=["film_grain"], version="1.0.0")
class FilmGrainInvocation(BaseInvocation):
    """Adds film grain to an image"""

    image: ImageField = InputField(description="The image to add film grain to", default=None)
    amount_1: int = InputField(ge=0, le=800, description="Amount of the first noise layer", default=100)
    amount_2: int = InputField(ge=0, le=800, description="Amount of the second noise layer", default=50)
    seed_1: Optional[int] = InputField(ge=0, le=SEED_MAX, description="The first seed to use (omit for random)")
    seed_2: Optional[int] = InputField(ge=0, le=SEED_MAX, description="The second seed to use (omit for random)")
    blur_1: float = InputField(ge=0, le=100, description="The strength of the first noise blur", default=0.5)
    blur_2: float = InputField(ge=0, le=100, description="The strength of the second noise blur", default=0.5)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        mode = image.mode

        if mode == "RGBA":
            image = image.convert("RGB")

        rng = np.random.default_rng(seed=self.seed_1 if self.seed_1 is not None else get_random_seed())
        noise_1 = rng.normal(0, 1, (image.size[1], image.size[0], 3)) * 127.5 * (self.amount_1 / 800.0)
        noise_1 = noise_1 + 127.5
        noise_1 = Image.fromarray(noise_1.astype("uint8"), "RGB")
        noise_1 = noise_1.filter(ImageFilter.GaussianBlur(radius=self.blur_1))

        rng = np.random.default_rng(seed=self.seed_2 if self.seed_2 is not None else get_random_seed())
        noise_2 = rng.normal(0, 1, (image.size[1], image.size[0], 3)) * 127.5 * (self.amount_2 / 800.0)
        noise_2 = noise_2 + 127.5
        noise_2 = Image.fromarray(noise_2.astype("uint8"), "RGB")
        noise_2 = noise_2.filter(ImageFilter.GaussianBlur(radius=self.blur_2))

        image = ImageChops.overlay(image, noise_1)
        image = ImageChops.overlay(image, noise_2)

        if mode == "RGBA":
            image = image.convert("RGBA")

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=None,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image.width,
            height=image.height,
        )
