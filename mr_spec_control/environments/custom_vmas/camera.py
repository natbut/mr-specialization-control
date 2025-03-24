import typing
from typing import Callable, List, Union, Tuple
import torch
from PIL import Image
import numpy as np

from vmas.simulator.core import Box, Entity, World
from vmas.simulator.sensors import Sensor
from vmas.simulator.utils import Color

from vmas.simulator import rendering

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class TopDownCamera(Sensor):
    def __init__(
        self,
        world: World,
        frame_x_dim: float,
        frame_y_dim: float,
        resolution: Tuple[int, int] = (64, 64),
        center: Union[None, torch.Tensor] = None,
        render_color: Union[Color, Tuple[float, float, float]] = Color.LIGHT_GREEN,
        alpha: float = 0.1,
        entity_filter: Callable[[Entity], bool] = lambda _: True,
    ):
        """
        Creates a top-down camera sensor that captures a rendered view of the environment.

        Parameters:
        - world: The VMAS world object.
        - x_dim, y_dim: Dimensions of the camera view.
        - resolution: Output image resolution (width, height).
        - center: The center of the camera view (default is the agent's position).
        - render_color: Color of rendered entities.
        - alpha: Transparency of rendered entities.
        - entity_filter: Function to filter entities to render.
        """
        super().__init__(world)
        self.frame_x_dim = frame_x_dim
        self.frame_y_dim = frame_y_dim
        self.resolution = resolution
        self.center = center  # Defaults to world center if None
        self.render_color = render_color
        self.alpha = alpha
        self.entity_filter = entity_filter

        if self.center is None:
            self.center = torch.zeros((world.batch_dim, 2), device=world.device)

        self.viewer= rendering.Viewer(self.resolution[0], self.resolution[1], visible=False)

        self._last_image = self._produce_image()

    def _produce_image(self):
        views = []

        for env in range(self._world.batch_dim):
            c_x, c_y = self.center[env]

            self.viewer.set_bounds(
                c_x - self.frame_x_dim / 2, c_x + self.frame_x_dim / 2,
                c_y - self.frame_y_dim / 2, c_y + self.frame_y_dim / 2
            )

            # Render filtered entities
            for entity in self._world._agents + self._world._landmarks:
                if not self.entity_filter(entity):
                    continue

                pos = entity.state.pos[env]  # Extract position for current env
                if c_x - self.frame_x_dim / 2 <= pos[0] <= c_x + self.frame_x_dim / 2 and \
                   c_y - self.frame_y_dim / 2 <= pos[1] <= c_y + self.frame_y_dim / 2:

                    # Create geometry for rendering
                    geom = entity.shape.get_geometry()
                    geom.set_color(*entity.color, alpha=1.0)
                    xform = rendering.Transform()
                    xform.set_translation(pos[0], pos[1])
                    geom.add_attr(xform)
                    self.viewer.add_onetime(geom)

            # Render and store result
            image = self.viewer.render(return_rgb_array=True)
            views.append(image)

        return torch.from_numpy(np.stack(views, axis=0))


    def to(self, device: torch.device):
        self._last_image = self._last_image.to(device)

    def measure(self, new_center=None):
        """Generates a top-down view of the environment as an image."""

        if new_center is None:
            self.center = self.agent.state.pos
        else:
            self.center = new_center

        self._last_image = self._produce_image()

        return self._last_image

    def save_image(self, filename: str):
        """Saves the last rendered images to a file."""
        if self._last_image is None:
            raise ValueError("No image has been captured yet. Call measure() first.")

        for i, im_arr in enumerate(self._last_image):
            image = Image.fromarray(np.uint8(im_arr))  # Convert NumPy array to an image
            image.save(filename+str(i)+".png")
            print(f"Top-down view saved to {filename}")

    def render(self, env_index: int = 0) -> "List[Geom]":
        """Returns a rendering of the image view region"""
        if self._last_image is None:
            return []

        region = Box(self.frame_x_dim/2, self.frame_y_dim/2).get_geometry()
        region.set_color(*self.render_color.value, self.alpha)
        pos_region = self.agent.state.pos[env_index]
        xform = rendering.Transform()
        xform.set_translation(*pos_region)
        region.add_attr(xform)

        return [region]
