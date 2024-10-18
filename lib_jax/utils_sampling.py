import jax
from functools import partial


class sampler_from_data:
    def __init__(self, x, key):
        self.x = x
        self.setup()
        self.key = key

    def setup(self):
        @partial(jax.jit, static_argnums=1)
        def generate_samples(key, num_samples):
            if num_samples >= len(self.x):
                points = jax.random.choice(key, self.x, (len(self.x),), replace=False)
            else:
                points = jax.random.choice(key, self.x, (num_samples,), replace=False)
            return points

        # define samples generator
        self.generate_samples = generate_samples
