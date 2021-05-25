from generative_models import generate_lca
import numpy as np

p = np.array([0.3, 0.7])
x_acc = np.array([0., 0.])
d = generate_lca(10, p, x_acc, 0.3, 0.2, 0.2, 0.4)
print(d)
breakpoint()
