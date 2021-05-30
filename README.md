# nam-pytorch
Unofficial PyTorch implementation of Neural Additive Models (NAM) by Agarwal, et al. [[`abs`](https://arxiv.org/abs/2004.13912), [`pdf`](https://arxiv.org/pdf/2004.13912.pdf)]

<img width=600 src="https://github.com/rish-16/nam-pytorch/raw/main/pic.png" />

---

## Installation

You can access `nam-pytorch` via `pip`:

```bash
$ pip install nam-pytorch
```

## Usage

```python
import torch 
from nam_pytorch import NAM

nam = NAM(
    num_features=784,
    link_func="sigmoid"
)

images = torch.rand(32, 784)
pred = nam(images) # [32, 1]
```

## Contributing
As always, if there are any issues with / suggestions for the code, feel free to raise an issue or submit a PR.

## License
[MIT](https://github.com/rish-16/nam-pytorch/blob/main/LICENSE)
