from cmx import doc, md, csv
doc @ """
# MPI and pyTorch Threading Performance

TorchScript threading performance can be found [here](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)

To test:
"""
with doc:
  import timeit
  runtimes = []
  threads = [1] + [t for t in range(2, 49, 2)]
  for t in threads:
      torch.set_num_threads(t)
      r = timeit.timeit(setup = "import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=100)
      runtimes.append(r)
  # ... plotting (threads, runtimes) ...
  
