# Annealed and Transformed Variational Inference 

> Official code for [Fast Compartment Model Calibration using Annealed and Transformed Variational Inference]

![image1](title/Model_Outline4.PNG)

ATVI adopts surjective function and temperature annealing to avoid inferential challenges in a naive VI approach in calibration tasks. 

This code relies on the NF model package from the SurVAE library.  To start with, you must install the SurVAE library. 

#### Installation

In the folder containing `setup.py`, run
```
pip install .
```

### Experiments
To train or evaluate model, please check /experiments.

#### Acknowledgements

I would like to thank the authors who provide useful repositories. Mainly, 

* https://github.com/didriknielsen/survae_flows
* https://github.com/rtqichen/torchdiffeq

