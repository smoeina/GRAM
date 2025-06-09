# GRAM
This repository provides the implementation for the algorithms presented in the paper "GRAM: An interpretable approach for graph anomaly detection using gradient attention maps".


## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Overview

This repository provides the implementation for the algorithms presented in the paper "GRAM: An interpretable approach for graph anomaly detection using gradient attention maps".
In this work, we proposed the GRAM method as an interpretable approach for anomaly detection for GAD. Specifically, for datasets that consist of both normal and abnormal graph samples and the goal is to distinguish abnormal graphs, we train a VGAE model in an unsupervised manner and then use its encoder to extract graph-level features for computing the anomaly scores. We compare the GRAM method with the following baseline methods:

- **GCNAE** [[1]](https://arxiv.org/pdf/1611.07308).
- **DOMINANT** [[2]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67).
- **GAAN** [[3]](https://dl.acm.org/doi/abs/10.1145/3340531.3412070).
- **CONAD** [[4]](https://par.nsf.gov/servlets/purl/10357529).
- **OC-GNN** [[5]](https://link.springer.com/content/pdf/10.1007/s00521-021-05924-9.pdf).

## Usage

### Running Experiments

1. **Dataset**
   
    PTC[[6]](https://watermark.silverchair.com/bioinformatics_17_1_107.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA40wggOJBgkqhkiG9w0BBwagggN6MIIDdgIBADCCA28GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMQOk2EaaDF1GBDh-pAgEQgIIDQKMevIP1VPlUR2mKepBa4SzH0TjxikC8lmGOD-2frJ0fCO7kg7Q3lc-gXZ-ipe9oVnIp7VfdWMbUiHhKaAccr5PI4mHuTA0u9-QECRRKJjw73qv08p1MsfsbHGF6_G3zB5KIPfzQezkVCfoYvyNgEMos0uvsFyCbBcA-ldIe47bZETKljVYLlgLCq9O9oAFqnI2cFy1QJ9WYtLXYkWpm-oLFOX1Acf8Z9Oo9f6RlxRJR2sKmqi3wGzRdWqU_tps9dUblru9V5Hnt8f_fcQD6BPfOx2U68aQnEDPhPEpujh7m-pT-9RaB177SOSK8bxJlTQn_CrYNq6KXOAnl0usDKs2U_cyBuiN7TfL4WhtZD914maHHlbq1I5dBlqvfSfALC7H8MZf5R0Mch-asn-ZcuVTVGAWsjFBIEjHMhhLG9zM63asbGC9wpSILqg6sBbAfDDtGFlN_NH9CcL6NPnUyVoLkwuJPa-5He1UcT0ADOBSXoZkKLc2C0r4yGiAXzm7lyz0Yf21YFgS0UfK7ZZhA50HyfSC4nwX_nqHROF6LHplCXC07ppcWFzCeF2Yn9kMOx1ryVcFYwtg2QvjjGw_ia9gfVoxDntSSdZse4gdsB3ga1BdDcy6h6VgT_LMGJ0ZfuBTb7EL-7xh-0qzjxgsozs-QawgwsJT0QmP43lT2TvDO73znDwDkFGuhVgZqkrrTEnfA8Y0KwMN6Qx9dH0NHsRgkXioS_CjP9Usrb6NgktvjsHYLJo87zrSiNGJXWlJV0sRza5pkDyri_GZa0Z_x0pj1CLDBQ157HjcaB8y75upDoq_V6HlNutOmNBSDAKgMrNM1hx9miEUv3uS_oX9Sx0aQ32afn614YNqVqaBtfaziKq-qLzich8YfzlUgGZzpERwOTy9zAGwqv6BM4icQ9qfWMOx3A2OQFhsJ4DYsGxImIOrbMdnWlRadBVjLc4pwb7MVi12nCIJgr0MO-Gwz5Ka3CypG7-zJP8PVYblpEHwJChE4yTH8IoxWLM_nnJfKWC-aaPKH-umZMvzxs3OgPgWIvGzz4cjLwuy6CU50nC0OF91CAV14uAs_n-iX3oROoW-K2crgDPM_KlOLAlrLj4o): PTC is a dataset used for predicting the carcinogenicity of organic compounds. It contains chemical compounds labeled as either carcinogenic or non-carcinogenic based on their structural features.

2. **Code**

  Use the main.py to run the experiments.

   

## Acknowledgements

This work is partially supported by National Natural Science Foundation of China (Grant No. 12301117) and the WHU–DKU Collaborative Research Seed, China under Grant WHUDKUZZJJ202207.


## References

[1] [T. N. Kipf and M. Welling, “Variational graph auto-encoders,” NIPS Workshop on Bayesian Deep Learning, 2016.](https://arxiv.org/pdf/1611.07308)

[2] [K. Ding, J. Li, R. Bhanushali, and H. Liu, “Deep anomaly detection on attributed networks,” in Proc. SIAM Int. Conf. Data Mining. SIAM, 2019, pp. 594–602.](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67)

[3] [Z. Chen, B. Liu, M. Wang, P. Dai, J. Lv, and L. Bo, “Generative adversarial attributed network anomaly detection,” in Proc. 29th ACM Int. Conf. Inf. Knowl. Manage., 2020, pp. 1989–1992.](https://dl.acm.org/doi/abs/10.1145/3340531.3412070)

[4] [Z. Xu, X. Huang, Y. Zhao, Y. Dong, and J. Li, “Contrastive attributed network anomaly detection with data augmentation,” in Proc. 26th Pacific-Asia Conf. Knowl. Discov. Data Mining (PAKDD). Springer, 2022, pp. 444–457.](https://par.nsf.gov/servlets/purl/10357529)

[5] [X. Wang, B. Jin, Y. Du, P. Cui, Y. Tan, and Y. Yang, “One-class graph neural networks for anomaly detection in attributed networks,” Neural Comput. Appl., vol. 33, pp. 12 073–12 085, 2021.](https://link.springer.com/content/pdf/10.1007/s00521-021-05924-9.pdf)

[6] [C. Helma, R. D. King, S. Kramer, and A. Srinivasan, “The predictive toxicology challenge 2000–2001,” Bioinformatics, vol. 17, no. 1, pp. 107–108, 2001.](https://watermark.silverchair.com/bioinformatics_17_1_107.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA40wggOJBgkqhkiG9w0BBwagggN6MIIDdgIBADCCA28GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMQOk2EaaDF1GBDh-pAgEQgIIDQKMevIP1VPlUR2mKepBa4SzH0TjxikC8lmGOD-2frJ0fCO7kg7Q3lc-gXZ-ipe9oVnIp7VfdWMbUiHhKaAccr5PI4mHuTA0u9-QECRRKJjw73qv08p1MsfsbHGF6_G3zB5KIPfzQezkVCfoYvyNgEMos0uvsFyCbBcA-ldIe47bZETKljVYLlgLCq9O9oAFqnI2cFy1QJ9WYtLXYkWpm-oLFOX1Acf8Z9Oo9f6RlxRJR2sKmqi3wGzRdWqU_tps9dUblru9V5Hnt8f_fcQD6BPfOx2U68aQnEDPhPEpujh7m-pT-9RaB177SOSK8bxJlTQn_CrYNq6KXOAnl0usDKs2U_cyBuiN7TfL4WhtZD914maHHlbq1I5dBlqvfSfALC7H8MZf5R0Mch-asn-ZcuVTVGAWsjFBIEjHMhhLG9zM63asbGC9wpSILqg6sBbAfDDtGFlN_NH9CcL6NPnUyVoLkwuJPa-5He1UcT0ADOBSXoZkKLc2C0r4yGiAXzm7lyz0Yf21YFgS0UfK7ZZhA50HyfSC4nwX_nqHROF6LHplCXC07ppcWFzCeF2Yn9kMOx1ryVcFYwtg2QvjjGw_ia9gfVoxDntSSdZse4gdsB3ga1BdDcy6h6VgT_LMGJ0ZfuBTb7EL-7xh-0qzjxgsozs-QawgwsJT0QmP43lT2TvDO73znDwDkFGuhVgZqkrrTEnfA8Y0KwMN6Qx9dH0NHsRgkXioS_CjP9Usrb6NgktvjsHYLJo87zrSiNGJXWlJV0sRza5pkDyri_GZa0Z_x0pj1CLDBQ157HjcaB8y75upDoq_V6HlNutOmNBSDAKgMrNM1hx9miEUv3uS_oX9Sx0aQ32afn614YNqVqaBtfaziKq-qLzich8YfzlUgGZzpERwOTy9zAGwqv6BM4icQ9qfWMOx3A2OQFhsJ4DYsGxImIOrbMdnWlRadBVjLc4pwb7MVi12nCIJgr0MO-Gwz5Ka3CypG7-zJP8PVYblpEHwJChE4yTH8IoxWLM_nnJfKWC-aaPKH-umZMvzxs3OgPgWIvGzz4cjLwuy6CU50nC0OF91CAV14uAs_n-iX3oROoW-K2crgDPM_KlOLAlrLj4o)
