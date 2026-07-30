Copyright (2025) Databricks, Inc.

This Software includes software developed at Databricks (https://www.databricks.com/) and its use is subject to the included LICENSE file.
By using this repository and the notebooks within, you consent to Databricks collection and use of usage and tracking information in accordance with our privacy policy at www.databricks/privacypolicy.

## Third-party components

This accelerator demonstrates tabular foundation models from third-party vendors. The
Databricks LICENSE applies to the code in this repository; each vendor's model, weights,
and client library are governed by their own licenses, which you must comply with
separately. The notebooks download and/or call these components at runtime — they are
not redistributed in this repository.

- **TabPFN** (Prior Labs) — consumed via the `tabpfn-client` package
  (Apache License 2.0, https://github.com/priorlabs/tabpfn-client). The hosted model is
  subject to Prior Labs' terms; some Prior Labs model weights are separately gated /
  non-commercial. See https://docs.priorlabs.ai/.

- **TabFM** (Google) — self-hosted weights from Hugging Face
  (`google/tabfm-1.0.0-pytorch`, `google/tabfm-1.0.0-jax`). The **weights** are released
  under the **TabFM Non-Commercial License v1.0** (`tabfm-non-commercial-v1.0`) — for
  non-commercial, evaluation-only use; commercial/production use requires a separate
  license from Google. The source code at https://github.com/google-research/tabfm is
  Apache License 2.0. License: https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE.

- **TabICL** (soda-inria / INRIA) — self-hosted weights via the `tabicl` package
  (**BSD 3-Clause License**, https://github.com/soda-inria/tabicl). The repository bundles
  Prior Labs-derived code under Apache License 2.0 in `src/tabicl/forecast`.

Trademarks are the property of their respective owners.
