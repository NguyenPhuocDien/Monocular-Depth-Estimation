# Comparability Report

The current evidence supports environment and startup comparability only, not a
paper-result reproduction claim. Official Eigen train/test lists are
byte-identical, and every protocol-sensitive Kaggle argument matches official.

The known 45 test rows without GT match the reference behavior printed in the
official PyTorch README. Final comparability requires full 50-epoch training,
checkpoint lineage and official evaluation metrics.
