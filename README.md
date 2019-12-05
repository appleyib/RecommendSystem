# RecommendSystem
A Recommended System based on MovieLens data.

-cfn: add collaborative features (optional)
-contn: use SVD to decompose content features (optional)
N: the number of columns for latent collaborative features after SVD
M: the number of columns for content features after SVD

command line: python training.py [-cfn N] [contn M]

table:
rf_max_depth	rf_n	cf_n	cf_filter	accuracy
2		20	NA			0.70297
2		20	20	70		0.70280
2		20	200	70		0.70170
5		20	100	55		0.70080