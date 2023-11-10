
## The effect of feature normalization methods in radiomics

This repository contains the code for the paper
'The effect of feature normalization methods in radiomics', to be published.

The results are in ./results, the figures for the paper are in ./paper.

To re-generate the results:
- python3 ./experiment.py

Modify ./parameters.py to your needs. It will take around 10 days on a
32-core CPU.

To re-evaluate:
- python3 ./evaluate.py

Note that you can re-evaluate without generating, since the results are
store within this repository.

To regenerate figures:
- python3 ./generateFigures.py




## Normalization

Scheme A: Apply normalization upfront to all data
Scheme B: Apply normalization inside CV (=correct)

#
