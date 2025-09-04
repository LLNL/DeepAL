# DeepActiveLearning

This project is an implementation of the Deep Active Learning (DeepAL) framwork from the paper Deep Active Learning based Experimental Design to Uncover Synergistic Genetic Interactions for Host Targeted Therapeutics


# Dependencies 
The required library for this project includes:
matplotlib, numpy, scipy, pandas, scikit_learn, torch, seaborn, torch-geometric. See requirements.txt for the minimal package versions.

# Setup
To set up the package as a python module run "pip install ." in terminal 

# Running the simple experiments
Step 1: Generate the simulated data which include generating
       (1) The simulated interaction matrix (up to number of replicates selected)
       (2) The simulated hetergeneous knolwedge graph 
python simulated_data.py

Step 2: Run the DeepAL framework (in terminal)
    (1) base model:
        cd scripts
        bash run.sh
    (2) ensemble model:
        cd scripts
        bash run_mp.sh
        
Step 3: Run the post analysis notebook (notebooks/result.ipynb)


# Acknowledgments

DeepAL has been developed under the financial support of:

- LLNL Laboratory Directed Research and Development Program 


# Contributors
DeepAL is written by Haonan Zhu(zhu18@llnl.gov), Mary Silva(silva223@llnl.gov) and Jose Cadena(cadenapico1@llnl.gov) from LLNL and has received important feedback from Braden Soper (LLNL), Priyadip Ray (LLNL), and Jeff Drocco (LLNL).

# Citing DeepAL

If you are referencing DeepAL in a publication, please cite the following paper:

* Haonan Zhu, Mary Silva, Jose Cadena, Braden Soper, Michał Lisicki, Braian Peetoom, Sergio E. Baranzini, 
Shivshankar Sundaram, Priyadip Ray and Jeff Drocco [**Deep Active Learning based Experimental Design to
Uncover Synergistic Genetic Interactions for Host Targeted Therapeutics**](https://arxiv.org/abs/2502.02552), 2025. LLNL-JRNL-872100. 

Or, see`DeepAL.bib` for the raw BibTeX file.

# Copyright
Copyright (c) 2023-2025, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-2010881. DeepAL is free software; you can modify it and/or redistribute it under the terms of the MIT license. All new contributions must be made under this license. See COPYRIGHT for complete copyright and license information.
