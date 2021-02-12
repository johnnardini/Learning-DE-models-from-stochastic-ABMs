# Learning DE models for stochastic ABMs
 Code for "Learning differential equation models from stochastic agent-based model simulations," by John Nardini, Ruth Baker, Mat Simpson, and Kevin Flores. Article available at https://arxiv.org/abs/2011.08255 .
 
 All code is implemented using Python (version 3.7.3). The folders listed in this repository implement code to run agent-based models (ABMs), perform equation learning (EQL), or store previously-computed data from ABM simulations.
 
 The **ABM** folder contains the code used to implement the ABMs from our study, which include the birth-death-migration (BDM) process and the susceptible-infected-recovered (SIR) model.  The jupyter notebook `plot_ABM_DE_BDM.ipynb` can be used to simulate the BDM ABM and compare its output to its mean-field model. The jupyter notebook `plot_ABM_DE_SIR.ipynb` can be used to simulate the BDM ABM and compare its output to its mean-field model. Both of these notebooks rely on `ABM_package.py` to implement the ABMs.
 

 
 Please contact John Nardini at jtnardin@ncsu.edu if you have any questions, thank you.
