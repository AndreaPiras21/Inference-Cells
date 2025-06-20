# Inference-Cells
This repository contains some of the scripts used in the project entitled: "Inference of lineage hierarchies, growth, and drug response mechanisms in cell populations without tracking."

The following scripts are included:

## EGS_script
EGS_script is a PDF file containing a script written in Mathematica. It implements a version of our Bayesian inference method for a Galton-Watson model considering only a single phenotype. The method uses a Gibbs sampler algorithm that leverages information derived from the probability generating function. The script also includes several examples where we compare the numerical marginals obtained using this Bayesian approach with the analytical marginals derived from our exact analytical method.

## BI_GW1_analytical_script
A script demonstrating Bayesian inference using our analytical method. This method derives a mathematical expression for the likelihood from the data, based on a 1-type Galton-Watson model. It then applies Bayes' rule to compute the posterior distribution of the model parameters, and finally derives all the marginal distributions and estimators.

## BI_GW2_analytical_script
This code extend the "BI_GW1_analytical_script" to the case of two phenotypes 

## GW1_generalizable 
A script to simulate a 1-type Galton-Watson model 

## GW2_generalizable 
A script to simulate a 2-type Galton-Watson model

## MCEM_algorithm_GW1
This script implement the Monte-Carlo Expection-Maximization algorithm described in the main text, using a 1-type Galton-Watson model


