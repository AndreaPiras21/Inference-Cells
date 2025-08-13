# Inference of lineage hierarchies, growth and drug response mechanisms in cancer cell populations -- without tracking
This repository contains sample scripts referenced in the paper entitled: "Inference of lineage hierarchies, growth, and drug response mechanisms in cell populations without tracking", posted on BiorXiv:

The following scripts are included:

## EGS_script
EGS_script is a PDF file containing a script written in Mathematica. It implements Bayesian inference for a Bienaymé-Galton-Watson branching process considering only a single type (or phenotype). The method uses a Gibbs sampler algorithm that leverages information derived from the probability generating function. The script also includes several examples where we compare the obtained marginal distribution for the parameters to those obtained from exact analytical Bayesian inference.

## BI_GW1_analytical_script
A script written in Mathematica demonstrating Bayesian inference on Bienaymé-Galton-Watson branching processes using an analytical method. This method derives a mathematical expression for the likelihood from the data, based on a 1-type Bienaymé-Galton-Watson model. It then applies Bayes' rule to compute the posterior distribution of the model parameters, and finally derives all the marginal distributions and estimators.

## BI_GW2_analytical_script
This code written in Mathematica extends the "BI_GW1_analytical_script" to the case of two phenotypes.

## GW1_generalizable 
A Python script to numerically simulate a 1-type Bienaymé-Galton-Watson branching process.

## GW2_generalizable 
A Python script to numerically simulate a 2-type Bienaymé-Galton-Watson branching process.

## MCEM_algorithm_GW1
This script written in Python implements the Monte-Carlo Expection-Maximization algorithm described in the main text, using a 1-type Bienaymé-Galton-Watson branching process.


