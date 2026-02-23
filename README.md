# SWA-Orchestrator-for-SGD
A class for scheduling an SGD optimizer and orchestrating SWA

## Stochastic Gradient Descent
Uses the timm library to create a Cosine LR Scheduler with a warm up period. Includes patience logic to end the SGD period on plateau.

## Stochastic Weighted Averaging
Once SGD has plateaued according to the patience parameter, SWA begins. SWA terminates upon convergence or at the end of the SWA training period.
