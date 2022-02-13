# COVID-19 case estimation under non-pharmaceutical policies (e.g., social distancing)

This repository provides a reference implementation of methodology and the data of following paper:

> Preparing for a future COVID-19 wave: insights and limitations from a data-driven evaluation of non-pharmaceutical interventions in Germany
> Ashwin Aravindakshan, Jörn Boehnke, Ehsan Gholami, Ashutosh Nayak  
> 2020. See https://www.nature.com/articles/s41598-020-76244-6

This code, provides tools to predict the effect of various Non-Pharmaceutical Interventions (NPI) by estimating the daily number of infected cases of a disease. The data provided is for COVID-19 in Germany for the duration of Feb 18, 2020 to May 7, 2020.

For inqueries please contact Ehsan Gholami (contact: egholami@ucdavis.edu).

## Citing

If you find this code/data useful for your research, please consider citing the following paper:

    @article{aravindakshan2020preparing,
      title={Preparing for a future COVID-19 wave: insights and limitations from a data-driven evaluation of non-pharmaceutical interventions in Germany},
      author={Aravindakshan, Ashwin and Boehnke, J{\"o}rn and Gholami, Ehsan and Nayak, Ashutosh},
      journal={Scientific reports},
      volume={10},
      number={1},
      pages={1--14},
      year={2020},
      publisher={Nature Publishing Group}
    }

## Basic Usage

To estimate the daily cases, in directory `Daily Case Prediciton` execute:  

`python3 predict.py`

To predict various scenarios daily cases, replace `social_distancing.py` and transportation files with corresponding files for desired scenario and run same command.
    
The output will be both printed on screen and saved to directory `output`

## Abstract

To contain the COVID-19 pandemic, several governments introduced strict Non-Pharmaceutical Interventions (NPI) that restricted movement, public gatherings, national and international travel, and shut down large parts of the economy. Yet, the impact of the enforcement and subsequent loosening of these policies on the spread of COVID-19 is not well understood. Accordingly, we measure the impact of NPI on mitigating disease spread by exploiting the spatio-temporal variations in policy measures across the 16 states of Germany. This quasi-experiment identifies each policy’s effect on reducing disease spread. We adapt the SEIR (Susceptible-Exposed- Infected-Recovered) model for disease propagation to include data on daily confirmed cases, intra- and inter-state movement, and social distancing. By combining the model with measures of policy contributions on mobility reduction, we forecast scenarios for relaxing various types of NPIs. Our model finds that, in Germany, policies that mandated contact restrictions (e.g., movement in public space limited to two persons or people co-living), initial business closures (e.g., restaurant closures), stay-at-home orders (e.g., prohibition of non-essential trips), non-essential services (e.g., florists, museums) and retail outlet closures led to the sharpest drops in movement within and across states. Contact restrictions were the most effective at lowering infection rates, while border closures had only minimal effects at mitigating the spread of the disease, even though cross-border travel might have played a role in seeding the disease in the population. We believe that a deeper understanding of the policy effects on mitigating the spread of COVID-19 allows a more accurate forecast of the disease spread when NPIs are (partially) loosened, and thus also better informs policymakers towards making appropriate decisions.
