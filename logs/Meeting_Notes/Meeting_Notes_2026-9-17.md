# Meeting Notes: 2026-09-17

## People present

- Jarrod Bieber
- Dr. Billy Quarles

## Main topics discussed

- Reviewed updates to the experimental notebook v6
    - Hyperparameter experiment shows that a conservative learning rate works best
    - Fixes to the shape of the ground truth and model prediction plots for ap/abin vs lambda_p
- Reviewed the initial progress of the Manuscript, utilizing the AAS Overleaf template, and building out a skeleton for Section 2, with the first 2 subsections written.
- Discussed why a_p should be set to 1.0, not a_bin
- Discussed secular forcing as described in Quarles et al. 2026
- Ultimate goal is to predict the full mu vs e_bin map, (Quarles 2020 fig 7), and possibly also incorporate inclination as well.

## Decisions made

- 

## Questions or confusion

- 

## Tasks assigned to me

- Try different color map for ap/abin vs lambda_p plots, to see if easier to read (try BWR, a couple others from Matplotlib)
- Change the definition of a_p = 1.0 rather tha a_bin. This should also make the short integrations much faster (maybe short integrations can be pushed to 10,000 years?). Radius of the star will matter less. No need to have a tidal model due to close approach of the planet. A planet ejection should not depend on a_crit, but rather related to a_bin.
- Incorporate secular forcing of the planet eccentricity into our initial condition model. See Quarles et al. 2026 section 2.1.
- Continue writing Section 2 of the Manuscript

## Tasks assigned to others

- N/A

## Next meeting or deadline

- 2026-9-24

## Next steps before the next meeting

- 
