# Meeting Notes: 2026-10-06

## People present

- Jarrod Bieber
- Dr. Billy Quarles

## Main topics discussed

- Presented the results of my latest "v10" Colab notebook which utilizes a denser sampling of training systems in the region of the parameter space with low mass ratio $\mu$ and low orbital eccentricity $e_{\rm bin}$. This results in training on more systems with a higher critical semi-major axis $a_c$. This required artificially "enriching" my previously generated training dataset with more sample high-$a_c$ systems, produced in Colab with short 100-orbit REBOUND integrations to provide the necessary dynamical features.
- Unlike the previous incarnation of the notebook, Model B is not retrained over the entire $6 \times 6$ grid of ($e_{\rm bin}, \mu$) combinations. This was likely introducing data leakage before.
- Quarles et al. 2020's Figure 7a is reproduced using both Model B with the dense sampling training, and Model C with the dense sampling and 100-orbit dynamical features. Both models do a much better job at reproducing the Quarles figure and predicting $a_c$ values. The $R^2$ Coefficient of Determination score improves dramatically from 0.5620 in the previous notebook to 0.9347 / 0.9583 for Model B / Model C predictions in the new notebook.


## Decisions made

- We want to expand the models to 

## Questions or confusion

- 

## Tasks assigned to me

- 

## Tasks assigned to others

- N/A

## Next meeting or deadline

- 2026-10-15

## Next steps before the next meeting

- 
