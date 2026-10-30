# Meeting Notes: 2026-10-29

## People present

- Jarrod Bieber
- Dr. Billy Quarles

## Main topics discussed

- The Google Colab results of the first training dataset that I produced in Regulus.
- Ways to improve the prediction parity plot

## Decisions made

- Our main research phase will conclude by around February, and I will start writing up a paper around that time.
- We need to find some useful element or procedure to include in my paper, other than the main scientific finding, that may be able to boost my paper's citation count over time.

## Questions or confusion

- I had some initial confusion about how ingesting different proportions of the Quarles et al. 2020 700 million file Zenodo dataset as part of generating my 200,000-row training dataset would be useful.
- I had some initial confusion trying to understand how incremnting our test data configurations by cosine squared would lead to a more dense sampling of inclinations in the mid-range of inclinations (~ 90 degrees).

## Tasks assigned to me

- generate a plot of numbers 0 to 1 uniformly, representing the cosine squared, multiply by a random 1 or -1, and then take the arc-cosine square root, and plot the histogram. Convert from radians to degrees. See if they bunch up around midrange inclinations. (90 degrees)
- Assuming the above method works, apply the cosine squared function to sample more midrange inclinations, along with the low mu sampling
- Find a way to intuitively plot the varying dominance / strength of the features
- Have the dataset generation script evaluate the feature importance of individual sub grids of mu and e_bin, and write to a file as it goes.

## Tasks assigned to others

- N/A

## Next meeting or deadline

- 2026-11-03 (Tuesday)

## Next steps before the next meeting

- I should first focus on improving the parity plot in my notebook (improve the model's prediction capability). Other aspects such as data ingestion rates and portraying feature dominance can come later.
