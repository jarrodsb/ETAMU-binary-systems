# Meeting Notes: 2026-08-27

## People present

- Jarrod Bieber
- Dr. Billy Quarles

## Main topics discussed

- High level discussion of SPOCK, cimcumstellar binary systems, and the project scope
- Discussed the toy classifier model / package that I built over the summer
- MEGNO was used in the early 2000's because it didn't take much integration time to see changes
- Better to use eccentricity, a1(1+e1) ~ a2(1-e2) as a feature
- It takes longer integration time to see changes for eccentricity
- The file structure of my summer project is fine for now
- Dr Quarles shared a ChatGPT query with me that outlines the next steps
- Dr Quarles shared a dataset with me that can be used as the Test portion of the Train/Test split (Using it for training would result in overfitting, since the dataset already has the "answer" for long integrations)

## Decisions made

- Remove MEGNO, it's too ambiguous of a feature for training our SPOCK-like model
- Keep project on local machine for now, instead of GitHub. uploading things to ETAMU OneDrive is okay for sharing
- We want to use XGBoost classifier like SPOCK uses
- AI usage for this research project should be done entirely in Temporary Mode, etc. so that whichever LLM I use doesn't train on my research

## Questions or confusion

- 

## Tasks assigned to me

- Follow the Starting Advice document that Dr Quarles generated with ChatGPT
- Prepare a notebook that experiments with the predictive power of short-term dynamics (Delta_e, Delta_i)
- Need to get access to the ETAMU cluster for training datasets with different collections of features than what exists currently (The Train portion of Train/Test split)

## Tasks assigned to others

- 

## Next meeting or deadline

- 2026-09-03

## Next steps before the next meeting

- 
