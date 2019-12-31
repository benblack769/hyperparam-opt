# hyperparam-opt
Multifidelity hyperparameter optimization for machine learning and multi-scale algorithms

###

https://arxiv.org/pdf/1406.7758.pdf

Length scale heuristic:

Basically gradient descent to ensure points are not selected redundantly

Set length scale very high.

    if a point is selected within the natural noise level (learned by sklearn optimizer)

        For each dimension, reduce dimension, see how it affects noise level.(shrinking can just be a 0.9 multiplications)

        Shrink length scale of dimension which reduces noise level the most.

    repeat until selected point is not within natural noise level

Error target heuristic

Idea is to try to have equal cost allocated to all fidelity levels

    ratio = cost_low/cost_highest
    do lowest fidelity function 2/ratio[low] times.

    Set error target very low

    if higher fidelity level has not been reached in the last 1/ratio[fid] times, then increase error target such that it would have been executed in the last 1/ratio[fid] times.
