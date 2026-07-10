import numpy as np

def conduct_rejection_sampling(response_candidates, response_rewards, num_samples, beta=0.05):
    """Official Acceptance-Rejection Sampler from LTO paper"""
    candidates = {c: r for c, r in zip(range(len(response_candidates)), response_rewards)}
    accepted = []

    while len(accepted) < num_samples:
        if not candidates:
            break
        max_reward = max(candidates.values())
        to_remove = []
        for c, r in candidates.items():
            u = np.random.uniform()
            if u >= np.exp((r - max_reward) / beta):
                continue
            accepted.append(c)
            to_remove.append(c)

        if len(accepted) == num_samples:
            break

        for c in to_remove:
            if c in candidates:
                candidates.pop(c)

    return [response_candidates[idx] for idx in accepted]s
