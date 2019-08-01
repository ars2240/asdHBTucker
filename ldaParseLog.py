import re
import matplotlib.pyplot as plt
import numpy as np


nTopics = 20
p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")

ll = np.zeros(10)

for i in range(1, 11):
    ofname = 'cancer_py_gen_gvLDA_' + str(nTopics) + '_' + str(i)
    matches = [p.findall(l) for l in open('logs/' + ofname + '.log')]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    perplexity = [float(t[1]) for t in tuples]
    likelihood = [float(t[0]) for t in tuples]
    iter = list(range(0, len(tuples) * 10, 10))
    plt.plot(iter, likelihood, c="black")
    plt.ylabel("log likelihood")
    plt.xlabel("iteration")
    plt.title("Topic Model Convergence")
    plt.grid()
    plt.savefig('plots/' + ofname + '_convergence_likelihood.png')
    plt.close()
    ll[i-1] = likelihood[-1]

print(np.mean(ll))
