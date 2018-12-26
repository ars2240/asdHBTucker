import re
import matplotlib.pyplot as plt


nTopics = 20
p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")

for i in range(1, 11):
    ofname = 'cancer_py_gvLDA_' + str(nTopics) + '_' + str(i)
    matches = [p.findall(l) for l in open('logs/' + ofname + '.log')]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    perplexity = [float(t[1]) for t in tuples]
    liklihood = [float(t[0]) for t in tuples]
    iter = list(range(0, len(tuples) * 10, 10))
    plt.plot(iter, liklihood, c="black")
    plt.ylabel("log likelihood")
    plt.xlabel("iteration")
    plt.title("Topic Model Convergence")
    plt.grid()
    plt.savefig('plots/' + ofname + '_convergence_likelihood.png')
    plt.close()
