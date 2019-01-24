# lda.py
#
# Author: Adam Sandler
# Date: 1/22/19
#
# Computes LDA decomposition
#
# Modified from onlineldavb & gensim source code
# https://github.com/blei-lab/onlineldavb/blob/master/onlineldavb.py
#
# Dependencies:
#   Packages: numpy, scipy

import numpy as np
from scipy.special import gammaln, psi

np.random.seed(100000001)
meanchangethresh = 0.001


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if len(alpha.shape) == 1:
        return psi(alpha) - psi(np.sum(alpha))
    return psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]


def update_dir_prior(prior, N, logphat, rho):
    """Update a given prior using Newton's method, described in
    `J. Huang: "Maximum Likelihood Estimation of Dirichlet Distribution Parameters"
    <http://jonathan-huang.org/research/dirichlet/dirichlet.pdf>`_.

    Parameters
    ----------
    prior : list of float
        The prior for each possible outcome at the previous iteration (to be updated).
    N : int
        Number of observations.
    logphat : list of float
        Log probabilities for the current estimation, also called "observed sufficient statistics".
    rho : float
        Learning rate.

    Returns
    -------
    list of float
        The updated prior.

    """
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * psi(np.sum(prior))
    q = -N * psi(prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    if all(rho * dprior + prior > 0):
        prior += rho * dprior

    return prior


class LDA(object):
    def __init__(self, K, V=0, alpha=None, eta=None, kappa=1, tau0=1):
        # K = number of topics
        # alpha = prior for documents over topics
        # eta = prior for topics over words
        # tau0 = a (positive) learning parameter that downweights early iterations
        # kappa = learning rate: exponential decay rate---should be between
        #              (0.5, 1.0] to guarantee asymptotic convergence.

        self.K = K  # save number of topics

        # adjust alpha to right length & format
        if alpha is None:
            self.alpha = 1/self.K*np.ones(self.K)
        elif len(alpha) == 1:
            self.alpha = alpha*np.ones(self.K)
        elif len(alpha) == self.K:
            self.alpha = np.array(alpha)
        else:
            raise Exception('Improper length of alpha')

        self.V = V  # size of vocabulary
        self.D = 0  # number of documents
        self.tau0 = tau0 + 1
        self.eta = eta  # prior over words
        self.kappa = kappa  # learning rate
        self.updatect = 0
        self.rhot = pow(self.tau0 + self.updatect, -self.kappa)
        self.eps = 1e-100  # epsilon, slight perturbation
        self.iterations = 100  # number of iterations

        # Initialize the variational distribution q(beta|lambda)
        self.lam = np.random.gamma(100., 1. / 100., (self.K, self.V))
        self.Elogbeta = dirichlet_expectation(self.lam)
        self.expElogbeta = np.exp(self.Elogbeta)

    def do_e_step(self, x):

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1 * np.random.gamma(100., 1. / 100., (self.D, self.K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        sstats = np.zeros(self.lam.shape)
        # Now, for each document d update that document's gamma and phi
        for d in range(0, self.D):
            # These are mostly just shorthand (but might help cache locality)
            ids = np.nonzero(x[d, :])[0]
            cts = x[d, ids]
            gammad = gamma[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = np.dot(expElogthetad, expElogbetad) + self.eps
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + self.eps
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(gammad - lastgamma))
                if meanchange < meanchangethresh:
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self.expElogbeta

        return gamma, sstats

    def update_lambda(self, x):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.
        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.
        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.
        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self.tau0 + self.updatect, -self.kappa)
        self.rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        gamma, sstats = self.do_e_step(x)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(x, gamma)
        # Update lambda based on documents.
        self.lam = self.lam * (1 - rhot) + rhot * (self.eta + sstats)
        self.Elogbeta = dirichlet_expectation(self.lam)
        self.expElogbeta = np.exp(self.Elogbeta)
        self.updatect += 1

        return gamma, bound

    def approx_bound(self, x, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.
        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.

        score = 0
        Elogtheta = dirichlet_expectation(gamma)

        # E[log p(docs | theta, beta)]
        for d in range(0, self.D):
            ids = np.nonzero(x[d, :])[0]
            cts = np.array(x[d, ids])
            phinorm = np.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self.Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
            score += np.sum(cts * phinorm)
        #             oldphinorm = phinorm
        #             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
        #             print oldphinorm
        #             print n.log(phinorm)
        #             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self.alpha - gamma) * Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self.alpha))
        score += sum(gammaln(np.sum(self.alpha)) - gammaln(np.sum(gamma, axis=1)))

        # Compensate for the subsampling of the population of documents
        score = score

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + np.sum((self.eta - self.lam) * self.Elogbeta)
        score = score + np.sum(gammaln(self.lam) - gammaln(self.eta))
        score = score + np.sum(gammaln(np.sum(self.eta)) - gammaln(np.sum(self.lam, axis=1)))

        return score

    def update_alpha(self, gammat):
        """Update parameters for the Dirichlet prior on the per-document topic weights.

        Parameters
        ----------
        gammat : numpy.ndarray
            Previous topic weight parameters.
        rho : float
            Learning rate.

        Returns
        -------
        numpy.ndarray
            Sequence of alpha parameters.

        """

        N = float(len(gammat))
        logphat = sum(dirichlet_expectation(gamma) for gamma in gammat) / N

        self.alpha = update_dir_prior(self.alpha, N, logphat, self.rhot)

        return self.alpha

    def inference(self, x, collect_sstats=False):
        """Given a chunk of sparse document vectors, estimate gamma (parameters controlling the topic weights)
        for each document in the chunk.

        This function does not modify the model The whole input chunk of document is assumed to fit in RAM;
        chunking of a large corpus must be done earlier in the pipeline. Avoids computing the `phi` variational
        parameter directly using the optimization presented in
        `Lee, Seung: Algorithms for non-negative matrix factorization"
        <https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf>`_.

        Parameters
        ----------
        x : matrix of counts where rows are documents & columns are words
        collect_sstats : bool, optional
            If set to True, also collect (and return) sufficient statistics needed to update the model's topic-word
            distributions.

        Returns
        -------
        (numpy.ndarray, {numpy.ndarray, None})
            The first element is always returned and it corresponds to the states gamma matrix. The second element is
            only returned if `collect_sstats` == True and corresponds to the sufficient statistics for the M step.

        """

        D = x.shape[0]  # number of documents

        # Initialize the variational distribution q(theta|gamma) for the chunk
        gamma = 1 * np.random.gamma(100., 1. / 100., (D, self.K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta)
        else:
            sstats = None
        converged = 0

        # Now, for each document d update that document's gamma and phi
        # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
        # Lee&Seung trick which speeds things up by an order of magnitude, compared
        # to Blei's original LDA-C code, cool!).
        for d in range(0, D):
            ids = np.nonzero(x[d, :])[0]
            cts = x[d, ids]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]

            # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
            # phinorm is the normalizer.
            phinorm = np.dot(expElogthetad, expElogbetad) + self.eps

            # Iterate between gamma and phi until convergence
            for _ in range(self.iterations):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + self.eps
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(gammad - lastgamma))
                if meanchange < meanchangethresh:
                    converged += 1
                    break
            gamma[d, :] = gammad
            if collect_sstats:
                # Contribution of document d to the expected sufficient
                # statistics for the M step.
                sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)

        if collect_sstats:
            # This step finishes computing the sufficient statistics for the
            # M step, so that
            # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
            # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
            sstats *= self.expElogbeta

        return gamma, sstats

    def train(self, x, iters=100):
        # for training LDA decompisition
        # x = dataset, where documents are rows and word counts are columns
        # iters = number of iterations

        x = np.array(x)
        self.V = x.shape[1]  # size of vocabulary
        self.D = x.shape[0]  # number of documents
        self.iterations = iters  # number of iterations

        # adjust eta to right length & format
        if self.eta is None:
            self.eta = 1 / self.V * np.ones(self.V)
        elif len(self.eta) == 1:
            self.eta = self.eta * np.ones(self.V)
        elif len(self.eta) == self.V:
            self.eta = np.array(self.eta)
        else:
            raise Exception('Improper length of eta')

        # Initialize the variational distribution q(beta|lambda)
        self.lam = np.random.gamma(100., 1. / 100., (self.K, self.V))
        self.Elogbeta = dirichlet_expectation(self.lam)
        self.expElogbeta = np.exp(self.Elogbeta)

        for i in range(iters):
            gamma, _ = self.update_lambda(x)
            self.update_alpha(gamma)

        docTop = gamma/np.sum(gamma, axis=1)[:, None]
        topWord = self.lam/np.sum(self.lam, axis=0)[None, :]

        return docTop, topWord

    def predict(self, x, iters=10):
        # for predicting topics of new documents

        self.iterations = iters  # number of iterations

        gamma, _ = self.inference(x)
        docTop = gamma/np.sum(gamma, axis=1)[:, None]
        return docTop


