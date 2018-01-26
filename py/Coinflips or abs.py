#Simulating Data Tool

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

n = 10
pcoin = 0.62 # actual value of p for coin
results = st.bernoulli(pcoin).rvs(n)
h = sum(results)
print("We observed %s heads out of %s"%(h,n))
print(results)

#Null Hypothesis
p = 0.5 #assumes this is a Fair Coin 
rv = st.binom(n,p) #this accumlates bernoulli trials over the n runs
mu = rv.mean()
sd = rv.std()
print("The expected distribution for a fair coin is mu=%s, sd=%s"%(mu,sd))

#Binomial Test checks to see if two propotions are the same 
#checks to see if when we sum # of heads and compare to the number of flips and compares it to Pvalue. 0.05 also known as type 1 error
#or expects that 5times of 100 we get a false positive
print("binomial test p-value: %s"%st.binom_test(h, n, p))

#Ztest "normal theory" to further validate
z = (h-0.5-mu)/sd
print("normal approximation p-value: - %s"%(2*(1 - st.norm.cdf(z))))

#Permutation (Simulation) Test simulates data then calculates the P from it 
nsamples = 100000
xs = np.random.binomial(n, p, nsamples)
print("simulation p-value - %s"%(2*np.sum(xs >= h)/(xs.size + 0.0)))

#Maximum Likelihood Estimation (MLE) Super common method for freqentist inference 
#does not test, instead we are fitting to estimate P. Sum Results/Leangth of results
#the Bootstrap helps us resample to check the p value and gets us a confidence interval 95% confident falls between ( #, #)
bs_samples = np.random.choice(results, (nsamples, len(results)), replace=True)
bs_ps = np.mean(bs_samples, axis=1)
bs_ps.sort()

print("Maximum Likelihood Estimate: %s"%(np.sum(results)/float(len(results))))
print("Bootstrap CI: (%.4f, %.4f)" % (bs_ps[int(0.025*nsamples)], bs_ps[int(0.975*nsamples)]))

#Bayesian Estamition - this is how you do it all at once:
#when we do this we get back a distrobution not a P
#called Postier 
#magicline plt.show

fig  = plt.figure()
ax = fig.add_subplot(111)

a, b = 10, 10
prior = st.beta(a, b)
post = st.beta(h+a, n-h+b)
ci = post.interval(0.95)
map_ =(h+a-1.0)/(n+a+b-2.0)

xs = np.linspace(0, 1, 100)
ax.plot(prior.pdf(xs), label='Prior')
ax.plot(post.pdf(xs), label='Posterior')
ax.axvline(mu, c='red', linestyle='dashed', alpha=0.4)
ax.set_xlim([0, 100])
ax.axhline(0.3, ci[0], ci[1], c='black', linewidth=2, label='95% CI');
ax.axvline(n*map_, c='blue', linestyle='dashed', alpha=0.4)
ax.legend()
plt.savefig("coin-toss.png")
plt.show







