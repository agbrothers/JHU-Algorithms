import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## LOCAL PACKAGES
import utils
from neighbors import closest_pairs


## ANALYSIS PLOTS

def test_worst_case(n=100):
    cases = range(10, n+1, 10)
    comparisons = []
    m_cases = []
    for n in cases:
        p = utils.sample_p(n)
        m = utils.combinations(n)
        _, k = closest_pairs(p, m)
        comparisons.append(k)
        m_cases.append(m)

    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(111)

    cases = np.array(cases)
    m_cases = np.array(m_cases)
    # program = ax.plot(cases, comparisons, color="orange", label="Program Time")
    program = ax.scatter(cases, comparisons, marker=".", linewidths=0.1, color="#66ABF7", label="Program Time")
    m_log_m = ax.plot(cases, m_cases*np.log2(m_cases), color="#F76666", label="M log M Time")
    m2_log_m = ax.plot(cases, 2*m_cases*np.log2(m_cases), color="#2F2F2F", label="2M log M Time",  alpha=0.8)
        
    fig.suptitle("Time Complexity Analysis", fontsize=10, fontweight='normal', alpha=0.5, y=0.96)
    ax.set_ylabel("Comparisons", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.set_xlabel("Number of Points", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.tick_params(axis='both', which='major', labelsize=3)
    ax.legend(loc=1, prop={'size': 4})

    fig.canvas.draw()
    fig.savefig(f"worst_case_{cases[-1]}.png")
    
    return
    

def test_values_of_m(n=100):
    p = utils.sample_p(n)
    cases = range(10,n+1,5)
    M_comparisons = []
    M_3_4_comparisons = []
    M_2_comparisons = []
    M_4_comparisons = []
    M_8_comparisons = []
    M_cases = []
    for n in cases:
        p = utils.sample_p(n)
        p = utils.sample_p_flat(n)
        M = utils.combinations(n)
        M_cases.append(M)

        _, k = closest_pairs(p, M)
        M_comparisons.append(k)

        _, k = closest_pairs(p, 3*M//4)
        M_3_4_comparisons.append(k)

        _, k = closest_pairs(p, M//2)
        M_2_comparisons.append(k)

        _, k = closest_pairs(p, M//4)
        M_4_comparisons.append(k)

        _, k = closest_pairs(p, M//8)
        M_8_comparisons.append(k)

    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(111)

    cases = np.array(cases)
    M_cases = np.array(M_cases)
    M_ = ax.scatter(cases, M_comparisons, marker=".", edgecolors=None, linewidths=0.001, color="#FFB637", label="M")
    M34 = ax.scatter(cases, M_3_4_comparisons, marker=".", edgecolors=None, linewidths=0.01, color="#FFDD26", label="0.75 M")
    M2 = ax.scatter(cases, M_2_comparisons, marker=".", edgecolors=None, linewidths=0.01, color="#9EF766", label="0.5 M")
    M4 = ax.scatter(cases, M_4_comparisons, marker=".", edgecolors=None, linewidths=0.01, color="#66ABF7", label="0.25 M")
    M8 = ax.scatter(cases, M_8_comparisons, marker=".", edgecolors=None, linewidths=0.01, color="#9A82ED", label="0.125 M")
    # mlogm = ax.plot(cases, M_cases*np.log2(M_cases), color="#F76666", label="M log M Time")
    m2logm = ax.plot(cases, 2*M_cases*np.log2(M_cases), color="#2F2F2F", label="2M log M Time")

    fig.suptitle("Comparisons for M = bin(n, 2)", fontsize=10, fontweight='normal', alpha=0.5, y=0.96)
    ax.set_ylabel("Comparisons", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.set_xlabel("Number of Points", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.tick_params(axis='both', which='major', labelsize=3)
    ax.legend(loc=1, prop={'size': 4})


    fig.canvas.draw()
    fig.savefig(f"m_analysis.png")
    return
    
    COLORS = {
        "blue": "#66ABF7",
        "red": "#F76666",
        "neutral": "#BDBDBD",
        "green": "#9EF766",
        "purple": "#9A82ED",
        "orange": "#FFB637",
        "yellow": "#FFDD26",
        "pink": "#FBAFF9",
        "black": "#2F2F2F",
    }

if __name__ == "__main__":

    test_worst_case(10)
    test_values_of_m(100)
