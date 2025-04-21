import pandas as pd
import scipy.stats as stats
import numpy as np

top_cs_schools = [
    "Massachusetts Institute of Technology",
    "University of California, Berkeley",
    "Stanford University",
    "Carnegie Mellon University",
    "Princeton University"
]

print("Doctorate Program Results")
print("-" * 15)
print("Top CS Universities:")
print(*top_cs_schools, sep="\n")
print("")

caltech_doctoral = pd.read_csv("caltech_grad.csv")
ut_doctoral = pd.read_csv("ut_grad.csv")

#z-score for 95% confidence interval
z_95 = stats.norm.ppf(1-0.05/2)

def count_top_schools(counts):
    sum = 0
    for school in top_cs_schools:
        sum += counts.get(school, default=0)

    return sum

def confidence_interval(data, level):
    counts = data[level].value_counts()

    top_schools_count = count_top_schools(counts)
    n = counts.sum()
    p_hat = top_schools_count/n
    se = np.sqrt(p_hat * (1-p_hat)/n)

    me = z_95 * se

    return p_hat-me, p_hat+me


low, high = confidence_interval(caltech_doctoral, "doctorate")
print(f"Caltech sends {low*100:.1f}% to {high*100:.1f}% to top CS doctorate programs (95% confidence)")

low, high = confidence_interval(ut_doctoral, "doctorate")
print(f"UT sends {low*100:.1f}% to {high*100:.1f}% to top CS doctorate programs (95% confidence)")


def diff_proportions(data_1, data_2, level):
    counts = data_1[level].value_counts()

    top_schools_count = count_top_schools(counts)
    n1 = counts.sum()
    p_hat1 = top_schools_count/n1

    counts = data_2[level].value_counts()

    top_schools_count = count_top_schools(counts)
    n2 = counts.sum()
    p_hat2 = top_schools_count/n2

    se = np.sqrt(p_hat1*(1-p_hat1)/n1 + p_hat2*(1-p_hat2)/n2)
    
    z = (p_hat1 - p_hat2)/se

    p = 1 - stats.norm.cdf(z)

    diff_p_hat = p_hat1 - p_hat2
    me = z_95 * se

    return p, diff_p_hat-me, diff_p_hat+me


p, low, high = diff_proportions(caltech_doctoral, ut_doctoral, "doctorate")

if p < 0.05:
    print(f"Caltech undergrads have a higher chance of going to top PhD programs compared to UT undergrads (p={p}).")
    print(f"The difference in proportions is {low * 100:.1f} to {high * 100:.1f} pp (95% confidence).")
else:
    print(f"There is not a significant difference in PhD results (p={p}).")

print("")
print("Industry Results")
print("-" * 15)

industry_data = pd.read_csv("industry.csv")

def salary(school):
    n = industry_data[school].sum()
    mean = (industry_data[school] * industry_data["salary"]).sum() / n

    stddev = (industry_data["salary"] - mean) ** 2
    stddev *= industry_data[school]
    stddev = np.sqrt(stddev.sum()/n)
    se = stddev / np.sqrt(n)

    t_95 = stats.t.ppf(1-0.05/2, n-1)
    me = se * t_95

    return mean-me, mean+me

low, high = salary("Caltech")
print(f"Caltech CS undergrads earn {low:.1f}k to {high:.1f} (95% confidence)")

low, high = salary("UT")
print(f"UT CS undergrads earn {low:.1f}k to {high:.1f} (95% confidence)")
print("Note: Data is sourced from Linkedin, not starting salaries")

def diff_means():
    n1 = industry_data["Caltech"].sum()
    mean1 = (industry_data["Caltech"] * industry_data["salary"]).sum() / n1

    stddev1 = (industry_data["salary"] - mean1) ** 2
    stddev1 *= industry_data["Caltech"]
    stddev1 = np.sqrt(stddev1.sum()/n1)

    n2 = industry_data["UT"].sum()
    mean2 = (industry_data["UT"] * industry_data["salary"]).sum() / n2

    stddev2 = (industry_data["salary"] - mean2) ** 2
    stddev2 *= industry_data["UT"]
    stddev2 = np.sqrt(stddev2.sum()/n2)

    se = np.sqrt(stddev1**2/n1 + stddev2**2/n2)

    df = np.min([n1, n2]) - 1
    t_95 = stats.t.ppf(1-0.05/2, df)
    diff_x_bar = mean1 - mean2
    
    me = se * t_95

    t = diff_x_bar / se

    p = 1 - stats.t.cdf(t, df)

    return p, diff_x_bar-me, diff_x_bar+me

p, low, high = diff_means()

if p < 0.05:
    print(f"Caltech CS undergrads earn more than UT CS undergrads (p={p}).")
    print(f"The difference in salary is {low:.1f}k to {high:.1f}k (95% confidence).")
else:
    print(f"There is not a significant difference in salary (p={p}).")