import pandas as pd
import numpy as np
import statsmodels as sm
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
from linearmodels import IV2SLS
from linearmodels import IVLIML
from linearmodels.iv.results import IVModelComparison
import matplotlib.pyplot as plt
from auxiliary.project_auxiliary_plot import *

def get_figure1(country_data):
    country_data=country_data[country_data["tyr05_n"].notna()]
    result1=smf.ols(formula="logpgdp05 ~ tyr05_n  ",data=country_data).fit(cov_type='HC3')
    result2=smf.ols(formula="logpgdp05 ~ ruleoflaw  ",data=country_data).fit(cov_type='HC3')
    result3=smf.ols(formula="ruleoflaw ~ tyr05_n",data=country_data).fit(cov_type='HC3')

    fit_1=result1.fittedvalues
    fit_2=result2.fittedvalues
    fit_3=result3.fittedvalues

    fig, ax = plt.subplots(1,3,figsize=(25,10))

    country_data.plot(kind="scatter",x="tyr05_n",y="logpgdp05",ax=ax[0])
    country_data.plot(kind="scatter",x="ruleoflaw",y="logpgdp05",ax=ax[1])
    country_data.plot(kind="scatter",x="tyr05_n",y="ruleoflaw",ax=ax[2])
    
    ax[0].set_xlabel("Years of schooling",fontsize=15)
    ax[0].set_ylabel("Log GDP per capita",fontsize=15)
    ax[0].set_title("Figure 1-A : Relationship between Log GDP per capita and years of schooling",fontsize=11)
  
    ax[1].set_xlabel("Rule of law",fontsize=15)
    ax[1].set_ylabel("Log GDP per capita",fontsize=15)
    ax[1].set_title("Figure 1-B: Relationship between Log GDP per capita and the rule of law index",fontsize=11)
    
    ax[2].set_xlabel("Years of schooling",fontsize=15)
    ax[2].set_ylabel("Rule of law",fontsize=15)
    ax[2].set_title("Figure 1-C : Relationship between rule of law and years of schooling",fontsize=11)
    
    ax[0].plot(country_data["tyr05_n"],fit_1,linestyle=":",color="r",label="Fitted values")
    ax[1].plot(country_data["ruleoflaw"],fit_2,linestyle=":",color="r", label="fitted values")
    ax[2].plot(country_data["tyr05_n"],fit_3,linestyle=":",color="r", label="fitted values")

    for i , code in enumerate(list(country_data["code"])):
        x0=list(country_data["tyr05_n"])
        x1=list(country_data["ruleoflaw"])
        y = list(country_data["logpgdp05"])
        ax[0].annotate(code,(x0[i],y[i]))
        ax[1].annotate(code,(x1[i],y[i]))
        ax[2].annotate(code,(x0[i],x1[i]))
    
    return ax