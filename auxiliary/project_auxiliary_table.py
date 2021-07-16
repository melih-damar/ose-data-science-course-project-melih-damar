"""This module contains auxiliary functions for tables presented in the main notebook."""
pip install stargazer

import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels as sm
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
from linearmodels.iv.model import IV2SLS
from linearmodels.iv.model import IVLIML
from linearmodels.iv.results import IVModelComparison

from auxiliary import *

def prepare_country_data(df):
    country_data=df[df["tyr05_n"].notna()]
    country_data = df.rename(columns={"code":"Country Name",
                                      "logpgdp05":"Log GDP per capita",
                                      "africa":"Africa",
                                      "lat_abst":"Latitude",
                                      "asia":"Asia",
                                      "baseco":"Base Sample",
                                      "f_brit":"British colony",
                                      "f_french":"French colony",
                                      "malfal94":"Malaria Index ",
                                      "lpd1500s":"Log population density 1500",
                                      "prienr1900":"Primary school enrollment 1900",
                                      "america":"America",
                                      "ruleoflaw":"Rule of law",
                                      "tyr05_n":"Years of schooling",
                                      "lcapped":"Log capped potential settler mortality",
                                      "dummy_dennis":"Dummy for different source of Protestant missions",
                                      "protmiss":"Protestant missionaries in the early twentieth century"})
    country_data=country_data[country_data["Years of schooling"].notna()]
    country_data = country_data[["Log GDP per capita",
                                 "Years of schooling",
                                 "Rule of law",
                                 "Primary school enrollment 1900",
                                 "Protestant missionaries in the early twentieth century",   
                                 "Log capped potential settler mortality",
                                 "Log population density 1500",
                                 "Dummy for different source of Protestant missions", 
                                 "Latitude","British colony","French colony","Africa","Asia","America"]]
    
    
    return country_data

def prepare_region_data(df):
    """" This functions prepares data to construct summary statistics"""
    
    df=df[df["yearsed"].notna()&df["lgdp"].notna()&df["capital_old"].notna()]
    
    region_data = df.rename(columns={ "lgdp":"Log GDP per capita",
                                     "yearsed":"Years of schooling",
                                     "temp_avg":"Temperature",
                                      "invdistcoast":"Inverse distance to coast",
                                      "landlocked":"Landlocked region",
                                      "miss_presence":"Presence of Protestant missionaries in early twentieth century",
                                      "capital_old":"Capital city",
                                      "lpopd_i":"Log population density before colonization"})
     ## Getting the colums which is related with 
    region_data = region_data[["Log GDP per capita",
                               "Years of schooling",
                               "Temperature",
                               "Inverse distance to coast",
                               "Landlocked region",
                               "Presence of Protestant missionaries in early twentieth century",
                               "Capital city", 
                               "Log population density before colonization"]]
    return region_data



def get_summary_statistics(country_data,region_data):
    ## Preparing data
    
    prepared_country_data= prepare_country_data(country_data)
    prepared_region_data=prepare_region_data(region_data)
   
    #get summary statistisc
    
    prepared_country_data = prepared_country_data.describe().T
    prepared_region_data = prepared_region_data.describe().T
    
    #get related summary statistisc
    
    prepared_country_data = prepared_country_data[["count","mean","std"]]
    prepared_country_data = prepared_country_data.rename(columns={"count":"Observations","mean":"Mean","std":"SD"})
    prepared_region_data = prepared_region_data[["count","mean","std"]]
    prepared_region_data = prepared_region_data.rename(columns={"count":"Observations","mean":"Mean","std":"SD"})
    
    summary_statistics = pd.concat([prepared_country_data,prepared_region_data],axis=0,keys=["Cross-country sample","Cross-region sample"])
    
    return summary_statistics
    
    
def get_table2(country_data):
    country_data=country_data[country_data["tyr05_n"].notna()]
    result1=smf.ols(formula="logpgdp05 ~ tyr05_n  ",data=country_data).fit(cov_type='HC3')
    result2=smf.ols(formula="logpgdp05 ~ ruleoflaw  ",data=country_data).fit(cov_type='HC3')
    result3=smf.ols(formula="logpgdp05 ~ tyr05_n + ruleoflaw  ",data=country_data).fit(cov_type='HC3')
    result4=smf.ols(formula="logpgdp05 ~ tyr05_n +lat_abst ",data=country_data).fit(cov_type='HC3')
    result5=smf.ols(formula="logpgdp05 ~ ruleoflaw +lat_abst ",data=country_data).fit(cov_type='HC3')
    result6=smf.ols(formula="logpgdp05 ~ tyr05_n+ruleoflaw +lat_abst ",data=country_data).fit(cov_type='HC3')
    result7=smf.ols(formula="logpgdp05 ~ tyr05_n +lat_abst + africa+america+asia",data=country_data).fit(cov_type='HC3')
    result8=smf.ols(formula="logpgdp05 ~ ruleoflaw +lat_abst + africa+ america +asia",data=country_data).fit(cov_type='HC3')
    result9=smf.ols(formula="logpgdp05~tyr05_n+ruleoflaw+lat_abst+africa+america+asia",data=country_data).fit(cov_type='HC3')
    result10=smf.ols(formula="logpgdp05~tyr05_n+lat_abst+africa+america+asia+f_brit+f_french",data=country_data).fit(cov_type='HC3')
    result11=smf.ols(formula="logpgdp05~ruleoflaw+lat_abst+africa+america+asia+f_brit+f_french",data=country_data).fit(cov_type='HC3')
    result12=smf.ols(formula="logpgdp05~tyr05_n+ruleoflaw+lat_abst+africa+america+asia+f_brit+f_french",data=country_data).fit(cov_type='HC3')
    
    table=Stargazer([result1,result2,result3,result4,result5,result6,result7,result8,result9,result10,result11,result12])
    table.covariate_order(["tyr05_n","ruleoflaw","lat_abst" , "africa","america","asia","f_brit","f_french"])
    table.rename_covariates({"tyr05_n":"Years of schooling","ruleoflaw":"Rule of law","lat_abst":"Latitude",
                             "africa":"Africa","america":"America","asia":"Asia","f_brit":"British colony",
                             "f_french":"French Colony"})
    #table.title("Table 2 Ordinary least squares (OLS) cross-country regressions")
    table.dependent_variable_name("Dependent Variable: log GDP per capita")
    table.add_custom_notes(["These are OLS regressions with one observation per country",
                         "Standard errors robust against heteroscedasticity are in parentheses",
                        "Dependent variable: log GDP per capita in 2005 "])
    return table

def get_table3(country_data):
    country_data=country_data[country_data["tyr05_n"].notna()]
    result1=smf.ols(formula="prienr1870 ~ protmiss",data=country_data).fit(cov_type='HC1')
    result2=smf.ols(formula="prienr1870 ~ protmiss + lat_abst",data=country_data).fit(cov_type='HC1')
    result3=smf.ols(formula="prienr1870 ~ protmiss + lat_abst+africa+america",data=country_data).fit(cov_type='HC1')
    result4=smf.ols(formula="prienr1870 ~ protmiss + lat_abst+africa+america+f_french+f_brit",data=country_data).fit(cov_type='HC1')

    result5=smf.ols(formula="prienr1940 ~ protmiss",data=country_data).fit(cov_type='HC1')
    result6=smf.ols(formula="prienr1940 ~ protmiss+ lat_abst",data=country_data).fit(cov_type='HC1')
    result7=smf.ols(formula="prienr1940 ~ protmiss + lat_abst+africa+america",data=country_data).fit(cov_type='HC1')
    result8=smf.ols(formula="prienr1940 ~ protmiss + lat_abst+africa+america+f_french+f_brit",data=country_data).fit(cov_type='HC1')

    country_data1=country_data[(country_data["Yrsmis60"] < 90)&(country_data["protmiss"] != 0)]
    result9=smf.ols(formula="tyr05_n ~ protmiss",data=country_data1).fit(cov_type='HC1')
    result10=smf.ols(formula="tyr05_n ~ protmiss+ lat_abst",data=country_data1).fit(cov_type='HC1')
    result11=smf.ols(formula="tyr05_n ~ protmiss + lat_abst+africa+america",data=country_data1).fit(cov_type='HC1')
    result12=smf.ols(formula="tyr05_n ~ protmiss + lat_abst+africa+america+f_french+f_brit",data=country_data1).fit(cov_type='HC1')
    
    table=Stargazer([result1,result2,result3,result4,result5,result6,result7,result8,result9,result10,result11,result12])
    table.covariate_order(["protmiss","lat_abst" , "africa","america","f_french","f_brit"])
    table.rename_covariates({"protmiss":"Protestant missionaries in the early twentieth century","lat_abst":"Latitude",
                             "africa":"Africa","america":"America","f_brit":"British colony",
                             "f_french":"French Colony"})
    #table.title("Table 3 Falsification exercise, Protestant missionaries, cross-country sample")
    #table.dependent_variable_name("Dependent Variable: primary school enrollment in 1870","Dependent Variable: primary school enrollment in 1940","Dependent Variable: primary school enrollment in 2005")
    table.custom_columns(["Dependent variable: Primary school enrollment in 1870","Dependent variable: Primary school enrollment in 1940",
                      "Dependent variable: years of schooling in 2005"],[4,4,4])
    table.add_custom_notes(["These are OLS regressions with one observation per country",
                         "Standard errors robust against heteroscedasticity are in parentheses"])
    return table


def get_table4(country_data):
    country_data.loc[country_data["code"]=="HKG","dummy_dennis"]=0.0
    country_data=country_data[country_data["tyr05_n"].notna()]
    country_data.loc[:,"inter"]=1
    
    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis", data=country_data).fit(cov_type='HC3').predict()
    result1=smf.ols(formula="logpgdp05 ~ predic+dummy_dennis", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst", data=country_data).fit(cov_type='HC3').predict()
    result2=smf.ols(formula="logpgdp05 ~ predic+dummy_dennis+lat_abst", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia",                                      
                                          data=country_data).fit(cov_type='HC3').predict()
    result3=smf.ols(formula="logpgdp05 ~ predic+dummy_dennis+lat_abst+africa+america+asia", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit", 
                                          data=country_data).fit(cov_type='HC3').predict()
    result4=smf.ols(formula="logpgdp05 ~ predic+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3').predict()
    result5=smf.ols(formula="logpgdp05 ~ predic+dummy_dennis+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+lcapped+lpd1500s",
                                          data=country_data).fit(cov_type='HC3').predict()
    result6=smf.ols(formula="logpgdp05 ~ predic+dummy_dennis+lat_abst+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+lcapped+lpd1500s", 
                                          data=country_data).fit(cov_type='HC3').predict()
    result7=smf.ols(formula="logpgdp05 ~ predic+dummy_dennis+lat_abst+africa+america+asia+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')
    
    country_data.loc[:,"predic"]=smf.ols(formula="tyr05_n~prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit+lcapped+lpd1500s",
                                    data=country_data).fit(cov_type='HC3').predict()
    result8=smf.ols(formula="logpgdp05 ~ predic+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')
    
    table1=Stargazer([result1,result2,result3,result4,result5,result6,result7,result8])
    table1.covariate_order(["predic","dummy_dennis","lat_abst","africa","america","asia","f_french","f_brit","lcapped","lpd1500s"])
    table1.rename_covariates({"predic":"Years of schooling","dummy_dennis":"Dummy for different source of protestant missionaries","lat_abst":"Latitude",
                             "africa":"Africa","america":"America","asia":"Asia","f_brit":"British colony","f_french":"French Colony",
                        "lcapped":"Log capped potential settler mortality","lpd1500s":"log population density in 1500"})
    table1.dependent_variable_name("Dependent Variable: log GDP per capita in 2005")
    table1.show_r2=False
    table1.show_n =False
    table1.title("Table 4, Panel A: Second‐stage regressions")
    table1.custom_columns("2SLS")
    
    result111=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()
    result112=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis","lat_abst"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()
    result113=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis","lat_abst","africa","america","asia"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()                            
    result114=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis","lat_abst","africa","america","asia","f_french","f_brit"]],
                    endog=country_data["tyr05_n"],
                    instruments=country_data[["prienr1900","protmiss"]]).fit()
    result115=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis","lcapped","lpd1500s"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()
    result116=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis","lat_abst","lcapped","lpd1500s"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()                                          
    result117=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis","lat_abst","africa","america","asia","lcapped","lpd1500s"]],
                    endog=country_data["tyr05_n"],
                    instruments=country_data[["prienr1900","protmiss"]]).fit()
    result118=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis","lat_abst","africa","america","asia",
                                                                            "lcapped","lpd1500s","f_french","f_brit"]],
                     endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()
    
    table1.add_line("Over‐identification test (p‐value)",[over_iden(result111),over_iden(result112),over_iden(result113),over_iden(result114),
                                                          over_iden(result115),over_iden(result116),
                                                          over_iden(result117),over_iden(result118)])
    
    country_data.loc[country_data["code"]=="HKG","dummy_dennis"]=0.0
    result11= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis", data=country_data).fit(cov_type='HC3')

    result12= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst", data=country_data).fit(cov_type='HC3')

    result13= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia", data=country_data).fit(cov_type='HC3')

    result14= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit", data=country_data).fit(cov_type='HC3')

    result15= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    result16= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    result17= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    result18= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit+lcapped+lpd1500s",
                      data=country_data).fit(cov_type='HC3')
    
    table2=Stargazer([result11,result12,result13,result14,result15,result16,result17,result18])
    table2.covariate_order(["prienr1900","protmiss","dummy_dennis","lat_abst","africa","america","asia","f_brit","f_french","lcapped","lpd1500s"])
    table2.rename_covariates({"prienr1900":"Primary enrollment in 1900","protmiss":"Protestant missionaries in early 20th century",
                             "dummy_dennis":"Dummy for different source of protestant missionaries","lat_abst":"Latitude","africa":"Africa","america":"America",
                             "asia":"Asia","f_brit":"British colony","f_french":"French Colony","lcapped":"Log capped potentialsettler mortality",
                            "lpd1500s":"log population density in 1500"})
    table2.dependent_variable_name("Dependent Variable: Years of schooling in 2005 ")
    table2.title("Table 4, Panel B: First‐stage regressions")
    
    return [table1, table2]

def get_table4_LIML(df):
    df.loc[df["code"]=="HKG","dummy_dennis"]=0.0
    df=df[df["tyr05_n"].notna()]
    df.loc[:,"Intercept"]=1
    country_data = df.rename(columns={"logpgdp05":"Log GDP per capita",
                                      "africa":"Africa",
                                      "lat_abst":"Latitude",
                                      "asia":"Asia",
                                      "f_brit":"British colony",
                                      "f_french":"French colony",
                                      "lpd1500s":"Log population density 1500",
                                      "prienr1900":"Primary school enrollment 1900",
                                      "america":"America",
                                      "tyr05_n":"Years of schooling",
                                      "lcapped":"Log capped potential settler mortality",
                                      "dummy_dennis":"Dummy for different source of Protestant missions",
                                      "protmiss":"Protestant missionaries in the early twentieth century"})
    
    result115=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions"
                                                                                     ,"Log capped potential settler mortality",
                                                                                     "Log population density 1500"]],
                     endog=country_data["Years of schooling"],
                     instruments=country_data[["Primary school enrollment 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result116=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Log capped potential settler mortality",
                                                                                     "Log population density 1500"]],
                     endog=country_data["Years of schooling"],
                     instruments=country_data[["Primary school enrollment 1900","Protestant missionaries in the early twentieth century"]]).fit()                                          
    result117=IVLIML(dependent=country_data["Log GDP per capita"],
                     exog=country_data[["Intercept","Dummy for different source of Protestant missions","Latitude","Africa","America","Asia",
                                        "Log capped potential settler mortality",
                                        "Log population density 1500"]],
                    endog=country_data["Years of schooling"],
                    instruments=country_data[["Primary school enrollment 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result118=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia",
                                                                            "French colony","British colony","Log capped potential settler mortality",
                                                                                     "Log population density 1500"]],
                     endog=country_data["Years of schooling"],
                     instruments=country_data[["Primary school enrollment 1900","Protestant missionaries in the early twentieth century"]]).fit()
    
    table =IVModelComparison([result115,result116,result117,result118],precision="std_errors",stars=True)
    return table

def get_table5(country_data):
    country_data.loc[country_data["code"]=="HKG","dummy_dennis"]=0.0
    country_data=country_data[country_data["tyr05_n"].notna()]
    country_data.loc[:,"inter"]=1
    
    country_data.loc[:,"predic"]= smf.ols(formula="ruleoflaw ~lcapped+lpd1500s", data=country_data).fit(cov_type='HC3').predict()
    result1=smf.ols(formula="logpgdp05 ~ predic", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst", data=country_data).fit(cov_type='HC3').predict()
    result2=smf.ols(formula="logpgdp05 ~ predic+lat_abst", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst+africa+america+asia",                                     
                                          data=country_data).fit(cov_type='HC3').predict()
    result3=smf.ols(formula="logpgdp05 ~ predic+lat_abst+africa+america+asia", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst+africa+america+asia+f_french+f_brit", 
                                          data=country_data).fit(cov_type='HC3').predict()
    result4=smf.ols(formula="logpgdp05 ~ predic+lat_abst+africa+america+asia+f_french+f_brit", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+dummy_dennis+prienr1900+protmiss", data=country_data).fit(cov_type='HC3').predict()
    result5=smf.ols(formula="logpgdp05 ~ predic+dummy_dennis+prienr1900+protmiss", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst+dummy_dennis+prienr1900+protmiss",
                                          data=country_data).fit(cov_type='HC3').predict()
    result6=smf.ols(formula="logpgdp05 ~ predic+lat_abst+dummy_dennis+prienr1900+protmiss", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst+africa+america+asia+dummy_dennis+prienr1900+protmiss", 
                                          data=country_data).fit(cov_type='HC3').predict()
    result7=smf.ols(formula="logpgdp05 ~ predic+lat_abst+africa+america+asia+dummy_dennis+prienr1900+protmiss", data=country_data).fit(cov_type='HC3')
    
    country_data.loc[:,"predic"]=smf.ols(formula="ruleoflaw~lcapped+lpd1500s+lat_abst+africa+america+asia+f_french+f_brit+dummy_dennis+prienr1900+protmiss",
                                    data=country_data).fit(cov_type='HC3').predict()
    result8=smf.ols(formula="logpgdp05 ~ predic+lat_abst+africa+america+asia+f_french+f_brit+dummy_dennis+prienr1900+protmiss", 
                    data=country_data).fit(cov_type='HC3')
    
    table1=Stargazer([result1,result2,result3,result4,result5,result6,result7,result8])
    table1.covariate_order(["predic","lat_abst","africa","america","asia","f_french","f_brit","dummy_dennis","prienr1900","protmiss"])
    table1.rename_covariates({"predic":"Rule of law","lat_abst":"Latitude",
                             "africa":"Africa","america":"America","asia":"Asia","f_brit":"British colony","f_french":"French Colony",
                        "dummy_dennis":"Dummy for different source of protestant missions","prienr1900":"Primary enrollment in 1900",
                             "protmiss":"Protestant missionaries in early 2Oth century"})
    table1.dependent_variable_name("Dependent Variable: log GDP per capita in 2005")
    table1.show_r2=False
    table1.show_n =False
    table1.title("Table 5, Panel A: Second‐stage regressions")
    table1.custom_columns("2SLS")
    
    result111=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter"]],endog=country_data["ruleoflaw"],
                     instruments=country_data[["lcapped","lpd1500s"]]).fit()
    result112=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","lat_abst"]],endog=country_data["ruleoflaw"],
                     instruments=country_data[["lcapped","lpd1500s"]]).fit()
    result113=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","lat_abst","africa","america","asia"]],
                     endog=country_data["ruleoflaw"],
                     instruments=country_data[["lcapped","lpd1500s"]]).fit()                            
    result114=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","lat_abst","africa","america","asia","f_french","f_brit"]],
                    endog=country_data["ruleoflaw"],
                    instruments=country_data[["lcapped","lpd1500s"]]).fit()
    result115=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis","prienr1900","protmiss"]],endog=country_data["ruleoflaw"],
                     instruments=country_data[["lcapped","lpd1500s"]]).fit()
    result116=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","lat_abst","dummy_dennis","prienr1900","protmiss"]],
                     endog=country_data["ruleoflaw"],
                     instruments=country_data[["lcapped","lpd1500s"]]).fit()                                          
    result117=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","lat_abst","africa","america","asia",
                                                                            "dummy_dennis","prienr1900","protmiss"]],
                    endog=country_data["ruleoflaw"],
                    instruments=country_data[["lcapped","lpd1500s"]]).fit()
    result118=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","lat_abst","africa","america","asia",
                                                                            "dummy_dennis","prienr1900","protmiss","f_french","f_brit"]],
                     endog=country_data["ruleoflaw"],
                     instruments=country_data[["lcapped","lpd1500s"]]).fit()
    
    table1.add_line("Over‐identification test (p‐value)",[over_iden(result111),over_iden(result112),over_iden(result113),over_iden(result114),
                                                          over_iden(result115),over_iden(result116),
                                                          over_iden(result117),over_iden(result118)])
    
    result11= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    result12= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst", data=country_data).fit(cov_type='HC3')

    result13= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst+africa+america+asia", data=country_data).fit(cov_type='HC3')

    result14= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst+africa+america+asia+f_french+f_brit", data=country_data).fit(cov_type='HC3')

    result15= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+dummy_dennis+prienr1900+protmiss", data=country_data).fit(cov_type='HC3')

    result16= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst+dummy_dennis+prienr1900+protmiss", data=country_data).fit(cov_type='HC3')

    result17= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst+africa+america+asia+dummy_dennis+prienr1900+protmiss", data=country_data).fit(cov_type='HC3')

    result18= smf.ols(formula=" ruleoflaw ~ lcapped+lpd1500s+lat_abst+africa+america+asia+f_french+f_brit+dummy_dennis+prienr1900+protmiss",
                      data=country_data).fit(cov_type='HC3')
    
    table2=Stargazer([result11,result12,result13,result14,result15,result16,result17,result18])
    table2.covariate_order(["lcapped","lpd1500s","lat_abst","africa","america","asia","f_brit","f_french","dummy_dennis","prienr1900","protmiss"])
    table2.rename_covariates({"lcapped":"log capped potential settler mortality","lpd1500s":"log population density in 1500",
                             "lat_abst":"Latitude","africa":"Africa","america":"America",
                             "asia":"Asia","f_brit":"British colony","f_french":"French Colony","prienr1900":"Primary Enrollment in 1900",
                            "protmiss":"Protestant missionaries in early 2Oth century","dummy_dennis":"Dummy for different source of protestant missions"})
    table2.dependent_variable_name("Dependent Variable: Rule of law ")
    table2.title("Table 5, Panel B: First‐stage regressions")
    
    return [table1, table2]


def over_iden(result):
    p_val=round(result.wooldridge_overid.pval,2)
    return p_val

def get_table5_LIML(df):
    df.loc[df["code"]=="HKG","dummy_dennis"]=0.0
    df=df[df["tyr05_n"].notna()]
    df.loc[:,"Intercept"]=1
    country_data = df.rename(columns={"logpgdp05":"Log GDP per capita",
                                      "africa":"Africa",
                                      "lat_abst":"Latitude",
                                      "asia":"Asia",
                                      "f_brit":"British colony",
                                      "f_french":"French colony",
                                      "lpd1500s":"Log population density 1500",
                                      "prienr1900":"Primary school enrollment 1900",
                                      "america":"America",
                                      "ruleoflaw":"Rule of law",
                                      "lcapped":"Log capped potential settler mortality",
                                      "dummy_dennis":"Dummy for different source of Protestant missions",
                                      "protmiss":"Protestant missionaries in the early twentieth century"})
    
    result115=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions"
                                                                                     ,"Protestant missionaries in the early twentieth century",
                                                                                     "Primary school enrollment 1900"]],
                     endog=country_data["Rule of law"],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500"]]).fit()
    result116=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Protestant missionaries in the early twentieth century",
                                                                                     "Primary school enrollment 1900"]],
                     endog=country_data["Rule of law"],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500"]]).fit()                   
    result117=IVLIML(dependent=country_data["Log GDP per capita"],
                     exog=country_data[["Intercept","Dummy for different source of Protestant missions","Latitude","Africa","America","Asia",
                                        "Protestant missionaries in the early twentieth century",
                                        "Primary school enrollment 1900"]],
                    endog=country_data["Rule of law"],
                    instruments=country_data[["Log capped potential settler mortality","Log population density 1500"]]).fit()
    result118=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia",
                                                                            "French colony","British colony",
                                                                                     "Protestant missionaries in the early twentieth century",
                                                                                     "Primary school enrollment 1900"]],
                     endog=country_data["Rule of law"],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500"]]).fit()
    
    table =IVModelComparison([result115,result116,result117,result118],precision="std_errors",stars=True)
    return table

def get_table6(df):
    df.loc[df["code"]=="HKG","dummy_dennis"]=0.0
    country_data=df[df["tyr05_n"].notna()]
    country_data.loc[:,"inter"]=1
    
    country_data.loc[:,"predic_ruleoflaw"]= smf.ols(formula="ruleoflaw ~lcapped+lpd1500s+prienr1900+protmiss+dummy_dennis", 
                                                    data=country_data).fit(cov_type='HC3').predict()
    country_data.loc[:,"predic_schooling"]= smf.ols(formula="tyr05_n ~lcapped+lpd1500s+prienr1900+protmiss+dummy_dennis", 
                                                    data=country_data).fit(cov_type='HC3').predict()
    result1=smf.ols(formula="logpgdp05 ~ predic_ruleoflaw+predic_schooling+dummy_dennis", data=country_data).fit(cov_type='HC3')
    
    country_data.loc[:,"predic_ruleoflaw"]= smf.ols(formula="ruleoflaw ~lcapped+lpd1500s+prienr1900+protmiss+dummy_dennis+lat_abst", 
                                                    data=country_data).fit(cov_type='HC3').predict()
    country_data.loc[:,"predic_schooling"]= smf.ols(formula="tyr05_n ~lcapped+lpd1500s+prienr1900+protmiss+dummy_dennis++lat_abst", 
                                                    data=country_data).fit(cov_type='HC3').predict()
    result2=smf.ols(formula="logpgdp05 ~ predic_ruleoflaw+predic_schooling+dummy_dennis++lat_abst", data=country_data).fit(cov_type='HC3')
    
    country_data.loc[:,"predic_ruleoflaw"]= smf.ols(formula="ruleoflaw ~lcapped+lpd1500s+prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia", 
                                                    data=country_data).fit(cov_type='HC3').predict()
    country_data.loc[:,"predic_schooling"]= smf.ols(formula="tyr05_n ~lcapped+lpd1500s+prienr1900+protmiss+dummy_dennis++lat_abst+africa+america+asia", 
                                                    data=country_data).fit(cov_type='HC3').predict()
    result3=smf.ols(formula="logpgdp05 ~ predic_ruleoflaw+predic_schooling+dummy_dennis++lat_abst+africa+america+asia", data=country_data).fit(cov_type='HC3')
    
    country_data.loc[:,"predic_ruleoflaw"]= smf.ols(formula="ruleoflaw ~lcapped+lpd1500s+prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+f_brit+f_french", 
                                                    data=country_data).fit(cov_type='HC3').predict()
    country_data.loc[:,"predic_schooling"]= smf.ols(formula="tyr05_n ~lcapped+lpd1500s+prienr1900+protmiss+dummy_dennis++lat_abst+africa+america+asia+f_brit+f_french", 
                                                    data=country_data).fit(cov_type='HC3').predict()
    result4=smf.ols(formula="logpgdp05 ~ predic_ruleoflaw+predic_schooling+dummy_dennis++lat_abst+africa+america+asia+f_brit+f_french",
                    data=country_data).fit(cov_type='HC3')
    
    table=Stargazer([result1,result2,result3,result4])
    table.covariate_order(["predic_schooling","predic_ruleoflaw","dummy_dennis","lat_abst","africa","america","asia","f_brit","f_french"])
    table.rename_covariates({"predic_schooling":"Years of schooling","predic_ruleoflaw":"Rule of law",
                             "lat_abst":"Latitude","africa":"Africa","america":"America",
                             "asia":"Asia","f_brit":"British colony","f_french":"French Colony",
                             "dummy_dennis":"Dummy for different source of protestant missions"})
    table.dependent_variable_name("Dependent Variable: Log GDP per capita in 2005 ")
    table.title("Table 6, Panel A: Second‐stage regressions")
    
    result111=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","dummy_dennis"]],endog=country_data[["ruleoflaw","tyr05_n"]],
                     instruments=country_data[["lcapped","lpd1500s","protmiss","prienr1900"]]).fit()
    result112=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","lat_abst","dummy_dennis"]],endog=country_data[["ruleoflaw","tyr05_n"]],
                     instruments=country_data[["lcapped","lpd1500s","protmiss","prienr1900"]]).fit()
    result113=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","lat_abst","africa","america","asia","dummy_dennis"]],
                     endog=country_data[["ruleoflaw","tyr05_n"]],
                     instruments=country_data[["lcapped","lpd1500s","protmiss","prienr1900"]]).fit()                            
    result114=IV2SLS(dependent=country_data["logpgdp05"],exog=country_data[["inter","lat_abst","africa","america","asia","f_french","f_brit","dummy_dennis"]],
                    endog=country_data[["ruleoflaw","tyr05_n"]],
                    instruments=country_data[["lcapped","lpd1500s","protmiss","prienr1900"]]).fit()
   
    
    table.add_line("Over‐identification test (p‐value)",[over_iden(result111),over_iden(result112),over_iden(result113),over_iden(result114)])
    
    result11= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+lcapped+lpd1500s+dummy_dennis", data=country_data).fit(cov_type='HC3')

    result12= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    result13= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss++lcapped+lpd1500s+dummy_dennis+lat_abst+africa+america+asia", 
                      data=country_data).fit(cov_type='HC3')

    result14= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit+lcapped+lpd1500s", 
                      data=country_data).fit(cov_type='HC3')

    result15= smf.ols(formula=" ruleoflaw ~ prienr1900+protmiss+dummy_dennis+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    result16= smf.ols(formula=" ruleoflaw ~ prienr1900+protmiss+dummy_dennis+lat_abst+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    result17= smf.ols(formula=" ruleoflaw ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+lcapped+lpd1500s", 
                      data=country_data).fit(cov_type='HC3')

    result18= smf.ols(formula=" ruleoflaw ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit+lcapped+lpd1500s",
                      data=country_data).fit(cov_type='HC3')
    
    table2=Stargazer([result11,result12,result13,result14,result15,result16,result17,result18])
    table2.covariate_order(["prienr1900","protmiss","lcapped","lpd1500s","dummy_dennis","lat_abst","africa","america","asia","f_brit","f_french"])
    table2.rename_covariates({"prienr1900":"Primary enrollment in 1900","protmiss":"Protestant missionaries in early 20th century",
                             "dummy_dennis":"Dummy for different source of protestant missionaries","lat_abst":"Latitude","africa":"Africa","america":"America",
                             "asia":"Asia","f_brit":"British colony","f_french":"French Colony","lcapped":"Log capped potential settler mortality",
                            "lpd1500s":"log population density in 1500"})
    table2.custom_columns(["Dependent variable: years of schooling","Dependent variable: rule of law",],[4,4])
    table2.title("Table 6, Panel A: First‐stage regressions")
    
    
    return table,table2

def get_table6_LIML(df):
    df.loc[df["code"]=="HKG","dummy_dennis"]=0.0
    df=df[df["tyr05_n"].notna()]
    df.loc[:,"Intercept"]=1
    country_data = df.rename(columns={"logpgdp05":"Log GDP per capita",
                                      "africa":"Africa",
                                      "lat_abst":"Latitude",
                                      "asia":"Asia",
                                      "f_brit":"British colony",
                                      "f_french":"French colony",
                                      "lpd1500s":"Log population density 1500",
                                      "prienr1900":"Primary school enrollment 1900",
                                      "america":"America",
                                      "ruleoflaw":"Rule of law",
                                      "lcapped":"Log capped potential settler mortality",
                                      "dummy_dennis":"Dummy for different source of Protestant missions",
                                      "protmiss":"Protestant missionaries in the early twentieth century",
                                     "prienr1900":"Primary enrollment in 1900","tyr05_n":"Years of schooling"})
    
    result115=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions"]],
                     endog=country_data[["Rule of law","Years of schooling"]],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result116=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude"]],
                     endog=country_data[["Rule of law","Years of schooling"]],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()                   
    result117=IVLIML(dependent=country_data["Log GDP per capita"],
                     exog=country_data[["Intercept","Dummy for different source of Protestant missions","Latitude","Africa","America","Asia",]],
                    endog=country_data[["Rule of law","Years of schooling"]],
                    instruments=country_data[["Log capped potential settler mortality","Log population density 1500",
                                             "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result118=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia",
                                                                            "French colony","British colony",]],
                     endog=country_data[["Rule of law","Years of schooling"]],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    
    table =IVModelComparison([result115,result116,result117,result118],precision="std_errors",stars=True)
    return table

def get_table7(df):
    df.loc[df["code"]=="HKG","dummy_dennis"]=0.0
    df=df[df["tyr05_n"].notna()]
    df.loc[:,"Intercept"]=1
    country_data = df.rename(columns={"logpgdp05":"Log GDP per capita",
                                      "africa":"Africa",
                                      "lat_abst":"Latitude",
                                      "asia":"Asia",
                                      "f_brit":"British colony",
                                      "f_french":"French colony",
                                      "lpd1500s":"Log population density 1500",
                                      "prienr1900":"Primary school enrollment 1900",
                                      "america":"America",
                                      "ruleoflaw":"Rule of law",
                                      "lcapped":"Log capped potential settler mortality",
                                      "dummy_dennis":"Dummy for different source of Protestant missions",
                                      "protmiss":"Protestant missionaries in the early twentieth century",
                                     "prienr1900":"Primary enrollment in 1900","tyr05_n":"Years of schooling",
                                     "malfal94":"Falciparum malaria index 1994",
                                     "cath1900":"Catholic affiliation","prot1900":"Protestant affiliation","musl1900":"Muslim affiliation"})
    country_neo=country_data.loc[country_data["neoeuropes"]==0]
    
    result1=IVLIML(dependent=country_neo["Log GDP per capita"],exog=country_neo[["Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia"]],
                     endog=country_neo[["Rule of law","Years of schooling"]],
                     instruments=country_neo[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result2=IVLIML(dependent=country_neo["Log GDP per capita"],exog=country_neo[["Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia","British colony", "French colony"]],
                     endog=country_neo[["Rule of law","Years of schooling"]],
                     instruments=country_neo[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()    
    country_data1=country_data[country_data["Falciparum malaria index 1994"].notna()]
    result3=IVLIML(dependent=country_data1["Log GDP per capita"],
                     exog=country_data1[["Intercept","Dummy for different source of Protestant missions","Latitude","Africa","America","Asia",
                                        "Falciparum malaria index 1994"]],
                    endog=country_data1[["Rule of law","Years of schooling"]],
                    instruments=country_data1[["Log capped potential settler mortality","Log population density 1500",
                                             "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result4=IVLIML(dependent=country_data1["Log GDP per capita"],exog=country_data1[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia","French colony",
                                                                                   "British colony","Falciparum malaria index 1994"]],
                     endog=country_data1[["Rule of law","Years of schooling"]],
                     instruments=country_data1[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result5=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",                   
                                                                                "Latitude","Africa","America","Asia","temp1","temp2","temp3","temp4","temp5","humid1",
                                                                                  "humid2","humid3","humid4"]],
                     endog=country_data[["Rule of law","Years of schooling"]],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result6=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia","French colony",
                                                                                   "British colony","temp1","temp2","temp3","temp4","temp5","humid1",
                                                                                  "humid2","humid3","humid4"]],
                     endog=country_data[["Rule of law","Years of schooling"]],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result7=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia","Catholic affiliation",
                                                                                  "Protestant affiliation","Muslim affiliation"]],
                     endog=country_data[["Rule of law","Years of schooling"]],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result8=IVLIML(dependent=country_data["Log GDP per capita"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia","Catholic affiliation",
                                                                                    "French colony","British colony",
                                                                                  "Protestant affiliation","Muslim affiliation"]],
                     endog=country_data[["Rule of law","Years of schooling"]],
                     instruments=country_data[["Log capped potential settler mortality","Log population density 1500",
                                              "Primary enrollment in 1900","Protestant missionaries in the early twentieth century"]]).fit()
    
    table =IVModelComparison([result1,result2,result3,result4,result5,result6,result7,result8],precision="std_errors",stars=True)
    return table
    
    
def get_table8(df):
    df.loc[df["code"]=="HKG","dummy_dennis"]=0.0
    country_data=df[df["tyr05_n"].notna()]
    country_data.loc[:,"inter"]=1
    
    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis", data=country_data).fit(cov_type='HC3').predict()
    result1=smf.ols(formula="ruleoflaw ~ predic+dummy_dennis", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst", data=country_data).fit(cov_type='HC3').predict()
    result2=smf.ols(formula="ruleoflaw ~ predic+dummy_dennis+lat_abst", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia",                                      
                                          data=country_data).fit(cov_type='HC3').predict()
    result3=smf.ols(formula="ruleoflaw ~ predic+dummy_dennis+lat_abst+africa+america+asia", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit", 
                                          data=country_data).fit(cov_type='HC3').predict()
    result4=smf.ols(formula="ruleoflaw ~ predic+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3').predict()
    result5=smf.ols(formula="ruleoflaw ~ predic+dummy_dennis+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+lcapped+lpd1500s",
                                          data=country_data).fit(cov_type='HC3').predict()
    result6=smf.ols(formula="ruleoflaw ~ predic+dummy_dennis+lat_abst+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')

    country_data.loc[:,"predic"]= smf.ols(formula=" tyr05_n ~ prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+lcapped+lpd1500s", 
                                          data=country_data).fit(cov_type='HC3').predict()
    result7=smf.ols(formula="ruleoflaw ~ predic+dummy_dennis+lat_abst+africa+america+asia+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')
    
    country_data.loc[:,"predic"]=smf.ols(formula="tyr05_n~prienr1900+protmiss+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit+lcapped+lpd1500s",
                                    data=country_data).fit(cov_type='HC3').predict()
    result8=smf.ols(formula="ruleoflaw ~ predic+dummy_dennis+lat_abst+africa+america+asia+f_french+f_brit+lcapped+lpd1500s", data=country_data).fit(cov_type='HC3')
    
    table1=Stargazer([result1,result2,result3,result4,result5,result6,result7,result8])
    table1.covariate_order(["predic","dummy_dennis","lat_abst","africa","america","asia","f_french","f_brit","lcapped","lpd1500s"])
    table1.rename_covariates({"predic":"Years of schooling","dummy_dennis":"Dummy for different source of protestant missionaries","lat_abst":"Latitude",
                             "africa":"Africa","america":"America","asia":"Asia","f_brit":"British colony","f_french":"French Colony",
                        "lcapped":"Log capped potential settler mortality","lpd1500s":"log population density in 1500"})
    table1.dependent_variable_name("Dependent Variable: rule of law")
    table1.show_r2=False
    table1.title("Table 8, Effects of years of schooling on institutions, second-stage regression, cross-country sample ")
    table1.custom_columns("2SLS")
    
    result111=IV2SLS(dependent=country_data["ruleoflaw"],exog=country_data[["inter","dummy_dennis"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()
    result112=IV2SLS(dependent=country_data["ruleoflaw"],exog=country_data[["inter","dummy_dennis","lat_abst"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()
    result113=IV2SLS(dependent=country_data["ruleoflaw"],exog=country_data[["inter","dummy_dennis","lat_abst","africa","america","asia"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()                            
    result114=IV2SLS(dependent=country_data["ruleoflaw"],exog=country_data[["inter","dummy_dennis","lat_abst","africa","america","asia","f_french","f_brit"]],
                    endog=country_data["tyr05_n"],
                    instruments=country_data[["prienr1900","protmiss"]]).fit()
    result115=IV2SLS(dependent=country_data["ruleoflaw"],exog=country_data[["inter","dummy_dennis","lcapped","lpd1500s"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()
    result116=IV2SLS(dependent=country_data["ruleoflaw"],exog=country_data[["inter","dummy_dennis","lat_abst","lcapped","lpd1500s"]],endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()                                          
    result117=IV2SLS(dependent=country_data["ruleoflaw"],exog=country_data[["inter","dummy_dennis","lat_abst","africa","america","asia","lcapped","lpd1500s"]],
                    endog=country_data["tyr05_n"],
                    instruments=country_data[["prienr1900","protmiss"]]).fit()
    result118=IV2SLS(dependent=country_data["ruleoflaw"],exog=country_data[["inter","dummy_dennis","lat_abst","africa","america","asia",
                                                                            "lcapped","lpd1500s","f_french","f_brit"]],
                     endog=country_data["tyr05_n"],
                     instruments=country_data[["prienr1900","protmiss"]]).fit()
    
    table1.add_line("Over‐identification test (p‐value)",[over_iden(result111),over_iden(result112),over_iden(result113),over_iden(result114),
                                                          over_iden(result115),over_iden(result116),
                                                          over_iden(result117),over_iden(result118)])
    return table1

def get_table8_LIML(df):
    df.loc[df["code"]=="HKG","dummy_dennis"]=0.0
    df=df[df["tyr05_n"].notna()]
    df.loc[:,"Intercept"]=1
    country_data = df.rename(columns={"logpgdp05":"Log GDP per capita",
                                      "africa":"Africa",
                                      "lat_abst":"Latitude",
                                      "asia":"Asia",
                                      "f_brit":"British colony",
                                      "f_french":"French colony",
                                      "lpd1500s":"Log population density 1500",
                                      "prienr1900":"Primary school enrollment 1900",
                                      "america":"America",
                                      "tyr05_n":"Years of schooling",
                                      "lcapped":"Log capped potential settler mortality",
                                      "dummy_dennis":"Dummy for different source of Protestant missions",
                                      "protmiss":"Protestant missionaries in the early twentieth century",
                                     "ruleoflaw":"Rule of law"})
    
    result115=IVLIML(dependent=country_data["Rule of law"],exog=country_data[["Intercept","Dummy for different source of Protestant missions"
                                                                                     ,"Log capped potential settler mortality",
                                                                                     "Log population density 1500"]],
                     endog=country_data["Years of schooling"],
                     instruments=country_data[["Primary school enrollment 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result116=IVLIML(dependent=country_data["Rule of law"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Log capped potential settler mortality",
                                                                                     "Log population density 1500"]],
                     endog=country_data["Years of schooling"],
                     instruments=country_data[["Primary school enrollment 1900","Protestant missionaries in the early twentieth century"]]).fit()                                          
    result117=IVLIML(dependent=country_data["Rule of law"],
                     exog=country_data[["Intercept","Dummy for different source of Protestant missions","Latitude","Africa","America","Asia",
                                        "Log capped potential settler mortality",
                                        "Log population density 1500"]],
                    endog=country_data["Years of schooling"],
                    instruments=country_data[["Primary school enrollment 1900","Protestant missionaries in the early twentieth century"]]).fit()
    result118=IVLIML(dependent=country_data["Rule of law"],exog=country_data[["Intercept","Dummy for different source of Protestant missions",
                                                                                     "Latitude","Africa","America","Asia",
                                                                            "French colony","British colony","Log capped potential settler mortality",
                                                                                     "Log population density 1500"]],
                     endog=country_data["Years of schooling"],
                     instruments=country_data[["Primary school enrollment 1900","Protestant missionaries in the early twentieth century"]]).fit()
    
    table =IVModelComparison([result115,result116,result117,result118],precision="std_errors",stars=True)
    return table
    
    
def get_table9(df):
    region_data=df[df["yearsed"].notna()&df["lgdp"].notna()]
    dummy_countryFE=pd.get_dummies(region_data["bbb"]) # Dummy variable for each country
    
    result1=smf.ols(formula="lgdp~yearsed+dummy_countryFE",data=region_data).fit(cov_type='HC3')
    
    region_data1=region_data[region_data["yearsed"].notna()&region_data["lgdp"].notna()&region_data["capital_old"].notna()]
    dummy1=pd.get_dummies(region_data1["bbb"])
    result2=smf.ols(formula="lgdp~yearsed+dummy1",data=region_data1).fit(cov_type='HC3')
    result3=smf.ols(formula="lgdp~yearsed+dummy1+capital_old",
                    data=region_data1).fit(cov_type='HC3')
    result4=smf.ols(formula="lgdp~yearsed+dummy1+capital_old+invdistcoast+invdis2+landlocked",
                    data=region_data1).fit(cov_type='HC3')
    result5=smf.ols(formula="lgdp~yearsed+dummy1+capital_old+invdistcoast+invdis2+landlocked+temp_avg+temp2",
                    data=region_data1).fit(cov_type='HC3')
    result6=smf.ols(formula="lgdp~yearsed+dummy1+capital_old+invdistcoast+invdis2+landlocked+temp_avg+temp2+lpopd_i",
                    data=region_data1).fit(cov_type='HC3')
    table=Stargazer([result1,result2,result3,result4,result5,result6])
    table.covariate_order(["yearsed","capital_old","invdistcoast","invdis2",
                           "landlocked","temp_avg","temp2","lpopd_i"])
    table.rename_covariates({"yearsed":"Years of schooling","capital_old":"Capital city","invdistcoast":"Inverse distance to coast","invdis2":"Squared inverse distance to coast",
                            "landlocked":"State without a sea costline dummy","temp_avg":"Average yearly temperature (Celsius)",
                            "temp2":"Squared average yearly temperature (Celsius)","lpopd_i":"Log population density in 1500"})
    table.dependent_variable_name("Dependent Variable: log GDP per capita")
    return table 

def get_table10(df):
    region_data=df[df["yearsed"].notna()&df["lgdp"].notna()&df["capital_old"].notna()]

    dummy=pd.get_dummies(region_data["bbb"]) # Dummy variable for each country
   
    region_data.loc[:,"predic"]=smf.ols(formula="yearsed~miss_presence+dummy",data=region_data).fit(cov_type='HC3').predict()  
    result1=smf.ols(formula="lgdp~predic+dummy",data=region_data).fit(cov_type='HC3')

    region_data.loc[:,"predic"]=smf.ols(formula="yearsed~miss_presence+dummy+capital_old",data=region_data).fit(cov_type='HC3').predict()  
    result2=smf.ols(formula="lgdp~predic+dummy+capital_old",data=region_data).fit(cov_type='HC3')

    region_data.loc[:,"predic"]=smf.ols(formula="yearsed~miss_presence+dummy+capital_old",data=region_data).fit(cov_type='HC3').predict()  
    result3=smf.ols(formula="lgdp~predic+dummy+capital_old",data=region_data).fit(cov_type='HC3')
    
    region_data.loc[:,"predic"]=smf.ols(formula="yearsed~miss_presence+dummy+capital_old+invdistcoast+invdis2+landlocked",data=region_data).fit(cov_type='HC3').predict()  
    result4=smf.ols(formula="lgdp~predic+dummy+capital_old+invdistcoast+invdis2+landlocked",data=region_data).fit(cov_type='HC3')
    
    region_data.loc[:,"predic"]=smf.ols(formula="yearsed~miss_presence+dummy+capital_old+invdistcoast+invdis2+landlocked+temp_avg+temp2",data=region_data).fit(cov_type='HC3').predict()  
    result5=smf.ols(formula="lgdp~predic+dummy+capital_old+invdistcoast+invdis2+landlocked+temp_avg+temp2",data=region_data).fit(cov_type='HC3')
    region_data1=region_data[region_data["lpopd_i"].notna()]
    dummy1=pd.get_dummies(region_data1["bbb"])
    region_data1.loc[:,"predic"]=smf.ols(formula="yearsed~miss_presence+dummy1+capital_old+invdistcoast+invdis2+landlocked+temp_avg+temp2+lpopd_i",
                                        data=region_data1).fit(cov_type='HC3').predict()  
    result6=smf.ols(formula="lgdp~predic+dummy1+capital_old+invdistcoast+invdis2+landlocked+temp_avg+temp2+lpopd_i",data=region_data1).fit(cov_type='HC3')
    
    table1=Stargazer([result1,result2,result3,result4,result5,result6])
    table1.covariate_order(["predic","capital_old","invdistcoast","invdis2",
                           "landlocked","temp_avg","temp2","lpopd_i"])
    table1.rename_covariates({"predic":"Years of schooling","capital_old":"Capital city","invdistcoast":"Inverse distance to coast","invdis2":"Squared inverse distance to coast",
                            "landlocked":"State without a sea costline dummy","temp_avg":"Average yearly temperature (Celsius)",
                            "temp2":"Squared average yearly temperature (Celsius)","lpopd_i":"Log population density in 1500"})
    table1.dependent_variable_name("Dependent Variable: log GDP per capita")
    table1.show_n=False
    table1.show_r2=False
    table1.title("IV regressions, cross region")
    
    result1=smf.ols(formula="yearsed~miss_presence+dummy",data=region_data).fit(cov_type='HC3')
    result2=smf.ols(formula="yearsed~miss_presence+dummy+capital_old",data=region_data).fit(cov_type='HC3')
    result3=smf.ols(formula="yearsed~miss_presence+dummy+capital_old",data=region_data).fit(cov_type='HC3')
    result4=smf.ols(formula="yearsed~miss_presence+dummy+capital_old+invdistcoast+invdis2+landlocked",data=region_data).fit(cov_type='HC3')
    result5=smf.ols(formula="yearsed~miss_presence+dummy+capital_old+invdistcoast+invdis2+landlocked+temp_avg+temp2",data=region_data).fit(cov_type='HC3')
    result6=smf.ols(formula="yearsed~miss_presence+dummy+capital_old+invdistcoast+invdis2+landlocked+temp_avg+temp2+lpopd_i",
                                        data=region_data).fit(cov_type='HC3')
    
    table2=Stargazer([result1,result2,result3,result4,result5,result6])
    table2.covariate_order(["miss_presence","capital_old","invdistcoast","invdis2",
                           "landlocked","temp_avg","temp2","lpopd_i"])
    table2.rename_covariates({"miss_presence":"Protestant missionaries in early twentieth century",
                              "capital_old":"Capital city","invdistcoast":"Inverse distance to coast","invdis2":"Squared inverse distance to coast",
                            "landlocked":"State without a sea costline dummy","temp_avg":"Average yearly temperature (Celsius)",
                            "temp2":"Squared average yearly temperature (Celsius)","lpopd_i":"Log population density in 1500"})
    table2.dependent_variable_name("Dependent Variable: Years of Capital")
    table2.title("First-stage regressions")
    
    
    return table1,table2