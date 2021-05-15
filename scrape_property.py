from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import pandas as pd
import string
import json
from slimit import ast
from slimit.parser import Parser
from slimit.visitors import nodevisitor
import time


driver = webdriver.Firefox()

Project = []
Specification = []
Floor = []
Area = []
Price = []
#containers = soup.find_all('div', class_="pageComponent srpTuple__srpTupleBox srp")

for j in range(1,22):
    time.sleep(15)
    url_s = "https://www.99acres.com/property-in-hadapsar-pune-ffid-page-"+str(j)+"?src=CLUSTER"
    rr = requests.get(url_s)
    soup = BeautifulSoup(rr.text,"html.parser")
    containers = soup.find_all('div', class_="pageComponent srpTuple__srpTupleBox srp")
    #url_F = soup.find_all("script",type="application/ld+json")#60 to -1
    #urls = [el['url'] for el in json.loads(str(url_F))['itemListElement']]
    print("Page "+str(j)+" done")
    for i in range(len(containers)):
        Project.append(containers[i].table.find('td', class_="list_header_bold srpTuple__spacer10").text)
        Specification.append(containers[i].h2.text)
        Area.append(containers[i].table.find('td', id="srp_tuple_primary_area").text)
        Price.append(containers[i].table.find('td', class_="srpTuple__midGrid title_semiBold srpTuple__spacer16").text)

df = pd.DataFrame({'Project':Project,'Specification':Specification,'Area':Area,'Value':Price})
df.to_csv('Data.csv',index=False,encoding="utf-8")
