import requests
from bs4 import BeautifulSoup
import csv

web = 'https://www.ptt.cc/bbs/Soft_Job/index.html'
content = requests.get(web)            #以get的方式取得網頁
content.text
soup = BeautifulSoup(content.text, "html.parser")
soup_list = soup.findAll('div', class_="title")

print(soup_list)



with open('ptt_soft_job_titles.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['標題'])  
    

    for item in soup_list:
        title = item.text.strip()  
        writer.writerow([title])
