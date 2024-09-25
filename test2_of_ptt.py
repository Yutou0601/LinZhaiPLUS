import requests
from bs4 import BeautifulSoup


web = 'https://www.ptt.cc/bbs/Soft_Job/index.html'
content = requests.get(web)            #以get的方式取得網頁
content.text
soup = BeautifulSoup(content.text, "html.parser")
soup_list = soup.findAll('div', class_="title")

print(soup_list)
