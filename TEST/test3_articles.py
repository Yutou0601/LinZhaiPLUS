import requests
from bs4 import BeautifulSoup
url = "https://www.ptt.cc/bbs/Soft_Job/M.1423129894.A.186.html"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'lxml')
title = soup.title.string

articles = soup.find_all('div', 'push')

with open('movie_message.txt','w') as f:
    f.write("文章標題: " + title + "\n\n")
    print("文章標題: " + title + "\n\n")
    for article in articles:
        #去除掉冒號和左右的空白
        messages = article.find('span','f3 push-content').getText().replace(':','').strip()
        print(messages)
        f.write(messages + "\n")
