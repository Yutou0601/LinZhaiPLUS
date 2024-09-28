import requests
import pandas as pd
from bs4 import BeautifulSoup
import selenium
import re
from datetime import datetime

now = datetime.now()
formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')

class House_Data :
    def __init__(self,name,size,age,price,tags ) -> None:
        self.Name = name
        self.Price= price
        self.Size = size
        self.Age  = age
        self.Tags = tags


    def Save(self):
        import csv
        with open('data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            csvfile.seek(0, 2)
            writer.writerow([self.Name]) 
            writer.writerow([self.Price + ' NT$']) 
            writer.writerow([self.Size +' 坪']) 
            writer.writerow([self.Age  +' 年']) 
            writer.writerow(self.Tags)
            writer.writerow([])
            writer.writerow([])
 
    def Append(self):
        import csv
        with open('data.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            writer.writerow(['-------------------------'])
            writer.writerow([self.Name]) 
            writer.writerow([self.Price + ' NT$']) 
            writer.writerow([self.Size +' 坪']) 
            writer.writerow([self.Age  +' 年']) 
            writer.writerow(self.Tags)
            writer.writerow(['-------------------------'])
            writer.writerow([])
            writer.writerow([])


req = requests.get('https://www.rakuya.com.tw/sell_item/info?ehid=00b118225976043&from=list_regular')
html = req.content.decode()
soup = BeautifulSoup(html,'html.parser')
title = soup.find('title').text



content = soup.find('script', string=lambda string: string and 'window.tmpDataLayer' in string)
data = content.string.strip()   # 擷取JavaScript物件的內容


# 用正則表達式提取 item_name 和 price
item_name = re.search(r'"item_name":"(.*?)"', data).group(1)
price     = re.search(r'"price":(\d+)', data).group(1)
age       = re.search(r'"age":(\d+\.?\d*)', data).group(1)
size      = re.search(r'"object_main_size":(\d+\.?\d*)', data).group(1)
floor     = re.search(r'"object_floor":(\d+)', data).group(1)
bedroom   = re.search(r'"bedrooms":(\d+)', data).group(1)
loc       = re.search(r'"item_category":"(.*?)"', data).group(1)
tags      = re.search(r'"object_tag":"(.*?)"',data).group(1).split(',')
# -----資訊整理-----


thehouse =  House_Data(name= item_name,
                       price=price,
                       size=size,
                       age=age,
                       tags=tags)
 
屋子  =  House_Data(name= item_name,
                       price=price,
                       size=size,
                       age=age,
                       tags=tags)

print(f"""
{title}

Name     : {item_name} 
Price    : {price} 
Age      : {age}
Size     : {size}
Floors   : {floor}
Bedroom  : {bedroom}
Location : {loc} 
Tags     : {tags}

Alright ~ Done !!  
-- {formatted_time}  --
 """)
thehouse.Save()
#thehouse.Append()