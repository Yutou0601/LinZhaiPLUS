import csv
import sqlite3

# 資料庫路徑
database_path = "data/database.db"

 
def get_db_connection():
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row   
    return conn

def define_sql():
    sql = """
    CREATE TABLE IF NOT EXISTS listings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        price INTEGER,
        size INTEGER,
        age INTEGER,
        floors INTEGER,
        bedroom INTEGER,
        city TEXT,
        location TEXT,
        district TEXT,
        house_type TEXT,
        pattern TEXT,
        tags TEXT,
        environment TEXT,
        url TEXT,
        image TEXT
    );
    """
    return sql

 
def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(define_sql())   
    conn.commit()
    conn.close()
    print("Table 'listings' has been created (if it didn't exist).")

 
def insert_listing(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    
     
    city = data['City'] if data['City'] else ''
    
     
    location_parts = data['Location'].split('市', 1)  
    
     
    if len(location_parts) == 2:
        city = location_parts[0] + '市'   
        district = location_parts[1].split('區', 1)[0] + '區'   
    else:
        city = location_parts[0]
        district = ''

     
    title = data['Name'] if data['Name'] else ''
    
     
    price = int(float(data['Price'])) if data['Price'] else 0
    size = int(float(data['Size'])) if data['Size'] else 0
    age = int(float(data['Age'])) if data['Age'] else 0
    floors = int(float(data['Floors'])) if data['Floors'] else 0
    bedroom = int(float(data['Bedroom'])) if data['Bedroom'] else 0
    
    house_type = data['HouseType'] if data['HouseType'] else ''
    pattern = data['Pattern'] if data['Pattern'] else ''
    tags = data['Tags'] if data['Tags'] else ''
    environment = data['Environment'] if data['Environment'] else ''
    url = data['Url'] if data['Url'] else ''
    image = data['Image_name'] if data['Image_name'] else ''
    
    
    description = ''   

     
    cursor.execute(""" 
        INSERT INTO listings (title, description, price, size, age,
                    floors, bedroom, city, location, district,
                    house_type, pattern, tags, environment, url,
                    image)
        VALUES (?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,  
                ?, ?, ?, ?, ?,
                ?)
    """, 
    (title, description, price, size, age, floors, bedroom, city,
          data['Location'], district, house_type, pattern, tags, environment, url, image))
    
    conn.commit()
    conn.close()

 
def import_csv_to_db(csv_file_path):
    create_table()  
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
           
            insert_listing(row)
            print(f"Inserted: {row['Name']}")

 
if __name__ == "__main__":
    csv_file_path = 'script/House_Rent_Info.csv'  
    import_csv_to_db(csv_file_path)
