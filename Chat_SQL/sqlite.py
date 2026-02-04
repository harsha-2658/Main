import sqlite3

# connection
connection=sqlite3.connect("student.db")

# cursor object to create record
cursor=connection.cursor()

# create table
table_info="""
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)

# insert some more records

cursor.execute(''' Insert into STUDENT values("Gaurav",'Artificial Inteligence','A',99) ''')
cursor.execute(''' Insert into STUDENT values("Harsha",'Computer Science','B',90) ''')
cursor.execute(''' Insert into STUDENT values("Charan",'Electrical','B',89) ''')
cursor.execute(''' Insert into STUDENT values("Ashish",'Information Technology','A',80) ''')

# display all the records
print("The inserted records are ")
data=cursor.execute('''select * from STUDENT''')
for row in data:
    print(row)

# commit changes in database
connection.commit()
connection.close()