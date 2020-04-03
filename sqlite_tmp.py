import sqlite3

conn = sqlite3.connect('cleaned_text.db')                               
c = conn.cursor()

# Create the table to hold the data if it doesn't already exist
c.execute('''CREATE TABLE IF NOT EXISTS tblCleanedText
                (id INTEGER PRIMARY KEY, 
                 cleaned_text TEXT)''')

# Stick data into the table
txt = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin cursus luctus posuere"
# txt = txt.split()

c.execute('''INSERT INTO tblCleanedText (cleaned_text)
            VALUES (?)''', [txt])
conn.commit()

# Retrieve the data and print it out
c.execute('select * From tblCleanedText')
print(c.fetchall())
conn.close()