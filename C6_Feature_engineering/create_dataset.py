import pandas as pd 

headers = ['date','customer_id','cat1','cat2','cat3','num1']
row1 = ['2016-09-01','146361','2','2','0','-0.518679']
row2 = ['2017-04-01','180838','4','1','0','0.415853']
row3 = ['2017-08-01','157857','3','3','1','-2.061687']
row4 = ['2017-12-01','159772','5','1','1','-0.276558']
row5 = ['2017-09-01','80014','3','2','1','-1.456827']

table =[row1,row2,row3,row4,row5]
df = pd.DataFrame(table, columns = headers)
df.to_csv('date_data.csv',index = False)