from WindPy import w 
import pandas as pd 
import datetime
import pyecharts as pe
import pyecharts.options as opts
from pyecharts_snapshot.main import make_a_snapshot
import torchvision

# 连接WIND
w.start()
bar=pe.charts.Bar()

if w.isconnected()==True:
    print("WIND连接成功")

# 获取当前时间

#def get_past_anual_data()
len=3
df = pd.DataFrame(columns = ["Code","Year","EBITDA"]) 
now_time=datetime.datetime.now()
year=now_time.year

# 绘制ebitda数据。目前仅支持绘制历史图

sec_code="600305.SH"
index="ebitda2"

#print(select)
for i in range(1,len+1):
    select="rptDate="+str(year-i)+"1231;rptType=1"
    res1=w.wss(sec_code, index, select, usedf=True)
    #print(res1[1])
    #print(res1[1].iloc[0])
    df.loc[i-1,'EBITDA']=res1[1].iloc[0,0]
    df.loc[i-1,'Code']=sec_code
    df.loc[i-1,'Year']=year-i

print(df.head())
    # print(res1[1].iloc[0])
bar.add_xaxis(df['Year'].iloc[0:])
bar.add_yaxis("EBITDA",df['EBITDA'].iloc[0:])
bar.set_global_opts(title_opts=opts.TitleOpts(title=index,subtitle=sec_code))
bar.render("EBIDTA.html")
#bar.add('EBITDA',df["Year"].iloc[0:],df["EBITDA"].iloc[0:],is_datazoom_show=True)

