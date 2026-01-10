#导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#导入数据
df = pd.read_csv('/Users/caojun/Downloads/vgsales.csv')

# 获取当前目录
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 中文字符设置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#数据处理
df.dropna(how='any',inplace=True)
df.shape
df['Year'] = pd.to_datetime(df['Year'].round().astype(int), format='%Y').dt.year

# 定义添加数据标签的函数
def add_labels(ax, data):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2, height),
                    xytext=(0,6),textcoords = 'offset points',ha='center', va='bottom')
                    
# 所有年份的游戏类型销售数据                   
loveG = pd.pivot_table(df,values='Global_Sales',index='Year',columns='Genre',aggfunc='sum').sum().sort_values(ascending=False)
loveG = pd.DataFrame(data  = loveG,columns = ['Genre_Global_Sales'])
loveG = loveG.reset_index().rename(columns={'index': 'Genre'})
# 最近5年的游戏类型销售数据
loveG5 = pd.pivot_table(df,values='Global_Sales',index='Year',columns='Genre',aggfunc='sum').iloc[-5:,:].sum().sort_values(ascending=False)
loveG5 = pd.DataFrame(data  = loveG5,columns = ['Genre_Global_Sales'])
loveG5 = loveG5.reset_index().rename(columns={'index': 'Genre'})
# 游戏类型销售数据可视化
palette = sns.color_palette('Set2', n_colors=len(loveG))
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20, 8),dpi=100)
plt.subplots_adjust(hspace=0.4,top=0.95,bottom = 0.1)
# 所有年份的游戏类型销售数据可视化
ax1 = sns.barplot(x='Genre', y='Genre_Global_Sales', data=loveG, palette=palette, hue='Genre', legend=False, ax=ax1)
add_labels(ax1, loveG['Genre_Global_Sales'])
ax1.set_title('所有年份游戏类型销售数据可视化')
ax1.set_xlabel('游戏类型')
ax1.set_ylabel('全球销售额 (百万美元)')
# 最近5年的游戏类型销售数据可视化
ax2 = sns.barplot(x='Genre', y='Genre_Global_Sales', data=loveG5, palette=palette, hue='Genre', legend=False, ax=ax2)
add_labels(ax2, loveG5['Genre_Global_Sales'])
ax2.set_title('最近5年游戏类型销售数据可视化')
ax2.set_xlabel('游戏类型')
ax2.set_ylabel('全球销售额 (百万美元)')
# 保存图片
plt.savefig(os.path.join(current_dir, '游戏类型销售数据.png'), dpi=300, bbox_inches='tight')

# 所有年份的游戏平台销售数据可视化
loveP = pd.pivot_table(df,values='Global_Sales',index='Year',columns='Platform',aggfunc='sum').sum().sort_values(ascending=False)
loveP = pd.DataFrame(data = loveP,columns=['Global_Sales'])
loveP = loveP[loveP['Global_Sales']>1]
loveP = loveP.reset_index().rename(columns={'index': 'Platform'})
# 最近5年的游戏平台销售数据可视化
loveP5 = pd.pivot_table(df,values='Global_Sales',index='Year',columns='Platform',aggfunc='sum').iloc[-5:,:].sum().sort_values(ascending=False)
loveP5 = pd.DataFrame(data = loveP5,columns=['Global_Sales'])
loveP5 = loveP5[loveP5['Global_Sales']>1]
loveP5 = loveP5.reset_index().rename(columns={'index': 'Platform'})
# 游戏平台销售数据可视化
palette = sns.color_palette('Set2', n_colors=len(loveP))
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20, 8),dpi=100)
plt.subplots_adjust(hspace=0.4,top=0.95,bottom = 0.1)
# 所有年份的游戏平台销售数据可视化
ax1 = sns.barplot(x='Platform', y='Global_Sales', data=loveP, palette=palette, hue='Platform', legend=False, ax=ax1)
add_labels(ax1, loveP['Global_Sales'])
ax1.set_title('所有年份游戏平台销售数据可视化')
ax1.set_xlabel('游戏平台')
ax1.set_ylabel('全球销售额 (百万美元)')
# 最近5年的游戏平台销售数据可视化
ax2 = sns.barplot(x='Platform', y='Global_Sales', data=loveP5, palette=palette, hue='Platform', legend=False, ax=ax2)
add_labels(ax2, loveP5['Global_Sales']) 
ax2.set_title('最近5年游戏平台销售数据可视化')
ax2.set_xlabel('游戏平台')
ax2.set_ylabel('全球销售额 (百万美元)')
# 保存图片
plt.savefig(os.path.join(current_dir, '游戏平台销售数据.png'), dpi=300, bbox_inches='tight')

# 游戏平台销售数据，游戏发行年份，游戏发行商排名可视化
pd.set_option('display.max_columns', None)
df.groupby('Platform')['Rank'].min().to_frame().sort_values('Rank').T
df.groupby('Year')['Rank'].min().to_frame().sort_values('Rank').T
df.groupby('Publisher')['Rank'].min().to_frame().sort_values(by='Rank').T

# 每年游戏发行数量可视化
palette = sns.color_palette('Set2', n_colors=len(df.Year.unique()))
fig, ax1 = plt.subplots(1,1,figsize=(20, 8),dpi=120)
plt.subplots_adjust(hspace=0.4,top=0.95,bottom = 0.1)
count = df['Year'].value_counts()
count = pd.DataFrame(count)
ax1 = sns.barplot(x='Year', y='count', data = count, palette = palette, hue = 'Year', legend = False, ax = ax1)
add_labels(ax1, count['count'])
ax1.set_title('每年游戏发行数量可视化')
ax1.set_xlabel('年份')
ax1.set_ylabel('游戏发行数量')
# 保存图片
plt.savefig(os.path.join(current_dir, '每年游戏发行数量.png'), dpi=300, bbox_inches='tight')

# 各地区游戏销售数据可视化
qushi = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales','Global_Sales']
qushi = pd.pivot_table(df, values=qushi, index='Year', aggfunc='sum')
fig, ax = plt.subplots(figsize=(20, 8),dpi=120)
qushi.T
sns.lineplot( data=qushi, palette='Set2')
plt.title('各地区游戏销售数据可视化')
plt.xlabel('年份')
plt.ylabel('销售额 (百万美元)')
plt.legend(title='地区')
# 保存图片
plt.savefig(os.path.join(current_dir, '各地区游戏销售数据.png'), dpi=300, bbox_inches='tight')

# 游戏销售数据相关系数热力图可视化
fig = plt.figure(figsize=(20, 8),dpi=120)
matrix = df.corr(numeric_only=True)
sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('游戏销售数据相关系数热力图可视化')
plt.xlabel('游戏销售数据指标')
plt.ylabel('游戏销售数据指标')
# 保存图片
plt.savefig(os.path.join(current_dir, '游戏销售数据相关系数热力图.png'), dpi=300, bbox_inches='tight')

# 游戏销售数据，游戏发行年份，游戏发行商排名可视化
P = ['Nintendo', 'Electronic Arts', 'Activision', 'Sony Computer Entertainment','Ubisoft']
Dp = df[df['Publisher'].isin(P)]
Dp = pd.pivot_table(Dp, values='Global_Sales', index='Year', columns='Publisher', aggfunc='sum')
ax = Dp.plot(figsize=(20, 8), marker='o', markersize=8, linewidth=2)
# 标记每个发行商的最大值
for publisher in Dp.columns:
    max_idx = Dp[publisher].idxmax()
    max_val = Dp[publisher].max()
    ax.annotate(f'{max_val:.1f}', 
               xy=(max_idx, max_val), 
               xytext=(0, 10), 
               textcoords='offset points',
               ha='center', va='bottom',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
plt.title('游戏销售数据，游戏发行年份，游戏发行商排名可视化', fontsize=16, fontweight='bold')
plt.subplots_adjust(hspace=0.4,top=0.95,bottom = 0.1)
plt.xlabel('年份', fontsize=14)
plt.ylabel('销售额 (百万美元)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='游戏发行商', fontsize=12, title_fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 保存图片
plt.savefig(os.path.join(current_dir, '游戏销售数据，游戏发行年份，游戏发行商排名可视化.png'), dpi=300, bbox_inches='tight')

# 游戏类型和发行商销售数据可视化
region_sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
# 创建游戏类型和发行商的销售数据透视表
Dpg = pd.pivot_table(data=df, 
                    values=region_sales_columns, 
                    index=['Genre', 'Publisher'], 
                    aggfunc='sum')
# 只保留销售额较高的发行商（例如：全球销售额大于10百万美元）
Dpg = Dpg[Dpg['Global_Sales'] > 10].copy()
# 按全球销售额降序排序
Dpg.sort_values(by='Global_Sales', ascending=False, inplace=True)
# 重置索引以便绘图
Dpg_reset = Dpg.reset_index()
# 重新组织数据：以地区为x轴，发行商为堆叠元素
# 选择前10个发行商进行可视化
Dpg_top10 = Dpg_reset.head(10)
# 定义销售地区
regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
region_names = ['北美', '欧洲', '日本', '其他地区']
# 准备数据：按地区和发行商分组的销售额
region_publisher_sales = []
for region in regions:
    region_sales = []
    for _, row in Dpg_top10.iterrows():
        region_sales.append(row[region])
    region_publisher_sales.append(region_sales)
# 绘制以地区为x轴的堆叠柱状图
plt.figure(figsize=(16, 9), dpi=120)
# 创建堆叠柱状图：x轴为地区，堆叠为不同发行商
bottom = [0] * len(regions)
colors = plt.cm.Set3.colors[:len(Dpg_top10)]  # 使用不同颜色区分发行商
for i, (publisher, sales_data) in enumerate(zip(Dpg_top10['Publisher'], zip(*region_publisher_sales))):
    plt.bar(range(len(regions)), sales_data, color=colors[i], label=publisher, bottom=bottom, width=0.8)
    # 更新底部位置
    bottom = [bottom[j] + sales_data[j] for j in range(len(regions))]
# 设置x轴标签为地区名称
plt.xticks(range(len(regions)), region_names, fontsize=14)
# 设置图表标题和标签
plt.title('主要游戏发行商各地区销售数据（按全球销售额排序）', fontsize=16, fontweight='bold')
plt.xlabel('销售地区', fontsize=14)
plt.ylabel('销售额（百万美元）', fontsize=14)
plt.yticks(fontsize=12)
# 添加图例和格线
plt.legend(title='游戏发行商', fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 调整布局，为右侧图例留出空间
plt.tight_layout(rect=[0, 0, 0.85, 1])
# 保存图片
plt.savefig(os.path.join(current_dir, '游戏类型和发行商销售数据.png'), dpi=300, bbox_inches='tight')