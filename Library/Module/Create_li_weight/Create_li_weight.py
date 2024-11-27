import pandas as pd

# 定义颜色分配函数
def assign_color(weight):
    if weight >= 8:
        return '紅色'
    elif 6 <= weight < 8:
        return '橙色'
    elif 4 <= weight < 6:
        return '黃色'
    elif 2 <= weight < 4:
        return '綠色'
    else:
        return '無色'

# 读取数据
input_file = r'C:\Users\User\OneDrive\桌面\LinZhaiPLUS專案\LinZhaiPLUS\Library\Module\Create_li_weight\Taipei_Li_data.csv'
df = pd.read_csv(input_file)

# 定义行政区的权重
district_weights = {
    '松山區': 7.69,
    '信義區': 10,
    '大安區': 9.74,
    '中山區': 7.74,
    '中正區': 9.05,
    '大同區': 6.8,
    '萬華區': 5.34,
    '文山區': 5.66,
    '南港區': 8.06,
    '內湖區': 6.59,
    '士林區': 7.25,
    '北投區': 6.43,
}

# 计算每个行政区的平均人口数
district_avg_population = df.groupby('行政區')['人口數-合計'].mean().to_dict()

# 初始化权重和颜色列表
weights = []
colors = []

# 计算每个里的权重和颜色
for idx, row in df.iterrows():
    district = row['行政區']
    li = row['里別']
    li_population = row['人口數-合計']
    avg_population = district_avg_population[district]
    district_weight = district_weights[district]
    
    # 根据人口比例调整权重
    li_weight = district_weight * (li_population / avg_population)
    
    # 确保权重在0到10之间
    li_weight = min(li_weight, 10)
    li_weight = round(li_weight, 2)
    
    # 分配颜色
    li_color = assign_color(li_weight)
    
    weights.append(li_weight)
    colors.append(li_color)

# 将权重和颜色添加到数据框
df['權重'] = weights
df['顏色'] = colors

# 准备输出数据
output_df = df[['行政區', '里別', '權重', '顏色']]
output_df.insert(0, '縣市', '台北市')
output_df.columns = ['縣市', '區', '里', '權重', '顏色']

# 保存到 CSV 文件
output_file = r'C:\Users\User\OneDrive\桌面\LinZhaiPLUS專案\LinZhaiPLUS\Library\Module\Create_li_weight\Taipei_Li_weight.csv'
output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
