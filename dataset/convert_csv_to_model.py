import pandas as pd
'''将原本csv文件的每一行重新排序，使得偶数索引的值在前，奇数索引的值在后，然后保存到新的csv文件，也就是前100个数值是x坐标，后100个数值是y坐标'''
def reorder_csv(file_path, output_path):
    # 读取CSV文件，每行有200个空格分隔的数值
    data = pd.read_csv(file_path, header=None, sep='\\s+', engine='python')

    # 重新排序每一行的数值
    def reorder_row(row):
        even_indices = row[::2]  # 偶数索引的值
        odd_indices = row[1::2]  # 奇数索引的值
        return pd.Series(list(even_indices) + list(odd_indices))

    # 对每一行应用重新排序函数
    reordered_data = data.apply(reorder_row, axis=1)

    # 保存重新排序后的数据到新的CSV文件
    reordered_data.to_csv(output_path, header=False, index=False, sep=' ')
if __name__ == '__main__':
    # 输入文件路径
    input_file = 'pedestrians_positions_MI.csv'
    # 输出文件路径
    output_file = 'pedestrians_positions_MI_converted.csv'

    # 调用函数重新排序CSV文件
    reorder_csv(input_file, output_file)
