import csv
import time
import subprocess
from datetime import datetime

# 读取 CSV 文件
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

# 发送负载数据给节点
def send_load_to_node(load, node, index):
    # 使用 subprocess 执行 vegeta 命令
    subprocess.run(f"echo 'GET http://{node}' | vegeta attack -rate={load} -duration=60s | vegeta report > /root/load_report/report/{node}_report_{index}.txt &", shell=True)


def clean_load(load):
    return ''.join(filter(lambda x: x.isdigit() or x == '.', load))


# 主函数
def main():
    csv_file = '/root/load_report/1240data.csv'  # CSV 文件路径
    data = read_csv(csv_file)

    # 从 CSV 文件中提取四列数据，分别对应四个节点的负载
    node1_load = [row[0] for row in data]
    node2_load = [row[1] for row in data]
    node3_load = [row[2] for row in data]
    node4_load = [row[3] for row in data]
    #print(len(node1_load))
    #print(node1_load)
    #exit(1)
    # 记录开始时间
    # start_time = time.time()

    # 每分钟发送一次负载
    index = 0
    while index < len(node1_load):
        print(datetime.now())
        # 发送负载给每个节点
        print(f"sent load to node1 {node1_load[index]}")
        send_load_to_node(int(float(clean_load(node1_load[index]))), "192.168.31.101",index)
        print(f"sent load to node2 {node2_load[index]}")
        send_load_to_node(int(float(clean_load(node2_load[index]))), "192.168.31.102",index)
        print(f"sent load to node7 {node3_load[index]}")
        send_load_to_node(int(float(clean_load(node3_load[index]))), "192.168.31.107",index)
        print(f"sent load to node8 {node4_load[index]}")
        send_load_to_node(int(float(clean_load(node4_load[index]))), "192.168.31.108",index)
        index += 1
        # 等待一分钟
        time.sleep(60)

if __name__ == "__main__":
    main()

