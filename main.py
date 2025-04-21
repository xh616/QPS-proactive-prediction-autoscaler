# import requests
import threading
import pandas as pd
import csv
import numpy as np
import torch
import json
import math
import subprocess
import time
import pytz
from threading import Thread
import RWKV_Class
from Informer import Informer
from RWKV_Model import RWKV
# from Informer import Informer
import kubernetes as k8s
from kubernetes import config, client, watch
from kubernetes.client import AppsV1Api
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import StandardScaler

qps_port = 3000
consul_port = 8500
namespace = "k8s"
replica_qps_up_limit = 800
replica_qps_down_limit = 500

node_pro_url = {
    "node1": "http://192.168.31.101:9090",
    "node2": "http://192.168.31.102:9090",
    "node7": "http://192.168.31.107:9090",
    "node8": "http://192.168.31.108:9090",
}

node_pod_counts = {
    "node1": 0,
    "node2": 0,
    "node7": 0,
    "node8": 0
}
node_name_ip_mapping = {}
pod_name_ip_mapping = {}

history_data_node1 = []
history_data_node2 = []
history_data_node7 = []
history_data_node8 = []


def get_statefulset(namespace='k8s'):
    k8s.config.load_kube_config(config_file=".kube/config")
    v1_apps = AppsV1Api()
    statefulset = v1_apps.list_namespaced_stateful_set(namespace)
    # sts1 = v1_apps.stateful_set
    # print(statefulset)
    return statefulset


def doScale(sts_name, namespace, number_of_pod):
    if number_of_pod > 0:
        scale_up_statefulset(sts_name, namespace, number_of_pod)
    elif number_of_pod == 0:
        pass
    else:
        scale_down_statefulset(sts_name, namespace, number_of_pod)


def get_pod_count(statefulset):
    pod_count = {}
    for statefulset in statefulset.items:
        metadata = statefulset.metadata
        pod_count[metadata.name] = statefulset.status.replicas
    return pod_count


def get_next_qps_index(existing_pod_names):
    # 如果没有已有的 pod，返回 1
    if not existing_pod_names:
        return 0
    # 提取每个 pod 名称中的数字，并找到最大值
    existing_indices = [int(name.split("qps-")[1]) for name in existing_pod_names]
    max_index = max(existing_indices)
    # 返回下一个 QPS 的 index
    return max_index + 1


def get_all_pods(namespace):
    v1 = client.CoreV1Api()
    pod_list = v1.list_namespaced_pod(namespace)
    pod_names = [pod.metadata.name for pod in pod_list.items if "qps" in pod.metadata.name]
    return pod_names


def scale_up_statefulset(sts_name, namespace, replicas):
    appsv1 = client.AppsV1Api()
    # statefulset.itmes.spec.replicas = replicas
    # appsv1.replace_namespaced_stateful_set(name=statefulset, namespace=namespace, body=statefulset)
    statefulset = get_statefulset()
    for sts in statefulset.items:
        if sts.metadata.name == sts_name:
            sts.spec.replicas += replicas
            if sts.spec.replicas > 6:
                sts.spec.replicas = 6
            print(f"-------->StatefulSet: {sts.metadata.name} after add replicas: {sts.spec.replicas}<--------")
            appsv1.replace_namespaced_stateful_set(name=sts.metadata.name, namespace=namespace, body=sts)


def scale_down_statefulset(sts_name, namespace, replicas):
    appsv1 = client.AppsV1Api()
    # statefulset.itmes.spec.replicas = replicas
    # appsv1.replace_namespaced_stateful_set(name=statefulset, namespace=namespace, body=statefulset)
    statefulset = get_statefulset()
    for sts in statefulset.items:
        if sts.metadata.name in sts_name:
            sts.spec.replicas += replicas
            if sts.spec.replicas < 1:
                sts.spec.replicas = 1
            print(f"-------->StatefulSet: {sts.metadata.name} after delete replicas: {sts.spec.replicas}<--------")
            appsv1.replace_namespaced_stateful_set(name=sts.metadata.name, namespace=namespace, body=sts)


def scale_statefulset(statefulset, namespace, replicas):
    appsv1 = client.AppsV1Api()
    # statefulset.itmes.spec.replicas = replicas
    # appsv1.replace_namespaced_stateful_set(name=statefulset, namespace=namespace, body=statefulset)
    for sts in statefulset.items:
        sts.spec.replicas = replicas
        appsv1.replace_namespaced_stateful_set(name=sts.metadata.name, namespace=namespace, body=sts)


def get_http_requests_total(prometheus_url):
    # 该方法没有使用 Prometheus Fed
    prom = PrometheusConnect(url=prometheus_url)

    # 查询 http_requests_total 指标
    # query = 'http_requests_total{job="%s"}' % job
    query = 'rate(http_requests_total[1m])'
    # end_time = datetime.utcnow()
    # start_time = end_time - timedelta(hours=1)
    result = prom.custom_query_range(query, start_time=datetime.now(), end_time=datetime.now(), step='15s')
    print(f"-------->Prometheus query :{result}")
    # node_qps_total = {}
    # node_pod_total = {}
    pod_qps_total = 0
    for sample in result:
        metric = sample['metric']
        pod_qps_value = float(sample['values'][0][1])
        pod_qps_total += pod_qps_value
        # node = metric['node']
        # if node in node_pod_total:
        #     node_pod_total[node] += 1
        # else:
        #     node_pod_total[node] = 1
        # # pod_name = metric['instance'].split(":")[0]
        #
        # if node in node_qps_total:
        #     node_qps_total[node] += pod_qps_value
        # else:
        #     node_qps_total[node] = pod_qps_value
    return pod_qps_total


def calculate_requests_percentage(http_requests_total, pod_count):
    node_requests_total = {}

    for instance, total in http_requests_total.items():
        node = instance.split('_')[0]  # Extract node name from instance
        node_pod = instance.split(':')[0]
        node_requests_total.setdefault(node, 0)
        node_requests_total[node] += total
    # print(instance)
    # print(total)
    # Calculate percentage
    percentage = {node: total / sum(node_requests_total.values()) for node, total in node_requests_total.items()}

    return node_requests_total, percentage


def update_node_pod_counts(node, pod_count):
    # 更新字典中指定节点的 Pod 个数
    global node_pod_counts
    node_pod_counts[node] = pod_count


def update_all_node_pod_counts():
    global node_pod_counts
    for key, _ in node_pod_counts.items():
        node_pod_counts[key] = get_node_pod(key)


def update_all_pod_ip(all_pods):
    global pod_name_ip_mapping
    for pod in all_pods:
        pod_name_ip_mapping[pod] = get_pod_ip(pod)


def get_node_pod(node):
    podnum = 0
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace)
    # print(pods)
    for pod in pods.items:
        if pod.status.phase == "Running" and node in pod.spec.node_name:
            podnum += 1
    return podnum


def get_pod_ip(pod):
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace)
    for p in pods.items:
        if pod in p.metadata.name:
            pod_ip = p.status.pod_ip
    return pod_ip


def init_check(event, namespace):
    global pod_name_ip_mapping
    """初始化检查，保证每个节点最初都只有一个副本"""
    pod_name = event['object'].metadata.name
    bind_node = pod_name.split("qps")[0]
    if node_pod_counts[bind_node] < 1:
        pod_ip = bind(event['object'], bind_node, namespace)
        pod_name_ip_mapping[pod_name] = pod_ip
        node_ip = node_name_ip_mapping[bind_node]
        register_new_replica_in_prometheus(bind_node, node_ip, pod_ip, pod_name, namespace)
        register_new_replica_in_nginx(node_ip, pod_ip)
        update_node_pod_counts(bind_node, 1)
        flag = "bind"
    else:
        update_all_node_pod_counts()
        flag = "None"
    print("---->Finish init check for node<----")
    print(f"---->{node_pod_counts}<----")
    return flag


def list_watch(namespace):
    w = watch.Watch()
    v1 = client.CoreV1Api()

    for event in w.stream(v1.list_namespaced_pod, namespace):
        print(
            "---->监听到资源: {} 名称: {} 变化类型：{}<----".format(event['object'].kind, event['object'].metadata.name,
                                                                   event['type']))
        # print(event)
        if event['object'].status.phase == "Pending" and event['type'] == "ADDED" and \
                event['object'].spec.scheduler_name == "qps":
            try:
                # pod_name = event['object'].metadata.name
                init_flag = init_check(event, namespace)

                if init_flag == "bind":
                    continue
                else:
                    global pod_name_ip_mapping
                    pod_name = event['object'].metadata.name
                    # bind_node = choose_node()
                    bind_node = pod_name.split("qps")[0]
                    node_ip = node_name_ip_mapping[bind_node]
                    # pod_ip = threading.Thread(target=bind, args=(event['object'], bind_node, namespace),)
                    pod_ip = bind(event['object'], bind_node, namespace)
                    pod_name_ip_mapping[pod_name] = pod_ip
                    # pod_ip = schedule(event['object'],bind_node)
                    # pod_ip = bind(pod_name, "node1")
                    # TODO: 获得新副本ip并修改bind_node上Prometheus的配置文件进行热更新
                    register_new_replica_in_prometheus(bind_node, node_ip, pod_ip, pod_name, namespace)
                    # TODO: 同时修改bind_node上Nginx的upstream热更新负载均衡
                    register_new_replica_in_nginx(node_ip, pod_ip)
                    update_all_node_pod_counts()
            except client.rest.ApiException as e:
                print("exception:" + json.loads(e.body)['message'])
                pass

        if event['object'].kind == "Pod" and event['type'] == "DELETED" and \
                event['object'].spec.scheduler_name == "qps":
            global node_pod_counts
            pod_name = event['object'].metadata.name
            bind_node = pod_name.split("qps")[0]
            node_ip = node_name_ip_mapping[bind_node]
            pod_ip = pod_name_ip_mapping[pod_name]
            # TODO:取消删除pod时对Prometheus的请求是否能让下次更快读取到负载数值？
            # deregister_replica_in_prometheus(node_ip, pod_name)
            # TODO: 同时修改bind_node上Nginx的upstream热更新负载均衡
            deregister_replica_in_nginx(node_ip, pod_ip)

            # node_pod_counts[bind_node] -= 1
            update_all_node_pod_counts()
        # if event['type'] == "DELETED" and event['object'].spec.scheduler_name == "qps":
        #     delete(event['object'])


def register_new_replica_in_prometheus(node_name, node_ip, pod_ip, pod_name, namespace):
    # parameter = (f'{{"id": "qps{next_index}","name": "qps{next_index}","address": "{pod_ip}",'
    #              f'"port": {qps_port},"tags": ["qps","{node_name}"],'
    #              f'"checks": [{{"http": "http://{pod_ip}:{qps_port}/metrics", "interval": "5s"}}]}}')
    try:
        parameter = {
            # "id": f"qps{next_index}",
            # "name": f"qps{next_index}",
            "id": pod_name,
            "name": pod_name,
            "address": pod_ip,
            "port": qps_port,
            "tags": ["qps", node_name],
            "checks": [{"http": f"http://{pod_ip}:{qps_port}/metrics", "interval": "5s"}]
        }
        print(parameter)
        # curl -X PUT -d '{"id": "qps0","name": "qps0","address": "192.168.31.101","port": 3000,"tags": ["qps","node1"], "checks": [{"http": "http://192.168.168.31:3000/metrics", "interval": "5s"}]}' http://192.168.31.101:8500/v1/agent/service/register
        # 通过consul注册副本服务
        # curl_command = f'curl -X PUT -d \'{parameter}\' http://{node_ip}:{consul_port}/v1/agent/service/register'
        url = f"http://{node_ip}:{consul_port}/v1/agent/service/register"
        response = requests.put(url, json=parameter)
        print(response.text)
        # subprocess.run(curl_command, shell=True)
    except:
        pass

def deregister_replica_in_prometheus(node_ip, pod_name):
    try:
        consul_url = f"http://{node_ip}:{consul_port}/v1/agent/service/deregister/{pod_name}"
        response = requests.put(consul_url)
        print(response.text)
        print(f"{pod_name} deregistered in prometheus")
    except:
        pass


def register_new_replica_in_nginx(node_ip, pod_ip):
    try:
        parameter = {
            "weight": 1,
            "max_fails": 2,
            "fail_timeout": 10,
            "down": 0
        }
        # 构造 URL
        url = f'http://{node_ip}:{consul_port}/v1/kv/upstreams/app/{pod_ip}:{qps_port}'
        response = requests.put(url, json=parameter)
        print(response.text)
    except:
        pass


def deregister_replica_in_nginx(node_ip, pod_ip):
    try:
        consul_url = f'http://{node_ip}:{consul_port}/v1/kv/upstreams/app/{pod_ip}:{qps_port}'
        response = requests.delete(consul_url)
        print(response.text)
        print(f"{pod_ip}:{qps_port} deregistered in Nginx")
    except:
        pass
    # parameter = {
    #     "weight": 0,
    #     "max_fails": 2,
    #     "fail_timeout": 10,
    #     "down": 0
    # }
    # # 构造 URL
    # url = f'http://{node_ip}:{consul_port}/v1/kv/upstreams/app/{pod_ip}:{qps_port}'
    # response = requests.put(url, json=parameter)
    # print(response.text)
    # return


def schedule(pod):
    return ""


def bind(pod, node_name, namespace):
    # Must use V1Pod instance
    v1 = client.CoreV1Api()
    target = client.V1ObjectReference(api_version='v1', kind='Node', name=node_name)
    meta = client.V1ObjectMeta()
    meta.name = pod.metadata.name
    body = client.V1Binding(target=target, metadata=meta)
    global node_pod_counts
    # print("target:{}".format(str(target)))
    try:
        api_response = v1.create_namespaced_pod_binding(name=pod.metadata.name, namespace=namespace,
                                                        body=body, _preload_content=False)
        print(f"---->Bind_API_RESPONSE:{api_response}<----")
        while 1:
            bound_pod = v1.read_namespaced_pod(name=pod.metadata.name, namespace=namespace)
            # print(bound_pod)
            pod_ip = bound_pod.status.pod_ip
            if pod_ip != None:
                break
        node_pod_counts[node_name] += 1
        print(f"---->binding pod ip: {pod_ip}<----")
        return pod_ip
        # print(pod)
        # while pod.status.phase != "Running" and pod.status.pod_ip is None:
        #     time.sleep(1)
        # pod_ip = pod.state.pod_ip
        # print(pod_ip)
        # return pod_ip
    except Exception as e:
        print("Warning when calling CoreV1Api->create_namespaced_pod_binding: %s\n" % e)
        pass


def make_decision(node_qps_total):
    return 1


def k8s_nodes_available():
    global node_name_ip_mapping
    v1 = client.CoreV1Api()
    for n in v1.list_node().items:
        if not n.spec.unschedulable:
            for status in n.status.conditions:
                if (status.status == "True" and status.type == "Ready"
                        and 'node-role.kubernetes.io/edge' in n.metadata.labels.keys()):
                    node_name_ip_mapping[n.metadata.name] = n.status.addresses[0].address


def calculateTheta(replicas):
    # 根据单个副本数能处理的最大副本数计算当前副本数能够处理的qps阈值
    # 例如单个850 两个就是850*2
    theta = replicas * replica_qps_up_limit
    return theta


def getMinReplicas(node_qps_total):
    return math.ceil(node_qps_total / replica_qps_up_limit)


def get_model_pre(history_data):
    data = np.array(history_data)
    scaler = StandardScaler()
    # df_raw = pd.read_csv("./output3.csv")
    # train_data = df_data[border1s[0]:border2s[0]]
    train_data = pd.read_csv("./train_data.csv")
    tdata = np.array(train_data)
    scaler.fit(tdata)
    print(history_data.shape)
    sdata = scaler.transform(history_data).reshape(1, 120, 4)

    parameters = RWKV_Class.Model_Param()
    # model = RWKV.Model(parameters)
    model = Informer.Model(parameters)
    # model.load_state_dict(torch.load('RWKV_Model/checkpoint.pth', map_location='cpu'))
    model.load_state_dict(torch.load('Informer/checkpoint.pth', map_location='cpu'))
    # print(model)
    # d = []

    # data = np.random.rand(1, 120, 4)
    # print(data.shape)
    data = torch.from_numpy(sdata).float()
    # print(data)
    with torch.no_grad():
        pred = model(data)

    # predicted_data = []  # 存储预测数据的列表
    # current_data = sdata  # 当前数据为初始数据

    # 循环预测多个时间步
    # for _ in range(4):
    #     # 使用模型预测下一个时间步的数据
    #     with torch.no_grad():
    #         # 将当前数据转换为张量
    #         x = torch.tensor(current_data, dtype=torch.float32)
    #         # 使用模型进行预测
    #         prediction = model(x)
    #         # 将预测结果转换为 numpy 数组并添加到预测数据列表中
    #         predicted_data.append(prediction.squeeze().numpy())
    #
    #     # 更新当前数据，将预测结果添加到当前数据末尾，并删除第一个元素，实现滚动
    #     current_data = np.concatenate((current_data[:, 60:, :], prediction.detach().numpy()), axis=1)

    # print(len(current_data))
    # print(current_data[0].shape)

    futuredata = pred.detach().numpy()
    # predicted_data = np.concatenate(predicted_data, axis=1)
    ffdata = futuredata.reshape(60, 4)
    # print(ffdata.shape)
    fdata = scaler.inverse_transform(ffdata)
    print(fdata.shape)
    print(fdata.dtype)
    print(type(fdata))
    print(fdata)
    output_file = 'output_Informer.csv'
    data_to_write = fdata.reshape(60, 4)
    df = pd.DataFrame(data_to_write)
    df.to_csv(output_file, index=False)
    return


def record_data():
    for _, url in node_pro_url:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=1)
        # PromQL 查询语句：计算一分钟内指标的均值
        query = 'sum(rate(http_requests_total[1m]))'
        # 查询参数
        params = {
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': 60  # 查询步长，这里设置为 60 秒，即查询一分钟内的平均值
        }
        query_url = f"{url}/api/v1/query"
        response = requests.get(query_url, params=params)
        data = response.json()
        # print(data)
        # # 获取查询结果中的均值
        average_value = data['data']['result'][0]['value'][1]
        # print(average_value)
        if average_value > 10:
            history_data.append()
            print(f"---->history data added len{len(history_data)}<----")


def countdown_timer(duration):
    global history_data
    start_time = time.time()
    end_time = start_time + duration
    while time.time() < end_time:
        remaining_time = int(end_time - time.time())
        # print(f"Time remaining: {remaining_time} seconds", end='\r')
        time.sleep(1)
    print(f"---->add data<----")

    print(f"---->history data: {history_data}<----")



def predict_future(model, initial_data, future_steps=4):
    predicted_data = []  # 存储预测数据的列表
    current_data = initial_data  # 当前数据为初始数据

    # 循环预测多个时间步
    for _ in range(future_steps):
        # 使用模型预测下一个时间步的数据
        with torch.no_grad():
            # 将当前数据转换为张量
            x = torch.tensor(current_data, dtype=torch.float32)
            # 使用模型进行预测
            prediction = model(x)
            # 将预测结果转换为 numpy 数组并添加到预测数据列表中
            predicted_data.append(prediction.squeeze().numpy())

        # 更新当前数据，将预测结果添加到当前数据末尾，并删除第一个元素，实现滚动
        current_data = np.concatenate((current_data[:, 1:, :], prediction.detach().numpy().reshape(1, 1, -1)), axis=1)

    return np.array(predicted_data)



def write_to_file():
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for node, url in node_pro_url.items():
            replica_count = node_pod_counts[node]
            file_path = f"./{node}_pod_numbert.txt"
            with open(file_path, 'a') as file:
                file.write(f"{current_time} - Node: {node}, Replicas: {replica_count}\n")
        time.sleep(60)

def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def pd_read_csv(file_path):
    data = pd.read_csv(file_path, dtype=float, header=None)
    return data


def clean_load(load):
    return ''.join(filter(lambda x: x.isdigit() or x == '.', load))

if __name__ == "__main__":

    data = pd_read_csv("./1220-1340.csv")
    # data = pd_read_csv("./1280-1400.csv")
    # data = pd_read_csv("./1340-1460.csv")
    # data = pd_read_csv("./1400-1520.csv")
    # get_model_pre(data)
    # exit(1)

    Thread(target=countdown_timer, args=(60,)).start()
    index = 0
    csv_file ="./predicted_data.csv"
    all_predicted_data = read_csv(csv_file)
    node_load = dict()
    node_load["node1"] = [row[0] for row in all_predicted_data]
    node_load["node2"] = [row[1] for row in all_predicted_data]
    node_load["node7"] = [row[2] for row in all_predicted_data]
    node_load["node8"] = [row[3] for row in all_predicted_data]
    # print(node_load["node1"][index])
    # exit(1)
    k8s.config.load_kube_config(config_file=".kube/config")
    k8s_nodes_available()
    print(node_name_ip_mapping)
    all_pods = get_all_pods(namespace)
    print(all_pods)
    update_all_pod_ip(all_pods)
    # next_index = get_next_qps_index(all_pods)
    # print(next_index)
    update_all_node_pod_counts()
    # update_all_pod_ip()
    # Thread(target=list_watch, args=("k8s",)).start()
    Thread(target=write_to_file, ).start()
    # list_watch("k8s")
    # exit(0)
    node_qps_total = dict()
    # node_pod_count = dict()

    while index < 240:
        statefulset = get_statefulset()
        # exit(0)
        # print(f"{statefulset}")
        pod_count = get_pod_count(statefulset)
        # print("-------->Pod Count:", pod_count)
        # print(f"-------->Pod ip:{pod_name_ip_mapping}")
        # 遍历url 获取不同节点的qps数据，据此做副本扩缩容
        for node, url in node_pro_url.items():
            node_pod_total = node_pod_counts[node]
            qps_total = get_http_requests_total(url)
            node_qps_total[node] = qps_total
            print(f"-------->pod_qps_total: {node_qps_total}<----")
            print(f"-------->Node Pod Count: {node_pod_counts}<----")
            if node_pod_counts[node] > 0:
                # threshold = calculateTheta(node_pod_total)
                minReplicas = getMinReplicas(qps_total)
                print(f"-------->Node {node} 监控数据需要副本数: {minReplicas}<----")
                addPredData = getMinReplicas(float(clean_load(node_load[node][index])))
                print(f"节点:: {node} 预测负载:: {node_load[node][index]} 预测需要副本数:: {addPredData}")
                finalReplicas = math.ceil(minReplicas * 1 + addPredData * 0)
                print(f"finalReplicas:: {finalReplicas}")
                # numberOfPodToChange = minReplicas - node_pod_total
                numberOfPodToChange = finalReplicas - node_pod_total
                print(
                    f"-------->Node {node} need minReplicas: {finalReplicas} && Number to change: {numberOfPodToChange}<----")
                sts_name = f"{node}qps"
                threading.Thread(target=doScale, args=(sts_name, namespace, numberOfPodToChange),).start()
                # doScale(sts_name, namespace, numberOfPodToChange)
        update_all_node_pod_counts()
        index += 1
        # 控制扩缩容的时间
        time.sleep(59)
    exit(1)
        # exit(0)
        # node_replica_to_change = make_decision(pod_qps_total)
        # node_requests_total, percentage = calculate_requests_percentage(http_requests_total, node_pod_count)
        # print("Node Requests Total:", node_requests_total)
        # print("Percentage:", percentage)
        # decision = make_decision()
        # if decision == 1:
        #     scale_statefulset(statefulset,"default",3)
