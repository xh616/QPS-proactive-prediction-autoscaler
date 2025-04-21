from kubernetes import client, config, watch
from kubernetes import client, config
import time
import kubernetes as k8s

def choose_pod_to_delete():
    # 编写删除 Pod 的逻辑
    pass

def delete_pod(pod_name):
    core_api = client.CoreV1Api()
    try:
        # 调用 Kubernetes API 删除 Pod
        core_api.delete_namespaced_pod(name=pod_name, namespace="default")

        print(f"Pod {pod_name} deleted successfully.")
    except Exception as e:
        print(f"Failed to delete Pod {pod_name}: {e}")

def handle_event(event):
    if event['type'] != 'MODIFIED':
        print("handle_111111111111")

    statefulset = event['object']
    current_replicas = statefulset.spec.replicas
    # previous_replicas = event['previous_object'].spec.replicas
    print(current_replicas)
    # delete_pod("qps-3")
    # 检查副本数量是否减少
    # if current_replicas < previous_replicas:
    #     # 根据自定义策略选择要删除的 Pod
    #     # 这里需要您根据自己的实际需求编写选择逻辑
    #     pod_name = "qps-3"
    #     delete_pod(pod_name)


def main():
    # 加载 Kubernetes 配置
    k8s.config.load_kube_config(config_file=".kube/config")

    # 创建 Kubernetes API 客户端
    core_api = client.CoreV1Api()
    statefulset_api = client.AppsV1Api()

    # 监听 StatefulSet 对象的变更事件
    w = watch.Watch()
    for event in w.stream(statefulset_api.list_namespaced_stateful_set, namespace='default'):
        print(event)
        print("111111111111")
        handle_event(event)
        time.sleep(1)


if __name__ == "__main__":
    main()