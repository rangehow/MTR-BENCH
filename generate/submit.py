#encoding=utf-8
import subprocess
import argparse
import json
from datetime import datetime

def add_time(taskname: str):
    """taskname后面增加时间戳

    Args:
        taskname (str): 原始任务名

    Returns:
        _str: 增加时间戳后的任务名
    """
    if taskname == None:
        taskname = "NONAME"
    current_time = datetime.now()
    # 去除秒后精度
    current_time = str(current_time).split(".")[0]
    # 去除空格
    current_time = "-".join(current_time.split(" "))
    # 去除":"
    current_time = "-".join(current_time.split(":"))

    return taskname + "-" + current_time

# "default"是triton2.0.0+flash_attn2。百川模型的fa2依赖triton2.0.0，如果升级为2.1.0会报错(flash_attention相关报错)
# "flash_attention1-moe"是triton2.1.0+flash_attn1。moe相关精度验证的回归测试依赖triton2.1.0
# llama类的（包括llama类型的moe模型）模型推荐使用flash_attention2-triton2.1.0
images = {
    "default" : "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/data-hadoop-hdp_pt1.13.1-py39-nccl2.14-cuda11.7-flashatten2.4.2-fastdispatch-bc771c09",
    "flash_attention1" : "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/data-hadoop-hdp_pt1.13.1-py39-nccl2.14-cuda11.7-flashatten1.0.9-fastdispatch-27134a57",
    "flash_attention1-moe" : "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.phxmlp.mtjupyter.singleuser/hdp_training_pytorch1.13.1_cuda11.7_ggm_fusegrad_fa1_tt21_rope_fusion_28c48d3b",
    "flash_attention-1.0.7" : "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/data-hadoop-hdp_pt1.13.1-py39-nccl2.14-cuda11.7-flashatten1.0.7-d6a74aa2",
    "flash_attention2-triton2.1.0": "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.phxmlp.mtjupyter.singleuser/hdp_training_tf1.13_pytorch1.13.1_cuda11.7_python39_ggm_fusegrad_fa2_rope_fusion_8ba74052",
    "taming" : "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.phxmlp.mtjupyter.singleuser/aipnlp_training_pytorch1.13.1_cuda11.7_python39_raw_2a052d2a",
    "taming_multimodal_flash_attn" : "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.phxmlp.mtjupyter.singleuser/aipnlp_training_pytorch1.13.1_cuda11.7_python39_multimodal_flash_attn_b48069d4"
}

# h800_image = "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/data-hadoop-hdp_hadoop-hdp_adoop-hdp_aipnlp_training_pytorch2.1.0_cuda12.2.1_taming_transformer_python39-1d6dd6db"
h800_image = "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.phxmlp.mtjupyter.singleuser/aipnlp_training_pytorch2.1.0_cuda12.2.1_python39_taming_moe_0830_v2_5b02255b"
# h910b_image = "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/data-hadoop-hdp_megatron_npu:v1-2ae76a33"
h910b_image = "registryonline-hulk.sankuai.com/custom_prod/com.sankuai.phxmlp.mtjupyter.singleuser/hdp_training_aipnlp_ab1e7f87"

def adapt_to_h800(template: str):
    template = template.replace("worker.gcores80g = ", "worker.gcoresh800-80g = ")
    # replace all image to h800
    for image in images.values():
        template = template.replace(f"afo.docker.image.name = {image}", f"afo.docker.image.name = {h800_image}")
    template = template.replace(
        "afo.dolphinfs.otherusers = hadoop-llm-mm-data,hadoop-aipnlp,hadoop-vacv,hadoop-mlp-ckpt,hadoop-vision-data,hadoop-speech",
        "afo.dolphinfs.otherusers = hadoop-aipnlp,hadoop-mlp-ckpt,hadoop-hldy-nlp"
    )
    template = template.replace(
        "worker.script = ",
        "worker.script = source submit/env_configure.sh && "
    )
    # 使用怀来的hdfs写
    template = template.replace(
        "# afo.app.env.DFS_CLIENT_WRITE_ZONE = hlsc", 
        "afo.app.env.DFS_CLIENT_WRITE_ZONE = hlsc"
    )
    return template

def adapt_to_h910b(template: str):
    template = template.replace("worker.gcores80g = ", "worker.gcoresh910-64g = ")
    # replace all image to h910b
    for image in images.values():
        template = template.replace(f"afo.docker.image.name = {image}", f"afo.docker.image.name = {h910b_image}")
    template = template.replace(
        "worker.script = ",
        "worker.script = export HCCL_RDMA_TC=100 && export HCCL_RDMA_SL=3 && export HCCL_ASYNC_ERROR_HANDLING=0 && source /usr/local/Ascend/ascend-toolkit/set_env.sh && export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH && "
    )
    return template

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, required=False, default="examples/configs/experiment_configs/default_config.json")
    parser.add_argument("--model_config", type=str, required=False, default="examples/configs/model_configs/llama_config.json")
    parser.add_argument("--train_config", type=str,  required=False, default="examples/configs/train_configs/default_config.json")
    parser.add_argument("--config", type=str, required=False, default="examples/configs/default_config.json")
    parser.add_argument('--queue', '-q', type=str, required=False, default="root.zw05_training_cluster.hadoop-aipnlp.assistant")
    parser.add_argument('--account', '-a', type=str, required=False)
    parser.add_argument('--usergroup', '-u', type=str, required=False, default=None)
    parser.add_argument('--taskname', '-n', type=str, required=False, default=None)
    parser.add_argument('--gpu', '-g', type=int, required=False, default=8, help="numbers of gpus each worker")
    parser.add_argument('--workers', '-w', type=int, required=False, default=1, help="numbers of workers")
    parser.add_argument('--run_env', type=str, required=False, default="mt", choices=["mt", "tx"])
    parser.add_argument("--image", type=str, choices=["default", "flash_attention1", "flash_attention1-moe", "flash_attention-1.0.7", "flash_attention1-triton2.0.0", "flash_attention2-triton2.1.0", "taming"], default=None)
    parser.add_argument('--report_to', type=str, required=False, default="kinet", help="only support kinet now")
    parser.add_argument('--client_git_revision_publish', type=str, required=False, default="false", help="report git")
    parser.add_argument('--project_name', type=str, required=False, default="default", help="project name, default "
                                                                                            "set to 'default'")
    parser.add_argument('--preprocess_wait_time', '-s', type=str, required=False, default="", help="if start with data preprocessing, other processes but 0 will "
                                                                                                 "wait several seconds to avoid TCPStore timeout.")
    parser.add_argument('--max_retry', type=int, required=False, default=8, help="max numbers of retry")
    parser.add_argument('--param_report', type=str, required=False, default="N", help="whether report params and grads, Y/N, or set a integer as log interval")
    parser.add_argument('--hope_tensorboard', type=str, required=False, default="native", help="report to which tensorboard: native/hope")
    parser.add_argument('--pretrain', action='store_true', help="enable pretrain test mode.")
    parser.add_argument('--legacy_checkpoint', action='store_true', help="Load the legacy model checkpoint when use mcore models.")
    parser.add_argument('--use_mcore_models', action='store_true', help="Use the legacy code.")
    parser.add_argument('--npu', action='store_true', help="Use NPU.")
    parser.add_argument('--num-vcores', type=int, required=False, default=None, help="numbers of vcores each worker")
    parser.add_argument('--memory', type=int, required=False, default=None, help="memory of each worker GB")
    parser.add_argument('--profiler-on', action='store_true', help="enable profiler")
    args = parser.parse_args()

    if args.image is None:
        args.image = "default"

    # 从队列中抽取hadoop租户
    usergroup = args.usergroup
    if usergroup == None:
        usergroup = args.queue.split('.')[-2]

    # 增加时间戳
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
    if args.taskname == None:
        args.taskname = "NONAME"
    taskname = args.taskname + "-" + formatted_time

    if args.config is not None:
        print("采用单文件配置")
        extra_script = ""
        if 'deepseek-v2' in args.taskname:
            print("Will install transformers==4.39.3 for deepseek-v2")
            extra_script += f"pip3 install transformers==4.39.3 && "
        
        print("Will install timm==0.4.12 for lavit")
        extra_script += f"pip3 install timm==0.4.12 && "
        
        if args.image == "taming":
            with open(args.config, "r") as f:
                configs = json.load(f)
                if "image-mask-type" in configs and "use-flash-attn" in configs:
                    args.image = "taming_multimodal_flash_attn"
        else:
            extra_script += f"pip3 install hope==3.6.5 -i http://pypi.sankuai.com/simple/ --trusted-host pypi.sankuai.com --force-reinstall --no-cache && " \
                 f"pip3 install safetensors && "

        script = extra_script + \
                 f"export MEGATRON_NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 && " + \
            f"python3 examples/auto_pretrain.py " + \
            f"--config {args.config} " + \
            f"--id {taskname} " + \
            f"--run_env {args.run_env} " + \
            f"--usergroup {usergroup} " + \
            f"--param_report {args.param_report} " + \
            f"--hope_tensorboard {args.hope_tensorboard} " + \
            (f"--preprocess_wait_time {args.preprocess_wait_time} " if args.preprocess_wait_time != "" else "") + \
            (f"--pretrain " if args.pretrain else "") + \
            (f"--use_mcore_models " if args.use_mcore_models else "") + \
            (f"--legacy_checkpoint " if args.legacy_checkpoint else "")
    else:
        raise Exception("不再支持多文件配置")

    workers = args.workers
    gpu = args.gpu
    if args.run_env == "tx":
        vcore = gpu * 12
    else:
        vcore = gpu * 10
    if args.npu:
        memory = gpu * 50000
    else:
        memory = gpu * 100000

    if gpu == 0:
        vcore = 48 if args.num_vcores is None else args.num_vcores
        memory = vcore * 4096 if args.memory is None else args.memory * 1024
        print(f"workers = {workers}, GPU = 0, vcore = {vcore}, memory = {memory/1024} GB")
        print("仅使用CPU做数据预处理，建议使用CPU队列，如：root.zw05_training_cluster.hadoop-aipnlp.cpu_job")

    template_hope_path = "submit/template.hope"
    if args.run_env == "tx":
        template_hope_path = "submit/template_tx.hope"
    with open(template_hope_path) as f:
        template = f.read()

    template = template.replace("${queue}", args.queue)
    if args.account is not None:
        print("--account -a参数已经废弃，不再需要填写，会根据hope登陆找到对应的用户名")
    template = template.replace("${usergroup}", usergroup)
    template = template.replace("${script}", script)
    template = template.replace("${gpu}", str(gpu))
    template = template.replace("${vcore}", str(vcore))
    template = template.replace("${memory}", str(memory))
    template = template.replace("${workers}", str(workers))
    template = template.replace("${image}", images[args.image])
    template = template.replace("${max_retry}", str(args.max_retry))
    template = template.replace("${last_worker}", f"worker_{workers-1}")
    template = template.replace("${project_name}", args.project_name)
    template = template.replace("${client_git_revision_publish}", args.client_git_revision_publish)
    if args.profiler_on:
        template = template.replace("${profiler_on}", "true")
        template = template.replace("${profiler_id}", taskname)
    else:
        template = template.replace("${profiler_on}", "false")
        template = template.replace("${profiler_id}", "")

    # gaoxi(20231129): H800机器暂时通过机房识别，未来可能需要采用更科学的办法
    if args.queue.startswith("root.hldy_training_cluster"):
        print("识别到H800队列，自动对提交模版进行适配")
        template = adapt_to_h800(template)

    if args.npu:
        print("使用Ascend 910B 机器")
        template = adapt_to_h910b(template)

    hope_file = f"auto_generate/{taskname}.hope"
    with open(hope_file, "w") as f:
        f.write(template)
    # print(hope_file)

    cmd = "hope run " + hope_file
    result = subprocess.check_call(cmd, shell=True)

    print(f"保存提交hope文件到{hope_file}，使用以下命令继续训练：\n{cmd}")
