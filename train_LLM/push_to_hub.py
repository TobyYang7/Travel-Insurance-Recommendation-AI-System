from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'ef53e3ed-cd68-4c13-aa47-1c12bf7f1dbc'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="TobyYang7/InsuranceGPT",
    model_dir="/home/zhangmin/toby/IBA_Project_24spr/exp_model/InsuranceGPT"
)
