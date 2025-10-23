import timm
from pprint import pprint

model_names = timm.list_models(pretrained=True)
print("支持的预训练模型数量：%s" % len(model_names))

strs = '*resne*t*'
model_names = timm.list_models(strs)
print("通过通配符 %s 查询到的可用模型：%s" % (strs, len(model_names)))

model_names = timm.list_models(strs, pretrained=True)
print("通过通配符 %s 查询到的可用预训练模型：%s" % (strs, len(model_names)))