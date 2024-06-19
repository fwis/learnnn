import torch
from model import ForkNet, ResNet
 
model = ResNet()
checkpoint = torch.load(r'T3\Frame\ckpt\ResNet2.pth')
model.eval()
dummy_input = torch.randn(1, 1, 100, 100)

torch.onnx.export(model, dummy_input, r'T3\Frame\ckpt\ResNet2.onnx',verbose=True)

# 多输入,opset_version根据torch的版本决定，转onnx的时候提示的
# torch.onnx.export(model, [dummy_input,dummy_input_2], 'name.onnx',verbose=True,opset_version=9/10/11)
 
 
# 指定输入和输出导出onnx
# input_names = [ "input_1"]
# output_names = [ "output1" ]
# torch.onnx.export(model, (dummy_input1, dummy_input2), "name.onnx", verbose=True, input_names=input_names, output_names=output_names)
 