import torch

# define a floating point model
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x
torch.backends.quantized.engine = 'qnnpack'
# create a model instance
model_fp32 = M()
# create a quantized model instance
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

# run the model
input_fp32 = torch.randn(4, 4, 4, 4)
res = model_int8(input_fp32)
print(res)



# backend = "qnnpack"
# torch.backends.quantized.engine = backend

# def print_model_size(mdl):
#     torch.save(mdl.state_dict(), "tmp.pt")
#     print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
#     os.remove('tmp.pt')

# dtype=torch.float32
# inputs=(torch.rand((2, 4, 32, 32),dtype=dtype),torch.rand(2,dtype=dtype),torch.rand((2,77,768),dtype=dtype))
# inputs=torch.rand(2, 3, 256, 256)

# model.qconfig = torch.quantization.get_default_qconfig(backend)
# model_static_quantized = torch.quantization.prepare(model, inplace=False)
# model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
# model_static_quantized.eval()
# model_static_quantized(inputs)



# scripted_model = torch.jit.trace(model_static_quantized, inputs)
# print_model_size(scripted_model)
# model_static_quantized = optimize_for_mobile(model_static_quantized, backend='cpu')
# print(torch.jit.export_opnames(model_static_quantized))
# model_static_quantized._save_for_lite_interpreter('./ccc.pt')

