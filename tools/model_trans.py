import linger
import torch
import torch.nn as nn
from transformer_model import transformer



src = torch.randint(1, (1, 32), device='cuda:0')
src_mask = None
tgt = torch.randint(1, (1, 32), device='cuda:0')
tgt_mask = None


net = transformer().cuda()
linger.SetFunctionBmmQuant(True)
replace_tuple = (nn.Linear, nn.Embedding, nn.LayerNorm)
net = linger.init(net, quant_modules=replace_tuple, mode=linger.QuantMode.QValue)
net.load_state_dict(torch.load("/data2/user/dwzhang/offline_translation/checkpoints/model_quant_epoch_16.pth"), False)
net.eval()

def trans_encoder():
    with torch.no_grad():
        torch.onnx.export(net.encoder, 
                        (src, src_mask), 
                        '/data2/user/dwzhang/offline_translation/checkpoints/encoder.onnx',
                        export_params=True, 
                        opset_version=12, 
                        do_constant_folding=True, 
                        input_names = ['en_input'],
                        output_names = ['memory'],
                        dynamic_axes = {'en_input': {1:'en_len'},
                                        },
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    print("Finish_trans_encoder!")

def trans_decoder():
    with torch.no_grad():
        enc_out = net.encoder(src, src_mask)
        decoder_input = (tgt, enc_out, src_mask, tgt_mask) 
        torch.onnx.export(net.decoder, 
                        decoder_input, 
                        '/data2/user/dwzhang/offline_translation/checkpoints/decoder.onnx',
                        export_params=True, 
                        opset_version=12, 
                        do_constant_folding=True, 
                        input_names = ['de_input', 'memory'],
                        output_names = ['prob'],
                        dynamic_axes = {'de_input': {1:'de_len'},
                                        'memory':{1:'memory_len'}
                                        },
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    print("Finish_trans_decoder!")

if __name__ == "__main__":
    trans_encoder()
    trans_decoder()

# tpacker -g encoder.onnx -d True -c en_len=32 -o encoder.bin
# tpacker -g decoder.onnx -d True -c de_len=32,memory_len=32 -o decoder.bin