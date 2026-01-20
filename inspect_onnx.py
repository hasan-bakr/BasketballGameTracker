
import onnxruntime as ort
import os

decoder_path = r"models/sam2_onnx/decoder.onnx"
encoder_path = r"models/sam2_onnx/encoder.onnx"

print(f" Inspecting: {decoder_path}")

try:
    sess = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])
    print("\n🔍 Decoder Inputs:")
    for inp in sess.get_inputs():
        print(f"   Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
    
    print("\n🔍 Encoder Inputs:")
    sess_enc = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
    for inp in sess_enc.get_inputs():
        print(f"   Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
        
except Exception as e:
    print(f"Error: {e}")
