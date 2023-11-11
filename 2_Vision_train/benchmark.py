from ultralytics.utils.benchmarks import benchmark
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# V5 ----------------------------------------------------------------------------------------------------------------------------------------------------------------


#python benchmarks.py --weights v5m_100.pt --img 640 --data custom.yaml  --device 0
# map50 = 0.189, map50:95 = 0.0608


#python benchmarks.py --weights v5x_100.pt --img 640 --data custom.yaml  --device 0
# map50 = 0.983, map50:95 = 0.63


# V7 ----------------------------------------------------------------------------------------------------------------------------------------------------------------


#python test.py --data custom.yaml  --device 0 --img 640 --batch 32 --device 0 --weights v7x_100.pt
# map50 = 0.98, map50:95 = 0.6


# V8 ----------------------------------------------------------------------------------------------------------------------------------------------------------------


#benchmark(model='C:/Users/admin/Desktop/V8/v8s_50_32_v1.pt', data='C:/Users/admin/Desktop/data_V3/data.yaml', imgsz=640, half=False, device=0)
# map50 = 0.153, map50:95 = 0.0632


#benchmark(model='C:/Users/admin/Desktop/V8/v8s_50_32_v2.pt', data='C:/Users/admin/Desktop/data_V3/data.yaml', imgsz=640, half=False, device=0)
# map50 = 0.977, map50:95 = 0.653


#benchmark(model='C:/Users/admin/Desktop/V8/v8s_50_32_v3.pt', data='C:/Users/admin/Desktop/data_V3/data.yaml', imgsz=640, half=False, device=0)
# map50 = 0.981, map50:95 = 0.654