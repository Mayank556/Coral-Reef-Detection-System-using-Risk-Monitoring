import torch

def check_nans():
    state = torch.load('./outputs/best_model.pth', map_location='cpu', weights_only=True)
    has_nan = False
    for k, v in state.items():
        if torch.isnan(v).any():
            print(f"NaN found in {k}")
            has_nan = True
        if torch.isinf(v).any():
            print(f"Inf found in {k}")
            has_nan = True
            
    if not has_nan:
        print("No NaNs or Infs found in the model weights/buffers!")

if __name__ == '__main__':
    check_nans()
