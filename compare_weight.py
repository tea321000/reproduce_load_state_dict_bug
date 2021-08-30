import oneflow.nn as nn
import torch


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))

    def forward(self, x):
        return self.proj(x)


class PtStem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))

    def forward(self, x):
        return self.proj(x)


def oneflow_weight_load(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    for name, weights in state_dict.items():
        npy = weights.detach().numpy()
        state_dict[name] = weights.detach().numpy()
    model.load_state_dict(state_dict)
    of_state_dict = model.state_dict()
    # for name, _ in state_dict.items():
    #     assert numpy.allclose(of_state_dict[name].numpy(), state_dict[name], rtol=1e-06, atol=1e-06)
    print("Load pretrained weights from {}".format(checkpoint_path))
    return of_state_dict


def torch_weight_load(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    return model.state_dict()


def compare(torch_dict, of_dict):
    for torch_key, of_key in zip(torch_dict, of_dict):
        value = abs(torch_dict[torch_key].numpy() - of_dict[of_key].numpy())
        print(value.mean())
        if abs(value.mean()) > 0.0000001:
            print(torch_key)


if __name__ == '__main__':
    of_model = Stem()
    of_state = oneflow_weight_load(of_model, "new.pt")
    pt_model = PtStem()
    pt_state = torch_weight_load(pt_model, "new.pt")
    # compare weight
    compare(pt_state, of_state)
