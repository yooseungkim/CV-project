import timm.layers.attention
import inspect

print(inspect.getsource(timm.layers.attention.Attention.forward))
