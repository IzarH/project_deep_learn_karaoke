from openunmix12.model import OpenUnmix

dummy_model=OpenUnmix()

pytorch_total_params = sum(p.numel() for p in dummy_model.parameters())

print("num of weights modified: ",pytorch_total_params)

# how many weights (trainable parameters) we have in our model? 
num_trainable_params = sum([p.numel() for p in dummy_model.parameters() if p.requires_grad]) 

print("num trainable weights modified: ", num_trainable_params)
