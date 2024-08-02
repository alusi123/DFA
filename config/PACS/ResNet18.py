config = {}

batch_size = 16
epoch = 50


lr = 0.001
lr_decay_rate = 0.1

num_classes = 7
num_domains = 3
config["T"] = 10.0
config["k"] = 308
config["dc"] = True

config["seed"] = None
config["hard_mask"] = False


config["batch_size"] = batch_size
config["epoch"] = epoch
config["num_classes"] = num_classes
config["num_domains"] = num_domains


inv_warmup = {
    "warmup": True,
    "lam_const": 1,
    "warmup_epoch": 5,
    "warmup_type": "sigmoid",
}
cl = {
    "cd": True,
    "cr": True
}
cl_warmup = {
    "warmup": True,
    "lam_const": 0.001,
    "warmup_epoch": 5,
    "warmup_type": "sigmoid",
}
config["inv_warmup"] = inv_warmup
config["cl_warmup"] = cl_warmup
config["cl"] = cl


# data configs
data_opt = {
    "image_size": 224,
    "use_crop": True,
    "jitter": 0.4,
    "from_domain": "all",
    "alpha": 1.0,
    "random": True
}

config["data_opt"] = data_opt


# network configs
networks = {}

encoder = {
    "name": "resnet18",
}
networks["encoder"] = encoder

classifier = {
    "name": "base",
    "in_dim": 512,
    "num_classes": num_classes
}
networks["classifier"] = classifier

domain_classifier = {
    "name": "domain_classifier",
    "in_dim": 512,
    "num_classes": num_domains
}
networks["domain_classifier"] = domain_classifier

predictor = {
    "name": "predictor",
    "in_dim": 512,
    "pre_dim": 512
}
networks["predictor"] = predictor

config["networks"] = networks

# optimizer configs
optimizer = {}

encoder_optimizer = {
    "optim_type": "sgd",
    "lr": lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "step",
    "lr_decay_step": int(epoch * 0.8),
    "lr_decay_rate": lr_decay_rate
}
optimizer["encoder_optimizer"] = encoder_optimizer

classifier_optimizer = {
    "optim_type": "sgd",
    "lr": 10*lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "step",
    "lr_decay_step": int(epoch * 0.8),
    "lr_decay_rate": lr_decay_rate
}
optimizer["classifier_optimizer"] = classifier_optimizer

config["optimizer"] = optimizer

seeds = {
    'art_painting': '7015728724524346394',
    'cartoon': '14757653597613772096',
    'photo': '5952586897060022513',
    'sketch': '9619485306267467659'
}
config["seeds"] = seeds