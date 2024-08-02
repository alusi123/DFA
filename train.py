import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data import *
from utils.Logger import Logger
from utils.tools import *
from models.classifier import Masker, Predictor

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from utils.loss import SupConLoss

class Trainer:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        
        # networks
        self.encoder_invariant = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.encoder_s0 = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.encoder_s1 = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.encoder_s2 = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.domain_classifier = get_classifier_from_config(self.config["networks"]["domain_classifier"]).to(device)

        self.classifier_superior = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)
        self.classifier_inferior = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)

        feature_dim = self.config["networks"]["classifier"]["in_dim"]
        self.masker = Masker(in_dim = feature_dim, num_classes = feature_dim, middle = 4 * feature_dim, 
                             k = self.config["k"], hard = self.config["hard_mask"]).to(device)
        self.predictor_domain = Predictor(in_dim = feature_dim * 2, out_dim = feature_dim).to(device)
        self.predictor_casual = Predictor(in_dim = feature_dim * 2, out_dim = feature_dim).to(device)
        self.SupConLoss = SupConLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_dc = nn.BCELoss()
        # turn on train mode# define loss function (criterion) and optimizer
        self.criterion_fact = nn.CosineSimilarity(dim=1)
        self.fact = nn.MSELoss()

        # optimizers
        self.encoder_invariant_optim, self.encoder_invariant_sched = \
            get_optim_and_scheduler(self.encoder_invariant, self.config["optimizer"]["encoder_optimizer"])
        self.encoder_s0_optim, self.encoder_s0_sched = \
            get_optim_and_scheduler(self.encoder_s0, self.config["optimizer"]["encoder_optimizer"])
        self.encoder_s1_optim, self.encoder_s1_sched = \
            get_optim_and_scheduler(self.encoder_s1, self.config["optimizer"]["encoder_optimizer"])
        self.encoder_s2_optim, self.encoder_s2_sched = \
            get_optim_and_scheduler(self.encoder_s2, self.config["optimizer"]["encoder_optimizer"])
        self.domain_classifier_optim, self.domain_classifier_sched = \
            get_optim_and_scheduler(self.domain_classifier, self.config["optimizer"]["classifier_optimizer"])
        
        self.classifier_superior_optim, self.classifier_superior_sched = \
            get_optim_and_scheduler(self.classifier_superior, self.config["optimizer"]["classifier_optimizer"])
        self.classifier_inferior_optim, self.classifier_inferior_sched = \
            get_optim_and_scheduler(self.classifier_inferior, self.config["optimizer"]["classifier_optimizer"])
        
        self.masker_optim, self.masker_sched = \
            get_optim_and_scheduler(self.masker, self.config["optimizer"]["classifier_optimizer"])
        
        self.predictor_optim_d, self.predictor_sched_d = \
            get_optim_and_scheduler(self.predictor_domain, self.config["optimizer"]["classifier_optimizer"])
        self.predictor_optim_c, self.predictor_sched_c = \
            get_optim_and_scheduler(self.predictor_casual, self.config["optimizer"]["classifier_optimizer"])
        
        # dataloaders
        # self.train_loader = get_train_dataloader(args=self.args, config=self.config)
        self.train_loader = get_fourier_train_dataloader(args=self.args, config=self.config)
        # self.train_loader = get_fourier_train_dataloader_old(args=self.args, config=self.config)
        self.val_loader = get_val_dataloader(args=self.args, config=self.config)
        self.test_loader = get_test_loader(args=self.args, config=self.config)
        self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}

    def _do_epoch(self):
        self.encoder_invariant.train()
        self.encoder_s1.train()
        self.encoder_s2.train()
        self.encoder_s0.train()
        self.domain_classifier.train()
        self.classifier_superior.train()
        self.classifier_inferior.train()
        self.masker.train()
        self.predictor_domain.train()
        self.predictor_casual.train()

        for it, ((images, label), domains) in enumerate(self.train_loader):

            images = torch.cat(images, dim=0).to(self.device) 
            label = torch.cat(label, dim=0).to(self.device) 
            domains = torch.cat(domains, dim=0).to(self.device) 
            batch_size = images.shape[0]

            if self.args.target in pacs_dataset:
                label -= 1
                
            # zero grad
            self.encoder_invariant_optim.zero_grad()
            self.encoder_s1_optim.zero_grad()
            self.encoder_s2_optim.zero_grad()
            self.encoder_s0_optim.zero_grad()
            self.domain_classifier_optim.zero_grad()
            self.classifier_superior_optim.zero_grad()
            self.classifier_inferior_optim.zero_grad()
            self.masker_optim.zero_grad()
            self.predictor_optim_d.zero_grad()
            self.predictor_optim_c.zero_grad()

            # forward
            loss_dict = {}
            correct_dict = {}
            num_samples_dict = {}
            total_loss = 0.0

            index = torch.arange(batch_size)
            batch0 = images[index[domains == 0]]
            batch1 = images[index[domains == 1]]
            batch2 = images[index[domains == 2]]
            labels0 = label[index[domains == 0]]
            labels1 = label[index[domains == 1]]
            labels2 = label[index[domains == 2]]
            labels = torch.cat((labels0, labels1, labels2), dim=0)
            num0 = len(batch0)
            num1 = len(batch1)
            num2 = len(batch2)



            batch = torch.cat((batch0, batch1, batch2), dim=0)
            features_invariant = self.encoder_invariant(batch) 
            ########################################## step 1 ########################################
            
            features_s0 = self.encoder_s0(batch0) 
            features_s1 = self.encoder_s1(batch1) 
            features_s2 = self.encoder_s2(batch2) 
            features_specific = torch.cat((features_s0, features_s1, features_s2), dim=0)

            # domain classifier
            scores_spe = self.domain_classifier(features_specific, False) 

            domain_label = torch.cat((torch.zeros(num0), torch.ones(num1), torch.ones(num2) * 2), dim=0).to(self.device)

            loss_domain_s = self.criterion(scores_spe, domain_label.long())

            loss_spe = loss_domain_s

        
            loss_dict["domain_spe"] = loss_spe.item()
            
            
            total_loss += loss_spe

            # backward
            total_loss.backward()

            # update
            self.encoder_s1_optim.step()
            self.encoder_s2_optim.step()
            self.encoder_s0_optim.step()
            self.domain_classifier_optim.step()

            ####################################### step 2 ########################################
            total_loss = 0.0

            scores_inv = self.domain_classifier(features_invariant, False) # [2*bs, 3]
            prob = F.softmax(scores_inv, dim=-1)
            log_prob = F.log_softmax(scores_inv, dim=-1)
            entorpy = torch.sum(prob * log_prob, dim=-1)
            loss_domain_i = entorpy.mean()
            loss_inv = (loss_domain_i) + 3 / torch.e


            invariant_weight = self.config["inv_warmup"]["lam_const"]
            if self.config["inv_warmup"]["warmup"] is True:
                invariant_weight = get_current_consistency_weight(epoch=self.current_epoch,
                                                        weight=self.config["inv_warmup"]["lam_const"],
                                                        rampup_length=self.config["inv_warmup"]["warmup_epoch"],
                                                        rampup_type=self.config["inv_warmup"]["warmup_type"])
                
            loss_dict["domain_inv"] = loss_inv.item()
            loss_inv = loss_inv * invariant_weight
            loss_dict["/i"] = loss_inv.item()
            
            total_loss += loss_inv

            # print domain classifier loss
            if it % 30 == 0:
                s_i1 = torch.softmax(scores_inv, dim=1)
                s_s1 = torch.softmax(scores_spe, dim=1)
                for i in range(1):
                    self.logger.print_log(f" ")
                    self.logger.print_log(f"num domain sample : 0:{num0}, 1:{num1}, 2:{num2}")
                    self.logger.print_log(f"Domain Invariant : {s_i1[0][0]:.5f}, {s_i1[0][1]:.5f}, {s_i1[0][2]:.5f}")
                    self.logger.print_log(f"Domain Specific : {s_s1[0][0]:.5f}, {s_s1[0][1]:.5f}, {s_s1[0][2]:.5f}")

            # Mask
            masks_superior = self.masker(features_invariant.detach())
            masks_inferior = torch.ones_like(masks_superior) - masks_superior
            if self.current_epoch <= 5:
                masks_superior = torch.ones_like(features_invariant.detach())
                masks_inferior = torch.ones_like(features_invariant.detach())
            # masked features
            features_superior = features_invariant * masks_superior
            features_inferior = features_invariant * masks_inferior
            # classifier
            scores_superior = self.classifier_superior(features_superior)
            scores_inferior = self.classifier_inferior(features_inferior)

            # classification loss for superior feature
            loss_cls_superior = self.criterion(scores_superior, labels)
            loss_dict["superior"] = loss_cls_superior.item()
            correct_dict["superior"] = calculate_correct(scores_superior, labels)
            num_samples_dict["superior"] = int(scores_superior.size(0))

            # classification loss for inferior feature
            loss_cls_inferior = self.criterion(scores_inferior, labels)
            loss_dict["inferior"] = loss_cls_inferior.item()
            correct_dict["inferior"] = calculate_correct(scores_inferior, labels)
            num_samples_dict["inferior"] = int(scores_inferior.size(0))

            total_loss += loss_cls_superior + loss_cls_inferior
            
            if self.current_epoch > 5 and (self.config["cl"]["cd"] or self.config["cl"]["cr"]):

                # domain related hard feature
                feature_hard_domain = torch.zeros((batch_size, 1024)).to(self.device)
                p = F.softmax(scores_spe, dim=1)
                imformation_entropy = - p * torch.log(p)
                ie = torch.min(imformation_entropy, dim=1)[0]
                for i in range(num0):
                    p0 = ie[i]
                    p1 = ie[i + num0]
                    p2 = ie[i + num0 + num1]
                    seleceted_index0 = 1 if p1 < p2 else 2
                    seleceted_index1 = 0 if p0 < p2 else 2
                    seleceted_index2 = 0 if p0 < p1 else 1
                    feature_hard_domain[i] = torch.cat([features_superior[i], features_specific[i + num0 * seleceted_index0].detach()])
                    feature_hard_domain[i + num0] = torch.cat([features_superior[i + num0], features_specific[i + num0 * seleceted_index1].detach()])
                    feature_hard_domain[i + num0 + num1] = torch.cat([features_superior[i + num0 + num1], features_specific[i + num0 * seleceted_index2].detach()])

                feature_hard_domain = self.predictor_domain(feature_hard_domain)
                loss_cl1 = 0
                if self.config["cl"]["cd"]:
                    loss_cl1 = self.SupConLoss(torch.cat([features_invariant.unsqueeze(1), feature_hard_domain.unsqueeze(1)], dim=1), labels)

                # causal related hard feature
                feature_hard_causal = torch.zeros((batch_size, 1024)).to(self.device)
                prob_superior = F.softmax(scores_superior, dim=-1)

                for i in range(len(labels)):
                    prob_superior[i, labels[i]] = 0
                
                aug_class = torch.max(prob_superior, dim=1)[1] 

                prob_inferior = F.softmax(scores_inferior, dim=-1)
                imformation_entropy = - prob_inferior * torch.log(prob_inferior)

                select_index0 = torch.min(imformation_entropy[:num0], dim=0)[1] 
                select_index1 = torch.min(imformation_entropy[num0:num0+num1], dim=0)[1] 
                select_index2 = torch.min(imformation_entropy[num0+num1:], dim=0)[1] 

                for i in range(num0):
                    index0 = select_index0[aug_class[i]]
                    index1 = select_index1[aug_class[i+num0]]
                    index2 = select_index2[aug_class[i+num0+num1]]
                    feature_hard_causal[i] = torch.cat([features_superior[i], features_inferior[index0].detach()])
                    feature_hard_causal[i + num0] = torch.cat([features_superior[i + num0], features_inferior[index1 + num0].detach()])
                    feature_hard_causal[i + num0 + num1] = torch.cat([features_superior[i], features_inferior[index2 + num0 + num1].detach()])

                feature_hard_causal = self.predictor_casual(feature_hard_causal)
                loss_cl2 = 0
                if self.config["cl"]["cr"]:
                    loss_cl2 = self.SupConLoss(torch.cat([features_invariant.unsqueeze(1), feature_hard_causal.unsqueeze(1)], dim=1), labels)

                loss_cl = loss_cl1 + loss_cl2
                
                loss_dict["cl"] = loss_cl.item()
                if self.config["cl_warmup"]["warmup"] is True:
                    const_weight_cl = get_current_consistency_weight(epoch=self.current_epoch,
                                                                        weight=self.config["cl_warmup"]["lam_const"],
                                                                        rampup_length=self.config["cl_warmup"]["warmup_epoch"],
                                                                        rampup_type=self.config["cl_warmup"]["warmup_type"])
                else:
                    const_weight_cl = self.config["cl_warmup"]["lam_const"]
                loss_cl = loss_cl * const_weight_cl
                loss_dict["/cl"] = loss_cl.item()
                total_loss += loss_cl


            loss_dict["total"] = total_loss.item()

            # backward
            total_loss.backward()

            # update
            self.encoder_invariant_optim.step()
            self.classifier_superior_optim.step()
            self.classifier_inferior_optim.step()
            self.predictor_optim_c.step()
            self.predictor_optim_d.step()

            ##################################### step 3 #####################################
            self.masker_optim.zero_grad()
            features_invariant = self.encoder_invariant(batch)
            masks_superior = self.masker(features_invariant.detach())
            masks_inferior = torch.ones_like(masks_superior) - masks_superior
            features_superior = features_invariant * masks_superior
            features_inferior = features_invariant * masks_inferior
            scores_superior = self.classifier_superior(features_superior)
            scores_inferior = self.classifier_inferior(features_inferior)


            loss_cls_superior = self.criterion(scores_superior, labels)
            loss_cls_inferior = self.criterion(scores_inferior, labels)
            total_loss = 0.5*loss_cls_superior - 0.5*loss_cls_inferior

            total_loss.backward()
            self.masker_optim.step()

            self.global_step += 1

            # record
            self.logger.log(
                it=it,
                iters=len(self.train_loader),
                losses=loss_dict,
                samples_right=correct_dict,
                total_samples=num_samples_dict
            )

        # turn on eval mode
        self.encoder_invariant.eval()
        self.classifier_superior.eval()
        self.masker.eval()

        # evaluation
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct = self.do_eval(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {'class': class_acc})
                self.results[phase][self.current_epoch] = class_acc

            # save from best model
            if self.results['test'][self.current_epoch] >= self.best_acc:
                self.best_acc = self.results['test'][self.current_epoch]
                self.best_epoch = self.current_epoch + 1
                self.logger.save_best_model(self.encoder_invariant, self.classifier_superior, self.best_acc)

    def do_eval(self, loader):
        correct = 0
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            features = self.encoder_invariant(data)
            scores = self.classifier_superior(features)
            correct += calculate_correct(scores, labels)
        return correct


    def do_training(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)
        self.logger.save_config()
        
        self.epochs = self.config["epoch"]
        self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}

        self.best_acc = 0
        self.best_epoch = 0

        for self.current_epoch in range(self.epochs):

            self.encoder_invariant_sched.step()
            self.encoder_s1_sched.step()
            self.encoder_s2_sched.step()
            self.encoder_s0_sched.step()
            self.domain_classifier_sched.step()
            self.classifier_superior_sched.step()
            self.classifier_inferior_sched.step()
            self.masker_sched.step()
            self.predictor_sched_c.step()
            self.predictor_sched_d.step()


            self.logger.new_epoch([group["lr"] for group in self.encoder_invariant_optim.param_groups])
            self._do_epoch()
            self.logger.finish_epoch()

        # save from best model
        val_res = self.results['val']
        test_res = self.results['test']
        self.logger.save_best_acc(val_res, test_res, self.best_acc, self.best_epoch - 1)

        return self.logger

def get_args():
    source = ["photo", "cartoon", "art_painting", "sketch"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=[source[1], source[2], source[3]], choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", default=source[0], choices=available_datasets, help="Target")

    parser.add_argument("--input_dir", default= 'dataset/PACS/datalists', help="The directory of dataset lists")
    parser.add_argument("--image_dir", default= 'dataset/PACS/kfold/', help="The directory of image lists")
    parser.add_argument("--output_dir", default='outputs', help="The directory to save logs and models")
    parser.add_argument("--config", default="PACS/ResNet18", help="Experiment configs")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True, help="If true will save tensorboard compatible logs")

    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--exp_name", default=" ", type=str, help="exp name")

    args = parser.parse_args()
    meta_name = args.config.replace("/", "_")
    args.output_dir = os.path.join(args.output_dir, meta_name)
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    print(f"\nOutput Dir : {args.output_dir}")
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config

def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("invariant weight:", config["inv_warmup"]["lam_const"])
    print("contrastive cd:", config["cl"]["cd"])
    print("contrastive cr:", config["cl"]["cr"])
    print("cl weight:", config["cl_warmup"]["lam_const"])

    if args.seed is not None:
        seed = config["seeds"][args.target]
        torch.manual_seed(seed)
        setup_seed()
    else:
        seed = torch.initial_seed()
        torch.manual_seed(seed)
        setup_seed()
    config["seed"] = seed
    print("seed:", seed)

    trainer = Trainer(args, config, device)
    trainer.do_training()

def setup_seed(seed = 11):
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
