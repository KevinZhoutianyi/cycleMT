import torch
from utils import *
from cycle import *
from basic_model import *
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from datasets import load_metric
@torch.no_grad()
def my_test(loader,model,tokenizer,logging,wandb):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GAB_acc = 0
    GBA_acc = 0
    counter = 0
    GAB_metric_sacrebleu =  load_metric('sacrebleu')
    GBA_metric_sacrebleu =  load_metric('sacrebleu')
    GAB = model.G_AB
    GBA = model.G_BA
    DA = model.D_A
    DB = model.D_B
    GAB.eval()
    GBA.eval()
    DA.eval()
    DB.eval()
    for step,batch in enumerate(loader):
        a = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)#en
        a_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        b = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)#de    
        b_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)


        
        GAB_loss = GAB.forward(a,a_attn,b,b_attn).loss
        GBA_loss = GBA.forward(b,b_attn,a,a_attn).loss
        GAB_acc+= GAB_loss.item()
        GBA_acc+= GBA_loss.item()
        counter+= 1

        a_generate = GAB.test_generate(a)[:,1:].contiguous()
        b_generate  = GBA.test_generate(b)[:,1:].contiguous()



        a_label_decoded = tokenizer.batch_decode(a,skip_special_tokens=True)
        b_label_decoded =  tokenizer.batch_decode(b,skip_special_tokens=True)
        a_pred_decoded = tokenizer.batch_decode(a_generate,skip_special_tokens=True)
        b_pred_decoded = tokenizer.batch_decode(b_generate,skip_special_tokens=True)


        b_pred_str = [x  for x in b_pred_decoded]
        a_label_str = [[x] for x in a_label_decoded]
        a_pred_str = [x  for x in a_pred_decoded]
        b_label_str = [[x] for x in b_label_decoded]

        GAB_metric_sacrebleu.add_batch(predictions=a_pred_str, references=b_label_str)
        GBA_metric_sacrebleu.add_batch(predictions=b_pred_str, references=a_label_str)

        if  step%100==0:
            # logging.info(f"a{a}")
            # logging.info(f"a_attn{a_attn}")
            # logging.info(f"b{b}")
            # logging.info(f"b_attn{b_attn}")
            # logging.info(f"a_generate{a_generate}")
            # logging.info(f"(a_generate>0.5).long()){(a_generate>0.5).long()}")
            # logging.info(f"b_generate{b_generate}")
            # logging.info(f"(b_generate>0.5).long(){(b_generate>0.5).long()}")
            
            b_dis  = DA(b,b_attn)
            a_dis  = DB(a,a_attn)
            a_pred_dis  = DA(a_generate,(a_generate>0.5).long())
            b_pred_dis  = DB(b_generate,(b_generate>0.5).long())
            logging.info("DB_a_: {}".format(''.join(map(lambda x: str(x.item())[:5]+',  ', a_dis))))
            logging.info("DB_pred_dis: {}".format(''.join(map(lambda x:  str(x.item())[:5]+',  ', b_pred_dis))))
            logging.info("DA_b: {}".format(''.join(map(lambda x:  str(x.item())[:5]+',  ', b_dis))))
            logging.info("DA_pred_dis: {}".format(''.join(map(lambda x:  str(x.item())[:5]+',  ', a_pred_dis))))

            logging.info(f'GABloss:\t{GAB_loss.item()}')
            logging.info(f'GBAloss:\t{GBA_loss.item()}')
            logging.info(f'a_decoded[:2]:{a_label_decoded[:2]}')
            logging.info(f'pred_b_decoded[:2]:{b_pred_decoded[:2]}')
            logging.info(f'b_decoded[:2]:{b_label_decoded[:2]}')
            logging.info(f'pred_a_decoded[:2]:{a_pred_decoded[:2]}')

    logging.info('computing score...') 
    GAB_sacrebleu_score = GAB_metric_sacrebleu.compute()
    GBA_sacrebleu_score = GBA_metric_sacrebleu.compute()
    logging.info('%s GAB sacreBLEU : %f',GAB.name,GAB_sacrebleu_score['score'])#TODO:bleu may be wrong cuz max length
    logging.info('%s GBA sacreBLEU : %f',GBA.name,GBA_sacrebleu_score['score'])#TODO:bleu may be wrong cuz max length
    logging.info('%s GAB test loss : %f',GAB.name,GAB_acc/(counter))
    logging.info('%s GBA test loss : %f',GBA.name,GBA_acc/(counter))
    wandb.log({'GAB sacreBLEU':GAB_sacrebleu_score['score']})
    wandb.log({'GBA sacreBLEU':GBA_sacrebleu_score['score']})
    wandb.log({'GAB test loss':GAB_acc/(counter)})
    wandb.log({'GBA test loss':GBA_acc/(counter)})