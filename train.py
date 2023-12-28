import time
import torch
import os
from parser import args
from lib.loss import SimpleLossCompute

start = time.time()
train_loss_list = []
evaluate_loss_list = []

def run_epoch(data, model, loss_compute, epoch, train=True):
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i , batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            totle_time = '%d h %d min %d s'%(hours, minutes, seconds)
            if train:
                print("Epoch:%d, Batch:%3d, Loss:%.3f, Time_use:%s, lr:%.6f" %
                    (epoch+1, i - 1, loss / batch.ntokens, totle_time, loss_compute.opt.rate()))
            else:
                print("Epoch:%d, Batch:%3d, Loss:%.3f, Time_use:%s"%
                    (epoch+1, i - 1, loss / batch.ntokens, totle_time))
            tokens = 0

    return total_loss / total_tokens


def train(data, model, criterion, optimizer, quant=False):
    for epoch in range(args.epochs):
        
        model.train()
        train_loss = run_epoch(data.train_data, model, SimpleLossCompute(criterion, optimizer), epoch)
        train_loss_list.append(train_loss.item())
        if not quant:
            model.eval()
            print('>>>>> Evaluate')
            min_evaluate_loss = 1e9
            evaluate_loss = run_epoch(data.dev_data, model, SimpleLossCompute(criterion, None), epoch, False)
            if evaluate_loss < min_evaluate_loss:
                save_path = os.path.join(args.save_file, 'model_best.pth')
                torch.save(model.state_dict(), save_path)
                min_evaluate_loss = evaluate_loss
            evaluate_loss_list.append(evaluate_loss.item())
            print('<<<<< Evaluate loss: %f' % evaluate_loss)

            if (epoch+1)%25 == 0:
                save_path = os.path.join(args.save_file, 'model_epoch%d.pth'%(epoch+1))
                torch.save(model.state_dict(), save_path)
        else:
            min_train_loss = 1e9
            if train_loss < min_train_loss:
                save_path = os.path.join(args.save_file, 'model_quant_best.pth')
                torch.save(model.state_dict(), save_path)
                min_train_loss = train_loss

            if (epoch+1)%4 == 0:
                save_path = os.path.join(args.save_file, 'model_quant_epoch_%d.pth'%(epoch+1))
                torch.save(model.state_dict(), save_path)


    with open("./loss_save/train_loss.txt", 'w') as tl:
        tl.write(str(train_loss_list))
    with open("./loss_save/evaluate_loss.txt", 'w') as el:
        el.write(str(evaluate_loss_list))